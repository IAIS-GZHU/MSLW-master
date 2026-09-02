from collections.abc import Sequence

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel


class MSLW_Pipeline:

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder,
        tokenizer,
        unet: UNet2DConditionModel,
        scheduler: DDIMScheduler,
        watermark_generator,
        device,
        dtype=torch.float16,
    ):
        if not isinstance(scheduler, DDIMScheduler):
            raise TypeError("scheduler must be an instance of DDIMScheduler")
        if scheduler.config.prediction_type not in {"epsilon", "v_prediction"}:
            raise ValueError(
                "DDIM prediction_type must be 'epsilon' (usually SD1) or "
                "'v_prediction' (usually SD2)"
            )

        self.device = torch.device(device)
        self.dtype = dtype
        self.vae = vae.to(device=self.device, dtype=dtype).eval()
        self.text_encoder = text_encoder.to(device=self.device, dtype=dtype).eval()
        self.unet = unet.to(device=self.device, dtype=dtype).eval()
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.watermark_generator = watermark_generator.to(
            device=self.device, dtype=dtype
        ).eval()
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @staticmethod
    def _wavelet_filter(x: torch.Tensor) -> torch.Tensor:
        """Keep the Haar LL component without moving tensors off the device."""
        if x.ndim != 4 or x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError("watermark embeddings must be BCHW tensors with even dimensions")
        low_frequency = F.avg_pool2d(x, kernel_size=2, stride=2)
        return F.interpolate(low_frequency, scale_factor=2, mode="nearest")

    @staticmethod
    def _prompt_list(prompts, name: str) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        elif isinstance(prompts, Sequence):
            prompts = list(prompts)
        else:
            raise TypeError(f"{name} must be a string or a sequence of strings")
        if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError(f"{name} must contain at least one string")
        return prompts

    def _encode_prompt(self, prompts, cfg_scale, negative_prompts=None):
        prompts = self._prompt_list(prompts, "prompts")
        batch_size = len(prompts)

        if negative_prompts is None:
            negative_prompts = [""] * batch_size
        else:
            negative_prompts = self._prompt_list(negative_prompts, "negative_prompts")
            if len(negative_prompts) == 1 and batch_size > 1:
                negative_prompts *= batch_size
            if len(negative_prompts) != batch_size:
                raise ValueError("negative_prompts must match the prompts batch size")

        def encode(texts):
            tokens = self.tokenizer(
                texts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(self.device)
            attention_mask = None
            if getattr(self.text_encoder.config, "use_attention_mask", False):
                attention_mask = tokens.attention_mask.to(self.device)
            return self.text_encoder(
                input_ids,
                attention_mask=attention_mask,
                return_dict=False,
            )[0].to(dtype=self.dtype)

        prompt_embeds = encode(prompts)
        negative_prompt_embeds = encode(negative_prompts) if cfg_scale > 1.0 else None
        return prompt_embeds, negative_prompt_embeds

    def _validate_inputs(
        self, prompts, msg, tau, phi, height, width, latents, cfg_scale, num_steps
    ):
        prompts = self._prompt_list(prompts, "prompts")
        batch_size = len(prompts)
        latent_shape = (
            batch_size,
            self.unet.config.in_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )

        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        if height % self.vae_scale_factor or width % self.vae_scale_factor:
            raise ValueError(f"height and width must be divisible by {self.vae_scale_factor}")
        if msg is not None and (msg.ndim != 2 or msg.shape[0] != batch_size):
            raise ValueError("msg must have shape (batch_size, bit_length)")
        if latents is not None and tuple(latents.shape) != latent_shape:
            raise ValueError(f"latents must have shape {latent_shape}")
        return prompts, latent_shape

    @torch.inference_mode()
    def sample(
        self,
        prompts,
        msg,
        tau=100,
        phi=0.6,
        height=512,
        width=512,
        latents=None,
        negative_prompts=None,
        cfg_scale=7.5,
        num_steps=50,
        generator=None,
    ):
        prompts, latent_shape = self._validate_inputs(
            prompts, msg, tau, phi, height, width, latents, cfg_scale, num_steps
        )
        self.scheduler.set_timesteps(num_steps, device=self.device)

        prompt_embeds, negative_prompt_embeds = self._encode_prompt(
            prompts, cfg_scale, negative_prompts
        )
        do_cfg = cfg_scale > 1.0
        embeds = (
            torch.cat([negative_prompt_embeds, prompt_embeds])
            if do_cfg
            else prompt_embeds
        )

        if msg is None:
            w_emb = torch.zeros(latent_shape, device=self.device, dtype=self.dtype)
            w_emb_f = w_emb
        else:
            msg = msg.to(device=self.device, dtype=self.dtype)
            w_emb = self.watermark_generator(msg * 2.0 - 1.0).to(dtype=self.dtype)
            if tuple(w_emb.shape) != latent_shape:
                raise ValueError(
                    "watermark generator output has shape "
                    f"{tuple(w_emb.shape)}, expected {latent_shape}"
                )
            w_emb_f = self._wavelet_filter(w_emb)

        if latents is None:
            latents = torch.randn(
                latent_shape,
                generator=generator,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            latents = latents.to(device=self.device, dtype=self.dtype)
        latents = latents * self.scheduler.init_noise_sigma

        for timestep in self.scheduler.timesteps:
            model_input = torch.cat([latents, latents]) if do_cfg else latents
            model_input = self.scheduler.scale_model_input(model_input, timestep)
            model_output = self.unet(
                model_input,
                timestep,
                encoder_hidden_states=embeds,
                return_dict=False,
            )[0]

            if do_cfg:
                output_uncond, output_text = model_output.chunk(2)
                model_output = output_uncond + cfg_scale * (output_text - output_uncond)

            latents = self.scheduler.step(
                model_output,
                timestep,
                latents,
                eta=0.0,
                generator=generator,
                return_dict=True,
            ).prev_sample

            timestep_value = int(timestep.item())
            if msg is not None and timestep_value < tau:
                previous_timestep = (
                    timestep_value
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
                )
                if previous_timestep >= 0:
                    alpha_previous = self.scheduler.alphas_cumprod[previous_timestep]
                else:
                    alpha_previous = self.scheduler.final_alpha_cumprod
                latents = latents + (
                    alpha_previous.to(device=self.device, dtype=self.dtype).sqrt()
                    * phi
                    * w_emb_f
                )

        latents = latents + w_emb
        images = self.vae.decode(
            latents / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        return (images / 2 + 0.5).clamp(0, 1)

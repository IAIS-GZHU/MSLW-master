import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
from lpips import LPIPS


class ModelWrapper(pl.LightningModule):

    def __init__(
        self,
        vae,
        generator,
        extractor,
        noise_selector,
        bit_num=48,
        lr=1e-4,
        attack_start_epoch=1,
        attack_probabilities=(0.1, 0.25, 0.25, 0.1, 0.15, 0.15),
    ):
        super().__init__()
        if abs(sum(attack_probabilities) - 1.0) > 1e-6:
            raise ValueError("attack probabilities must sum to 1")
        if len(attack_probabilities) != len(noise_selector.transforms):
            raise ValueError("one attack probability is required per noise transform")

        self.save_hyperparameters(
            "bit_num", "lr", "attack_start_epoch", "attack_probabilities"
        )
        self.vae = vae
        self.vae.requires_grad_(False)
        self.generator = generator
        self.extractor = extractor
        self.noise_selector = noise_selector
        self.bit_num = bit_num
        self.lr = lr
        self.coef_img = 0                
        self.coef_lpips = 5                                       
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_loss.requires_grad_(False)
        self.attack_start_epoch = attack_start_epoch
        self.attack_probabilities = list(attack_probabilities)
        self.p = None

    def on_train_epoch_start(self):
        self.p = (
            self.attack_probabilities
            if self.current_epoch >= self.attack_start_epoch
            else None
        )

    def training_step(self, batch, batch_idx):
        batch_size = batch.shape[0]
        with torch.no_grad():
            latents = self.vae.encode(batch).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        messages = torch.randint(
            0, 2, (batch_size, self.bit_num), device=self.device
        ).float()
        residuals = self.generator(messages * 2.0 - 1.0)
        watermarked_latents = latents + residuals
        decoded_latents = watermarked_latents / self.vae.config.scaling_factor
        watermarked_images = self.vae.decode(decoded_latents).sample.clamp(-1.0, 1.0)

        images_01 = (watermarked_images + 1.0) / 2.0
        noised_images = self.noise_selector(images_01, self.p).clamp(0.0, 1.0)
        noised_images = noised_images * 2.0 - 1.0
        noised_latents = self.vae.encode(noised_images).latent_dist.sample()
        noised_latents = noised_latents * self.vae.config.scaling_factor

        logits = self.extractor(noised_latents)
        perceptual_loss = self.perceptual_loss(watermarked_images, batch).mean()
        bce_loss = F.binary_cross_entropy_with_logits(logits, messages)
        mse_loss = F.mse_loss(watermarked_images.float(), batch.float())
        loss = self.coef_img * (
            mse_loss + self.coef_lpips * perceptual_loss
        ) + bce_loss
        bit_accuracy = (
            ((torch.sigmoid(logits) > 0.5).float() == messages).float().mean()
        )

        if bce_loss < 0.5:
            self.coef_img = 0.5
            if bce_loss < 0.3:
                self.coef_img = 1.0

        self.log_dict(
            {
                "train_loss": loss,
                "train_bce_loss": bce_loss,
                "train_mse_loss": mse_loss,
                "train_lpips_loss": perceptual_loss,
                "train_bit_accuracy": bit_accuracy,
                "train_attack_enabled": float(self.p is not None),
            },
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch.shape[0]
        latents = self.vae.encode(batch).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        messages = torch.randint(
            0, 2, (batch_size, self.bit_num), device=self.device
        ).float()
        residuals = self.generator(messages * 2.0 - 1.0)
        watermarked_latents = latents + residuals
        decoded_latents = watermarked_latents / self.vae.config.scaling_factor
        watermarked_images = self.vae.decode(decoded_latents).sample.clamp(-1.0, 1.0)

        images_01 = (watermarked_images + 1.0) / 2.0
        noised_images = self.noise_selector(images_01, self.p).clamp(0.0, 1.0)
        noised_images = noised_images * 2.0 - 1.0
        noised_latents = self.vae.encode(noised_images).latent_dist.sample()
        noised_latents = noised_latents * self.vae.config.scaling_factor

        logits = self.extractor(noised_latents)
        perceptual_loss = self.perceptual_loss(watermarked_images, batch).mean()
        bce_loss = F.binary_cross_entropy_with_logits(logits, messages)
        mse_loss = F.mse_loss(watermarked_images.float(), batch.float())
        loss = self.coef_img * (
            mse_loss + self.coef_lpips * perceptual_loss
        ) + bce_loss
        bit_accuracy = (
            ((torch.sigmoid(logits) > 0.5).float() == messages).float().mean()
        )

        if batch_idx == 0 and self.logger is not None:
            grid = torchvision.utils.make_grid(images_01[:4], nrow=2)
            self.logger.experiment.add_image("val/watermarked_images", grid, self.global_step)

        self.log_dict(
            {
                "val_loss": loss,
                "val_bce_loss": bce_loss,
                "val_mse_loss": mse_loss,
                "val_lpips_loss": perceptual_loss,
                "val_bit_accuracy": bit_accuracy,
            },
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def configure_optimizers(self):
        parameters = (parameter for parameter in self.parameters() if parameter.requires_grad)
        return torch.optim.AdamW(parameters, lr=self.lr)

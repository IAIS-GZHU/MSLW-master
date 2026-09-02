import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm
from torchvision.utils import save_image

from model.watermark_models import WatermarkGenerator
from pipeline.mslw_pipe import MSLW_Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-path", required=True, type=Path)
    parser.add_argument(
        "--prompt-column",
        help="Name of the CSV column containing prompts (default: first column)",
    )
    parser.add_argument("--generator-path", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--message-dir", required=True, type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, help="Exclusive end index (default: all prompts)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bit-length", type=int, default=48)
    parser.add_argument("--tau", type=int, default=150)
    parser.add_argument("--phi", type=float, default=0.6)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prediction-type",
        choices=("epsilon", "v_prediction"),
        help="DDIM prediction type (default: use the model configuration)",
    )
    return parser.parse_args()


def load_prompts(path: Path, column: str | None) -> list[str]:
    if path.suffix.lower() == ".txt":
        prompts = path.read_text(encoding="utf-8-sig").splitlines()
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            column = column or reader.fieldnames[0]
            prompts = [row[column] for row in reader]
    else:
        raise ValueError("prompt file must be a TXT or CSV file")
    return [prompt.strip() for prompt in prompts if prompt and prompt.strip()]


def main():
    args = parse_args()
    prompts = load_prompts(args.prompt_path, args.prompt_column)
    end = len(prompts) if args.end is None else args.end

    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.message_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    random_generator = torch.Generator(device=device).manual_seed(args.seed)

    sd = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    scheduler_kwargs = {}
    if args.prediction_type is not None:
        scheduler_kwargs["prediction_type"] = args.prediction_type
    scheduler = DDIMScheduler.from_config(sd.scheduler.config, **scheduler_kwargs)

    watermark_generator = WatermarkGenerator(args.bit_length)
    state_dict = torch.load(args.generator_path, map_location="cpu", weights_only=True)
    watermark_generator.load_state_dict(state_dict)
    watermark_generator.requires_grad_(False)

    pipe = MSLW_Pipeline(
        vae=sd.vae,
        text_encoder=sd.text_encoder,
        tokenizer=sd.tokenizer,
        unet=sd.unet,
        scheduler=scheduler,
        watermark_generator=watermark_generator,
        device=device,
        dtype=dtype,
    )

    index = args.start
    with tqdm(total=end - args.start) as progress:
        while index < end:
            batch_prompts = prompts[index : min(index + args.batch_size, end)]
            messages = torch.randint(
                0,
                2,
                (len(batch_prompts), args.bit_length),
                generator=random_generator,
                device=device,
            )
            images = pipe.sample(
                prompts=batch_prompts,
                msg=messages,
                tau=args.tau,
                phi=args.phi,
                cfg_scale=args.guidance_scale,
                num_steps=args.num_steps,
                generator=random_generator,
            )

            messages = messages.to(torch.uint8).cpu().numpy()
            for offset, image in enumerate(images):
                name = f"{index + offset:05d}"
                save_image(image, args.image_dir / f"img_{name}.png")
                np.save(args.message_dir / f"msg_{name}.npy", messages[offset])

            index += len(batch_prompts)
            progress.update(len(batch_prompts))


if __name__ == "__main__":
    main()

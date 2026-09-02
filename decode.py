import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model.watermark_models import WatermarkExtractor


def positive_int(value: str) -> int:
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def non_negative_int(value: str) -> int:
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def existing_dir(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"directory does not exist: {path}")
    return path


def existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True, help="Identifier written to the result")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--image-dir", required=True, type=existing_dir)
    parser.add_argument("--message-dir", required=True, type=existing_dir)
    parser.add_argument("--extractor-path", required=True, type=existing_file)
    parser.add_argument("--result-path", required=True, type=Path)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--bit-length", type=positive_int, default=48)
    parser.add_argument("--image-size", type=positive_int, default=512)
    parser.add_argument("--num-workers", type=non_negative_int, default=4)
    return parser.parse_args()


class WatermarkDataset(Dataset):

    def __init__(self, image_dir: Path, message_dir: Path, image_size: int):
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
            ]
        )
        self.samples = []

        for image_path in sorted(image_dir.glob("img_*.png")):
            sample_id = image_path.stem.removeprefix("img_")
            message_path = message_dir / f"msg_{sample_id}.npy"
            if not message_path.is_file():
                raise FileNotFoundError(f"missing message for {image_path.name}: {message_path}")
            self.samples.append((image_path, message_path))

        if not self.samples:
            raise ValueError(f"no img_*.png files found in {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, message_path = self.samples[index]
        with Image.open(image_path) as image:
            image = self.transform(image.convert("RGB"))
        message = torch.from_numpy(np.load(message_path)).float()
        return image, message


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = WatermarkDataset(args.image_dir, args.message_dir, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae").to(device)
    vae.requires_grad_(False).eval()

    extractor = WatermarkExtractor(args.bit_length)
    state_dict = torch.load(args.extractor_path, map_location="cpu", weights_only=True)
    extractor.load_state_dict(state_dict)
    extractor.requires_grad_(False).eval().to(device)

    correct_bits = 0
    total_bits = 0
    with torch.inference_mode():
        for images, messages in tqdm(loader, desc="Decoding"):
            if messages.ndim != 2 or messages.shape[1] != args.bit_length:
                raise ValueError(
                    f"expected messages with shape [batch, {args.bit_length}], "
                    f"got {tuple(messages.shape)}"
                )

            images = images.to(device, non_blocking=True) * 2.0 - 1.0
            messages = messages.to(device, non_blocking=True)
            
            latents = vae.encode(images).latent_dist.mean * vae.config.scaling_factor
            predictions = (torch.sigmoid(extractor(latents)) > 0.5).to(messages.dtype)
            correct_bits += int((predictions == messages).sum().item())
            total_bits += messages.numel()

    accuracy = correct_bits / total_bits
    args.result_path.parent.mkdir(parents=True, exist_ok=True)
    with args.result_path.open("a", encoding="utf-8") as result_file:
        result_file.write(f"experiment_id: {args.experiment_id}\n")
        result_file.write(f"accuracy: {accuracy:.6f}\n")

    print(f"Experiment {args.experiment_id}: accuracy={accuracy:.6f}")


if __name__ == "__main__":
    main()

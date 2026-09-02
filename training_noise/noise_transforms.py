from collections.abc import Sequence

import kornia as K
import numpy as np
import torch
from torch import nn


def random_float(low: float, high: float) -> float:
    return float(np.random.rand() * (high - low) + low)


def random_int(low: int, high: int) -> int:
    return int(np.random.randint(low, high))


def _validate_range(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2 or values[0] >= values[1]:
        raise ValueError(f"{name} must contain two increasing values")
    return float(values[0]), float(values[1])


class Identity(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image


class RandomJpegCompression(nn.Module):
    def __init__(self, quality_range: Sequence[int] = (10, 50)):
        super().__init__()
        low, high = _validate_range(quality_range, "quality_range")
        self.quality_range = int(low), int(high)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        quality_value = random_int(*self.quality_range)
        quality = image.new_full((image.shape[0],), quality_value)
        return K.enhance.jpeg_codec_differentiable(image, jpeg_quality=quality)


class RandomGaussianBlur(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        blur = K.augmentation.RandomGaussianBlur(
            kernel_size=(3, 9), sigma=(1.0, 1.5), p=1.0
        )
        return blur(image)


class RandomGaussianNoise(nn.Module):
    def __init__(self, std_range: Sequence[float] = (0.05, 0.15)):
        super().__init__()
        self.std_range = _validate_range(std_range, "std_range")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        std = random_float(*self.std_range)
        normalized = image * 2.0 - 1.0
        noisy = (normalized + std * torch.randn_like(image)).clamp(-1.0, 1.0)
        return (noisy + 1.0) / 2.0


class RandomColorJitter(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        color_jitter = K.augmentation.ColorJiggle(
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.2, 0.2),
            p=1.0,
        )
        return color_jitter(image.clamp(0.0, 1.0))


def _random_crop_mask(image: torch.Tensor, crop_ratio: float) -> torch.Tensor:
    if not 0.0 < crop_ratio <= 1.0:
        raise ValueError("crop_ratio must be in (0, 1]")

    batch_size, _, height, width = image.shape
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)
    height_start = random_int(0, height - crop_height + 1)
    width_start = random_int(0, width - crop_width + 1)
    mask = image.new_zeros((batch_size, 1, height, width))
    mask[
        :,
        :,
        height_start : height_start + crop_height,
        width_start : width_start + crop_width,
    ] = 1.0
    return image * mask


class RandomCrop(nn.Module):
    def __init__(self, ratio_range: Sequence[float] = (0.5, 0.8)):
        super().__init__()
        self.ratio_range = _validate_range(ratio_range, "ratio_range")
        if not 0.0 < self.ratio_range[0] < self.ratio_range[1] <= 1.0:
            raise ValueError("ratio_range values must be in (0, 1]")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return _random_crop_mask(image, random_float(*self.ratio_range))

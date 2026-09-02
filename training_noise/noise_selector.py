from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from training_noise.noise_transforms import (
    Identity,
    RandomColorJitter,
    RandomCrop,
    RandomGaussianBlur,
    RandomGaussianNoise,
    RandomJpegCompression,
)


NOISE_TRANSFORM_REGISTRY = {
    "Identity": Identity,
    "Jpeg": RandomJpegCompression,
    "random_crop": RandomCrop,
    "GaussianBlur": RandomGaussianBlur,
    "GaussianNoise": RandomGaussianNoise,
    "ColorJitter": RandomColorJitter,
}


class NoiseSelector(nn.Module):

    def __init__(
        self,
        noise_transforms: Sequence[str | nn.Module],
        probabilities: Sequence[float],
    ):
        super().__init__()
        if not noise_transforms:
            raise ValueError("at least one noise transform is required")

        transforms = []
        for transform in noise_transforms:
            if isinstance(transform, str):
                try:
                    transform = NOISE_TRANSFORM_REGISTRY[transform]()
                except KeyError as error:
                    available = ", ".join(NOISE_TRANSFORM_REGISTRY)
                    raise ValueError(
                        f"unknown noise transform {transform!r}; available: {available}"
                    ) from error
            if not isinstance(transform, nn.Module):
                raise TypeError("noise transforms must be names or nn.Module instances")
            transforms.append(transform)

        self.transforms = nn.ModuleList(transforms)
        self.probabilities = self._validate_probabilities(probabilities)

    def _validate_probabilities(self, probabilities: Sequence[float]) -> np.ndarray:
        values = np.asarray(probabilities, dtype=np.float64)
        if values.ndim != 1 or len(values) != len(self.transforms):
            raise ValueError("one probability is required per noise transform")
        if not np.isfinite(values).all() or (values < 0.0).any():
            raise ValueError("noise probabilities must be finite and non-negative")
        if not np.isclose(values.sum(), 1.0):
            raise ValueError("noise probabilities must sum to 1")
        return values

    def forward(
        self,
        image: torch.Tensor,
        probabilities: Sequence[float] | None = None,
    ) -> torch.Tensor:
        sampling_probabilities = (
            self.probabilities
            if probabilities is None
            else self._validate_probabilities(probabilities)
        )
        index = int(np.random.choice(len(self.transforms), p=sampling_probabilities))
        return self.transforms[index](image)

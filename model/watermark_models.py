import torch
from torch import nn


class WatermarkGenerator(nn.Module):

    def __init__(self, bit_num: int = 48):
        super().__init__()
        if bit_num <= 0:
            raise ValueError("bit_num must be positive")

        self.fc = nn.Sequential(
            nn.Linear(bit_num, 16 * 16),
            nn.SiLU(),
            nn.Linear(16 * 16, 16 * 16),
        )
        self.convblock = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
        )

    def forward(self, message: torch.Tensor) -> torch.Tensor:
        features = self.fc(message)
        features = features.view(-1, 1, 16, 16)
        features = features.repeat(1, 1, 4, 4)
        return self.convblock(features)


class WatermarkExtractor(nn.Module):
    """Recover binary-message logits from four-channel latent features."""

    def __init__(self, bit_num: int = 48):
        super().__init__()
        if bit_num <= 0:
            raise ValueError("bit_num must be positive")

        self.convblock = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1),
            nn.Conv2d(4, 1, kernel_size=1, padding=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 16, 16 * 16),
            nn.SiLU(),
            nn.Linear(16 * 16, bit_num),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size = latents.shape[0]
        features = self.convblock(latents).squeeze(1)
        features = features.unfold(1, 16, 16).unfold(2, 16, 16)
        features = features.contiguous().view(batch_size, -1, 16, 16)
        features = features.mean(dim=1)
        features = torch.flatten(features, start_dim=1)
        return self.fc(features)

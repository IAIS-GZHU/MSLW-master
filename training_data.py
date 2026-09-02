from pathlib import Path

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def find_image_files(image_dir: Path, recursive: bool = False) -> list[Path]:
    
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_dir}")

    paths = image_dir.rglob("*") if recursive else image_dir.iterdir()
    image_paths = sorted(
        path
        for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"no supported images found in {image_dir}")
    return image_paths


class TrainingDataset(Dataset):

    def __init__(self, image_dir: Path, transform=None, recursive: bool = False):
        self.image_paths = find_image_files(image_dir, recursive)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image


class TrainingDataModule(pl.LightningDataModule):

    def __init__(
        self,
        image_dir: Path,
        batch_size: int = 8,
        num_workers: int = 4,
        image_size: int = 512,
        train_ratio: float = 0.95,
        split_seed: int = 42,
        recursive: bool = False,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be in (0, 1)")

        self.image_dir = Path(image_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.split_seed = split_seed
        self.recursive = recursive
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if self.train_dataset is not None:
            return

        dataset = TrainingDataset(
            self.image_dir,
            transform=self.transform,
            recursive=self.recursive,
        )
        train_size = int(self.train_ratio * len(dataset))
        val_size = len(dataset) - train_size
        if train_size == 0 or val_size == 0:
            raise ValueError("the dataset must contain at least two images")

        generator = torch.Generator().manual_seed(self.split_seed)
        self.train_dataset, self.val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

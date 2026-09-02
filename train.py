import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch
from diffusers import AutoencoderKL
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from model.model_wrapper import ModelWrapper
from model.watermark_models import WatermarkExtractor, WatermarkGenerator
from training_data import TrainingDataModule
from training_noise.noise_selector import NoiseSelector


DEFAULT_NOISE_TRANSFORMS = (
    "Identity",
    "Jpeg",
    "random_crop",
    "GaussianBlur",
    "GaussianNoise",
    "ColorJitter",
)
DEFAULT_ATTACK_PROBABILITIES = (0.1, 0.25, 0.25, 0.1, 0.15, 0.15)


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


def probability(value: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def validate_probabilities(parser, name, probabilities, layer_count):
    if len(probabilities) != layer_count:
        parser.error(f"{name} requires one value per noise transform")
    if abs(sum(probabilities) - 1.0) > 1e-6:
        parser.error(f"{name} must sum to 1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search image-dir recursively",
    )
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("--log-name", default="exp1")
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
    )
    parser.add_argument("--max-epochs", type=positive_int, default=3)
    parser.add_argument(
        "--attack-start-epoch",
        type=non_negative_int,
        default=1,
        help="Zero-based epoch at which attack sampling starts",
    )
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--bit-length", type=positive_int, default=48)
    parser.add_argument("--image-size", type=positive_int, default=512)
    parser.add_argument("--num-workers", type=non_negative_int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--noise-transforms", nargs="+", default=DEFAULT_NOISE_TRANSFORMS
    )
    parser.add_argument(
        "--initial-noise-probabilities",
        nargs="+",
        type=probability,
        help="Initial probabilities (default: Identity only)",
    )
    parser.add_argument(
        "--attack-probabilities",
        nargs="+",
        type=probability,
        default=DEFAULT_ATTACK_PROBABILITIES,
        help="Noise probabilities used from attack-start-epoch onward",
    )
    args = parser.parse_args()
    if args.attack_start_epoch >= args.max_epochs:
        parser.error("attack-start-epoch must be smaller than max-epochs")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        parser.error("train-ratio must be in (0, 1)")

    if args.initial_noise_probabilities is None:
        identity_indices = [
            index
            for index, name in enumerate(args.noise_transforms)
            if name == "Identity"
        ]
        if len(identity_indices) != 1:
            parser.error("noise-transforms must contain exactly one Identity transform")
        args.initial_noise_probabilities = [0.0] * len(args.noise_transforms)
        args.initial_noise_probabilities[identity_indices[0]] = 1.0

    validate_probabilities(
        parser,
        "initial-noise-probabilities",
        args.initial_noise_probabilities,
        len(args.noise_transforms),
    )
    validate_probabilities(
        parser,
        "attack-probabilities",
        args.attack_probabilities,
        len(args.noise_transforms),
    )
    return args


class Checkpoint(pl.Callback):

    def __init__(self, directory: Path):
        super().__init__()
        self.directory = Path(directory)
        self.best_loss = float("inf")
        self.best_bce = float("inf")

    def _save(self, module: ModelWrapper, suffix: str):
        self.directory.mkdir(parents=True, exist_ok=True)
        models = (("generator", module.generator), ("extractor", module.extractor))
        for name, model in models:
            filenames = {
                "best": f"{name}.pth",
                "best_bce": f"{name}_bestbce.pth",
                "last": f"{name}_last.pth",
            }
            path = self.directory / filenames[suffix]
            temporary_path = path.with_suffix(".pth.tmp")
            state_dict = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            torch.save(state_dict, temporary_path)
            temporary_path.replace(path)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        if "val_loss" not in metrics or "val_bce_loss" not in metrics:
            return

        self._save(pl_module, "last")

        val_loss = metrics["val_loss"].item()
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self._save(pl_module, "best")

        val_bce = metrics["val_bce_loss"].item()
        if val_bce < self.best_bce:
            self.best_bce = val_bce
            self._save(pl_module, "best_bce")


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)

    vae = AutoencoderKL.from_pretrained(args.model_id, subfolder="vae")
    generator = WatermarkGenerator(args.bit_length)
    extractor = WatermarkExtractor(args.bit_length)

    noise_selector = NoiseSelector(
        args.noise_transforms, args.initial_noise_probabilities
    ).eval()
    noise_selector.requires_grad_(False)

    model = ModelWrapper(
        vae,
        generator,
        extractor,
        noise_selector,
        bit_num=args.bit_length,
        lr=args.learning_rate,
        attack_start_epoch=args.attack_start_epoch,
        attack_probabilities=args.attack_probabilities,
    )

    data_module = TrainingDataModule(
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        train_ratio=args.train_ratio,
        split_seed=args.seed,
        recursive=args.recursive,
    )

    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.log_name)
    checkpoint_callback = Checkpoint(args.checkpoint_dir)
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        precision="32",
        devices=1,
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_checkpointing=False,
        val_check_interval=0.25,
        gradient_clip_val=1.0,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()

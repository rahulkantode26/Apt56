"""Data loading and augmentation.

Reads an ImageFolder layout produced by prepare_data.py:

    processed/train/<class>/*.jpg
    processed/val/<class>/*.jpg

Class names (and their index order) are taken directly from the folder names,
so the pipeline works for any number of classes / diseases.
"""

from __future__ import annotations

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from . import config


def _train_transforms() -> transforms.Compose:
    # Aggressive-ish augmentation because these datasets are small. The
    # transforms are chosen to be plausible for real phone eye photos:
    # slight rotation/shift, flips, and mild colour/brightness variation.
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ]
    )


def _eval_transforms() -> transforms.Compose:
    # No augmentation for validation/inference — just resize + normalize.
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ]
    )


def build_dataloaders() -> tuple[DataLoader, DataLoader, list[str]]:
    """Return (train_loader, val_loader, class_names)."""
    train_ds = datasets.ImageFolder(config.TRAIN_DIR, transform=_train_transforms())
    val_ds = datasets.ImageFolder(config.VAL_DIR, transform=_eval_transforms())

    if train_ds.classes != val_ds.classes:
        raise ValueError(
            f"Train/val class mismatch: {train_ds.classes} vs {val_ds.classes}"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == "cuda"),
    )
    return train_loader, val_loader, train_ds.classes


def get_eval_transforms() -> transforms.Compose:
    """Exposed so inference (the app) uses the exact same preprocessing."""
    return _eval_transforms()

"""PyTorch datasets and data loaders for the prepared PlantVillage data."""

from __future__ import annotations

from pathlib import Path

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import IMG_SIZE

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_dataloaders(
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: str | Path,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation, and test loaders with train-only augmentation."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=evaluation_transform)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=evaluation_transform)
    loader_options = {"batch_size": batch_size, "num_workers": 0}
    return (
        DataLoader(train_dataset, shuffle=True, **loader_options),
        DataLoader(val_dataset, shuffle=False, **loader_options),
        DataLoader(test_dataset, shuffle=False, **loader_options),
    )

"""Train and evaluate a VGG16 classifier on the prepared PlantVillage dataset.

The script uses the same train/validation/test directory layout as
``dataset_loader.py`` and writes artifacts under ``vgg16/``.

Example:
    python train_vgg16.py --epochs 10
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from config import BATCH_SIZE, CLASS_NAMES, IMG_SIZE

ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = ROOT / "vgg16"
MODEL_DIR = ARTIFACT_ROOT / "models"
CHECKPOINT_DIR = ARTIFACT_ROOT / "checkpoints"
RESULT_DIR = ARTIFACT_ROOT / "results"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(batch_size: int, workers: int) -> tuple[dict[str, DataLoader], dict[str, int]]:
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
    dataset_root = ROOT / "dataset"
    datasets_by_split = {
        "train": datasets.ImageFolder(dataset_root / "train", train_transform),
        "val": datasets.ImageFolder(dataset_root / "val", evaluation_transform),
        "test": datasets.ImageFolder(dataset_root / "test", evaluation_transform),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets_by_split.items()
    }
    if set(datasets_by_split["train"].classes) != set(CLASS_NAMES):
        raise ValueError("Dataset class folders do not match class_indices.json/config.py.")
    CLASS_NAMES[:] = datasets_by_split["train"].classes
    return loaders, {split: len(dataset) for split, dataset in datasets_by_split.items()}


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += labels.size(0)
    return total_loss / total_items, total_correct / total_items


def save_checkpoint(path: Path, model: nn.Module, optimizer: optim.Optimizer, epoch: int, history: dict) -> None:
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "history": history},
        path,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            actual.extend(labels.numpy())
            predicted.extend(outputs.argmax(dim=1).cpu().numpy())
    return np.asarray(actual), np.asarray(predicted)


def write_results(model: nn.Module, loaders: dict[str, DataLoader], device: torch.device, history: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    actual, predicted = evaluate(model, loaders["test"], device)
    report = classification_report(actual, predicted, target_names=CLASS_NAMES, zero_division=0)
    (RESULT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    metrics = {
        "test_accuracy": float((actual == predicted).mean()),
        "classes": CLASS_NAMES,
        "history": history,
    }
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    cm = confusion_matrix(actual, predicted)
    fig, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(title="VGG16 confusion matrix", xlabel="Predicted label", ylabel="True label")
    axis.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=90, fontsize=6)
    axis.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=6)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(history["train_loss"], label="train loss")
    axis.plot(history["val_loss"], label="val loss")
    axis.set(xlabel="Epoch", ylabel="Loss", title="VGG16 training curves")
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    loaders, split_sizes = build_loaders(args.batch_size, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.2)
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_loss, best_state = float("inf"), None
    print(f"Device: {device}; split sizes: {split_sizes}")
    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        save_checkpoint(CHECKPOINT_DIR / "latest.pt", model, optimizer, epoch + 1, history)
        if val_loss < best_loss:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
            torch.save(best_state, MODEL_DIR / "vgg16_best.pth")
        print(f"Epoch {epoch + 1}/{args.epochs} - train loss {train_loss:.4f}, val loss {val_loss:.4f}, val accuracy {val_accuracy:.4f}")
    if best_state is None:
        raise RuntimeError("Training completed without a best model checkpoint.")
    model.load_state_dict(best_state)
    torch.save(best_state, MODEL_DIR / "vgg16_final.pth")
    write_results(model, loaders, device, history)
    print(f"Artifacts written to {ARTIFACT_ROOT}")


if __name__ == "__main__":
    main()

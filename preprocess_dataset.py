"""Prepare a stratified PlantVillage image dataset for PyTorch training.

Example:
    python preprocess_dataset.py --raw-dir "C:\\data\\PlantVillage"

The raw directory must contain one subdirectory per class. Images are resized
to 224x224 and written to train/val/test. ImageNet normalization and training
augmentation are applied by ``dataset_loader.get_dataloaders`` at load time.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

IMG_SIZE = 224
EXPECTED_CLASS_COUNT = 38
VALID_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
LOGGER = logging.getLogger(__name__)


def _image_files(directory: Path) -> Iterable[Path]:
    return (
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def _read_valid_images(class_dir: Path) -> list[Path]:
    valid: list[Path] = []
    for path in sorted(_image_files(class_dir)):
        try:
            with Image.open(path) as image:
                image.verify()
            valid.append(path)
        except (OSError, UnidentifiedImageError) as exc:
            LOGGER.warning("Skipping corrupted image %s: %s", path, exc)
    return valid


def _split_items(
    items: list[Path], rng: random.Random
) -> dict[str, list[Path]]:
    shuffled = items.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = round(n * 0.70)
    val_end = train_end + round(n * 0.15)
    # Keep every non-empty class represented in each split where its size allows.
    if n >= 3:
        train_end = max(1, min(train_end, n - 2))
        val_end = max(train_end + 1, min(val_end, n - 1))
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def _write_image(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS).save(
            destination, format="JPEG", quality=90, optimize=True
        )


def _write_config(output_dir: Path, class_names: list[str]) -> None:
    config = f'''"""Generated dataset configuration."""

from pathlib import Path

CLASS_NAMES = {class_names!r}
IMG_SIZE = {IMG_SIZE}
BATCH_SIZE = 32
DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"
TEST_DIR = DATASET_ROOT / "test"
'''
    (output_dir.parent / "config.py").write_text(config, encoding="utf-8")


def prepare_dataset(
    raw_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    clean_output: bool = False,
) -> None:
    """Validate, stratify, resize, and write a PlantVillage dataset."""
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")

    class_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    # Accept both the class-folder root and a PlantVillage parent containing
    # the common ``train`` subdirectory.
    if len(class_dirs) != EXPECTED_CLASS_COUNT and (raw_dir / "train").is_dir():
        raw_dir = raw_dir / "train"
        class_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    if len(class_dirs) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} class folders in {raw_dir}, "
            f"found {len(class_dirs)}"
        )
    class_names = [path.name for path in class_dirs]
    if len(set(class_names)) != EXPECTED_CLASS_COUNT:
        raise ValueError("Class folder names must be unique.")

    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    split_files: dict[str, dict[str, list[Path]]] = defaultdict(dict)
    for class_dir in class_dirs:
        files = _read_valid_images(class_dir)
        if not files:
            raise ValueError(f"Class folder contains no valid images: {class_dir}")
        splits = _split_items(files, rng)
        for split, paths in splits.items():
            split_files[split][class_dir.name] = paths
            if not paths:
                LOGGER.warning("Class %s has no images in %s split", class_dir.name, split)

    for split, classes in split_files.items():
        for class_name, paths in classes.items():
            for index, source in enumerate(paths):
                destination = output_dir / split / class_name / f"{index:06d}.jpg"
                _write_image(source, destination)

    class_indices = {name: index for index, name in enumerate(class_names)}
    (output_dir.parent / "class_indices.json").write_text(
        json.dumps(class_indices, indent=2) + "\n", encoding="utf-8"
    )
    _write_config(output_dir, class_names)
    LOGGER.info("Prepared %d classes in %s", len(class_names), output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare_dataset(
        args.raw_dir,
        args.output_dir,
        seed=args.seed,
        clean_output=args.clean_output,
    )


if __name__ == "__main__":
    main()

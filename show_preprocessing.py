"""Display and save metrics for a prepared PlantVillage dataset.

Run from the repository root:
    python show_preprocessing.py

Use ``--dataset-dir`` to inspect a different prepared dataset directory.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SPLITS = ("train", "val", "test")


def _image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file())


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{value} B"


def _class_counts(dataset_dir: Path, split: str, classes: list[str]) -> dict[str, int]:
    return {
        class_name: sum(
            1 for path in (dataset_dir / split / class_name).iterdir() if path.is_file()
        )
        for class_name in classes
    }


def inspect_dataset(dataset_dir: Path, metrics_path: Path) -> None:
    """Calculate, save, and print dataset summary metrics."""
    dataset_dir = dataset_dir.resolve()
    missing = [split for split in SPLITS if not (dataset_dir / split).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing split directories in {dataset_dir}: {', '.join(missing)}. "
            "Run preprocess_dataset.py first or pass --dataset-dir with the "
            "prepared dataset location."
        )

    classes = sorted(
        path.name for path in (dataset_dir / "train").iterdir() if path.is_dir()
    )
    all_files: dict[str, list[Path]] = {
        split: _image_files(dataset_dir / split) for split in SPLITS
    }
    counts_by_split = {
        split: _class_counts(dataset_dir, split, classes) for split in SPLITS
    }
    total = sum(len(files) for files in all_files.values())
    sizes = [path.stat().st_size for files in all_files.values() for path in files]
    dimensions: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    extensions: Counter[str] = Counter()
    invalid: list[tuple[Path, str]] = []

    for split, files in all_files.items():
        for path in files:
            extensions[path.suffix.lower()] += 1
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    dimensions[image.size] += 1
                    modes[image.mode] += 1
            except (OSError, UnidentifiedImageError) as exc:
                invalid.append((path, str(exc)))

    split_counts = {split: len(files) for split, files in all_files.items()}
    # Content-hash leakage check.
    # Relative-path comparison is unreliable when splits use an independent
    # index counter (000000.jpg, 000001.jpg …) restarted per split, because
    # train/cls/000000.jpg and val/cls/000000.jpg are different source images
    # that happen to share a relative path.  MD5 hashing compares actual bytes.
    import hashlib

    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    split_hashes: dict[str, dict[str, Path]] = {}
    for split, files in all_files.items():
        split_hashes[split] = {_md5(p): p for p in files}

    overlaps: dict[str, int] = {}
    for first_index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[first_index + 1:]:
            shared = split_hashes[first_split].keys() & split_hashes[second_split].keys()
            overlaps[f"{first_split}/{second_split}"] = len(shared)

    indices_path = dataset_dir.parent / "class_indices.json"
    indices = json.loads(indices_path.read_text(encoding="utf-8")) if indices_path.exists() else {}
    metrics = {
        "dataset_directory": str(dataset_dir),
        "classes": len(classes),
        "class_distribution": {
            class_name: {
                "total": sum(counts_by_split[split][class_name] for split in SPLITS),
                **{split: counts_by_split[split][class_name] for split in SPLITS},
            }
            for class_name in classes
        },
        "split_statistics": {
            split: {
                "images": split_counts[split],
                "percentage": round(split_counts[split] / total * 100, 2) if total else 0,
            }
            for split in SPLITS
        },
        "total_images": total,
        "invalid_or_corrupted_outputs": len(invalid),
        "extensions": dict(extensions),
        "dimensions": {str(key): value for key, value in dimensions.items()},
        "color_modes": dict(modes),
        "total_bytes": sum(sizes),
        "average_bytes": round(sum(sizes) / len(sizes)) if sizes else 0,
        "smallest_bytes": min(sizes) if sizes else 0,
        "largest_bytes": max(sizes) if sizes else 0,
        "content_hash_overlap": overlaps,
        "class_indices_entries": len(indices),
    }
    metrics_path = metrics_path.resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Metrics saved to: {metrics_path}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Classes: {len(classes)}")
    print(f"Total images: {total:,}")
    print(f"Split counts: {split_counts}")
    print(f"Invalid/corrupted outputs: {len(invalid)}")
    print(f"Dimensions: {dict(dimensions)}")
    print(f"Color modes: {dict(modes)}")
    print(f"Total size: {_format_bytes(sum(sizes))}")
    print("\nClass distribution:")
    for class_name, counts in metrics["class_distribution"].items():
        print(
            f"  {class_name}: total={counts['total']:,}, "
            f"train={counts['train']:,}, val={counts['val']:,}, test={counts['test']:,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Prepared dataset directory (default: dataset)",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("preprocessing_metrics.json"),
        help="Output JSON path (default: preprocessing_metrics.json)",
    )
    args = parser.parse_args()
    inspect_dataset(args.dataset_dir, args.metrics_path)


if __name__ == "__main__":
    main()

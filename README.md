# AgroDetect

AgroDetect provides utilities for preparing the PlantVillage image dataset and
loading it with PyTorch. The current repository focuses on dataset
preprocessing and data loaders for 38 plant health classes.

## Features

- Validates and reads class-organized image datasets.
- Skips corrupted or unsupported image files with warnings.
- Creates reproducible 70/15/15 train, validation, and test splits.
- Resizes images to `224x224` and saves them as optimized JPEG files.
- Provides PyTorch `DataLoader` instances with training augmentation and
  ImageNet normalization.

## Requirements

Install Python 3.10 or newer, then install the runtime dependencies:

```powershell
python -m pip install torch torchvision pillow
```

## Dataset preparation

The raw dataset must contain one directory for each of the 38 classes. For
example:

```text
PlantVillage/
├── Apple___Apple_scab/
│   ├── image001.jpg
│   └── image002.jpg
├── Apple___healthy/
└── ...
```

Prepare the dataset with:

```powershell
python preprocess_dataset.py --raw-dir "C:\data\PlantVillage" --clean-output
```

By default, the processed dataset is written to `dataset`. A different output
directory, or a deterministic split seed, can be provided:

```powershell
python preprocess_dataset.py `
  --raw-dir "C:\data\PlantVillage" `
  --output-dir "C:\data\AgroDetect\dataset" `
  --seed 42
```

The command also generates `config.py` and `class_indices.json` in the
repository directory. The generated dataset has this structure:

```text
dataset/
├── train/<class-name>/*.jpg
├── val/<class-name>/*.jpg
└── test/<class-name>/*.jpg
```

## Loading the data

Use `get_dataloaders` after preparing the dataset:

```python
from config import BATCH_SIZE, TEST_DIR, TRAIN_DIR, VAL_DIR
from dataset_loader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    BATCH_SIZE,
)
```

Training data uses random horizontal flips, rotations, and color jitter.
Validation and test data use resizing and ImageNet normalization only.

## Repository layout

| File | Description |
| --- | --- |
| `preprocess_dataset.py` | Validates, splits, resizes, and writes the dataset |
| `dataset_loader.py` | Builds the train, validation, and test PyTorch loaders |
| `config.py` | Dataset paths, class names, image size, and batch size |
| `class_indices.json` | Mapping from class names to numeric labels |

## Notes

The repository does not currently include a model training or inference
script. The generated `dataset/` directory is intentionally kept separate
from the source utilities and should not be committed to version control.

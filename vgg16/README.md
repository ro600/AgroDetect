# VGG16 training artifacts

Run the notebook `vgg16_training.ipynb` or:

```powershell
python train_vgg16.py --epochs 10
```

The script reads `dataset/train`, `dataset/val`, and `dataset/test`. It creates
the following files after training:

- `models/vgg16_best.pth` and `models/vgg16_final.pth`
- `checkpoints/latest.pt`
- `results/classification_report.txt`
- `results/metrics.json`
- `results/confusion_matrix.png`
- `results/training_curves.png`

The model uses ImageNet-pretrained VGG16 by default. Use `--no-pretrained` when
working offline and wanting to initialize the network randomly.

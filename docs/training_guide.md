# Training Guide

This tutorial demonstrates how to train a custom model using the built‑in
training pipeline.

## 1. Prepare a Dataset

```python
from wifi_activity_recognition.datasets import Dataset

train, val, test = Dataset.from_files(
    train_glob="data/train/*.npy",
    val_glob="data/val/*.npy",
    test_glob="data/test/*.npy",
)
```

Augment the data with built-in transforms if needed:

```python
from wifi_activity_recognition.datasets import transforms

train = transforms.add_gaussian_noise(train, std=0.01)
```

## 2. Configure the Experiment

Create a YAML file specifying the hardware, preprocessing steps, and model
architecture:

```yaml
hardware: intel5300
model: resnet
training:
  epochs: 10
  batch_size: 32
```

## 3. Run Training

Launch training from the command line:

```bash
python -m wifi_activity_recognition.cli train --config config.yaml
```

Checkpoints and metrics are saved under the `runs/` directory. Use the
`Trainer.load_checkpoint` method to resume or evaluate models later.

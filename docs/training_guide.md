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

During training the CLI prints epoch metrics and saves checkpoints under
`runs/`. To visualize learning curves, launch TensorBoard:

```bash
tensorboard --logdir runs
```

## 4. Evaluate a Saved Model

```bash
python -m wifi_activity_recognition.cli evaluate \
    --checkpoint runs/latest.ckpt \
    --data data/test
```

The output reports accuracy, precision, recall and F1.

## 5. Tune Hyper‑parameters

Key parameters are configured in the YAML file:

```yaml
training:
  epochs: 20
  batch_size: 64
  optimizer:
    lr: 0.001
  early_stopping:
    patience: 5
```

Adjust learning rate (`optimizer.lr`), batch size, or enable early stopping to
prevent overfitting.

## 6. Resume or Fine‑tune

```python
from wifi_activity_recognition.training.trainer import Trainer

trainer = Trainer.load_checkpoint("runs/latest.ckpt")
trainer.train(epochs=5)
```

These commands continue training from a saved state or evaluate directly.

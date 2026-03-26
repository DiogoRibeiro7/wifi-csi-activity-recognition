# Training Guide

This guide shows the training workflow that the current package actually supports: array-based datasets, explicit CLI flags, and the `Trainer` class.

## 1. Prepare data arrays

The built-in dataset loader expects a feature array and a label array on disk:

```python
import numpy as np

data = np.load("demo_data.npy")
labels = np.load("demo_labels.npy")

print(data.shape)
print(labels.shape)
```

For the CLI, both arrays are passed directly to `Dataset.from_files(data_path=..., labels_path=...)`.

## 2. Train from the CLI

```bash
python -m wifi_activity_recognition.cli train \
  --data demo_data.npy \
  --labels demo_labels.npy \
  --model cnn2d \
  --hardware esp32 \
  --epochs 5 \
  --batch-size 8 \
  --output runs/model.pt
```

The CLI:

- loads the arrays into a `Dataset`
- creates the requested model architecture
- trains with `Trainer`
- saves a structured model artifact

## 3. Evaluate from the CLI

```bash
python -m wifi_activity_recognition.cli evaluate \
  --model runs/model.pt \
  --data demo_data.npy \
  --labels demo_labels.npy \
  --hardware esp32 \
  --output runs/evaluation.json
```

This evaluates the dataset's `test` split and optionally writes the metrics report to JSON.

## 4. Use the Python API directly

```python
import numpy as np

from wifi_activity_recognition.datasets import Dataset, split_dataset
from wifi_activity_recognition.models import create_model
from wifi_activity_recognition.training import Trainer

data = np.load("demo_data.npy")
labels = np.load("demo_labels.npy")

train, val, test = split_dataset(data, labels, val_ratio=0.2, test_ratio=0.2)
dataset = Dataset(train=train, val=val, test=test)

model = create_model("cnn2d", num_classes=len(dataset.classes), in_channels=1)
trainer = Trainer(model=model, dataset=dataset, batch_size=8, learning_rate=1e-3)
trainer.train(epochs=5)

metrics = trainer.get_metrics()
print(metrics["train_accuracy"], metrics["val_accuracy"])
trainer.save_model("runs/model.pt")
```

## 5. Hyperparameter search

The current CLI includes a simple search command:

```bash
python -m wifi_activity_recognition.cli autotrain \
  --data demo_data.npy \
  --labels demo_labels.npy \
  --model cnn2d \
  --hardware esp32 \
  --epochs 1 \
  --learning-rates 1e-3,5e-4 \
  --batch-sizes 4,8 \
  --output runs/best_model.pt
```

This evaluates the requested combinations and saves the best structured model artifact.

## 6. Notes

- The current training path is array-based, not glob-based.
- The current CLI does not support a single `--config` training command.
- There is no `Trainer.load_checkpoint(...)` helper in the current implementation.

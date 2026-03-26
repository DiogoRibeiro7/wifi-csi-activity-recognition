# WiFi Activity Recognition

[![PyPI version](https://img.shields.io/pypi/v/wifi-activity-recognition.svg)](https://pypi.org/project/wifi-activity-recognition/)
[![CI](https://github.com/diogoribeiro7/wifi-csi-activity-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/diogoribeiro7/wifi-csi-activity-recognition/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

WiFi Activity Recognition is a Python package for CSI-based human activity recognition. It provides a hardware abstraction layer, preprocessing and feature utilities, train/evaluate/predict workflows, and research modules for adaptation and federated learning.

## Features

- Hardware abstraction for currently registered readers: Intel 5300, ESP32, Atheros AR9300, and Qualcomm.
- Model implementations including CNN2D, CNN3D, ResNet, Transformer, and Vision Transformer variants.
- CLI workflows for collection, training, prediction, evaluation, streaming, benchmarking, export, and visualization.
- Research utilities for domain adaptation, few-shot learning, and federated training.
- Dataset helpers and a PyTorch-based training loop for reproducible experiments.

## Installation

For the current codebase, Python 3.10 is the safest documented choice because that is what CI runs today.

```bash
pip install wifi-activity-recognition
```

For local development:

```bash
git clone https://github.com/diogoribeiro7/wifi-csi-activity-recognition.git
cd wifi-csi-activity-recognition
pip install -e .[dev,docs]
```

If you plan to train or run inference, install a compatible PyTorch build for your platform as well.

See [docs/installation.md](docs/installation.md) for environment and hardware notes.

## Quickstart

Create a small synthetic dataset:

```bash
python - <<'PY'
import numpy as np

rng = np.random.default_rng(42)
data = rng.random((24, 1, 8, 8), dtype=np.float32)
labels = rng.integers(0, 2, size=24, dtype=np.int64)

np.save("demo_data.npy", data)
np.save("demo_labels.npy", labels)
PY
```

Train a model:

```bash
python -m wifi_activity_recognition.cli train \
  --data demo_data.npy \
  --labels demo_labels.npy \
  --model cnn2d \
  --hardware esp32 \
  --epochs 1 \
  --batch-size 4 \
  --output demo_model.pt
```

Evaluate it:

```bash
python -m wifi_activity_recognition.cli evaluate \
  --model demo_model.pt \
  --data demo_data.npy \
  --labels demo_labels.npy \
  --hardware esp32
```

See [docs/quickstart.md](docs/quickstart.md) for a slightly fuller walkthrough.

## Python API

### Train a model

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
trainer = Trainer(model=model, dataset=dataset, batch_size=4)
trainer.train(epochs=1)
trainer.save_model("demo_model.pt")
```

### Run packet-level inference

```python
from wifi_activity_recognition.inference import ActivityRecognizer
from wifi_activity_recognition.models import load_model
from wifi_activity_recognition.utils.io import load_csi_data

model = load_model("demo_model.pt")
recognizer = ActivityRecognizer(model)
packets = load_csi_data("captured_packets.json")

label, confidence = recognizer.predict(packets[0])
print(label, confidence)
```

### Stream from hardware

```python
from wifi_activity_recognition.hardware import CSIReader

reader = CSIReader("esp32", {"sampling_rate": 100, "channel": 6})

with reader:
    for packet in reader.stream():
        print(packet.shape)
        break
```

## Hardware Status

The current registry-backed CLI and factory surface expose the hardware drivers that are actually registered at import time. At the moment that means Intel 5300, ESP32, Atheros AR9300, and Qualcomm. Broadcom and MediaTek are not enabled in the active registry yet.

Use the CLI to inspect the current environment:

```bash
python -m wifi_activity_recognition.cli info --hardware all
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Quickstart](docs/quickstart.md)
- [Hardware Setup](docs/hardware_setup.md)
- [Training Guide](docs/training_guide.md)
- [Deployment Guide](docs/deployment.md)
- [API Reference](docs/api_reference.md)

## License

Distributed under the [MIT License](LICENSE).

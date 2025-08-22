# WiFi Activity Recognition 📡

[![PyPI version](https://img.shields.io/pypi/v/wifi-activity-recognition.svg)](https://pypi.org/project/wifi-activity-recognition/)
[![CI](https://github.com/diogoribeiro7/wifi-csi-activity-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/diogoribeiro7/wifi-csi-activity-recognition/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

WiFi Activity Recognition is a research-grade yet production-ready Python library that uses WiFi Channel State Information (CSI) and modern computer vision techniques to detect human activities without cameras.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Hardware Compatibility](#hardware-compatibility)
- [Benchmarks](#benchmarks)
- [Code Examples](#code-examples)
- [Research Toolkit](#research-toolkit)
- [Deployment Options](#deployment-options)
- [Community](#community)
- [License & Citation](#license--citation)

## Features
- Supports Intel 5300, ESP32, Atheros, Qualcomm, Broadcom and MediaTek adapters.
- Provides state-of-the-art models including CNN2D/3D, ResNet and Vision Transformers.
- Real-time inference with sub -50 ms latency and performance monitoring.
- Advanced preprocessing such as filtering, calibration and multipath analysis.
- Privacy-preserving training with federated learning and domain adaptation.
- Ready for production through Docker/Kubernetes deployment and edge optimization.

## Installation
The library requires **Python 3.9+** and a C++ build toolchain.

```bash
pip install wifi-activity-recognition
```

To work from source or to access optional components, clone the repository and install the project in editable mode:

```bash
git clone https://github.com/diogoribeiro7/wifi-csi-activity-recognition.git
cd wifi-csi-activity-recognition
pip install -e .[dev]
```

Refer to the [installation guide](docs/installation.md) for platform-specific notes, hardware drivers and troubleshooting.

## Quickstart
Run the command-line quickstart to stream data from a supported device:

```bash
wifi-activity-recognition quickstart  # downloads demo data and runs a live example
```

Using the Python API:

```python
from wifi_activity_recognition.hardware import HardwareFactory
from wifi_activity_recognition.models import factory

reader = HardwareFactory.create("esp32")
model = factory.create("cnn2d", pretrained=True)

for csi in reader.stream():
    pred = model.predict(csi)
    print(pred.label, pred.confidence)
```

## Architecture
```mermaid
flowchart LR
    A[Hardware Drivers] --> B[CSIData]
    B --> C[Preprocessing]
    C --> D[Feature Extraction]
    D --> E[Models]
    E --> F[Streaming / Benchmarks / Deployment]
```

## Hardware Compatibility
| Hardware Platform | Subcarriers | Antennas | Difficulty | Notes |
|-------------------|-------------|----------|------------|-------|
| Intel 5300        | 30          | 1 -3      | 🟢 Easy    | Research standard |
| ESP32 / ESP32 -S2  | 64/128      | 1 -2      | 🟢 Easy    | Low -cost IoT boards |
| Atheros AR9300    | 56          | 1 -3      | 🟡 Medium  | Legacy NICs |
| Qualcomm Android  | 64 -256      | 1 -4      | 🟡 Medium  | Mobile devices |
| Broadcom          | 64 -256      | 1 -4      | 🔴 Hard    | Router firmware |
| MediaTek          | 64 -256      | 1 -4      | 🔴 Hard    | Emerging platform |

See the [hardware setup guide](docs/hardware_setup.md) for detailed instructions.

## Benchmarks
| Metric | Target | Achieved* |
|--------|--------|-----------|
| Accuracy (Widar3.0) | >90 % | 94 % |
| End -to -end latency (95th) | <25 ms | 22 ms |
| Memory during streaming | <128 MB | 96 MB |
| Cross -hardware variance | <2 % | 1.3 % |

\*See `benchmarks/performance_report.py` for full reproducible metrics.

## Code Examples
### Training
```python
from wifi_activity_recognition.training import Trainer
from wifi_activity_recognition.datasets import loaders
from wifi_activity_recognition.models import factory

dataset = loaders.load("widar", split=(0.7,0.2,0.1))
model = factory.create("resnet", num_classes=dataset.num_classes)
trainer = Trainer(model, device="cuda")
trainer.fit(dataset.train, val_data=dataset.val, epochs=20)
```

### Real -time Streaming
```python
from wifi_activity_recognition.inference import StreamingPipeline
pipeline = StreamingPipeline(reader, model)
pipeline.run()  # prints activity labels in real time
```

### Feature Extraction
```python
from wifi_activity_recognition.features import cv_transforms, time_domain
spec = cv_transforms.csi_to_spectrogram(csi)
features = time_domain.basic_stats(spec.amplitude)
```

## Research Toolkit
- Domain adversarial neural networks & CORAL alignment
- Few -shot learning: MAML, Prototypical & Relation Networks
- Federated learning: FedAvg, FedProx with differential privacy
- Cross -hardware adaptation and simulation utilities

## Deployment Options
- Docker / docker -compose for reproducible environments
- Kubernetes manifests with health probes and resource limits
- Edge runtimes for Raspberry Pi, NVIDIA Jetson and Android
- Automated model conversion (ONNX, TensorRT) and OTA updates

See the [deployment guide](docs/deployment.md) for details.

## Community
- [Contributing Guide](CONTRIBUTING.md)
- [CHANGELOG](CHANGELOG.md) and [AUTHORS](AUTHORS.md)
- Questions? Contact **Diogo Ribeiro** (<dfr@esmad.ipp.pt>) or open a GitHub issue.

## License & Citation
Distributed under the [MIT License](LICENSE).

```bibtex
@software{wifi_activity_recognition,
  title   = {WiFi Activity Recognition: A Universal Framework for CSI-based Human Activity Recognition},
  author  = {Ribeiro, Diogo},
  year    = {2025},
  url     = {https://github.com/diogoribeiro7/wifi-csi-activity-recognition}
}
```

**Happy sensing!** 🛰️

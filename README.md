# WiFi Activity Recognition 📡

[![PyPI version](https://img.shields.io/pypi/v/wifi-activity-recognition.svg)](https://pypi.org/project/wifi-activity-recognition/)
[![CI](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **research-grade** yet **production-ready** Python library for human activity recognition using WiFi Channel State Information (CSI) and modern computer vision models. Build smart environments that sense motion without cameras while retaining privacy.

---

## ✨ Highlights

- **6+ hardware platforms** &mdash; Intel 5300, ESP32, Atheros, Qualcomm, Broadcom, MediaTek
- **State‑of‑the‑art models** &mdash; CNN2D/3D, ResNet, Vision Transformers, ensembles
- **Real‑time inference** &mdash; sub‑50 ms latency with performance monitoring
- **Advanced preprocessing** &mdash; filtering, calibration, multipath analysis
- **Privacy‑preserving training** &mdash; federated learning, domain adaptation, few‑shot
- **Ready for production** &mdash; Docker/Kubernetes deployment and edge optimization

Target audiences include **researchers**, **IoT developers**, **students**, and **industry practitioners**.

---

## 🚀 Quickstart (5 minutes)

```bash
pip install wifi-activity-recognition
wifi-activity-recognition quickstart  # downloads demo data & runs a live example
```

```python
from wifi_activity_recognition.hardware import HardwareFactory
from wifi_activity_recognition.models import factory

reader = HardwareFactory.create("esp32")
model = factory.create("cnn2d", pretrained=True)

for csi in reader.stream():
    pred = model.predict(csi)
    print(pred.label, pred.confidence)
```

---

## 🧱 Architecture

```mermaid
flowchart LR
    A[Hardware Drivers] --> B[CSIData]
    B --> C[Preprocessing]
    C --> D[Feature Extraction]
    D --> E[Models]
    E --> F[Streaming / Benchmarks / Deployment]
```

---

## 📡 Hardware Compatibility

| Hardware Platform | Subcarriers | Antennas | Difficulty | Notes |
|-------------------|-------------|----------|------------|-------|
| Intel 5300        | 30          | 1‑3      | 🟢 Easy    | Research standard |
| ESP32 / ESP32‑S2  | 64/128      | 1‑2      | 🟢 Easy    | Low‑cost IoT boards |
| Atheros AR9300    | 56          | 1‑3      | 🟡 Medium  | Legacy NICs |
| Qualcomm Android  | 64‑256      | 1‑4      | 🟡 Medium  | Mobile devices |
| Broadcom          | 64‑256      | 1‑4      | 🔴 Hard    | Router firmware |
| MediaTek          | 64‑256      | 1‑4      | 🔴 Hard    | Emerging platform |

See [hardware setup](docs/hardware_setup.md) for instructions and troubleshooting.

---

## 📈 Benchmarks

| Metric | Target | Achieved* |
|--------|--------|-----------|
| Accuracy (Widar3.0) | >90 % | 94 % |
| End‑to‑end latency (95th) | <25 ms | 22 ms |
| Memory during streaming | <128 MB | 96 MB |
| Cross‑hardware variance | <2 % | 1.3 % |

\*See `benchmarks/performance_report.py` for full reproducible metrics.

---

## 🧪 Code Examples

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

### Real‑time Streaming
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

---

## 🧠 Research Toolkit

- Domain adversarial neural networks & CORAL alignment
- Few‑shot learning: MAML, Prototypical & Relation Networks
- Federated learning: FedAvg, FedProx with differential privacy
- Cross‑hardware adaptation and simulation utilities

---

## 🚀 Deployment Options

- **Docker / docker‑compose** for reproducible environments
- **Kubernetes** manifests with health probes and resource limits
- **Edge runtimes** for Raspberry Pi, NVIDIA Jetson, and Android
- Automated model conversion (ONNX, TensorRT) and OTA updates

See [deployment guide](docs/deployment.md) for details.

---

## 🤝 Community

- [Contributing Guide](CONTRIBUTING.md)
- [CHANGELOG](CHANGELOG.md) & [AUTHORS](AUTHORS.md)
- Questions? Contact **Diogo Ribeiro** (dfr@esmad.ipp.pt) or open a GitHub issue.

---

## 🔍 Why this library?

- ✅ **Hardware‑agnostic** CSI format and drivers
- ✅ **Cross‑platform** deployment from edge devices to cloud
- ✅ **Research‑friendly** with reproducible benchmarks and advanced algorithms
- ✅ **Privacy‑preserving** sensing without cameras

---

## 📄 License & Citation

Distributed under the [MIT License](LICENSE).

```bibtex
@software{wifi_activity_recognition,
  title   = {WiFi Activity Recognition: A Universal Framework for CSI-based Human Activity Recognition},
  author  = {Ribeiro, Diogo},
  year    = {2025},
  url     = {https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition}
}
```

---

**Happy sensing!** 🛰️

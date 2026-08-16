# WiFi Activity Recognition — Project Roadmap

This roadmap records what is built, what is next, and what is deliberately not
being worked on yet.

It was previously a plan written before the code existed, describing the
project as a planning-phase alpha targeting Q3 2025 with Phase 1 items
unchecked — while most of Phase 1 and Phase 2 had shipped. This version
describes the repository as it is.

## 🎯 Project Vision

A comprehensive, correct and honestly-documented WiFi sensing package for
activity recognition, usable by researchers who need results that survive peer
review and by practitioners who need software that installs and runs.

## 📍 Current status

**Version**: 0.2.0
**Published**: [PyPI](https://pypi.org/project/wifi-activity-recognition/) ·
[Zenodo DOI](https://doi.org/10.5281/zenodo.21935219)
**CI**: twelve jobs, green on Python 3.10–3.12
**Tests**: 350+, coverage 83%

The emphasis has shifted from adding features to making the existing surface
demonstrably correct. 0.2.0 was largely that work.

--------------------------------------------------------------------------------

## ✅ Shipped

### Core infrastructure

- [x] Base CSI reader interface and driver registry
- [x] Standardized `CSIData` format
- [x] Hardware profile configuration
- [x] Plugin architecture for drivers
- [x] Packaging that produces a complete, installable wheel
- [x] CI that builds the wheel, installs it clean and runs the console scripts

### Hardware

- [x] Intel 5300 driver, with file replay
- [x] ESP32 driver with real binary parsing and firmware detection
- [x] Atheros AR9300 driver
- [x] Qualcomm driver (network, no mock mode)
- [x] Headless mock capture for Intel 5300, ESP32 and Atheros
- [x] Real-device verification procedure and script
- [ ] Broadcom — no driver; listed in `PLANNED_HARDWARE`
- [ ] MediaTek — no driver; listed in `PLANNED_HARDWARE`

### Models

- [x] CNN2D, ResNet spectrogram
- [x] CNN3D and attention CNN3D
- [x] Vision Transformer
- [x] Ensemble
- [x] Transformer with sinusoidal positional encoding
- [x] Behaviour-level correctness tests across every registered family
- [x] Representation adapters so every registered model is usable for inference
- [ ] Multi-head attention across antennas

### Preprocessing and features

- [x] Filtering, normalization, calibration, outlier and artifact removal
- [x] Segmentation, multipath analysis
- [x] Time, frequency, spectrogram, wavelet, Doppler, fractal, graph and
      information-theoretic features
- [x] Temporal filtering along a real time axis, over packet sequences
- [x] Reference-signal and invariant validation for preprocessing

### Evaluation

- [x] Group-aware splitting (`split_dataset_by_groups`)
- [x] Leave-one-subject / session / environment-out (`leave_one_group_out`)
- [x] Grouped cross-validation via `StratifiedGroupKFold`
- [x] Documented evaluation protocol and reporting expectations

### Deployment and tooling

- [x] Docker images that build and run, verified in CI
- [x] Edge optimization: quantization, pruning, ONNX export
- [x] Kubernetes manifest (example, not a verified path)
- [x] Benchmarks for accuracy, latency and memory
- [x] Enforced performance regression policy
- [x] `wifi-har-quickstart` — full cycle in ~10s, no hardware
- [x] Blocking lint and format gate
- [x] Dependabot version updates

--------------------------------------------------------------------------------

## 🎯 Next: correctness foundations

Ordered by value. These are the items the project keeps working around.

### 1. `CSISequence` — a first-class temporal type

`CSIData` is a single packet with axes `(rx, tx, subcarrier)`. None is time.
Three separate pieces of work have now had to route around that: temporal
filters take packet lists, group metadata travels beside the arrays because
`Dataset` cannot hold it, and the inference adapters rebuild a time axis from a
list on every call.

- [ ] `CSISequence` with explicit time, timestamps and sampling frequency
- [ ] Subject, session, environment and device metadata on the sequence
- [ ] Antenna and subcarrier metadata
- [ ] Migrate preprocessing, features, datasets and inference onto it

### 2. Real dataset adapters

- [ ] Parse the Widar3 native format rather than delegating to the NumPy loader
- [ ] Parse SignFi `.mat` files
- [ ] Surface the subject and environment identifiers those datasets record, so
      group-aware evaluation works on public data

Without this, LOSO is a capability the package has but cannot yet apply to any
established benchmark.

### 3. Reproducible benchmark protocol

- [ ] Subject-independent and environment-independent baselines
- [ ] Macro-F1, balanced accuracy, confusion matrices, confidence intervals
- [ ] Per-fold results rather than means alone
- [ ] Published reference numbers with the protocol that produced them

### 4. Research modules on learned representations

`DomainAdapter.adapt_to_target` reduces each packet to a mean amplitude and
ignores its `method` argument; the few-shot layer does something similar.

- [ ] CORAL, MMD and DANN over learned CSI embeddings
- [ ] Few-shot learning on real representations
- [ ] Keep the namespace marked experimental until it does

### 5. Type safety

- [ ] Clear the 245 mypy errors tracked in `docs/lint_status.md`
- [ ] Make the type-check job blocking

--------------------------------------------------------------------------------

## 🔭 Later

- [ ] Multi-modal fusion validated end to end
- [ ] Federated learning beyond simulation
- [ ] An HTTP service interface, which would make the Kubernetes path real
- [ ] ARM image builds and Raspberry Pi verification
- [ ] TensorFlow parity, or removal of the unreachable TF variants

--------------------------------------------------------------------------------

## 🛠️ Standards

### Performance

Enforced by `tests/benchmarks/test_performance_regression.py` using
complexity, memory and relative checks rather than wall-clock thresholds, which
are unreliable on shared runners. See `docs/performance_policy.md`.

### Quality

| Standard | Status |
|---|---|
| Lint and format | enforced, blocking |
| Test coverage | 83% |
| Type hints on public APIs | partial — 245 mypy errors outstanding |
| Performance regression | enforced |
| Wheel installability | verified in CI |
| Container build and run | verified in CI |
| Real-device verification | procedure documented, no runs recorded |

--------------------------------------------------------------------------------

## 🤝 Contributing

Highest-value contributions right now:

- **Hardware owners**: run `scripts/verify_device.py` and record the result in
  `docs/hardware_verification.md`. No platform has a recorded run, so every
  hardware claim currently rests on mock coverage.
- **Researchers**: dataset adapters for Widar3 and SignFi, and baseline numbers
  under a subject-independent protocol.
- **Anyone**: the mypy backlog in `docs/lint_status.md` is ordered
  cheapest-first and is a good way in.

See [CONTRIBUTING.md](CONTRIBUTING.md).

--------------------------------------------------------------------------------

## 🔄 Keeping this current

This file is updated at each release. If it describes work that is already done,
or claims support that does not exist, that is a bug — please open an issue.

# API Reference

High‑level overview of the modules exposed by the package. Refer to the source
code for complete docstrings and advanced usage.

## Hardware

Classes implementing `CSIReaderBase` for different chipsets:

- `Intel5300Reader`
- `ESP32Reader`
- `AtherosReader`

## Preprocessing

Utilities to clean and segment CSI:

- `normalize_amplitude`, `normalize_phase`
- `butterworth_filter`, `kalman_filter`
- `segment_windows`

## Features

Feature extraction helpers used for downstream models:

- `rms_energy`, `zero_crossing_rate`
- `short_time_fourier_transform`
- `doppler_spectrum`

## Models

Model architectures registered in the factory:

- `cnn2d`, `resnet`, `cnn3d`, `transformer`
- `EnsembleModel` to average predictions

## Datasets

- `Dataset.from_files` to build train/validation/test splits
- Public dataset helpers: `load_widar`, `load_signfi`

## Training

- `Trainer` orchestrates epochs and checkpointing
- CLI `train` command provides end‑to‑end execution

## Utilities

- `config.load_config` and `validate_config`
- `io.save_csi` / `io.load_csi`
- `visualization.plot_heatmap`

## Deployment

- Docker and Kubernetes manifests under `deployment/`
- Raspberry Pi optimization script

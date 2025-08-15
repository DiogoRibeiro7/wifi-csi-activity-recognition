# Quickstart

This five‑minute guide demonstrates a minimal end‑to‑end workflow.

## 1. Install

Follow the [installation guide](installation.md) to set up the environment.

## 2. Collect Sample Data

```bash
python -m wifi_activity_recognition.cli listen --device intel5300 --duration 5
```

This saves a small recording under `data/raw/`.

## 3. Train a Model

```bash
python -m wifi_activity_recognition.cli train \
    --config configs/default.yaml \
    --data data/raw/
```

## 4. Evaluate

```bash
python -m wifi_activity_recognition.cli evaluate --checkpoint runs/latest.ckpt
```

The commands above gather data, train a default model, and report accuracy. For
more advanced usage, consult the training and deployment guides.

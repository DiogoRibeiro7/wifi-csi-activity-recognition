# Quickstart

This five‑minute guide demonstrates a minimal end‑to‑end workflow.

## 1. Install

Follow the [installation guide](installation.md) to set up the environment.

## 2. Obtain Example Data

Download the small sample dataset used in tests:

```bash
python -m wifi_activity_recognition.cli download-demo --out data/demo
```

Expected output:

```
Downloaded 3 recordings to data/demo
```

## 3. Evaluate a Pretrained Model

```bash
python -m wifi_activity_recognition.cli evaluate \
    --checkpoint examples/checkpoints/resnet_demo.ckpt \
    --data data/demo
```

Expected output:

```
Accuracy: 0.92
```

## 4. Train Your Own Model

```bash
python -m wifi_activity_recognition.cli train \
    --config configs/default.yaml \
    --data data/demo
```

During training you should see epoch‑wise metrics such as:

```
Epoch 1/5 - loss: 1.23 - acc: 0.55
```

## 5. Next Steps

The commands above download data, run inference using a pretrained checkpoint,
and train a small model. For more advanced usage, consult the training and
deployment guides.

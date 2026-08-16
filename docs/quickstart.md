# Quickstart

Run a complete training, evaluation and prediction cycle in about ten seconds,
without hardware and without downloading anything.

## 1. Install

```bash
pip install "wifi-activity-recognition[torch]"
```

Requires Python 3.10 or newer. For local development use an editable install
instead:

```bash
pip install -e ".[dev,docs,torch]"
```

## 2. Run the demo

```bash
wifi-har-quickstart
```

That single command generates a synthetic CSI dataset, trains a CNN2D model on
it, evaluates on a held-out split, saves a model artifact and runs a prediction
with the reloaded model:

```text
[1/5] Generating synthetic CSI...
      240 samples of shape (1, 8, 32) -> quickstart_demo
[2/5] Loading as a Dataset...
      144 train samples, 3 classes [0, 1, 2]
[3/5] Training cnn2d for 8 epochs...
      training  [####################################]  100%
[4/5] Evaluating on the held-out split...
      accuracy=1.000  f1=1.000
      saved model artifact -> quickstart_demo/demo_model.pt
[5/5] Predicting with the reloaded model...
      predicted class 0, actual 0

Quickstart complete (quickstart_demo).
```

The same run is also available as `python -m wifi_activity_recognition quickstart`.

### Why the accuracy is high

The synthetic task is genuinely learnable, not random: each class is encoded as
a different sine frequency across subcarriers. That is the point — a demo built
on random arrays with random labels would train to chance and tell you nothing
about whether your install works.

Options: `--epochs`, `--samples`, `--seed`, `--output-dir`. A fixed seed
produces identical data every run, so results are reproducible.

## 3. What it leaves behind

```text
quickstart_demo/
├── demo_data.npy     # (240, 1, 8, 32) float32 -- samples, channels, antennas, subcarriers
├── demo_labels.npy   # (240,) int -- class index per sample
└── demo_model.pt     # structured artifact: state_dict + model_spec + metadata
```

These are ordinary NumPy arrays and a standard model artifact, so every command
below works on them unchanged.

## 4. Use the individual commands

The quickstart runs the same APIs the real commands use. To drive them yourself:

```bash
wifi-har-train \
  --data quickstart_demo/demo_data.npy \
  --labels quickstart_demo/demo_labels.npy \
  --model cnn2d \
  --hardware esp32 \
  --epochs 8 \
  --batch-size 16 \
  --output my_model.pt
```

```bash
python -m wifi_activity_recognition evaluate \
  --model my_model.pt \
  --data quickstart_demo/demo_data.npy \
  --labels quickstart_demo/demo_labels.npy \
  --hardware esp32
```

`python -m wifi_activity_recognition --help` lists everything available:
`train`, `evaluate`, `predict`, `stream`, `collect`, `autotrain`, `benchmark`,
`export`, `visualize`, `info`, `live` and `quickstart`.

## 5. Move to your own data

Replace the demo arrays with your own. The shapes are the contract:

- **data**: `(samples, channels, antennas, subcarriers)`, float
- **labels**: `(samples,)`, integer class indices

To capture from a device instead, see [hardware_setup.md](hardware_setup.md).
Intel 5300, ESP32, Atheros AR9300 and Qualcomm have registered drivers;
`python -m wifi_activity_recognition info` prints what is available in your
install.

## Public datasets

Established CSI datasets exist — UT-HAR, SignFi, Widar 3.0 — but they are
gigabyte-scale, hosted on Google Drive, Box or IEEE DataPort, and several carry
licences that forbid redistribution. None of that suits a first-run experience,
which is why the quickstart is synthetic and self-contained. Loading a real
dataset is a separate step with its own requirements.

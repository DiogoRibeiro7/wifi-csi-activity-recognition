# Installation Guide

This guide covers the current package install flow and the minimum checks needed before using the CLI or Python API.

## 1. Python version

Use Python 3.10, 3.11, or 3.12.

## 2. Clone the repository

```bash
git clone https://github.com/diogoribeiro7/wifi-csi-activity-recognition.git
cd wifi-csi-activity-recognition
```

## 3. Create a virtual environment

### Linux/macOS/WSL

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 4. Install the package

For normal usage:

```bash
pip install wifi-activity-recognition
```

For local development:

```bash
pip install -e .[dev,docs]
```

If you plan to train or run inference, install a compatible PyTorch build for your platform after the base install.

## 5. Verify the software install

Run the lightweight test suite:

```bash
pytest -q
```

Then inspect the current hardware registry:

```bash
python -m wifi_activity_recognition.cli info --hardware all
```

## 6. Platform notes

- Linux or WSL is the most realistic environment for hardware-backed CSI workflows.
- Native Windows can be useful for development and non-hardware tests, but real driver support is limited.
- GPU support depends on the PyTorch build you install separately.

## 7. Hardware-specific setup

Use [hardware_setup.md](hardware_setup.md) for Intel 5300, ESP32, Atheros AR9300, and Qualcomm setup notes.

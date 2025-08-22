# Installation Guide

This guide walks through software and driver installation on **Linux**,
**macOS**, and **Windows** hosts. A Python 3.10+ environment is recommended.

## 1. Obtain the Source

```bash
git clone https://github.com/diogoribeiro7/wifi-csi-activity-recognition.git
cd wifi-csi-activity-recognition
```

## 2. Install System Prerequisites

| Platform | Packages |
| --- | --- |
| Ubuntu/Debian | `build-essential libpcap-dev python3-venv git` |
| Fedora | `gcc-c++ libpcap-devel python3-virtualenv git` |
| macOS (Homebrew) | `brew install libpcap python@3 git` |
| Windows (WSL) | install Ubuntu/Fedora packages inside WSL |

> **Note:** Native Windows lacks official CSI drivers; use **WSL** for full
> functionality.

## 3. Create a Virtual Environment

### Linux/macOS/WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell, limited functionality)

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 4. Install Python Dependencies

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## 5. Install Hardware Drivers

### Intel 5300
- Linux: build and install the modified `iwlwifi` driver from the
  [Linux 5300 CSI Tool](https://github.com/dhalperi/linux-80211n-csitool).
- Windows/macOS: not supported; use an ESP32 device instead.

### ESP32
- Flash the CSI-enabled firmware from
  [Espressif's repo](https://github.com/espressif/esp32-wifi-csi).
- Install USB‑serial drivers if required (`cp210x` on Windows/macOS).

### Atheros AR9300
- Ensure the `ath9k` driver is present (Linux kernel 3.2+).
- Load the module in monitor mode with `sudo modprobe ath9k`.

## 6. Final Verification

1. **Run tests** to verify software components:

   ```bash
   pytest -q
   ```

2. **Check hardware connectivity** using the CLI:

   ```bash
   python -m wifi_activity_recognition.cli info --device intel5300
   ```

   The command should display device information without errors.

## 7. Optional: GPU Support

If training on a GPU, install the appropriate PyTorch build from
[pytorch.org](https://pytorch.org/get-started/locally/) after activating the
environment.

Your system is now ready for data collection and model training.

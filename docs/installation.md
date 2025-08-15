# Installation Guide

Follow the steps below to set up the WiFi Activity Recognition package on
Linux, macOS, or Windows. A Python 3.10+ environment is recommended.

## 1. Obtain the Source

```bash
git clone https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition.git
cd wifi-csi-activity-recognition
```

## 2. Create a Virtual Environment

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## 4. Verify the Installation

Run the unit tests to ensure the environment is working correctly:

```bash
pytest -q
```

## 5. Optional: GPU Support

If training on a GPU, install the appropriate PyTorch build from
[pytorch.org](https://pytorch.org/get-started/locally/) after activating the
environment.

Your environment is now ready for experimentation and development.

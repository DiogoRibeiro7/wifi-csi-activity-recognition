import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    compute_spectrogram,
)


def test_compute_spectrogram_shape():
    signal = np.sin(2 * np.pi * 0.1 * np.arange(128))
    f, t, spec = compute_spectrogram(signal, fs=1.0, nperseg=32)
    assert spec.shape[0] == len(f)
    assert spec.shape[1] == len(t)

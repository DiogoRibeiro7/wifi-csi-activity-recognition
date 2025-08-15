"""Tests for Doppler spectrum computation."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    doppler_spectrum,
)


def test_doppler_spectrum_shape() -> None:
    """Doppler spectrum retains original array shape."""
    signal = np.random.randn(16, 2, 2)
    spec = doppler_spectrum(signal, axis=0)
    assert spec.shape == signal.shape

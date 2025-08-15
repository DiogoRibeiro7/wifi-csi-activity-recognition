"""Tests for time-domain feature utilities."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    compute_rms,
    zero_crossing_rate,
)


def test_compute_rms() -> None:
    """RMS computation returns expected value."""
    data = np.array([[1.0, 2.0, 2.0]])
    rms = compute_rms(data, axis=1)
    assert np.allclose(rms, np.sqrt((1 + 4 + 4) / 3))


def test_zero_crossing_rate() -> None:
    """Zero crossing rate counts sign changes."""
    signal = np.array([[1, -1, 1, -1]])
    zcr = zero_crossing_rate(signal, axis=1)
    assert np.isclose(zcr, 0.75)

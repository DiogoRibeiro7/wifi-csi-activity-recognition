"""Tests for calibration utilities."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)
from wifi_activity_recognition.preprocessing import (  # type: ignore  # noqa: E402
    phase_unwrap,
    remove_dc_offset,
)


def _make_csi(amplitude: np.ndarray, phase: np.ndarray) -> CSIData:
    n_rx, n_tx, n_sc = amplitude.shape
    return CSIData(
        timestamp=0.0,
        amplitude=amplitude,
        phase=phase,
        frequency=5.0,
        bandwidth=20.0,
        n_tx=n_tx,
        n_rx=n_rx,
        n_subcarriers=n_sc,
    )


def test_remove_dc_offset() -> None:
    """Remove DC offset along subcarriers."""
    amp = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    phase = np.zeros_like(amp)
    csi = _make_csi(amp, phase)
    calibrated = remove_dc_offset(csi, axis=-1)
    assert np.allclose(np.mean(calibrated.amplitude, axis=-1), [[0.0, 0.0]])


def test_phase_unwrap() -> None:
    """Unwrap phase along subcarriers."""
    amp = np.ones((1, 1, 3))
    phase = np.array([[[0.0, np.pi - 0.1, -np.pi + 0.1]]])
    csi = _make_csi(amp, phase)
    unwrapped = phase_unwrap(csi)
    diffs = unwrapped.phase[0, 0, 1:] - unwrapped.phase[0, 0, :-1]
    assert np.all(diffs < np.pi)

"""Tests for normalization utilities."""

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
    log_normalize,
    min_max_normalize,
    z_score_normalize,
)


def _make_csi(amplitude: np.ndarray) -> CSIData:
    phase = np.zeros_like(amplitude)
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


def test_min_max_normalize() -> None:
    """Normalize amplitude with min-max scaling along subcarriers."""
    amp = np.array([[[1.0, 3.0]]])
    csi = _make_csi(amp)
    norm = min_max_normalize(csi)
    assert np.allclose(norm.amplitude, [[[0.0, 1.0]]])


def test_min_max_normalize_axis() -> None:
    """Normalize amplitude with min-max scaling along receivers."""
    amp = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    csi = _make_csi(amp)
    norm = min_max_normalize(csi, axis=0)
    assert np.allclose(norm.amplitude, [[[0.0, 0.0]], [[1.0, 1.0]]])


def test_z_score_normalize() -> None:
    """Standardize amplitude along subcarriers."""
    amp = np.array([[[1.0, 3.0]]])
    csi = _make_csi(amp)
    norm = z_score_normalize(csi)
    assert np.allclose(np.mean(norm.amplitude, axis=-1), [[0.0]])
    assert np.allclose(np.std(norm.amplitude, axis=-1), [[1.0]])


def test_z_score_normalize_axis() -> None:
    """Standardize amplitude along receivers."""
    amp = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    csi = _make_csi(amp)
    norm = z_score_normalize(csi, axis=0)
    assert np.allclose(np.mean(norm.amplitude, axis=0), [[0.0, 0.0]])
    assert np.allclose(np.std(norm.amplitude, axis=0), [[1.0, 1.0]])


def test_log_normalize() -> None:
    """Log-scale amplitude values."""
    amp = np.array([[[1.0, np.e]]])
    csi = _make_csi(amp)
    norm = log_normalize(csi)
    assert np.allclose(norm.amplitude, [[[0.0, 1.0]]])

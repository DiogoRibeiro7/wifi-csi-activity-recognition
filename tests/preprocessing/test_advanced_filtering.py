"""Tests for advanced filtering utilities."""

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
    adaptive_wiener_filter,
    median_filter,
    multirate_resample,
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


def test_adaptive_wiener_filter_reduces_variance() -> None:
    """Wiener filter should reduce noise variance."""
    rng = np.random.default_rng(0)
    signal = np.ones((1, 1, 64))
    noisy = signal + rng.normal(scale=0.5, size=signal.shape)
    csi = _make_csi(noisy)
    filtered = adaptive_wiener_filter(csi, mysize=5)
    assert np.var(filtered.amplitude) < np.var(csi.amplitude)


def test_median_filter_validation() -> None:
    """Median filter rejects even kernel sizes."""
    csi = _make_csi(np.ones((1, 1, 8)))
    try:
        median_filter(csi, kernel_size=4)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_multirate_resample_shape() -> None:
    """Resampling updates the number of subcarriers."""
    csi = _make_csi(np.ones((1, 1, 8)))
    resampled = multirate_resample(csi, up=2, down=1)
    assert resampled.n_subcarriers == 16
    assert csi.n_subcarriers == 8

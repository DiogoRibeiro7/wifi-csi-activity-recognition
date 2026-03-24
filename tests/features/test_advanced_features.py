"""Tests for advanced feature extraction."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    sample_entropy,
    spectral_entropy,
    statistical_moments,
)
from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
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


def test_statistical_moments() -> None:
    """Moments match numpy calculations."""
    amp = np.array([[[1.0, 2.0, 3.0]]])
    csi = _make_csi(amp)
    moments = statistical_moments(csi)
    assert np.allclose(moments["mean"], [[2.0]])
    assert np.allclose(moments["variance"], [[2.0 / 3]])


def test_spectral_entropy_uniform() -> None:
    """Constant signal has near-zero spectral entropy."""
    amp = np.ones((1, 1, 8))
    csi = _make_csi(amp)
    ent = spectral_entropy(csi)
    assert ent < 0.1


def test_sample_entropy_constant() -> None:
    """Constant sequences yield low sample entropy."""
    amp = np.ones((1, 1, 20))
    csi = _make_csi(amp)
    ent = sample_entropy(csi, m=2, r=0.2)
    assert np.all(ent < 1e-6)


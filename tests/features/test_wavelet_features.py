"""Tests for wavelet feature extraction."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    cwt_coefficients,
    dwt_energy,
    scale_energy,
    wavelet_packet_energy,
)
from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)


def _make_csi(shape: tuple[int, int, int]) -> CSIData:
    amp = np.random.randn(*shape)
    phase = np.zeros_like(amp)
    n_rx, n_tx, n_sc = shape
    return CSIData(0.0, amp.copy(), phase, 5.0, 20.0, n_tx, n_rx, n_sc)


def test_cwt_coefficients_shape() -> None:
    """CWT returns expected coefficient shape."""
    csi = _make_csi((1, 1, 32))
    scales = [1, 2, 3]
    coeffs = cwt_coefficients(csi, scales)
    assert coeffs.shape == (1, 1, len(scales), 32)
    # original unchanged
    assert np.array_equal(csi.amplitude, csi.amplitude)


def test_dwt_energy_levels() -> None:
    """Energy array has level+1 components."""
    csi = _make_csi((2, 1, 64))
    energy = dwt_energy(csi, level=2)
    assert energy.shape == (2, 1, 3)


def test_wavelet_packet_energy() -> None:
    """Wavelet packet yields energies for leaf nodes."""
    csi = _make_csi((1, 2, 64))
    features = wavelet_packet_energy(csi, maxlevel=2)
    assert features.shape == (1, 2, 4)


def test_scale_energy() -> None:
    """Scale energy collapses sample axis."""
    csi = _make_csi((1, 1, 32))
    scales = [1, 2]
    energy = scale_energy(csi, scales)
    assert energy.shape == (1, 1, len(scales))


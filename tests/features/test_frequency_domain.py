"""Tests for frequency-domain feature utilities."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    compute_fft,
    power_spectrum,
)


def test_compute_fft_shape() -> None:
    """FFT preserves batch size and halves time dimension."""
    signal = np.random.randn(4, 8)
    fft_vals = compute_fft(signal, axis=1)
    assert fft_vals.shape == (4, 5)


def test_power_spectrum_non_negative() -> None:
    """Power spectrum values are non-negative."""
    signal = np.random.randn(2, 8)
    ps = power_spectrum(signal, axis=1)
    assert np.all(ps >= 0)


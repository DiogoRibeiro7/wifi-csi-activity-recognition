"""Tests for filtering utilities."""

import numpy as np

from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)
from wifi_activity_recognition.preprocessing import (  # type: ignore  # noqa: E402
    butterworth_filter,
    kalman_filter,
    moving_average_filter,
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


def test_moving_average_filter() -> None:
    """Smooth amplitude with a moving average filter."""
    amp = np.ones((1, 1, 10))
    amp[0, 0, 5] = 5
    csi = _make_csi(amp)
    filtered = moving_average_filter(csi, window_size=3)
    assert filtered.amplitude.shape == csi.amplitude.shape
    assert filtered.amplitude[0, 0, 5] < 5


def test_butterworth_filter() -> None:
    """Filter amplitude with a Butterworth filter."""
    fs = 100.0
    t = np.linspace(0, 1, int(fs))
    data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    amp = data.reshape(1, 1, -1)
    csi = _make_csi(amp)
    filtered = butterworth_filter(csi, cutoff=10, fs=fs)
    assert filtered.amplitude.shape == amp.shape


def test_kalman_filter() -> None:
    """Filter amplitude with a Kalman filter."""
    rng = np.random.default_rng(0)
    signal = np.sin(np.linspace(0, 2 * np.pi, 50))
    noisy = signal + rng.normal(scale=0.1, size=signal.shape)
    amp = noisy.reshape(1, 1, -1)
    csi = _make_csi(amp)
    filtered = kalman_filter(csi)
    mse_filtered = ((filtered.amplitude[0, 0] - signal) ** 2).mean()
    mse_noisy = ((noisy - signal) ** 2).mean()
    assert mse_filtered < mse_noisy

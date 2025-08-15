"""Tests for filtering utilities."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.preprocessing import (  # type: ignore  # noqa: E402
    butterworth_filter,
    kalman_filter,
    moving_average_filter,
)


def test_moving_average_filter() -> None:
    """Smooth data with a moving average filter along axis 0."""
    data = np.ones((10, 1))
    data[5] = 5
    filtered = moving_average_filter(data, window_size=3)
    assert filtered.shape == data.shape
    assert filtered[5, 0] < 5


def test_moving_average_filter_axis() -> None:
    """Smooth data with a moving average filter along axis 1."""
    data = np.ones((1, 10))
    data[0, 5] = 5
    filtered = moving_average_filter(data, window_size=3, axis=1)
    assert filtered.shape == data.shape
    assert filtered[0, 5] < 5


def test_butterworth_filter() -> None:
    """Filter data with a Butterworth filter along axis 0."""
    fs = 100.0
    t = np.linspace(0, 1, int(fs))
    data = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
    filtered = butterworth_filter(data[:, None], cutoff=10, fs=fs)
    assert filtered.shape == data[:, None].shape


def test_butterworth_filter_axis() -> None:
    """Filter data with a Butterworth filter along axis 1."""
    fs = 100.0
    t = np.linspace(0, 1, int(fs))
    data = (np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t))[None, :]
    filtered = butterworth_filter(data, cutoff=10, fs=fs, axis=1)
    assert filtered.shape == data.shape


def test_kalman_filter() -> None:
    """Filter data with a Kalman filter along axis 0."""
    rng = np.random.default_rng(0)
    signal = np.sin(np.linspace(0, 2 * np.pi, 50))
    noisy = signal + rng.normal(scale=0.1, size=signal.shape)
    filtered = kalman_filter(noisy[:, None])[:, 0]
    assert ((filtered - signal) ** 2).mean() < ((noisy - signal) ** 2).mean()


def test_kalman_filter_axis() -> None:
    """Filter data with a Kalman filter along axis 1."""
    rng = np.random.default_rng(0)
    signal = np.sin(np.linspace(0, 2 * np.pi, 50))[None, :]
    noisy = signal + rng.normal(scale=0.1, size=signal.shape)
    filtered = kalman_filter(noisy, axis=1)
    assert ((filtered - signal) ** 2).mean() < ((noisy - signal) ** 2).mean()

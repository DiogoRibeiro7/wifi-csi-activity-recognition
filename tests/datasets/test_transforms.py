"""Tests for dataset augmentation transforms."""

import numpy as np

from wifi_activity_recognition.datasets import (  # type: ignore  # noqa: E402
    add_noise,
    time_shift,
)


def test_add_noise_deterministic():
    """Noise addition is deterministic with fixed seed."""
    data = np.zeros((5, 3))
    noisy = add_noise(data, noise_std=0.1, random_state=42)
    noisy_again = add_noise(data, noise_std=0.1, random_state=42)
    assert np.allclose(noisy, noisy_again)
    assert not np.allclose(noisy, data)


def test_time_shift():
    """Default axis shift moves along time dimension."""
    data = np.arange(10).reshape(10, 1)
    shifted = time_shift(data, 2)
    assert shifted[0, 0] == data[-2, 0]


def test_time_shift_axis():
    """Axis parameter selects dimension to roll."""
    data = np.arange(6).reshape(2, 3)
    shifted = time_shift(data, 1, axis=1)
    assert np.array_equal(shifted[0], np.roll(data[0], 1))

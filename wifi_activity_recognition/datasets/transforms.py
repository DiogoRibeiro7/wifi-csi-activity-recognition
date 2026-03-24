"""Data augmentation helpers for CSI data."""

from typing import Optional

import numpy as np


def add_noise(
    data: np.ndarray, noise_std: float, random_state: Optional[int] = None
) -> np.ndarray:
    """Add Gaussian noise with standard deviation ``noise_std``.

    The function returns a new array with noise added while leaving the
    original ``data`` untouched. ``random_state`` can be used to make the
    operation deterministic.
    """
    rng = np.random.default_rng(random_state)
    noise = rng.normal(scale=noise_std, size=data.shape)
    return data + noise.astype(data.dtype)


def time_shift(data: np.ndarray, shift: int, axis: int = 0) -> np.ndarray:
    """Circularly shift CSI data along the given axis."""
    return np.roll(data, shift=shift, axis=axis)


__all__ = ["add_noise", "time_shift"]

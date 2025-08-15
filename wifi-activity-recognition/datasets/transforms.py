"""Data augmentation helpers for CSI data."""

from typing import Optional

import numpy as np


def add_noise(
    data: np.ndarray, noise_std: float, random_state: Optional[int] = None
) -> np.ndarray:
    """Add Gaussian noise to CSI data."""
    rng = np.random.default_rng(random_state)
    return data + rng.normal(scale=noise_std, size=data.shape)


def time_shift(data: np.ndarray, shift: int) -> np.ndarray:
    """Circularly shift CSI data along the time axis (axis 0)."""
    return np.roll(data, shift=shift, axis=0)


__all__ = ["add_noise", "time_shift"]

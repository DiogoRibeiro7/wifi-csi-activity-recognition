"""Synthetic CSI data generation utilities."""

from typing import Optional, Tuple

import numpy as np


def generate_synthetic_csi(
    num_samples: int,
    num_subcarriers: int,
    num_antennas: int = 1,
    num_classes: int = 2,
    noise_std: float = 0.0,
    random_state: Optional[int] = None,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic CSI data with sine-wave patterns for each class.

    ``dtype`` specifies the floating point type of the generated data.  Each
    sample contains ``num_antennas`` antennas and ``num_subcarriers``
    subcarriers.  ``labels`` are integer class indices in ``[0, num_classes)``.
    """
    rng = np.random.default_rng(random_state)
    data = np.zeros((num_samples, num_antennas, num_subcarriers), dtype=dtype)
    labels = rng.integers(0, num_classes, size=num_samples)
    for idx in range(num_samples):
        freq = labels[idx] + 1
        for ant in range(num_antennas):
            x = np.linspace(0, 2 * np.pi * freq, num_subcarriers)
            data[idx, ant] = np.sin(x).astype(dtype)
    if noise_std > 0:
        data += rng.normal(scale=noise_std, size=data.shape).astype(dtype)
    return data, labels


__all__ = ["generate_synthetic_csi"]

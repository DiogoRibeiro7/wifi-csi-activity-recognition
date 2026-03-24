"""Computer vision style transforms for CSI matrices."""

from __future__ import annotations

import numpy as np


def magnitude_to_uint8(magnitude: np.ndarray) -> np.ndarray:
    """Scale magnitude matrix to uint8 image representation."""
    mag = magnitude - magnitude.min()
    if mag.max() > 0:
        mag = mag / mag.max()
    return (mag * 255).astype(np.uint8)


__all__ = ["magnitude_to_uint8"]

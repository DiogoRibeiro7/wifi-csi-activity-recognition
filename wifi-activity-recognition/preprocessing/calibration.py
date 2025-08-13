"""Calibration routines for CSI data."""

from __future__ import annotations

import numpy as np


def remove_dc_offset(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Remove constant bias along a specified axis."""
    mean = np.mean(data, axis=axis, keepdims=True)
    return data - mean


def phase_unwrap(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """Unwrap phase along a specified axis."""
    return np.unwrap(data, axis=axis)

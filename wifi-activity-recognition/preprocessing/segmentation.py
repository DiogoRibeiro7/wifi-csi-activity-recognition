"""Segmentation utilities for time-series CSI data."""

from __future__ import annotations

import numpy as np


def segment_windows(
    data: np.ndarray, window_size: int, overlap: float = 0.5, axis: int = 0
) -> np.ndarray:
    """Segment data into overlapping windows.

    Parameters
    ----------
    data:
        Array containing the time dimension.
    window_size:
        Number of samples per window.
    overlap:
        Fractional overlap between consecutive windows in [0, 1).
    axis:
        Axis representing time. Defaults to 0.
    """
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    data_moved = np.moveaxis(data, axis, 0)
    step = int(window_size * (1 - overlap)) or 1
    windows = [
        data_moved[i : i + window_size]
        for i in range(0, len(data_moved) - window_size + 1, step)
    ]
    if not windows:
        raise ValueError("window_size larger than data length")
    segmented = np.stack(windows)
    return np.moveaxis(segmented, 1, axis + 1)

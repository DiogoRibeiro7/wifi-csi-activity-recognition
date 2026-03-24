"""Segmentation utilities for sequences of :class:`CSIData`."""

from __future__ import annotations

from typing import List, Sequence

from ..hardware.base import CSIData


def segment_windows(
    data: Sequence[CSIData],
    window_size: int,
    overlap: float = 0.5,
) -> List[List[CSIData]]:
    """Segment a CSI sequence into overlapping windows.

    Parameters
    ----------
    data:
        Sequence of CSI samples ordered in time.
    window_size:
        Number of samples per window.
    overlap:
        Fractional overlap between consecutive windows in ``[0, 1)``.

    Returns
    -------
    list of list of CSIData
        Segmented windows preserving original order.
    """
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    step = int(window_size * (1 - overlap)) or 1
    windows: List[List[CSIData]] = [
        list(data[i : i + window_size])
        for i in range(0, len(data) - window_size + 1, step)
    ]
    if not windows:
        raise ValueError("window_size larger than data length")
    return windows

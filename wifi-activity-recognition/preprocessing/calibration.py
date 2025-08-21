"""Calibration routines for :class:`CSIData`."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from ..hardware.base import CSIData


def remove_dc_offset(
    csi: CSIData,
    field: str = "amplitude",
    axis: int = -1,
) -> CSIData:
    """Remove constant bias from a field along a specified axis."""
    data = getattr(csi, field)
    mean = np.mean(data, axis=axis, keepdims=True)
    return replace(csi, **{field: data - mean})


def phase_unwrap(csi: CSIData, axes: Iterable[int] = (-1,)) -> CSIData:
    """Unwrap phase along one or more axes.

    Parameters
    ----------
    csi:
        Input CSI sample.
    axes:
        Iterable of axes along which to unwrap sequentially. Use ``(-1,)`` to
        unwrap across subcarriers or ``(0, 1, -1)`` to additionally unwrap
        across receive and transmit antennas.

    Returns
    -------
    :class:`CSIData`
        CSI sample with unwrapped phase.

    Raises
    ------
    ValueError
        If ``axes`` is empty or contains an out-of-range axis index.
    """
    axes = tuple(axes)
    if not axes:
        raise ValueError("axes cannot be empty")
    phase = csi.phase
    for ax in axes:
        if ax >= phase.ndim or ax < -phase.ndim:
            raise ValueError("axis out of range")
        phase = np.unwrap(phase, axis=ax)
    return replace(csi, phase=phase)

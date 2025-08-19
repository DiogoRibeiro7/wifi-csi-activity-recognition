"""Calibration routines for :class:`CSIData`."""

from __future__ import annotations

from dataclasses import replace

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


def phase_unwrap(csi: CSIData, axis: int = -1) -> CSIData:
    """Unwrap phase along a specified axis."""
    unwrapped = np.unwrap(csi.phase, axis=axis)
    return replace(csi, phase=unwrapped)

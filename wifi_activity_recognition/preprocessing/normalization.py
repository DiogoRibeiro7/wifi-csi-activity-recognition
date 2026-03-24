"""CSI normalization utilities operating on :class:`CSIData`."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from ..hardware.base import CSIData


def min_max_normalize(
    csi: CSIData,
    fields: Iterable[str] = ("amplitude",),
    axis: int = -1,
) -> CSIData:
    """Scale selected fields to the ``[0, 1]`` range.

    Parameters
    ----------
    csi:
        Input CSI sample.
    fields:
        Iterable of field names to normalise. Defaults to ``("amplitude",)``.
    axis:
        Axis along which to compute statistics. Defaults to the last axis
        (subcarriers).

    Returns
    -------
    CSIData
        New CSI sample with normalised fields.
    """
    updates = {}
    for field in fields:
        data = getattr(csi, field)
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
        denom = data_max - data_min
        denom[denom == 0] = 1
        updates[field] = (data - data_min) / denom
    return replace(csi, **updates)


def z_score_normalize(
    csi: CSIData,
    fields: Iterable[str] = ("amplitude",),
    axis: int = -1,
) -> CSIData:
    """Standardise selected fields using the z-score.

    Parameters
    ----------
    csi:
        Input CSI sample.
    fields:
        Iterable of field names to standardise. Defaults to ``("amplitude",)``.
    axis:
        Axis along which to compute statistics. Defaults to the last axis
        (subcarriers).

    Returns
    -------
    CSIData
        CSI sample with standardised fields.
    """
    updates = {}
    for field in fields:
        data = getattr(csi, field)
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        std[std == 0] = 1
        updates[field] = (data - mean) / std
    return replace(csi, **updates)


def log_normalize(
    csi: CSIData,
    field: str = "amplitude",
    eps: float = 1e-6,
) -> CSIData:
    """Apply logarithmic scaling to a single field.

    Parameters
    ----------
    csi:
        Input CSI sample.
    field:
        Field name to scale, typically ``"amplitude"``.
    eps:
        Small constant to avoid ``log(0)``.

    Returns
    -------
    CSIData
        CSI sample with log-scaled field.
    """
    data = getattr(csi, field)
    scaled = np.log(np.maximum(data, eps))
    return replace(csi, **{field: scaled})

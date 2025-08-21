"""Information theoretic features for CSI data."""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

from ..hardware.base import CSIData


def shannon_entropy(
    csi: CSIData,
    field: str = "amplitude",
    bins: int = 32,
    axis: int = -1,
) -> np.ndarray:
    """Compute Shannon entropy along ``axis``.

    Parameters
    ----------
    csi: CSIData
        Input data.
    field: str
        CSI attribute to analyse.
    bins: int
        Number of histogram bins.
    axis: int
        Axis representing the sequence dimension.
    """
    data = getattr(csi, field)
    data_moved = np.moveaxis(data, axis, 0)
    flat = data_moved.reshape(data_moved.shape[0], -1)
    ent = []
    for series in flat.T:
        hist, _ = np.histogram(series, bins=bins, density=True)
        hist = hist + 1e-12
        ent.append(float(entropy(hist)))
    return np.array(ent).reshape(data_moved.shape[1:])


def mutual_information(
    csi_x: CSIData,
    csi_y: CSIData,
    field: str = "amplitude",
    bins: int = 32,
) -> float:
    """Estimate mutual information between two CSI datasets."""
    x = getattr(csi_x, field).ravel()
    y = getattr(csi_y, field).ravel()
    n = len(x)
    bins = min(bins, max(2, int(np.sqrt(n))))
    x_binned = np.digitize(x, np.histogram_bin_edges(x, bins=bins))
    y_binned = np.digitize(y, np.histogram_bin_edges(y, bins=bins))
    return float(mutual_info_score(x_binned, y_binned))


__all__ = ["shannon_entropy", "mutual_information"]

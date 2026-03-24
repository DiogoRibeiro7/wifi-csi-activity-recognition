"""Correlation-based features for CSI data."""

from __future__ import annotations

import numpy as np


def correlation_matrix(signal: np.ndarray) -> np.ndarray:
    """Compute correlation matrix across the first dimension."""
    reshaped = signal.reshape(signal.shape[0], -1)
    return np.corrcoef(reshaped)


__all__ = ["correlation_matrix"]

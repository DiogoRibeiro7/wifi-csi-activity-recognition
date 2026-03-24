"""Tests for correlation-based features."""

from pathlib import Path

import numpy as np


from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    correlation_matrix,
)


def test_correlation_matrix_shape() -> None:
    """Correlation matrix uses first dimension as variables."""
    signal = np.random.randn(3, 10, 5)
    corr = correlation_matrix(signal)
    assert corr.shape == (3, 3)


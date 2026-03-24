"""Tests for correlation-based features."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    correlation_matrix,
)


def test_correlation_matrix_shape() -> None:
    """Correlation matrix uses first dimension as variables."""
    signal = np.random.randn(3, 10, 5)
    corr = correlation_matrix(signal)
    assert corr.shape == (3, 3)


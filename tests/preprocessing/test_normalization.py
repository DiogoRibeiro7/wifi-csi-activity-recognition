"""Tests for normalization utilities."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.preprocessing import (  # type: ignore  # noqa: E402
    log_normalize,
    min_max_normalize,
    z_score_normalize,
)


def test_min_max_normalize() -> None:
    """Normalize data with min-max scaling along axis 0."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    norm = min_max_normalize(data)
    assert np.allclose(norm, [[0, 0], [1, 1]])


def test_min_max_normalize_axis() -> None:
    """Normalize data with min-max scaling along axis 1."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    norm = min_max_normalize(data, axis=1)
    assert np.allclose(norm, [[0, 1], [0, 1]])


def test_z_score_normalize() -> None:
    """Standardize data along axis 0."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    norm = z_score_normalize(data)
    assert np.allclose(np.mean(norm, axis=0), [0, 0])
    assert np.allclose(np.std(norm, axis=0), [1, 1])


def test_z_score_normalize_axis() -> None:
    """Standardize data along axis 1."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    norm = z_score_normalize(data, axis=1)
    assert np.allclose(np.mean(norm, axis=1), [0, 0])
    assert np.allclose(np.std(norm, axis=1), [1, 1])


def test_log_normalize() -> None:
    """Log-scale non-negative values."""
    data = np.array([[1, 1], [np.e, np.e**2]], dtype=float)
    norm = log_normalize(data)
    assert np.allclose(norm, [[0, 0], [1, 2]])

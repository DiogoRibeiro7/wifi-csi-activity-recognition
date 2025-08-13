"""Tests for calibration utilities."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.preprocessing import (  # type: ignore  # noqa: E402
    phase_unwrap,
    remove_dc_offset,
)


def test_remove_dc_offset() -> None:
    """Remove DC offset along axis 0."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    calibrated = remove_dc_offset(data)
    assert np.allclose(np.mean(calibrated, axis=0), [0, 0])


def test_remove_dc_offset_axis() -> None:
    """Remove DC offset along axis 1."""
    data = np.array([[1, 2], [3, 4]], dtype=float)
    calibrated = remove_dc_offset(data, axis=1)
    assert np.allclose(np.mean(calibrated, axis=1), [0, 0])


def test_phase_unwrap() -> None:
    """Unwrap phase along axis 0."""
    data = np.array([[0], [np.pi - 0.1], [-np.pi + 0.1]])
    unwrapped = phase_unwrap(data)
    assert np.all(unwrapped[1:] - unwrapped[:-1] < np.pi)


def test_phase_unwrap_axis() -> None:
    """Unwrap phase along axis 1."""
    data = np.array([[0, np.pi - 0.1, -np.pi + 0.1]])
    unwrapped = phase_unwrap(data, axis=1)
    assert np.all(unwrapped[:, 1:] - unwrapped[:, :-1] < np.pi)

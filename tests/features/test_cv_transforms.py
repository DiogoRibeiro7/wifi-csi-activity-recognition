"""Tests for computer-vision transforms."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.features import (  # type: ignore  # noqa: E402
    magnitude_to_uint8,
)


def test_magnitude_to_uint8_range() -> None:
    """Conversion scales data to 0-255 uint8 range."""
    mat = np.array([[0.0, 1.0], [0.5, 0.2]])
    img = magnitude_to_uint8(mat)
    assert img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255

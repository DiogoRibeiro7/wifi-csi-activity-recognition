"""Tests for segmentation utilities."""

import sys
import types
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.preprocessing import (  # type: ignore  # noqa: E402
    segment_windows,
)


def test_segment_windows() -> None:
    """Segment along axis 0 with overlapping windows."""
    data = np.arange(10)[:, None]
    segments = segment_windows(data, window_size=4, overlap=0.5)
    assert segments.shape == (4, 4, 1)
    assert np.array_equal(segments[0, :, 0], [0, 1, 2, 3])


def test_segment_windows_axis() -> None:
    """Segment along axis 1 with overlapping windows."""
    data = np.arange(10)[None, :]
    segments = segment_windows(data, window_size=4, overlap=0.5, axis=1)
    assert segments.shape == (4, 1, 4)
    assert np.array_equal(segments[0, 0, :], [0, 1, 2, 3])

"""Tests for circular buffer management."""

import sys
import types
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
if "wifi_activity_recognition" not in sys.modules:
    package = types.ModuleType("wifi_activity_recognition")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.inference import (  # type: ignore  # noqa: E402
    CircularBuffer,
)


def test_circular_buffer_overflow_and_pop() -> None:
    """Circular buffer drops oldest elements when full."""
    buf: CircularBuffer[int] = CircularBuffer(2)
    assert buf.append(1) is False
    assert buf.append(2) is False
    assert buf.is_full()
    assert buf.append(3) is True  # overflow drops oldest
    assert len(buf) == 2
    assert buf.pop() == 2
    assert buf.pop() == 3
    assert buf.pop() is None

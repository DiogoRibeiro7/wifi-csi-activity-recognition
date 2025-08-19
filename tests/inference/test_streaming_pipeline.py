"""Integration tests for the streaming pipeline."""

# isort: skip_file
import sys
import types
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
if "wifi_activity_recognition" not in sys.modules:
    package = types.ModuleType("wifi_activity_recognition")
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules["wifi_activity_recognition"] = package
    hw = types.ModuleType("wifi_activity_recognition.hardware")
    hw.__path__ = [str(PACKAGE_ROOT / "hardware")]
    sys.modules["wifi_activity_recognition.hardware"] = hw
    package.hardware = hw

from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
    CSIReaderBase,
    HardwareConfig,
)
from wifi_activity_recognition.inference import (  # type: ignore  # noqa: E402
    ActivityRecognizer,
    StreamingPipeline,
)
from wifi_activity_recognition.models.cnn2d import (  # type: ignore  # noqa: E402
    CNN2DModel,
)


class DummyReader(CSIReaderBase):
    """Reader that serves packets from a deque for testing."""

    def __init__(self, packets: deque[CSIData]):
        """Store packets to be served on ``read_packet`` calls."""
        super().__init__(HardwareConfig(100, 1, 20.0, [0]))
        self.packets = packets

    def connect(self) -> bool:
        """Simulate establishing a hardware connection."""
        self._is_connected = True
        return True

    def disconnect(self) -> None:
        """Simulate closing the hardware connection."""
        self._is_connected = False

    def start_streaming(self) -> None:  # pragma: no cover - no-op
        """Start streaming (noop for dummy)."""
        pass

    def calibrate(self) -> bool:
        """No calibration required for dummy reader."""
        return True

    def stop_streaming(self) -> None:  # pragma: no cover - no-op
        """Stop streaming (noop for dummy)."""
        pass

    def read_packet(self) -> CSIData | None:
        """Return the next packet or ``None`` when exhausted."""
        return self.packets.popleft() if self.packets else None

    def get_hardware_info(self) -> dict:  # pragma: no cover - unused
        """Return hardware metadata."""
        return {}


@pytest.fixture()
def csi_packet() -> CSIData:
    """Create a random CSI packet for tests."""
    amp = np.random.rand(1, 1, 30).astype(np.float32)
    phase = np.random.rand(1, 1, 30).astype(np.float32)
    return CSIData(0.0, amp, phase, 2.4, 20.0, 1, 1, 30)


def test_streaming_pipeline_runs(csi_packet: CSIData) -> None:
    """Pipeline produces predictions and records latency."""
    packets = deque([csi_packet for _ in range(5)])
    reader = DummyReader(packets)
    model = CNN2DModel(num_classes=2)
    recognizer = ActivityRecognizer(model, class_names=["a", "b"])
    pipeline = StreamingPipeline(reader, recognizer, buffer_size=2, smoothing=1)
    pipeline.start()
    time.sleep(0.1)
    result = pipeline.get_latest()
    pipeline.stop()
    assert result is not None
    label, conf, ts = result
    assert label in {"a", "b"}
    assert 0.0 <= conf <= 1.0
    assert pipeline.monitor.latencies

"""Integration tests for the streaming pipeline."""

# isort: skip_file
import sys
import types
import time
from collections import deque
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi_activity_recognition"
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


class ConstantRecognizer:
    """Recognizer returning predefined outputs for testing."""

    def __init__(self, outputs: deque[tuple[str, float]], delay: float = 0.0) -> None:
        """Store outputs and optional delay."""
        self.outputs = outputs
        self.delay = delay

    def predict(self, _: CSIData) -> tuple[str, float]:  # pragma: no cover - trivial
        """Return the next predetermined prediction."""
        time.sleep(self.delay)
        return self.outputs.popleft() if self.outputs else ("a", 0.5)


def test_confidence_and_transition_smoothing(csi_packet: CSIData) -> None:
    """Low-confidence predictions yield 'unknown' and transitions require stability."""
    packets = deque([csi_packet for _ in range(4)])
    reader = DummyReader(packets)
    outputs = deque(
        [
            ("a", 0.4),  # below threshold -> unknown
            ("b", 0.9),  # first b
            ("b", 0.9),  # second b -> stable
            ("c", 0.9),  # new label but only one -> remain b
        ]
    )
    recognizer = ConstantRecognizer(outputs)
    pipeline = StreamingPipeline(
        reader,
        recognizer,  # type: ignore[arg-type]
        buffer_size=4,
        smoothing=1,
        confidence_threshold=0.5,
        transition_smoothing=2,
    )
    pipeline.start()
    for _ in range(50):
        if pipeline.monitor.processed >= 4:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    result = pipeline.get_latest()
    pipeline.stop()
    assert result is not None
    label, conf, _ = result
    assert label == "b"
    assert conf <= 1.0


def test_run_sync_and_drop_monitoring(csi_packet: CSIData) -> None:
    """Synchronous mode returns results and tracks dropped packets."""
    packets = deque([csi_packet for _ in range(3)])
    reader = DummyReader(packets)
    outputs = deque([("a", 0.9) for _ in range(3)])
    recognizer = ConstantRecognizer(outputs)
    pipeline = StreamingPipeline(
        reader,
        recognizer,  # type: ignore[arg-type]
        buffer_size=1,
        smoothing=1,
    )
    results = pipeline.run_sync(3)
    assert len(results) == 3
    assert pipeline.monitor.processed == 3
    assert pipeline.monitor.dropped == 0


def test_buffer_overflow_records_drop(csi_packet: CSIData) -> None:
    """Overflowing the buffer increments the dropped counter."""
    packets = deque([csi_packet for _ in range(10)])
    reader = DummyReader(packets)
    outputs = deque([("a", 0.9) for _ in range(10)])
    recognizer = ConstantRecognizer(outputs, delay=0.01)
    pipeline = StreamingPipeline(
        reader,
        recognizer,  # type: ignore[arg-type]
        buffer_size=2,
        smoothing=1,
    )
    pipeline.start()
    time.sleep(0.2)
    pipeline.stop()
    assert pipeline.monitor.dropped > 0


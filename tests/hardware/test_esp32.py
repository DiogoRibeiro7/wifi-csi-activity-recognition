"""Tests for the simulated ESP32 hardware driver."""

import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make the package importable despite repository layout using hyphenated name
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.hardware import (  # type: ignore  # noqa: E402
    CSIReader,
    ESP32Reader,
    HardwareConfig,
)
from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)


class DummySerial:
    """Simple serial port mock."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the mock serial port."""
        self.is_open = True
        self.read_calls = 0

    def readline(self) -> bytes:
        """Return a dummy line of bytes."""
        self.read_calls += 1
        return b"dummy\n"

    def close(self) -> None:
        """Close the mock serial port."""
        self.is_open = False


@pytest.fixture(autouse=True)
def mock_serial(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the serial.Serial class used by the driver."""
    monkeypatch.setattr(
        "wifi_activity_recognition.hardware.esp32.serial.Serial",
        DummySerial,
    )


@pytest.fixture
def config() -> HardwareConfig:
    """Return a sample hardware configuration for the ESP32 reader."""
    return HardwareConfig(
        sampling_rate=100.0,
        channel=6,
        bandwidth=20.0,
        antenna_config=[1],
        calibration_required=True,
        buffer_size=10,
        timeout=1.0,
        additional_params={
            "serial_port": "/dev/ttyUSB0",
            "baud_rate": 115200,
            "n_tx": 1,
            "n_rx": 1,
            "n_subcarriers": 64,
        },
    )


def test_connect_and_disconnect(config: HardwareConfig) -> None:
    """The reader connects and disconnects correctly."""
    reader = ESP32Reader(config)
    assert not reader.is_connected
    assert reader.connect() is True
    assert reader.is_connected
    reader.disconnect()
    assert not reader.is_connected


def test_streaming_and_read_packet(config: HardwareConfig) -> None:
    """Streaming produces CSI packets with expected shape."""
    reader = ESP32Reader(config)
    reader.connect()
    assert reader.read_packet() is None
    reader.start_streaming()
    packet = reader.read_packet()
    assert isinstance(packet, CSIData)
    assert packet.amplitude.shape == (1, 1, 64)
    assert packet.phase.shape == (1, 1, 64)
    reader.stop_streaming()
    assert reader.read_packet() is None
    reader.disconnect()


def test_read_batch(config: HardwareConfig) -> None:
    """read_batch returns the requested number of packets."""
    reader = ESP32Reader(config)
    reader.connect()
    reader.start_streaming()
    batch = reader.read_batch(3)
    assert len(batch) == 3
    assert all(isinstance(p, CSIData) for p in batch)
    reader.stop_streaming()
    reader.disconnect()


def test_calibration_and_info(config: HardwareConfig) -> None:
    """Calibration updates state and hardware info is correct."""
    reader = ESP32Reader(config)
    assert not reader.state.calibrated
    reader.calibrate()
    assert reader.state.calibrated
    info = reader.get_hardware_info()
    assert info["n_subcarriers"] == 64


def test_factory_creation(config: HardwareConfig) -> None:
    """Factory function instantiates ESP32Reader correctly."""
    config_dict = {
        "sampling_rate": config.sampling_rate,
        "channel": config.channel,
        "bandwidth": config.bandwidth,
        "antenna_config": config.antenna_config,
        "calibration_required": config.calibration_required,
        "buffer_size": config.buffer_size,
        "timeout": config.timeout,
        "additional_params": {
            "serial_port": "/dev/ttyUSB0",
            "baud_rate": 115200,
            "n_tx": 1,
            "n_rx": 1,
            "n_subcarriers": 64,
        },
    }
    reader = CSIReader("esp32", config_dict)
    assert isinstance(reader, ESP32Reader)

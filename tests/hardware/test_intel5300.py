"""Tests for the simulated Intel 5300 hardware driver."""

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
    HardwareConfig,
    Intel5300Reader,
)
from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)


@pytest.fixture
def config() -> HardwareConfig:
    """Return a sample hardware configuration for the Intel 5300 reader."""
    return HardwareConfig(
        sampling_rate=1000.0,
        channel=6,
        bandwidth=20.0,
        antenna_config=[1, 2, 3],
        calibration_required=True,
        buffer_size=10,
        timeout=1.0,
        additional_params={"n_tx": 3, "n_rx": 3, "n_subcarriers": 30},
    )


def test_connect_and_disconnect(config: HardwareConfig) -> None:
    """The reader connects and disconnects correctly."""
    reader = Intel5300Reader(config)
    assert not reader.is_connected
    assert reader.connect() is True
    assert reader.is_connected
    reader.disconnect()
    assert not reader.is_connected


def test_streaming_and_read_packet(config: HardwareConfig) -> None:
    """Streaming produces CSI packets with expected shape."""
    reader = Intel5300Reader(config)
    reader.connect()
    # Should not return data when not streaming
    assert reader.read_packet() is None
    reader.start_streaming()
    packet = reader.read_packet()
    assert isinstance(packet, CSIData)
    assert packet.amplitude.shape == (3, 3, 30)
    assert packet.phase.shape == (3, 3, 30)
    reader.stop_streaming()
    assert reader.read_packet() is None
    reader.disconnect()


def test_read_batch(config: HardwareConfig) -> None:
    """read_batch returns the requested number of packets."""
    reader = Intel5300Reader(config)
    reader.connect()
    reader.start_streaming()
    batch = reader.read_batch(5)
    assert len(batch) == 5
    assert all(isinstance(p, CSIData) for p in batch)
    reader.stop_streaming()
    reader.disconnect()


def test_calibration_and_info(config: HardwareConfig) -> None:
    """Calibration updates state and hardware info is correct."""
    reader = Intel5300Reader(config)
    assert not reader.state.calibrated
    reader.calibrate()
    assert reader.state.calibrated
    info = reader.get_hardware_info()
    assert info["n_subcarriers"] == 30


def test_factory_creation(config: HardwareConfig) -> None:
    """Factory function instantiates Intel5300Reader correctly."""
    config_dict = {
        "sampling_rate": config.sampling_rate,
        "channel": config.channel,
        "bandwidth": config.bandwidth,
        "antenna_config": config.antenna_config,
        "calibration_required": config.calibration_required,
        "buffer_size": config.buffer_size,
        "timeout": config.timeout,
        "additional_params": {"n_tx": 3, "n_rx": 3, "n_subcarriers": 30},
    }
    reader = CSIReader("intel_5300", config_dict)
    assert isinstance(reader, Intel5300Reader)

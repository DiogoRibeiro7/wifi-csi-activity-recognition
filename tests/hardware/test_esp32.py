"""Tests for the simulated ESP32 hardware driver."""

import struct

import numpy as np
import pytest

from wifi_activity_recognition.hardware import CSIReader, ESP32Reader, HardwareConfig
from wifi_activity_recognition.hardware.base import CSIData


class DummySerial:
    """Simple serial port mock for synthetic mode."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the dummy serial port."""
        self.is_open = True
        self.read_calls = 0

    def readline(self) -> bytes:
        """Return a dummy line of bytes."""
        self.read_calls += 1
        return b"dummy\n"

    def close(self) -> None:
        """Close the dummy port."""
        self.is_open = False


class RealSerial:
    """Serial mock delivering predefined binary data."""

    def __init__(self, data: bytes, *args, **kwargs) -> None:
        """Initialize with preloaded binary ``data``."""
        self.is_open = True
        self._buffer = bytearray(data)

    def read(self, size: int = 1) -> bytes:
        """Read ``size`` bytes from the buffer."""
        chunk = self._buffer[:size]
        del self._buffer[:size]
        return bytes(chunk)

    def close(self) -> None:
        """Close the port."""
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


def _build_packet(version: int, *, n_sub: int = 4, endian_flag: int = 0) -> bytes:
    """Construct a binary CSI packet for tests."""
    board_id = 0
    n_tx = 1
    n_rx = 1
    timestamp = 1234
    if version == 1:
        header = struct.pack("<BBBBHI", version, board_id, n_tx, n_rx, n_sub, timestamp)
    else:
        header = struct.pack(
            "<BBBBHIB", version, board_id, n_tx, n_rx, n_sub, timestamp, endian_flag
        )
    # IQ pairs: (1,1), (2,-2), ...
    iq = []
    for i in range(1, n_sub + 1):
        iq.extend([i, -i])
    fmt = ("<" if endian_flag == 0 else ">") + "h" * (2 * n_sub)
    iq_bytes = struct.pack(fmt, *iq)
    return bytes([version]) + header + iq_bytes


@pytest.mark.parametrize("version,endian_flag", [(1, 0), (2, 1)])
def test_real_mode_parsing(
    monkeypatch: pytest.MonkeyPatch,
    config: HardwareConfig,
    version: int,
    endian_flag: int,
) -> None:
    """Reader parses real binary packets for supported firmware versions."""
    data = _build_packet(version, endian_flag=endian_flag)
    monkeypatch.setattr(
        "wifi_activity_recognition.hardware.esp32.serial.Serial",
        lambda *a, **k: RealSerial(data),
    )
    cfg = HardwareConfig(
        sampling_rate=config.sampling_rate,
        channel=config.channel,
        bandwidth=config.bandwidth,
        antenna_config=config.antenna_config,
        calibration_required=config.calibration_required,
        buffer_size=config.buffer_size,
        timeout=config.timeout,
        additional_params=dict(config.additional_params, mode="real"),
    )
    reader = ESP32Reader(cfg)
    assert reader.connect() is True
    reader.start_streaming()
    pkt = reader.read_packet()
    assert pkt is not None
    assert pkt.metadata["firmware_version"] == f"v{version}"
    assert pkt.amplitude.shape == (1, 1, 4)
    # first subcarrier has values (1,1)
    assert np.isclose(pkt.amplitude[0, 0, 0], np.sqrt(2))
    reader.stop_streaming()
    reader.disconnect()


def test_unsupported_firmware(
    monkeypatch: pytest.MonkeyPatch, config: HardwareConfig
) -> None:
    """Unsupported firmware versions fail to connect."""
    data = bytes([9])  # unknown version byte
    monkeypatch.setattr(
        "wifi_activity_recognition.hardware.esp32.serial.Serial",
        lambda *a, **k: RealSerial(data),
    )
    cfg = HardwareConfig(
        sampling_rate=config.sampling_rate,
        channel=config.channel,
        bandwidth=config.bandwidth,
        antenna_config=config.antenna_config,
        calibration_required=config.calibration_required,
        buffer_size=config.buffer_size,
        timeout=config.timeout,
        additional_params=dict(config.additional_params, mode="real"),
    )
    reader = ESP32Reader(cfg)
    assert reader.connect() is False

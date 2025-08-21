"""Tests for the Qualcomm CSI reader."""

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
    QualcommReader,
)
from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)


class DummySocket:
    """Simple socket mock."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the mock socket."""
        self.connected = False
        self.closed = False

    def settimeout(self, _timeout: float) -> None:
        """Ignore timeout configuration."""

    def connect(self, addr) -> None:
        """Simulate a successful connection unless IP is invalid."""
        if addr[0] == "0.0.0.0":
            raise OSError("connection failed")
        self.connected = True

    def recv(self, _n: int) -> bytes:
        """Return dummy bytes."""
        return b"data"

    def close(self) -> None:
        """Mark the socket as closed."""
        self.closed = True


@pytest.fixture(autouse=True)
def mock_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch socket.socket used by the driver."""
    monkeypatch.setattr(
        "wifi_activity_recognition.hardware.qualcomm.socket.socket",
        DummySocket,
    )


@pytest.fixture
def base_config() -> HardwareConfig:
    """Return a sample hardware configuration for the Qualcomm reader."""
    return HardwareConfig(
        sampling_rate=100.0,
        channel=6,
        bandwidth=20.0,
        antenna_config=[1],
        calibration_required=True,
        buffer_size=5,
        timeout=1.0,
        additional_params={
            "device_ip": "127.0.0.1",
            "port": 9000,
            "n_tx": 1,
            "n_rx": 1,
            "n_subcarriers": 128,
        },
    )


@pytest.mark.parametrize("subcarriers", [64, 128, 256])
def test_streaming_and_packet(base_config: HardwareConfig, subcarriers: int) -> None:
    """Streaming yields packets with expected shapes for supported subcarriers."""
    base_config.additional_params["n_subcarriers"] = subcarriers
    reader = QualcommReader(base_config)
    assert reader.connect()
    reader.start_streaming()
    pkt = reader.read_packet()
    assert isinstance(pkt, CSIData)
    assert pkt.amplitude.shape == (1, 1, subcarriers)
    reader.stop_streaming()
    reader.disconnect()


def test_connect_failure(base_config: HardwareConfig) -> None:
    """Connection fails when device IP is missing."""
    base_config.additional_params.pop("device_ip")
    reader = QualcommReader(base_config)
    assert reader.connect() is False


def test_read_batch(base_config: HardwareConfig) -> None:
    """read_batch returns the requested number of packets."""
    reader = QualcommReader(base_config)
    reader.connect()
    reader.start_streaming()
    batch = reader.read_batch(2)
    assert len(batch) == 2
    assert all(isinstance(p, CSIData) for p in batch)
    reader.stop_streaming()
    reader.disconnect()


def test_factory_creation(base_config: HardwareConfig) -> None:
    """Factory function instantiates QualcommReader correctly."""
    cfg = base_config
    cfg_dict = {
        "sampling_rate": cfg.sampling_rate,
        "channel": cfg.channel,
        "bandwidth": cfg.bandwidth,
        "antenna_config": cfg.antenna_config,
        "calibration_required": cfg.calibration_required,
        "buffer_size": cfg.buffer_size,
        "timeout": cfg.timeout,
        "additional_params": cfg.additional_params,
    }
    reader = CSIReader("qualcomm", cfg_dict)
    assert isinstance(reader, QualcommReader)


def test_invalid_subcarriers(base_config: HardwareConfig) -> None:
    """Unsupported subcarrier counts raise a ValueError."""
    base_config.additional_params["n_subcarriers"] = 100
    with pytest.raises(ValueError):
        QualcommReader(base_config)

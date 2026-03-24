"""Tests for the Intel 5300 CSI reader."""

from pathlib import Path

import numpy as np
import pytest

from wifi_activity_recognition.hardware import (
    CSIReader,
    HardwareConfig,
    Intel5300Reader,
)
from wifi_activity_recognition.hardware.base import CSIData


@pytest.fixture
def mock_config() -> HardwareConfig:
    """Return configuration for mock mode."""
    return HardwareConfig(
        sampling_rate=1000.0,
        channel=6,
        bandwidth=20.0,
        antenna_config=[1, 2, 3],
        calibration_required=True,
        buffer_size=10,
        timeout=1.0,
        additional_params={
            "n_tx": 3,
            "n_rx": 3,
            "n_subcarriers": 30,
            "mode": "mock",
        },
    )


class DummyIntel:
    """Minimal ``csiread.Intel`` replacement returning deterministic data."""

    def __init__(
        self, *_: object, nrxnum: int = 1, ntxnum: int = 1, **__: object
    ) -> None:
        """Initialize the dummy reader."""
        self.count = 1
        self.timestamp_low = np.array([12345], dtype=float)
        phase = np.linspace(0, np.pi, 30)
        amp = np.linspace(1, 2, 30)
        csi = amp * np.exp(1j * phase)
        self.csi = csi[None, :, None, None].astype(np.complex64)

    def read(self) -> None:  # pragma: no cover - nothing to do
        """No-op read method for interface compatibility."""
        return None


@pytest.fixture
def file_config(tmp_path: Path) -> HardwareConfig:
    """Return configuration for file mode using a dummy capture."""
    dummy_path = tmp_path / "dummy.dat"
    dummy_path.write_bytes(b"dummy")
    return HardwareConfig(
        sampling_rate=1000.0,
        channel=6,
        bandwidth=20.0,
        antenna_config=[1],
        calibration_required=True,
        buffer_size=10,
        timeout=1.0,
        additional_params={
            "n_tx": 1,
            "n_rx": 1,
            "n_subcarriers": 30,
            "mode": "file",
            "file_path": str(dummy_path),
        },
    )


def test_mock_streaming(mock_config: HardwareConfig) -> None:
    """Streaming in mock mode produces packets with expected shape."""
    reader = Intel5300Reader(mock_config)
    reader.connect()
    assert reader.read_packet() is None
    reader.start_streaming()
    pkt = reader.read_packet()
    assert isinstance(pkt, CSIData)
    assert pkt.amplitude.shape == (3, 3, 30)
    reader.stop_streaming()
    reader.disconnect()


def test_file_parsing(
    file_config: HardwareConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parse file capture and calibrate CSI."""
    monkeypatch.setattr(
        "wifi_activity_recognition.hardware.intel5300.csiread",
        types.SimpleNamespace(Intel=DummyIntel),
    )
    reader = Intel5300Reader(file_config)
    reader.connect()
    reader.start_streaming()
    pkt = reader.read_packet()
    assert pkt is not None
    assert np.isclose(pkt.amplitude.min(), 0.0)
    assert np.isclose(pkt.amplitude.max(), 1.0)
    assert np.allclose(pkt.phase, 0.0, atol=1e-6)
    assert reader.read_packet() is None  # end of file
    reader.stop_streaming()
    reader.disconnect()


def test_missing_file_raises(mock_config: HardwareConfig) -> None:
    """Check that missing file path raises ``FileNotFoundError``."""
    mock_config.additional_params.update({"mode": "file", "file_path": "bad.dat"})
    reader = Intel5300Reader(mock_config)
    with pytest.raises(FileNotFoundError):
        reader.connect()


def test_factory(mock_config: HardwareConfig) -> None:
    """Factory function instantiates Intel5300Reader in mock mode."""
    config_dict = {
        "sampling_rate": mock_config.sampling_rate,
        "channel": mock_config.channel,
        "bandwidth": mock_config.bandwidth,
        "antenna_config": mock_config.antenna_config,
        "calibration_required": mock_config.calibration_required,
        "buffer_size": mock_config.buffer_size,
        "timeout": mock_config.timeout,
        "additional_params": mock_config.additional_params,
    }
    reader = CSIReader("intel_5300", config_dict)
    assert isinstance(reader, Intel5300Reader)

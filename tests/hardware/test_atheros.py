"""Tests for the Atheros AR9300 hardware driver."""

import sys
import types
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "wifi-activity-recognition"
package = types.ModuleType("wifi_activity_recognition")
package.__path__ = [str(PACKAGE_ROOT)]
sys.modules["wifi_activity_recognition"] = package

from wifi_activity_recognition.hardware import (  # type: ignore  # noqa: E402
    AtherosReader,
    CSIReader,
    HardwareConfig,
)
from wifi_activity_recognition.hardware.base import (  # type: ignore  # noqa: E402
    CSIData,
)


def _config() -> HardwareConfig:
    """Return a sample configuration for tests."""
    return HardwareConfig(
        sampling_rate=1000.0,
        channel=6,
        bandwidth=20.0,
        antenna_config=[1, 2],
        calibration_required=True,
        buffer_size=5,
        timeout=1.0,
        additional_params={"n_tx": 2, "n_rx": 2, "n_subcarriers": 56},
    )


def test_streaming_and_packet() -> None:
    """Streaming produces packets with expected shape."""
    cfg = _config()
    reader = AtherosReader(cfg)
    assert reader.connect()
    reader.start_streaming()
    pkt = reader.read_packet()
    assert isinstance(pkt, CSIData)
    assert pkt.amplitude.shape == (2, 2, 56)
    reader.stop_streaming()
    reader.disconnect()


def test_factory() -> None:
    """Factory creates an Atheros reader instance."""
    cfg = _config()
    cfg_dict = {
        "sampling_rate": cfg.sampling_rate,
        "channel": cfg.channel,
        "bandwidth": cfg.bandwidth,
        "antenna_config": cfg.antenna_config,
        "calibration_required": cfg.calibration_required,
        "buffer_size": cfg.buffer_size,
        "timeout": cfg.timeout,
        "additional_params": {"n_tx": 2, "n_rx": 2, "n_subcarriers": 56},
    }
    reader = CSIReader("atheros", cfg_dict)
    assert isinstance(reader, AtherosReader)

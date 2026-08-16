"""Every mock driver must produce packets with no device attached.

A mock reader that still needs hardware is not a mock. The ESP32 driver opened
a serial port in ``connect()`` regardless of mode, so it failed on any machine
without an ESP32 plugged in -- which meant no driver ran headless, and neither
CI nor a container could exercise the capture path end to end.

These tests run the drivers exactly as an unattended environment would: no
device, and no serial or socket object patched in. See
docs/hardware_verification.md for what each platform supports.
"""

from __future__ import annotations

import pytest

from wifi_activity_recognition.hardware import CSIReader
from wifi_activity_recognition.hardware.base import CSIData, validate_csi_data

# Platforms whose mock mode is expected to work with nothing attached.
# Qualcomm is absent deliberately: it connects over TCP to a device IP and has
# no mock mode at all.
HEADLESS_PLATFORMS = ["intel_5300", "esp32", "atheros_ar9300"]


def _reader(platform: str):
    return CSIReader(
        platform,
        {
            "sampling_rate": 100,
            "channel": 6,
            "additional_params": {"mode": "mock"},
        },
    )


@pytest.mark.regression
@pytest.mark.parametrize("platform", HEADLESS_PLATFORMS)
def test_mock_driver_connects_without_hardware(platform: str) -> None:
    """connect() must succeed with no device present."""
    reader = _reader(platform)
    try:
        assert (
            reader.connect() is True
        ), f"{platform} mock mode could not connect without hardware"
        assert reader.is_connected
    finally:
        reader.disconnect()


@pytest.mark.regression
@pytest.mark.parametrize("platform", HEADLESS_PLATFORMS)
def test_mock_driver_produces_valid_packets(platform: str) -> None:
    """Packets must be well-formed CSIData, not placeholders."""
    reader = _reader(platform)
    try:
        assert reader.connect()
        reader.start_streaming()

        packet = reader.read_packet()
        assert isinstance(packet, CSIData), f"{platform} returned {type(packet)}"
        assert packet.amplitude.shape == packet.shape
        assert packet.phase.shape == packet.shape
        # Would fail on NaN amplitude, out-of-range phase or an absurd timestamp.
        assert validate_csi_data(packet), f"{platform} produced invalid CSI"
    finally:
        reader.stop_streaming()
        reader.disconnect()


@pytest.mark.regression
@pytest.mark.parametrize("platform", HEADLESS_PLATFORMS)
def test_mock_driver_streams_continuously(platform: str) -> None:
    """The stream iterator must yield repeatedly without a device."""
    reader = _reader(platform)
    try:
        assert reader.connect()
        collected = []
        for packet in reader.stream():
            collected.append(packet)
            if len(collected) >= 5:
                reader.stop_streaming()
        assert len(collected) == 5, f"{platform} yielded {len(collected)} packets"
    finally:
        reader.disconnect()


@pytest.mark.regression
def test_esp32_mock_opens_no_serial_port() -> None:
    """The specific regression: mock mode must not touch the serial layer.

    Asserting on the absence of the port object rather than only on behaviour,
    because a driver that opens a real port and merely tolerates failure would
    still pass the tests above on a machine that happens to have one attached.
    """
    reader = _reader("esp32")
    try:
        assert reader.connect()
        assert (
            reader._serial is None
        ), "mock mode opened a serial port; it must not require a device"
    finally:
        reader.disconnect()


@pytest.mark.regression
def test_qualcomm_refuses_to_connect_without_a_device_address() -> None:
    """Qualcomm has no mock mode and must fail clearly rather than pretend.

    Documented in docs/hardware_verification.md: it reaches an Android device
    over TCP, so there is nothing to fall back to. Returning False is the
    correct outcome, and this pins it so a future change cannot quietly start
    reporting a connection that does not exist.
    """
    reader = CSIReader("qualcomm", {"sampling_rate": 100, "channel": 6})
    assert reader.connect() is False
    assert reader.is_connected is False

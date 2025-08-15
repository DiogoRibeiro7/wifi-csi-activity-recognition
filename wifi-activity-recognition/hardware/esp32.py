"""Simulated ESP32 CSI reader implementation.

This module provides a driver that mimics the behaviour of an ESP32 device
with CSI capabilities. The implementation follows the
:class:`~wifi_activity_recognition.hardware.base.CSIReaderBase` interface and
uses a serial connection to read raw packets. The actual bytes received over
serial are not parsed but the driver generates synthetic CSI data for testing
and development purposes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - serial may not be installed in the environment
    import serial  # type: ignore
except ImportError:  # pragma: no cover

    class _DummySerialModule:  # type: ignore
        class Serial:  # type: ignore
            def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
                raise RuntimeError("pyserial is required for ESP32Reader")

    serial = _DummySerialModule()  # type: ignore

from .base import CSIData, CSIReaderBase, HardwareConfig


@dataclass
class ESP32State:
    """Internal state for the ESP32 driver."""

    calibrated: bool = False


class ESP32Reader(CSIReaderBase):
    """Driver for ESP32 CSI extraction using serial communication."""

    def __init__(self, config: HardwareConfig):
        """Initialize the ESP32 reader with hardware configuration."""
        super().__init__(config)
        params = config.additional_params or {}
        self.serial_port: str = str(params.get("serial_port", "/dev/ttyUSB0"))
        self.baud_rate: int = int(params.get("baud_rate", 115200))
        self.n_tx: int = int(params.get("n_tx", 1))
        self.n_rx: int = int(params.get("n_rx", len(config.antenna_config)))
        self.n_subcarriers: int = int(params.get("n_subcarriers", 64))
        self.state = ESP32State(calibrated=not config.calibration_required)
        self._serial: Optional[serial.Serial] = None

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------
    def connect(self) -> bool:  # type: ignore[override]
        """Establish a serial connection to the ESP32 device."""
        try:
            self._serial = serial.Serial(
                self.serial_port, self.baud_rate, timeout=self.config.timeout
            )
            self._is_connected = True
        except (serial.SerialException, OSError):
            self._serial = None
            self._is_connected = False
        return self._is_connected

    def disconnect(self) -> None:  # type: ignore[override]
        """Close the serial connection and reset state."""
        if self._serial and getattr(self._serial, "is_open", False):
            try:
                self._serial.close()
            except (serial.SerialException, OSError):  # pragma: no cover
                pass
        self._serial = None
        self._is_connected = False
        self._is_streaming = False

    # ------------------------------------------------------------------
    # Streaming Control
    # ------------------------------------------------------------------
    def start_streaming(self) -> None:  # type: ignore[override]
        """Begin reading CSI packets from the serial connection."""
        if not self._is_connected:
            raise RuntimeError("Hardware not connected")
        self._is_streaming = True

    def stop_streaming(self) -> None:  # type: ignore[override]
        """Stop reading CSI packets."""
        self._is_streaming = False

    # ------------------------------------------------------------------
    # Data Acquisition
    # ------------------------------------------------------------------
    def read_packet(self) -> Optional[CSIData]:  # type: ignore[override]
        """Read a single CSI packet from the ESP32.

        The method performs a serial read to emulate packet retrieval but the
        data is not parsed. Instead synthetic amplitude and phase matrices are
        generated to match the expected dimensions.
        """
        if not (self._is_connected and self._is_streaming and self._serial):
            return None

        # Perform a serial read to mimic hardware behaviour.
        try:
            _ = self._serial.readline()
        except (serial.SerialException, OSError):
            return None

        if not self.state.calibrated:
            self.calibrate()

        amplitude = np.abs(np.random.randn(self.n_rx, self.n_tx, self.n_subcarriers))
        phase = np.random.uniform(
            -np.pi, np.pi, size=(self.n_rx, self.n_tx, self.n_subcarriers)
        )
        timestamp = datetime.now().timestamp()
        frequency = 2412 + (self.config.channel - 1) * 5
        bandwidth = self.config.bandwidth

        csi = CSIData(
            timestamp=timestamp,
            amplitude=amplitude,
            phase=phase,
            frequency=frequency,
            bandwidth=bandwidth,
            n_tx=self.n_tx,
            n_rx=self.n_rx,
            n_subcarriers=self.n_subcarriers,
            metadata={"hardware": "ESP32"},
        )
        self._buffer.append(csi)
        if len(self._buffer) > self.config.buffer_size:
            self._buffer.pop(0)
        return csi

    # ------------------------------------------------------------------
    # Calibration & Info
    # ------------------------------------------------------------------
    def get_hardware_info(self) -> Dict[str, Any]:  # type: ignore[override]
        """Return basic hardware specifications."""
        return {
            "name": "ESP32",
            "n_tx": self.n_tx,
            "n_rx": self.n_rx,
            "n_subcarriers": self.n_subcarriers,
            "supports_calibration": False,
        }

    def calibrate(self) -> bool:  # type: ignore[override]
        """Mock calibration routine (ESP32 typically requires none)."""
        self.state.calibrated = True
        return True


__all__ = ["ESP32Reader"]

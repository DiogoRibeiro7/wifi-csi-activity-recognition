"""ESP32 CSI reader with real packet parsing and synthetic fallback.

This module implements a serial-based CSI reader for ESP32 boards. When
configured in ``real`` mode the driver expects binary packets produced by the
CSI-enabled ESP32 firmware and converts them into the standard
:class:`~wifi_activity_recognition.hardware.base.CSIData` structure. The
previous synthetic behaviour is preserved through a ``mock`` mode which
generates random CSI matrices and is useful for tests and development without
hardware.

Two firmware formats are currently supported:

``v1``
    Basic header containing board type, antenna counts and subcarrier count.
``v2``
    Extends v1 with an endianness flag for the IQ samples.

Both formats encode IQ pairs as 16-bit signed integers. The driver performs
firmware version detection on connection and validates compatibility before
streaming.
"""

from __future__ import annotations

import struct
import time
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
        self.mode: str = str(params.get("mode", "mock"))  # "mock" or "real"
        self.board: str = str(params.get("board", "esp32"))
        self.firmware_version: str = str(params.get("firmware_version", ""))
        self.endianness: str = str(params.get("endianness", "little"))
        self.max_retries: int = int(params.get("max_retries", 3))
        self.retry_interval: float = float(params.get("retry_interval", 0.1))
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
            if self.mode == "real":
                if not self._detect_firmware_version():
                    self.disconnect()
                    return False
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

        Behaviour depends on ``mode``:

        ``mock``
            Generates synthetic CSI matrices without parsing serial data.
        ``real``
            Parses binary packets emitted by the ESP32 CSI firmware and converts
            them into :class:`CSIData` instances. Packet parsing handles
            endianness, firmware differences and retry logic for robustness.
        """
        if not (self._is_connected and self._is_streaming and self._serial):
            return None

        if self.mode == "mock":
            try:
                _ = self._serial.readline()
            except (serial.SerialException, OSError):
                return None

            if not self.state.calibrated:
                self.calibrate()

            amplitude = np.abs(
                np.random.randn(self.n_rx, self.n_tx, self.n_subcarriers)
            )
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
                metadata={"hardware": "ESP32", "mode": "mock"},
            )
            self._buffer.append(csi)
            if len(self._buffer) > self.config.buffer_size:
                self._buffer.pop(0)
            return csi

        # Real mode -----------------------------------------------------
        header = self._read_exact(self._header_size())
        if header is None:
            return None

        try:
            if self.firmware_version == "v2":
                (
                    _version,
                    board_id,
                    n_tx,
                    n_rx,
                    n_sub,
                    timestamp,
                    endian_flag,
                ) = struct.unpack(self._header_fmt_v2(), header)
                endian = "<" if endian_flag == 0 else ">"
            else:  # default to v1
                (
                    _version,
                    board_id,
                    n_tx,
                    n_rx,
                    n_sub,
                    timestamp,
                ) = struct.unpack(self._header_fmt_v1(), header)
                endian = self._endian_prefix()
        except struct.error:
            return None

        total_samples = n_tx * n_rx * n_sub
        payload_len = 4 * total_samples
        payload = self._read_exact(payload_len)
        if payload is None or len(payload) != payload_len:
            return None

        try:
            iq = struct.unpack(f"{endian}{2 * total_samples}h", payload)
        except struct.error:
            return None

        i_vals = np.array(iq[0::2], dtype=np.float32)
        q_vals = np.array(iq[1::2], dtype=np.float32)
        complex_vals = (i_vals + 1j * q_vals).reshape(n_rx, n_tx, n_sub)
        amplitude = np.abs(complex_vals)
        phase = np.angle(complex_vals)

        frequency = 2412 + (self.config.channel - 1) * 5
        bandwidth = self.config.bandwidth
        if not self.state.calibrated:
            self.calibrate()

        csi = CSIData(
            timestamp=float(timestamp),
            amplitude=amplitude,
            phase=phase,
            frequency=frequency,
            bandwidth=bandwidth,
            n_tx=n_tx,
            n_rx=n_rx,
            n_subcarriers=n_sub,
            metadata={
                "hardware": "ESP32",
                "board": self._board_name(board_id),
                "firmware_version": self.firmware_version,
                "mode": "real",
            },
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
            "board": self.board,
            "firmware_version": self.firmware_version or "unknown",
            "supports_calibration": False,
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _detect_firmware_version(self) -> bool:
        """Determine firmware version from initial serial byte."""
        if self.firmware_version:
            return True
        if not self._serial:
            return False
        try:
            version_byte = self._serial.read(1)
        except (serial.SerialException, OSError):
            return False
        if not version_byte:
            return False
        version = int.from_bytes(version_byte, "little")
        if version not in (1, 2):
            return False
        self.firmware_version = f"v{version}"
        return True

    def _read_exact(self, num_bytes: int) -> Optional[bytes]:
        """Read exactly ``num_bytes`` from serial with retry logic."""
        if not self._serial:
            return None
        data = b""
        for _ in range(self.max_retries):
            try:
                chunk = self._serial.read(num_bytes - len(data))
            except (serial.SerialException, OSError):
                chunk = b""
            if chunk:
                data += chunk
            if len(data) >= num_bytes:
                return data
            time.sleep(self.retry_interval)
        return None

    def _header_size(self) -> int:
        return struct.calcsize(
            self._header_fmt_v2()
            if self.firmware_version == "v2"
            else self._header_fmt_v1()
        )

    @staticmethod
    def _header_fmt_v1() -> str:
        return "<BBBBHI"

    @staticmethod
    def _header_fmt_v2() -> str:
        return "<BBBBHIB"

    def _endian_prefix(self) -> str:
        return "<" if self.endianness == "little" else ">"

    @staticmethod
    def _board_name(board_id: int) -> str:
        mapping = {0: "esp32", 1: "esp32-s2", 2: "esp32-c3"}
        return mapping.get(board_id, "unknown")

    def calibrate(self) -> bool:  # type: ignore[override]
        """Mock calibration routine (ESP32 typically requires none)."""
        self.state.calibrated = True
        return True


__all__ = ["ESP32Reader"]

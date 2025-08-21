"""Qualcomm Android CSI reader implementation.

This driver simulates connection to Qualcomm-based Android devices capable of
CSI extraction. It follows
:class:`~wifi_activity_recognition.hardware.base.CSIReaderBase` and generates
synthetic CSI packets for development and testing. The reader
supports variable subcarrier counts and mimics network-based communication with
the mobile device.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - socket may not be available in test env
    import socket
except ImportError:  # pragma: no cover
    socket = None  # type: ignore

from .base import CSIData, CSIReaderBase, HardwareConfig

SUPPORTED_SUBCARRIERS = (64, 128, 256)


@dataclass
class QualcommState:
    """Internal state for the Qualcomm driver."""

    calibrated: bool = False


class QualcommReader(CSIReaderBase):
    """Driver for Qualcomm-based Android devices.

    Parameters are provided through
    :class:`~wifi_activity_recognition.hardware.base.HardwareConfig`.
    Expected keys in ``additional_params`` include ``device_ip`` or ``host`` and
    ``port`` for network communication as well as CSI dimensions. The driver
    performs basic error handling for connection failures and validates
    supported subcarrier counts.
    """

    def __init__(self, config: HardwareConfig):
        """Initialize the reader with hardware configuration."""
        super().__init__(config)
        params = config.additional_params or {}
        self.device_ip: Optional[str] = params.get("device_ip") or params.get("host")
        self.port: int = int(params.get("port", 9000))
        self.n_tx: int = int(params.get("n_tx", 1))
        self.n_rx: int = int(params.get("n_rx", len(config.antenna_config)))
        self.n_subcarriers: int = int(params.get("n_subcarriers", 128))
        if self.n_subcarriers not in SUPPORTED_SUBCARRIERS:
            raise ValueError(
                f"Unsupported subcarrier count: {self.n_subcarriers}. "
                f"Supported: {SUPPORTED_SUBCARRIERS}"
            )
        self.state = QualcommState(calibrated=not config.calibration_required)
        self._socket: Optional[socket.socket] = None if socket else None

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------
    def connect(self) -> bool:  # type: ignore[override]
        """Establish a network connection to the Android device."""
        if not self.device_ip or socket is None:
            self._is_connected = False
            return False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.config.timeout)
            sock.connect((self.device_ip, self.port))
            self._socket = sock
            self._is_connected = True
        except OSError:
            self._socket = None
            self._is_connected = False
        return self._is_connected

    def disconnect(self) -> None:  # type: ignore[override]
        """Close the network connection and reset state."""
        if self._socket:
            try:
                self._socket.close()
            except OSError:  # pragma: no cover
                pass
        self._socket = None
        self._is_connected = False
        self._is_streaming = False

    # ------------------------------------------------------------------
    # Streaming Control
    # ------------------------------------------------------------------
    def start_streaming(self) -> None:  # type: ignore[override]
        """Begin streaming CSI data."""
        if not self._is_connected:
            raise RuntimeError("Hardware not connected")
        self._is_streaming = True

    def stop_streaming(self) -> None:  # type: ignore[override]
        """Stop streaming CSI data."""
        self._is_streaming = False

    # ------------------------------------------------------------------
    # Data Acquisition
    # ------------------------------------------------------------------
    def read_packet(self) -> Optional[CSIData]:  # type: ignore[override]
        """Read a single CSI packet from the device."""
        if not (self._is_connected and self._is_streaming):
            return None
        if not self.state.calibrated:
            self.calibrate()
        if self._socket:
            try:
                _ = self._socket.recv(1024)
            except OSError:
                return None
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
            metadata={"hardware": "Qualcomm"},
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
            "name": "Qualcomm Android",
            "n_tx": self.n_tx,
            "n_rx": self.n_rx,
            "n_subcarriers": self.n_subcarriers,
            "supports_calibration": True,
        }

    def calibrate(self) -> bool:  # type: ignore[override]
        """Perform a dummy calibration routine."""
        self.state.calibrated = True
        return True


__all__ = ["QualcommReader"]

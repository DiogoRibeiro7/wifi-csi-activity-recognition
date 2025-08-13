"""Intel 5300 WiFi hardware driver for CSI extraction.

This module provides a simulated driver implementation for the Intel 5300
Network Interface Card (NIC). It follows the standardized interface defined by
:class:`~wifi_activity_recognition.hardware.base.CSIReaderBase` and generates
synthetic CSI data useful for development and testing.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .base import CSIData, CSIReaderBase, HardwareConfig


@dataclass
class Intel5300State:
    """Internal state for the Intel 5300 driver."""

    calibrated: bool = False


class Intel5300Reader(CSIReaderBase):
    """Driver for Intel 5300 CSI extraction.

    The real Intel 5300 requires a modified driver and specific firmware. This
    implementation simulates the behaviour of the hardware to facilitate unit
    testing and algorithm development without access to physical devices.
    """

    def __init__(self, config: HardwareConfig):
        """Initialize the Intel 5300 reader with hardware configuration."""
        super().__init__(config)
        params = config.additional_params or {}
        self.n_tx: int = int(params.get("n_tx", 3))
        self.n_rx: int = int(params.get("n_rx", len(config.antenna_config)))
        self.n_subcarriers: int = int(params.get("n_subcarriers", 30))
        self.state = Intel5300State(calibrated=not config.calibration_required)

    # ------------------------------------------------------------------
    # Connection Management
    # ------------------------------------------------------------------
    def connect(self) -> bool:  # type: ignore[override]
        """Establish connection to the (simulated) Intel 5300 hardware."""
        self._is_connected = True
        return True

    def disconnect(self) -> None:  # type: ignore[override]
        """Disconnect from hardware and reset state."""
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
        """Stop the CSI data stream."""
        self._is_streaming = False

    # ------------------------------------------------------------------
    # Data Acquisition
    # ------------------------------------------------------------------
    def read_packet(self) -> Optional[CSIData]:  # type: ignore[override]
        """Generate a synthetic CSI packet.

        Returns ``None`` when the device is not streaming. When calibration is
        required but has not yet been performed, the method automatically
        triggers :meth:`calibrate`.
        """
        if not (self._is_connected and self._is_streaming):
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
            metadata={"hardware": "Intel 5300"},
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
            "name": "Intel WiFi Link 5300",
            "n_tx": self.n_tx,
            "n_rx": self.n_rx,
            "n_subcarriers": self.n_subcarriers,
            "supports_calibration": True,
        }

    def calibrate(self) -> bool:  # type: ignore[override]
        """Perform a mock calibration routine."""
        # Simulate calibration delay
        random.random()  # no-op for deterministic behaviour in tests
        self.state.calibrated = True
        return True


__all__ = ["Intel5300Reader"]

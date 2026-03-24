"""Simulated Atheros AR9300 CSI reader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

from .base import CSIData, CSIReaderBase, HardwareConfig


@dataclass
class AtherosState:
    """Internal state for the mock driver."""

    calibrated: bool = False


class AtherosReader(CSIReaderBase):
    """Mock driver for the Atheros AR9300 chipset."""

    def __init__(self, config: HardwareConfig):
        """Initialize the reader with hardware configuration."""
        super().__init__(config)
        params = config.additional_params or {}
        self.n_tx: int = int(params.get("n_tx", 3))
        self.n_rx: int = int(params.get("n_rx", len(config.antenna_config)))
        self.n_subcarriers: int = int(params.get("n_subcarriers", 56))
        self.state = AtherosState(calibrated=not config.calibration_required)

    def connect(self) -> bool:  # type: ignore[override]
        """Mock establishing a hardware connection."""
        self._is_connected = True
        return True

    def disconnect(self) -> None:  # type: ignore[override]
        """Disconnect and reset internal state."""
        self._is_connected = False
        self._is_streaming = False

    def start_streaming(self) -> None:  # type: ignore[override]
        """Begin generating synthetic CSI packets."""
        if not self._is_connected:
            raise RuntimeError("Hardware not connected")
        self._is_streaming = True

    def stop_streaming(self) -> None:  # type: ignore[override]
        """Stop packet generation."""
        self._is_streaming = False

    def read_packet(self) -> Optional[CSIData]:  # type: ignore[override]
        """Generate a single synthetic CSI packet."""
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
            metadata={"hardware": "Atheros AR9300"},
        )
        self._buffer.append(csi)
        if len(self._buffer) > self.config.buffer_size:
            self._buffer.pop(0)
        return csi

    def get_hardware_info(self) -> Dict[str, Any]:  # type: ignore[override]
        """Return basic hardware specifications."""
        return {
            "name": "Atheros AR9300",
            "n_tx": self.n_tx,
            "n_rx": self.n_rx,
            "n_subcarriers": self.n_subcarriers,
            "supports_calibration": True,
        }

    def calibrate(self) -> bool:  # type: ignore[override]
        """Perform a dummy calibration routine."""
        self.state.calibrated = True
        return True


__all__ = ["AtherosReader"]

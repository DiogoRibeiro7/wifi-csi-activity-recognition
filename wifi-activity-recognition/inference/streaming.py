"""Utilities for streaming activity recognition."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from ..hardware.base import CSIData
from .postprocessing import apply_confidence_threshold
from .predictor import ActivityRecognizer


class StreamingPredictor:
    """Perform streaming predictions using a sliding window over CSI packets."""

    def __init__(
        self,
        recognizer: ActivityRecognizer,
        window_size: int = 100,
        threshold: float = 0.5,
    ) -> None:
        """Initialize the streaming predictor."""
        self.recognizer = recognizer
        self.window_size = window_size
        self.threshold = threshold
        self._buffer: Deque[CSIData] = deque(maxlen=window_size)

    def update(self, csi_data: CSIData) -> Optional[Tuple[str, float, float]]:
        """Update the predictor with a new CSI packet."""
        self._buffer.append(csi_data)
        if len(self._buffer) < self.window_size:
            return None

        amp = np.mean([d.amplitude for d in self._buffer], axis=0)
        phase = np.mean([d.phase for d in self._buffer], axis=0)
        agg = CSIData(
            timestamp=csi_data.timestamp,
            amplitude=amp,
            phase=phase,
            frequency=csi_data.frequency,
            bandwidth=csi_data.bandwidth,
            n_tx=csi_data.n_tx,
            n_rx=csi_data.n_rx,
            n_subcarriers=csi_data.n_subcarriers,
        )
        label, conf = self.recognizer.predict(agg)
        result = apply_confidence_threshold(conf, label, self.threshold)
        if result is None:
            return None
        chosen_label, confidence = result
        return chosen_label, confidence, csi_data.timestamp

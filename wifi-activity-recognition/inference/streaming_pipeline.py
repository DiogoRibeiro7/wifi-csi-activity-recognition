"""High-performance real-time streaming pipeline for activity recognition."""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from typing import Callable, Optional, Tuple

from ..hardware.base import CSIData, CSIReaderBase
from ..utils.performance_monitoring import PerformanceMonitor
from .buffer_management import CircularBuffer
from .predictor import ActivityRecognizer


class StreamingPipeline:
    """Coordinate multi-threaded acquisition and asynchronous inference."""

    def __init__(
        self,
        reader: CSIReaderBase,
        recognizer: ActivityRecognizer,
        buffer_size: int = 256,
        smoothing: int = 5,
        buffer_factory: Callable[[int], CircularBuffer[CSIData]] = CircularBuffer,
    ) -> None:
        """Create the pipeline with given reader, model and parameters."""
        self.reader = reader
        self.recognizer = recognizer
        self.buffer = buffer_factory(buffer_size)
        self._predictions: deque[str] = deque(maxlen=smoothing)
        self._latest: Optional[Tuple[str, float, float]] = None
        self._stop = threading.Event()
        self.monitor = PerformanceMonitor()
        self._acq_thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start streaming threads and hardware acquisition."""
        self.reader.connect()
        try:
            self.reader.start_streaming()
        except Exception:  # pragma: no cover - hardware may not support
            pass
        self._acq_thread.start()
        self._infer_thread.start()

    def stop(self) -> None:
        """Stop threads and disconnect hardware."""
        self._stop.set()
        self._acq_thread.join(timeout=1)
        self._infer_thread.join(timeout=1)
        try:
            self.reader.stop_streaming()
        except Exception:  # pragma: no cover
            pass
        try:
            self.reader.disconnect()
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    def _acquire_loop(self) -> None:
        while not self._stop.is_set():
            pkt = self.reader.read_packet()
            if pkt is not None:
                self.buffer.append(pkt)
            else:
                time.sleep(0.001)

    def _infer_loop(self) -> None:
        while not self._stop.is_set():
            pkt = self.buffer.pop()
            if pkt is None:
                time.sleep(0.001)
                continue
            start = time.perf_counter()
            label, conf = self.recognizer.predict(pkt)
            latency_ms = (time.perf_counter() - start) * 1000
            self.monitor.record_latency(latency_ms)
            self._predictions.append(label)
            smooth = Counter(self._predictions).most_common(1)[0][0]
            self._latest = (smooth, conf, pkt.timestamp)

    # ------------------------------------------------------------------
    def get_latest(self) -> Optional[Tuple[str, float, float]]:
        """Return the most recent smoothed prediction."""
        return self._latest

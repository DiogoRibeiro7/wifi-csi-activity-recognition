"""High-performance real-time streaming pipeline for activity recognition."""

from __future__ import annotations

import logging
import threading
import time
from collections import Counter, deque
from typing import Callable, List, Optional, Tuple

from ..hardware.base import CSIData, CSIReaderBase
from ..utils.performance_monitoring import PerformanceMonitor
from .buffer_management import CircularBuffer
from .predictor import ActivityRecognizer

logger = logging.getLogger(__name__)


class StreamingPipeline:
    """Coordinate multi-threaded acquisition and asynchronous inference."""

    def __init__(
        self,
        reader: CSIReaderBase,
        recognizer: ActivityRecognizer,
        buffer_size: int = 256,
        smoothing: int = 5,
        confidence_threshold: float = 0.0,
        transition_smoothing: int = 1,
        buffer_factory: Callable[[int], CircularBuffer[CSIData]] = CircularBuffer,
    ) -> None:
        """Create the pipeline with given reader, model and parameters."""
        self.reader = reader
        self.recognizer = recognizer
        self.buffer = buffer_factory(buffer_size)
        self.confidence_threshold = confidence_threshold
        self.transition_smoothing = max(1, transition_smoothing)
        self._predictions: deque[str] = deque(maxlen=smoothing)
        self._latest: Optional[Tuple[str, float, float]] = None
        self._stable_label = "unknown"
        self._transition_count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.monitor = PerformanceMonitor()
        self._acq_thread = threading.Thread(target=self._acquire_loop, daemon=True)
        self._infer_thread = threading.Thread(target=self._infer_loop, daemon=True)

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start streaming threads and hardware acquisition."""
        self.reader.connect()
        try:  # pragma: no cover - hardware may not support
            self.reader.start_streaming()
        except Exception:  # noqa: BLE001 - reader may not implement this
            logger.debug("reader operation not supported", exc_info=True)
        self._acq_thread.start()
        self._infer_thread.start()

    def stop(self) -> None:
        """Stop threads and disconnect hardware."""
        self._stop.set()
        self._acq_thread.join(timeout=1)
        self._infer_thread.join(timeout=1)
        try:  # pragma: no cover
            self.reader.stop_streaming()
        except Exception:  # noqa: BLE001 - reader may not implement this
            logger.debug("reader operation not supported", exc_info=True)
        try:  # pragma: no cover
            self.reader.disconnect()
        except Exception:  # noqa: BLE001 - reader may not implement this
            logger.debug("reader operation not supported", exc_info=True)

    # ------------------------------------------------------------------
    def _acquire_loop(self) -> None:
        while not self._stop.is_set():
            if not self.reader.is_connected:
                try:
                    self.reader.connect()
                except Exception:
                    time.sleep(0.1)
                    continue
            try:
                pkt = self.reader.read_packet()
            except Exception:
                self.monitor.record_dropped()
                time.sleep(0.01)
                continue
            if pkt is not None:
                if self.buffer.append(pkt):
                    self.monitor.record_dropped()
            else:
                time.sleep(0.001)

    def _infer_loop(self) -> None:
        while not self._stop.is_set():
            pkt = self.buffer.pop()
            if pkt is None:
                time.sleep(0.001)
                continue
            self._process_packet(pkt)

    # ------------------------------------------------------------------
    def _process_packet(self, pkt: CSIData) -> Tuple[str, float, float]:
        """Run model prediction and update internal state."""
        start = time.perf_counter()
        label, conf = self.recognizer.predict(pkt)
        latency_ms = (time.perf_counter() - start) * 1000
        self.monitor.record_latency(latency_ms)
        self.monitor.record_processed()
        if conf < self.confidence_threshold:
            label = "unknown"
        self._predictions.append(label)
        smooth = Counter(self._predictions).most_common(1)[0][0]
        if smooth != self._stable_label:
            self._transition_count += 1
            if self._transition_count >= self.transition_smoothing:
                self._stable_label = smooth
                self._transition_count = 0
        else:
            self._transition_count = 0
        with self._lock:
            self._latest = (self._stable_label, conf, pkt.timestamp)
        return self._latest

    # ------------------------------------------------------------------
    def get_latest(self) -> Optional[Tuple[str, float, float]]:
        """Return the most recent smoothed prediction."""
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    def run_sync(self, packets: int) -> List[Tuple[str, float, float]]:
        """Process a number of packets synchronously in the calling thread."""
        results: List[Tuple[str, float, float]] = []
        self.reader.connect()
        try:  # pragma: no cover - optional
            self.reader.start_streaming()
        except Exception:  # noqa: BLE001 - reader may not implement this
            logger.debug("reader operation not supported", exc_info=True)
        for _ in range(packets):
            try:
                pkt = self.reader.read_packet()
            except Exception:
                self.monitor.record_dropped()
                continue
            if pkt is None:
                continue
            results.append(self._process_packet(pkt))
        try:  # pragma: no cover
            self.reader.stop_streaming()
        except Exception:  # noqa: BLE001 - reader may not implement this
            logger.debug("reader operation not supported", exc_info=True)
        try:  # pragma: no cover
            self.reader.disconnect()
        except Exception:  # noqa: BLE001 - reader may not implement this
            logger.debug("reader operation not supported", exc_info=True)
        return results

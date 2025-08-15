"""Automated generation of consolidated performance reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from torch import nn

from wifi_activity_recognition.hardware.base import CSIData

from .accuracy_benchmark import Loader, run_accuracy_benchmark
from .latency_benchmark import measure_latency
from .memory_benchmark import measure_memory_usage


def generate_performance_report(
    model: nn.Module,
    dataloaders: Mapping[str, Loader],
    predictor: Callable[[CSIData], object],
    packets: Iterable[CSIData],
    consumer: Callable[[Iterable[CSIData]], object],
    output_path: str | Path,
    devices: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run benchmarks and persist a JSON performance report.

    Args:
        model: Model to evaluate.
        dataloaders: Mapping of dataset name to dataloader for accuracy.
        predictor: Callable used for latency measurements.
        packets: CSI packets for latency and memory benchmarks.
        consumer: Function consuming CSI packets for memory profiling.
        output_path: File to write the report to.
        devices: Optional list of devices for cross-platform accuracy tests.

    Returns:
        Dictionary containing benchmark results which is also written to
        ``output_path`` as JSON.
    """
    accuracy = run_accuracy_benchmark(model, dataloaders, devices=devices)
    latency = measure_latency(predictor, packets)
    memory = measure_memory_usage(consumer, packets)

    report = {
        "accuracy": accuracy,
        "latency_ms": latency,
        "memory_bytes": memory,
    }

    path = Path(output_path)
    path.write_text(json.dumps(report, indent=2))
    return report

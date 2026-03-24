"""Inference utilities for real-time activity recognition."""

from .buffer_management import CircularBuffer
from .latency_optimization import (
    dynamic_batch,
    prune_model,
    quantize_model,
    set_gpu_memory_limit,
)
from .predictor import ActivityRecognizer
from .streaming import StreamingPredictor
from .streaming_pipeline import StreamingPipeline

__all__ = [
    "ActivityRecognizer",
    "StreamingPredictor",
    "StreamingPipeline",
    "CircularBuffer",
    "quantize_model",
    "prune_model",
    "set_gpu_memory_limit",
    "dynamic_batch",
]

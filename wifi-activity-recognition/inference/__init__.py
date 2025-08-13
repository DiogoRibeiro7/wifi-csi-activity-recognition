"""Inference utilities for real-time activity recognition."""

from .predictor import ActivityRecognizer
from .streaming import StreamingPredictor

__all__ = ["ActivityRecognizer", "StreamingPredictor"]

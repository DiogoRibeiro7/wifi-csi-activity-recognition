"""Model implementations and helpers for WiFi activity recognition."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "CNN2DModel",
    "CNN2DTensorFlowModel",
    "ResNetSpectrogramModel",
    "ResNetSpectrogramTensorFlowModel",
    "CNN3DModel",
    "CNN3DTensorFlowModel",
    "AttentionCNN3DModel",
    "AttentionCNN3DTensorFlowModel",
    "EnsembleModel",
    "TransformerModel",
    "VisionTransformerModel",
    "VisionTransformerTensorFlowModel",
    "create_model",
    "list_available_models",
    "infer_model_spec",
    "build_model_artifact",
    "save_model_artifact",
    "load_model",
]

_LAZY_EXPORTS = {
    "CNN2DModel": (".cnn2d", "CNN2DModel"),
    "CNN2DTensorFlowModel": (".cnn2d", "CNN2DTensorFlowModel"),
    "ResNetSpectrogramModel": (".resnet", "ResNetSpectrogramModel"),
    "ResNetSpectrogramTensorFlowModel": (
        ".resnet",
        "ResNetSpectrogramTensorFlowModel",
    ),
    "CNN3DModel": (".cnn3d", "CNN3DModel"),
    "CNN3DTensorFlowModel": (".cnn3d", "CNN3DTensorFlowModel"),
    "AttentionCNN3DModel": (".advanced_cnn3d", "AttentionCNN3DModel"),
    "AttentionCNN3DTensorFlowModel": (
        ".advanced_cnn3d",
        "AttentionCNN3DTensorFlowModel",
    ),
    "EnsembleModel": (".ensemble", "EnsembleModel"),
    "TransformerModel": (".transformer", "TransformerModel"),
    "VisionTransformerModel": (".vision_transformer", "VisionTransformerModel"),
    "VisionTransformerTensorFlowModel": (
        ".vision_transformer",
        "VisionTransformerTensorFlowModel",
    ),
    "create_model": (".factory", "create_model"),
    "list_available_models": (".factory", "list_available_models"),
    "infer_model_spec": (".serialization", "infer_model_spec"),
    "build_model_artifact": (".serialization", "build_model_artifact"),
    "save_model_artifact": (".serialization", "save_model_artifact"),
    "load_model": (".serialization", "load_model"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose model classes and helper functions."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Return module attributes exposed by the lazy model API."""
    return sorted(set(globals()) | set(__all__))

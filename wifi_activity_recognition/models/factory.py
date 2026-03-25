"""Factory utilities for creating model instances by name."""

from __future__ import annotations

import importlib
from typing import Any, Dict

_MODEL_REGISTRY: Dict[str, tuple[str, str]] = {
    "cnn2d": (".cnn2d", "CNN2DModel"),
    "resnet": (".resnet", "ResNetSpectrogramModel"),
    "cnn3d": (".cnn3d", "CNN3DModel"),
    "attention_cnn3d": (".advanced_cnn3d", "AttentionCNN3DModel"),
    "ensemble": (".ensemble", "EnsembleModel"),
    "transformer": (".transformer", "TransformerModel"),
    "vit": (".vision_transformer", "VisionTransformerModel"),
}

_MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    "cnn2d": {
        "class_name": "CNN2DModel",
        "description": "2D convolutional baseline for CSI spectrogram inputs.",
    },
    "resnet": {
        "class_name": "ResNetSpectrogramModel",
        "description": "Residual network for CSI spectrogram classification.",
    },
    "cnn3d": {
        "class_name": "CNN3DModel",
        "description": "3D convolutional network for spatiotemporal CSI volumes.",
    },
    "attention_cnn3d": {
        "class_name": "AttentionCNN3DModel",
        "description": "3D CNN with attention blocks for temporal feature refinement.",
    },
    "ensemble": {
        "class_name": "EnsembleModel",
        "description": "Ensemble wrapper for combining multiple activity models.",
    },
    "transformer": {
        "class_name": "TransformerModel",
        "description": "Transformer-based sequence model for CSI activity recognition.",
    },
    "vit": {
        "class_name": "VisionTransformerModel",
        "description": "Vision Transformer model for CSI image-like representations.",
    },
}


def create_model(name: str, *args, **kwargs) -> Any:
    """Create a model instance from the registry.

    Parameters
    ----------
    name:
        Identifier of the model. Supported values are ``'cnn2d'``, ``'resnet'``,
        ``'cnn3d'``, ``'attention_cnn3d'``, ``'ensemble'``, ``'transformer'`` and
        ``'vit'``.
    *args, **kwargs:
        Passed to the model constructor.
    """
    try:
        module_name, class_name = _MODEL_REGISTRY[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown model '{name}'") from exc
    module = importlib.import_module(module_name, __package__)
    model_cls = getattr(module, class_name)
    return model_cls(*args, **kwargs)


def list_available_models() -> Dict[str, Dict[str, Any]]:
    """Return metadata for the registered model architectures."""
    available_models: Dict[str, Dict[str, Any]] = {}
    for name, (_, class_name) in _MODEL_REGISTRY.items():
        model_info = dict(_MODEL_METADATA.get(name, {}))
        model_info.setdefault("class_name", class_name)
        available_models[name] = model_info
    return available_models


__all__ = ["create_model", "list_available_models"]

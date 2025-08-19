"""Factory utilities for creating model instances by name."""

from __future__ import annotations

from typing import Callable, Dict

from torch import nn

from .advanced_cnn3d import AttentionCNN3DModel
from .cnn2d import CNN2DModel
from .cnn3d import CNN3DModel
from .ensemble import EnsembleModel
from .resnet import ResNetSpectrogramModel
from .transformer import TransformerModel

_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "cnn2d": CNN2DModel,
    "resnet": ResNetSpectrogramModel,
    "cnn3d": CNN3DModel,
    "attention_cnn3d": AttentionCNN3DModel,
    "ensemble": EnsembleModel,
    "transformer": TransformerModel,
}


def create_model(name: str, *args, **kwargs) -> nn.Module:
    """Create a model instance from the registry.

    Parameters
    ----------
    name:
        Identifier of the model. Supported values are ``'cnn2d'``, ``'resnet'``,
        ``'cnn3d'``, ``'attention_cnn3d'``, ``'ensemble'`` and ``'transformer'``.
    *args, **kwargs:
        Passed to the model constructor.
    """
    try:
        model_cls = _MODEL_REGISTRY[name.lower()]
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise ValueError(f"Unknown model '{name}'") from exc
    return model_cls(*args, **kwargs)


__all__ = ["create_model"]

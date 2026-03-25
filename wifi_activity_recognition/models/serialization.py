"""Model serialization helpers for WiFi activity recognition."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from .advanced_cnn3d import AttentionCNN3DModel
from .cnn2d import CNN2DModel
from .cnn3d import CNN3DModel
from .ensemble import EnsembleModel
from .factory import create_model
from .resnet import ResNetSpectrogramModel
from .transformer import TransformerModel
from .vision_transformer import VisionTransformerModel

MODEL_ARTIFACT_VERSION = 1
MODEL_ARTIFACT_TYPE = "wifi_ar_model"
PICKLED_MODEL_ARTIFACT_TYPE = "wifi_ar_pickled_model"


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module for DataParallel-wrapped models."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def infer_model_spec(model: nn.Module) -> tuple[str, Dict[str, Any]]:
    """Infer a factory model name and constructor kwargs from a model instance.

    Args:
        model: PyTorch model instance to serialize.

    Returns:
        Tuple containing the canonical model name and constructor kwargs.

    Raises:
        ValueError: If the model type is not recognized.
    """
    model = _unwrap_model(model)

    if isinstance(model, CNN2DModel):
        return "cnn2d", {
            "num_classes": model.classifier.out_features,
            "in_channels": model.features[0].in_channels,
        }

    if isinstance(model, CNN3DModel):
        return "cnn3d", {
            "num_classes": model.classifier.out_features,
            "in_channels": model.features[0].in_channels,
        }

    if isinstance(model, ResNetSpectrogramModel):
        return "resnet", {
            "num_classes": model.model.fc.out_features,
            "in_channels": model.model.conv1.in_channels,
            "pretrained": False,
        }

    if isinstance(model, AttentionCNN3DModel):
        dropout_layer = next(
            (layer for layer in model.features if isinstance(layer, nn.Dropout3d)),
            None,
        )
        return "attention_cnn3d", {
            "num_classes": model.classifier[1].out_features,
            "in_channels": model.features[0].in_channels,
            "dropout": dropout_layer.p if dropout_layer is not None else 0.3,
        }

    if isinstance(model, VisionTransformerModel):
        first_block = model.blocks[0]
        return "vit", {
            "num_classes": model.head.out_features,
            "in_channels": model.patch_embed.in_channels,
            "patch_size": model.patch_size,
            "dim": model.dim,
            "depth": len(model.blocks),
            "heads": first_block.attn.num_heads,
            "mlp_dim": first_block.mlp[0].out_features,
            "dropout": float(first_block.attn.dropout),
            "emb_dropout": model.dropout.p,
            "seq_to_seq": model.seq_to_seq,
        }

    if isinstance(model, TransformerModel):
        layer = model.encoder.layers[0]
        return "transformer", {
            "input_dim": model.input_proj.in_features,
            "num_classes": model.classifier.out_features,
            "d_model": model.input_proj.out_features,
            "nhead": layer.self_attn.num_heads,
            "num_layers": len(model.encoder.layers),
            "dim_feedforward": layer.linear1.out_features,
            "dropout": layer.dropout.p,
        }

    if isinstance(model, EnsembleModel):
        return "ensemble", {
            "num_classes": model.cnn2d.classifier.out_features,
            "in_channels": model.cnn2d.features[0].in_channels,
        }

    raise ValueError(f"Unsupported model type for serialization: {type(model)!r}")


def build_model_artifact(
    model: nn.Module,
    *,
    model_name: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    class_names: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a structured model artifact payload.

    Args:
        model: Model instance to serialize.
        model_name: Optional explicit factory model name.
        model_kwargs: Optional explicit constructor kwargs.
        class_names: Optional class-name mapping for inference.
        metadata: Optional extra metadata to persist with the artifact.

    Returns:
        Dictionary suitable for ``torch.save``.
    """
    base_model = _unwrap_model(model)
    try:
        inferred_name, inferred_kwargs = infer_model_spec(base_model)
    except ValueError:
        return {
            "artifact_type": PICKLED_MODEL_ARTIFACT_TYPE,
            "artifact_version": MODEL_ARTIFACT_VERSION,
            "model": base_model,
            "class_names": list(class_names) if class_names is not None else None,
            "metadata": dict(metadata or {}),
        }

    payload_name = model_name or inferred_name
    payload_kwargs = dict(model_kwargs or inferred_kwargs)

    return {
        "artifact_type": MODEL_ARTIFACT_TYPE,
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "model_name": payload_name,
        "model_kwargs": payload_kwargs,
        "state_dict": base_model.state_dict(),
        "class_names": list(class_names) if class_names is not None else None,
        "metadata": dict(metadata or {}),
    }


def save_model_artifact(
    model: nn.Module,
    path: str | Path,
    *,
    model_name: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    class_names: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a model artifact to disk.

    Args:
        model: Model instance to serialize.
        path: Destination checkpoint path.
        model_name: Optional explicit factory model name.
        model_kwargs: Optional explicit constructor kwargs.
        class_names: Optional class-name mapping for inference.
        metadata: Optional extra metadata to persist with the artifact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_model_artifact(
        model,
        model_name=model_name,
        model_kwargs=model_kwargs,
        class_names=class_names,
        metadata=metadata,
    )
    torch.save(payload, path)


def load_model(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    model_name: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Load a model from a structured artifact or legacy file.

    Args:
        path: Checkpoint path to load.
        map_location: Device mapping passed to ``torch.load``.
        model_name: Explicit model name used when loading a legacy raw state dict.
        model_kwargs: Constructor kwargs used with ``model_name`` for legacy state dicts.

    Returns:
        Reconstructed PyTorch model in evaluation mode.

    Raises:
        ValueError: If the checkpoint does not include enough metadata to rebuild the model.
    """
    obj = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(obj, nn.Module):
        obj.eval()
        return obj

    if isinstance(obj, dict) and obj.get("artifact_type") == MODEL_ARTIFACT_TYPE:
        artifact_model_name = str(obj["model_name"])
        artifact_kwargs = dict(obj.get("model_kwargs") or {})
        model = create_model(artifact_model_name, **artifact_kwargs)
        model.load_state_dict(obj["state_dict"])
        model.eval()
        return model

    if isinstance(obj, dict) and obj.get("artifact_type") == PICKLED_MODEL_ARTIFACT_TYPE:
        model = obj["model"]
        model.eval()
        return model

    if isinstance(obj, dict) and all(isinstance(v, torch.Tensor) for v in obj.values()):
        if model_name is None:
            raise ValueError(
                "Raw state-dict checkpoints require 'model_name' and matching "
                "'model_kwargs' to be provided for loading."
            )
        model = create_model(model_name, **(model_kwargs or {}))
        model.load_state_dict(obj)
        model.eval()
        return model

    raise ValueError(f"Unsupported model artifact format at '{path}'")

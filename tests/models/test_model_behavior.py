"""Behaviour-level correctness checks shared by every model family.

The existing per-model tests assert output shape and basic trainability. A
model can satisfy both and still be wrong in ways that surface much later:
non-deterministic inference, outputs that depend on batch composition,
parameters no gradient ever reaches, or checkpoints that do not round-trip.

Each family registered in the model factory is exercised here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest
import torch
from torch import nn

from wifi_activity_recognition.models.factory import create_model


@dataclass(frozen=True)
class ModelSpec:
    """How to build a model family and feed it a batch."""

    name: str
    kwargs: dict
    make_inputs: Callable[[int], tuple[torch.Tensor, ...]]

    def build(self) -> nn.Module:
        """Instantiate through the factory, exercising the registry too."""
        return create_model(self.name, **self.kwargs)


def _image(batch: int) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 1, 30, 50),)


def _volume(batch: int) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 1, 8, 30, 50),)


def _sequence(batch: int) -> tuple[torch.Tensor]:
    return (torch.randn(batch, 12, 16),)


def _image_and_volume(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.randn(batch, 1, 30, 50), torch.randn(batch, 1, 8, 30, 50))


SPECS = [
    ModelSpec("cnn2d", {"num_classes": 3}, _image),
    ModelSpec("resnet", {"num_classes": 3, "pretrained": False}, _image),
    ModelSpec("cnn3d", {"num_classes": 3}, _volume),
    ModelSpec("attention_cnn3d", {"num_classes": 3}, _volume),
    ModelSpec("vit", {"num_classes": 3}, _image),
    ModelSpec("transformer", {"input_dim": 16, "num_classes": 3}, _sequence),
    ModelSpec("ensemble", {"num_classes": 3}, _image_and_volume),
]

IDS = [spec.name for spec in SPECS]


def test_every_registered_family_is_covered() -> None:
    """Fail if a new architecture is registered without behaviour coverage."""
    from wifi_activity_recognition.models.factory import _MODEL_REGISTRY

    assert set(_MODEL_REGISTRY) == {spec.name for spec in SPECS}, (
        "model registry and behaviour-test specs have diverged; add a ModelSpec "
        "for any newly registered architecture"
    )


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_inference_is_deterministic(spec: ModelSpec) -> None:
    """The same input must produce the same logits in eval mode."""
    torch.manual_seed(0)
    model = spec.build().eval()
    inputs = spec.make_inputs(4)

    with torch.no_grad():
        first = model(*inputs)
        second = model(*inputs)

    assert torch.allclose(
        first, second, atol=1e-6
    ), f"{spec.name} gives different logits for identical input in eval mode"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_predictions_do_not_depend_on_batch_composition(spec: ModelSpec) -> None:
    """A sample's logits must be the same alone as inside a larger batch.

    Catches normalization layers using batch statistics at inference time,
    which silently makes predictions depend on whatever else was in the batch.
    """
    torch.manual_seed(0)
    model = spec.build().eval()
    inputs = spec.make_inputs(4)

    with torch.no_grad():
        batched = model(*inputs)
        alone = model(*(tensor[:1] for tensor in inputs))

    assert torch.allclose(
        batched[0], alone[0], atol=1e-5
    ), f"{spec.name} predictions depend on batch composition"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_gradients_reach_every_parameter(spec: ModelSpec) -> None:
    """No learnable parameter may be disconnected from the loss."""
    torch.manual_seed(0)
    model = spec.build()
    logits = model(*spec.make_inputs(2))
    nn.CrossEntropyLoss()(logits, torch.randint(0, 3, (2,))).backward()

    unreached = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    ]
    assert not unreached, f"{spec.name}: no gradient reached {unreached}"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_checkpoint_round_trip_preserves_predictions(spec: ModelSpec) -> None:
    """Saving and reloading weights must not change what the model predicts."""
    torch.manual_seed(0)
    original = spec.build().eval()
    inputs = spec.make_inputs(2)

    restored = spec.build()
    restored.load_state_dict(original.state_dict())
    restored.eval()

    with torch.no_grad():
        assert torch.allclose(
            original(*inputs), restored(*inputs), atol=1e-6
        ), f"{spec.name} predictions changed across a state_dict round trip"


@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_output_is_finite(spec: ModelSpec) -> None:
    """Logits must not contain NaN or infinity on ordinary input."""
    torch.manual_seed(0)
    model = spec.build().eval()

    with torch.no_grad():
        logits = model(*spec.make_inputs(4))

    assert torch.isfinite(logits).all(), f"{spec.name} produced non-finite logits"
    assert logits.shape == (4, 3), f"{spec.name} returned {tuple(logits.shape)}"


@pytest.mark.slow
@pytest.mark.parametrize("spec", SPECS, ids=IDS)
def test_model_can_reduce_loss_on_a_tiny_task(spec: ModelSpec) -> None:
    """Each family must be able to fit a small fixed batch.

    A model that compiles and produces the right shape can still fail to learn
    at all -- a detached graph or a dead activation looks identical to a
    working model under shape-only tests.
    """
    torch.manual_seed(0)
    model = spec.build()
    inputs = spec.make_inputs(4)
    targets = torch.tensor([0, 1, 2, 0])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    first = last = None
    for step in range(20):
        optimizer.zero_grad()
        loss = criterion(model(*inputs), targets)
        loss.backward()
        optimizer.step()
        if step == 0:
            first = loss.item()
        last = loss.item()

    assert (
        last < first
    ), f"{spec.name} loss did not decrease ({first:.4f} -> {last:.4f})"

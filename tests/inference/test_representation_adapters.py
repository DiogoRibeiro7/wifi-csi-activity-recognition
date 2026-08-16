"""Every registered model must be usable through ActivityRecognizer.

The recogniser built one CNN2D-shaped tensor for every model, so four of the
seven registered architectures raised on the first forward pass::

    cnn3d            ValueError: expected 5D input (got 4D input)
    attention_cnn3d  ValueError: expected 5D input (got 4D input)
    ensemble         TypeError: forward() missing 1 required positional argument
    transformer      AssertionError: query should be unbatched 2D or batched 3D

Registering a model did not make it usable for inference. These tests pin the
contract that it now does.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from wifi_activity_recognition.hardware.base import CSIData
from wifi_activity_recognition.inference import ActivityRecognizer
from wifi_activity_recognition.inference.adapters import (
    ADAPTERS_BY_MODEL,
    EnsembleAdapter,
    SequenceAdapter,
    SpectrogramAdapter,
    VolumeAdapter,
    adapter_for_model,
    adapter_for_model_name,
)
from wifi_activity_recognition.models.factory import _MODEL_REGISTRY, create_model

N_RX, N_TX, N_SUB = 3, 1, 30
WINDOW = 8


def _packet(seed: int = 0) -> CSIData:
    rng = np.random.default_rng(seed)
    amplitude = np.abs(rng.normal(size=(N_RX, N_TX, N_SUB))).astype(np.float32)
    return CSIData(
        timestamp=float(seed),
        amplitude=amplitude,
        phase=np.zeros_like(amplitude),
        frequency=5.0,
        bandwidth=20.0,
        n_tx=N_TX,
        n_rx=N_RX,
        n_subcarriers=N_SUB,
    )


def _window(size: int = WINDOW) -> list[CSIData]:
    return [_packet(i) for i in range(size)]


def _model(name: str):
    kwargs: dict = {"num_classes": 2}
    if name == "transformer":
        kwargs["input_dim"] = SequenceAdapter.feature_dim(_packet())
    if name == "resnet":
        kwargs["pretrained"] = False
    return create_model(name, **kwargs)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.parametrize("name", sorted(_MODEL_REGISTRY))
def test_every_registered_model_predicts_from_a_window(name: str) -> None:
    """Registration must imply usability for inference."""
    recognizer = ActivityRecognizer(_model(name), class_names=["a", "b"])

    label, confidence = recognizer.predict(_window())

    assert label in {"a", "b"}
    assert 0.0 <= confidence <= 1.0


def test_every_registered_model_has_an_adapter() -> None:
    """A new architecture must not be able to land without one."""
    assert set(_MODEL_REGISTRY) == set(ADAPTERS_BY_MODEL), (
        "model registry and adapter table have diverged: "
        f"{set(_MODEL_REGISTRY) ^ set(ADAPTERS_BY_MODEL)}"
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("cnn2d", SpectrogramAdapter),
        ("resnet", SpectrogramAdapter),
        ("vit", SpectrogramAdapter),
        ("cnn3d", VolumeAdapter),
        ("attention_cnn3d", VolumeAdapter),
        ("transformer", SequenceAdapter),
        ("ensemble", EnsembleAdapter),
    ],
)
def test_adapter_is_inferred_from_the_model_instance(name: str, expected) -> None:
    """The recogniser must pick the right representation unaided."""
    assert isinstance(adapter_for_model(_model(name)), expected)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_spectrogram_is_a_single_channel_image() -> None:
    """CNN2D-family input is ``(1, 1, H, W)``."""
    (tensor,) = SpectrogramAdapter()(_window())
    assert tensor.ndim == 4
    assert tensor.shape[:2] == (1, 1)
    assert tensor.shape[2] == N_SUB  # subcarriers on the height axis


def test_spectrogram_pads_a_narrow_image() -> None:
    """A single packet gives ``rx*tx`` columns, fewer than the models need."""
    (tensor,) = SpectrogramAdapter()(_packet())
    assert tensor.shape[3] >= SpectrogramAdapter.min_width


def test_sequence_is_time_major() -> None:
    """Transformer input is ``(1, T, F)`` with T the packet count."""
    packets = _window()
    (tensor,) = SequenceAdapter()(packets)
    assert tensor.shape == (1, len(packets), SequenceAdapter.feature_dim(packets[0]))


def test_volume_uses_time_as_depth() -> None:
    """CNN3D input is ``(1, 1, D, H, W)`` with D the packet count."""
    packets = _window()
    (tensor,) = VolumeAdapter()(packets)
    assert tensor.ndim == 5
    assert tensor.shape[0:3] == (1, 1, len(packets))
    assert tensor.shape[4] == N_SUB


def test_volume_tiles_the_antenna_axis_to_survive_pooling() -> None:
    """CNN3D halves each spatial axis twice; 3 antenna pairs would reach zero."""
    (tensor,) = VolumeAdapter()(_window())
    assert tensor.shape[3] >= VolumeAdapter.min_height


def test_ensemble_builds_both_representations() -> None:
    """EnsembleModel.forward takes a 2-D and a 3-D tensor."""
    tensors = EnsembleAdapter()(_window())
    assert len(tensors) == 2
    assert tensors[0].ndim == 4
    assert tensors[1].ndim == 5


# ---------------------------------------------------------------------------
# Errors that name the capture, not a torch internal
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_too_short_a_window_reports_the_packet_count() -> None:
    """A single packet has no time axis to build a volume from."""
    recognizer = ActivityRecognizer(_model("cnn3d"), class_names=["a", "b"])

    with pytest.raises(ValueError, match="needs at least 4 packets"):
        recognizer.predict(_packet())


def test_mismatched_packet_shapes_are_rejected() -> None:
    """Packets of differing geometry cannot be stacked."""
    odd = _packet()
    amplitude = np.abs(np.random.default_rng(1).normal(size=(3, 1, 64)))
    other = CSIData(
        timestamp=1.0,
        amplitude=amplitude,
        phase=np.zeros_like(amplitude),
        frequency=5.0,
        bandwidth=20.0,
        n_tx=1,
        n_rx=3,
        n_subcarriers=64,
    )
    with pytest.raises(ValueError, match="must share a shape"):
        VolumeAdapter()([odd, other] * 2)


def test_empty_input_is_rejected() -> None:
    """An empty window must raise rather than produce an empty tensor."""
    with pytest.raises(ValueError, match="no CSI packets"):
        SpectrogramAdapter()([])


def test_unknown_model_name_is_rejected() -> None:
    """Looking up an unregistered name names the known ones."""
    with pytest.raises(ValueError, match="no representation adapter"):
        adapter_for_model_name("does_not_exist")


def test_unknown_model_instance_falls_back_to_spectrogram() -> None:
    """Custom models keep the recogniser's original behaviour."""

    class CustomModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return x

    assert isinstance(adapter_for_model(CustomModel()), SpectrogramAdapter)


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_single_packet_prediction_still_works_for_2d_models() -> None:
    """The original single-packet path must not regress."""
    recognizer = ActivityRecognizer(_model("cnn2d"), class_names=["a", "b"])

    label, confidence = recognizer.predict(_packet())

    assert label in {"a", "b"}
    assert 0.0 <= confidence <= 1.0


def test_to_tensor_still_returns_a_tensor() -> None:
    """Callers using the private helper directly keep working."""
    recognizer = ActivityRecognizer(_model("cnn2d"), class_names=["a", "b"])

    tensor = recognizer._to_tensor(_packet())

    assert isinstance(tensor, torch.Tensor)
    assert tensor.ndim == 4


def test_an_explicit_adapter_overrides_inference() -> None:
    """Callers can supply their own representation."""
    recognizer = ActivityRecognizer(
        _model("cnn2d"), class_names=["a", "b"], adapter=SpectrogramAdapter()
    )
    assert isinstance(recognizer.adapter, SpectrogramAdapter)

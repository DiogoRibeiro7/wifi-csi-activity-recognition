"""Convert CSI packets into the tensor layout each model family expects.

The model factory registers architectures with incompatible input contracts:
CNN2D and ResNet want ``(B, C, H, W)``, CNN3D wants ``(B, C, D, H, W)``, the
Transformer wants ``(B, T, F)`` and the ensemble wants two tensors at once.
:class:`~wifi_activity_recognition.inference.predictor.ActivityRecognizer`
previously built one CNN2D-shaped tensor for every model, so four of the seven
registered architectures raised on the first forward pass::

    cnn3d            ValueError: expected 5D input (got 4D input)
    attention_cnn3d  ValueError: expected 5D input (got 4D input)
    ensemble         TypeError: forward() missing 1 required positional argument
    transformer      AssertionError: query should be unbatched 2D or batched 3D

An adapter owns that conversion, so a model and its representation travel
together and a mismatch is reported in terms of the capture rather than as a
shape error from inside torch.

Where a capture is smaller than an architecture requires, the antenna axis is
tiled -- following the precedent already set for the 2D path. Tiling is
replication, not information: it makes a model runnable, it does not make its
output meaningful on a capture that lacks the antennas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Union

import numpy as np
import torch

from ..hardware.base import CSIData

PacketInput = Union[CSIData, Sequence[CSIData]]


def _as_packets(data: PacketInput) -> list[CSIData]:
    """Normalise a single packet or a sequence into a list."""
    if isinstance(data, CSIData):
        return [data]
    packets = list(data)
    if not packets:
        raise ValueError("no CSI packets supplied")
    return packets


def _tile_to(array: np.ndarray, axis: int, minimum: int) -> np.ndarray:
    """Repeat ``axis`` until it reaches ``minimum`` samples."""
    size = array.shape[axis]
    if size >= minimum:
        return array
    repeats = int(np.ceil(minimum / size))
    tiled = np.repeat(array, repeats, axis=axis)
    return np.take(tiled, range(minimum), axis=axis)


class RepresentationAdapter(ABC):
    """Build the input tensors one model family expects."""

    #: Human-readable name used in error messages.
    name: str = "representation"
    #: Fewest packets this representation can be built from.
    min_packets: int = 1

    def __call__(self, data: PacketInput) -> Tuple[torch.Tensor, ...]:
        """Validate the input and build the model's tensors."""
        packets = _as_packets(data)
        if len(packets) < self.min_packets:
            raise ValueError(
                f"{self.name} needs at least {self.min_packets} packets, got "
                f"{len(packets)}. Capture a longer window, or use a model whose "
                "representation works on fewer frames."
            )
        shapes = {packet.shape for packet in packets}
        if len(shapes) != 1:
            raise ValueError(
                f"all packets must share a shape to build a {self.name}, got {shapes}"
            )
        return self.build(packets)

    @abstractmethod
    def build(self, packets: list[CSIData]) -> Tuple[torch.Tensor, ...]:
        """Return the positional tensors to pass to the model."""

    @staticmethod
    def _stack_amplitude(packets: list[CSIData]) -> np.ndarray:
        """Stack amplitudes into ``(time, rx, tx, subcarrier)``."""
        return np.stack([packet.amplitude for packet in packets], axis=0)


class SpectrogramAdapter(RepresentationAdapter):
    """``(1, 1, H, W)`` for CNN2D, ResNet and ViT.

    A single packet is laid out as subcarriers against antennas, preserving the
    behaviour these models were previously given. Several packets form the
    conventional CSI spectrogram: subcarriers against time.
    """

    name = "spectrogram"
    min_width = 8

    def build(self, packets: list[CSIData]) -> Tuple[torch.Tensor, ...]:
        """Build a single-channel 2-D image."""
        first = packets[0]
        if len(packets) == 1:
            # Subcarriers x antennas, matching the original single-packet path.
            image = np.transpose(first.amplitude, (2, 0, 1)).reshape(
                first.n_subcarriers, first.n_rx * first.n_tx
            )
        else:
            # Subcarriers x time: the usual CSI spectrogram, averaged over
            # antennas so the axes stay two-dimensional.
            stacked = self._stack_amplitude(packets)  # (T, rx, tx, sub)
            image = stacked.mean(axis=(1, 2)).T  # (sub, T)

        image = _tile_to(image, axis=1, minimum=self.min_width)
        tensor = torch.tensor(image, dtype=torch.float32)
        return (tensor.unsqueeze(0).unsqueeze(0),)


class SequenceAdapter(RepresentationAdapter):
    """``(1, T, F)`` for the Transformer.

    Time is the sequence axis and every antenna-subcarrier pair is a feature,
    so ``F`` must equal the model's ``input_dim``.
    """

    name = "sequence"

    def build(self, packets: list[CSIData]) -> Tuple[torch.Tensor, ...]:
        """Build a time-major feature sequence."""
        stacked = self._stack_amplitude(packets)  # (T, rx, tx, sub)
        sequence = stacked.reshape(len(packets), -1)  # (T, rx*tx*sub)
        tensor = torch.tensor(sequence, dtype=torch.float32)
        return (tensor.unsqueeze(0),)

    @staticmethod
    def feature_dim(packet: CSIData) -> int:
        """Return the ``input_dim`` a Transformer needs for this geometry."""
        return packet.n_rx * packet.n_tx * packet.n_subcarriers


class VolumeAdapter(RepresentationAdapter):
    """``(1, 1, D, H, W)`` for CNN3D and the attention 3-D CNN.

    Depth is time, height is the antenna pairs and width is subcarriers.
    ``CNN3DModel`` applies two ``MaxPool3d(2)`` stages, so each spatial axis
    must survive halving twice; real captures rarely have four antenna pairs,
    so that axis is tiled up to the minimum.
    """

    name = "volume"
    min_packets = 4
    min_height = 4

    def build(self, packets: list[CSIData]) -> Tuple[torch.Tensor, ...]:
        """Build a single-channel 3-D volume with time as depth."""
        stacked = self._stack_amplitude(packets)  # (T, rx, tx, sub)
        time, n_rx, n_tx, n_sub = stacked.shape
        volume = stacked.reshape(time, n_rx * n_tx, n_sub)  # (D, H, W)
        volume = _tile_to(volume, axis=1, minimum=self.min_height)
        tensor = torch.tensor(volume, dtype=torch.float32)
        return (tensor.unsqueeze(0).unsqueeze(0),)


class EnsembleAdapter(RepresentationAdapter):
    """``((1, 1, H, W), (1, 1, D, H, W))`` for the ensemble.

    ``EnsembleModel.forward`` takes a 2-D and a 3-D tensor, so it needs both
    representations built from the same packets.
    """

    name = "ensemble"
    min_packets = VolumeAdapter.min_packets

    def __init__(self) -> None:
        """Compose the two representations the ensemble consumes."""
        self._spectrogram = SpectrogramAdapter()
        self._volume = VolumeAdapter()

    def build(self, packets: list[CSIData]) -> Tuple[torch.Tensor, ...]:
        """Build the 2-D and 3-D inputs together."""
        return self._spectrogram.build(packets) + self._volume.build(packets)


#: Adapter for each name in the model registry.
ADAPTERS_BY_MODEL: dict[str, type[RepresentationAdapter]] = {
    "cnn2d": SpectrogramAdapter,
    "resnet": SpectrogramAdapter,
    "vit": SpectrogramAdapter,
    "cnn3d": VolumeAdapter,
    "attention_cnn3d": VolumeAdapter,
    "transformer": SequenceAdapter,
    "ensemble": EnsembleAdapter,
}


def adapter_for_model_name(name: str) -> RepresentationAdapter:
    """Return the adapter registered for a factory model name."""
    try:
        return ADAPTERS_BY_MODEL[name.lower()]()
    except KeyError as exc:
        raise ValueError(
            f"no representation adapter for model '{name}'. Known: "
            f"{sorted(ADAPTERS_BY_MODEL)}"
        ) from exc


def adapter_for_model(model: object) -> RepresentationAdapter:
    """Infer the adapter from a model instance by its class name.

    Falls back to the spectrogram layout for unrecognised models, which is what
    the recogniser did for everything before adapters existed.
    """
    class_to_name = {
        "CNN2DModel": "cnn2d",
        "ResNetSpectrogramModel": "resnet",
        "VisionTransformerModel": "vit",
        "CNN3DModel": "cnn3d",
        "AttentionCNN3DModel": "attention_cnn3d",
        "TransformerModel": "transformer",
        "EnsembleModel": "ensemble",
    }
    name = class_to_name.get(type(model).__name__)
    return adapter_for_model_name(name) if name else SpectrogramAdapter()


__all__ = [
    "ADAPTERS_BY_MODEL",
    "EnsembleAdapter",
    "RepresentationAdapter",
    "SequenceAdapter",
    "SpectrogramAdapter",
    "VolumeAdapter",
    "adapter_for_model",
    "adapter_for_model_name",
]

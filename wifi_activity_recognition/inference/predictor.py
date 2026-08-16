"""Model-based activity recognition utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from ..hardware.base import CSIData
from ..models import load_model
from .adapters import RepresentationAdapter, adapter_for_model


class ActivityRecognizer:
    """Run single-packet activity predictions using a trained model.

    Parameters
    ----------
    model:
        Either a :class:`torch.nn.Module` instance or path to a saved model.
    class_names:
        Optional mapping of class indices to human-readable labels.
    device:
        Torch device to run inference on. Defaults to CUDA when available.
    """

    def __init__(
        self,
        model: Union[str, Path, nn.Module],
        class_names: Optional[Sequence[str]] = None,
        device: Optional[str] = None,
        adapter: Optional[RepresentationAdapter] = None,
    ) -> None:
        """Initialize the recognizer with a model or model path.

        ``adapter`` converts CSI packets into the layout this model expects.
        Left unset it is inferred from the model class, so 3-D CNNs, the
        Transformer and the ensemble work without the caller doing anything;
        previously every model was handed a CNN2D-shaped tensor and four of the
        seven registered architectures raised on the first forward pass.
        """
        if isinstance(model, (str, Path)):
            self.model = load_model(model, map_location="cpu")
        else:
            self.model = model
        self.adapter = adapter or adapter_for_model(self.model)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        if class_names is None:
            num_classes = getattr(self.model, "classifier", None)
            if isinstance(num_classes, nn.Linear):
                n = num_classes.out_features
            else:
                n = 0
            class_names = [str(i) for i in range(n)]
        self.class_names: List[str] = list(class_names)

    def _to_tensor(self, csi_data: CSIData) -> torch.Tensor:
        """Convert a single packet into this model's input tensor.

        Retained for callers that used it directly. It returns the first tensor
        only, so it cannot express the ensemble's two inputs; use
        :meth:`predict` for that.
        """
        return self.adapter(csi_data)[0].to(self.device)

    @torch.no_grad()
    def predict(self, csi_data: Union[CSIData, Sequence[CSIData]]) -> Tuple[str, float]:
        """Predict an activity from one CSI packet or a sequence of them.

        Sequence input is what the 3-D and Transformer representations need:
        a single packet has no time axis to build a volume from. The adapter
        raises with the packet count when the capture is too short, rather than
        letting torch fail on a shape deep inside the model.
        """
        tensors = tuple(tensor.to(self.device) for tensor in self.adapter(csi_data))
        logits = self.model(*tensors)[0]
        probs = torch.softmax(logits, dim=0)
        conf, idx = torch.max(probs, dim=0)
        label = self.class_names[int(idx)] if self.class_names else str(int(idx))
        return label, float(conf)

"""Domain adaptation utilities for CSI-based activity models.

This module implements several state-of-the-art techniques for adapting models
trained on a source environment to new target domains. Methods include
statistic-based alignment (CORAL, MMD), adversarial learning via DANN, and
helpers for adapting CSI streams across heterogeneous hardware platforms.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from ..hardware.base import CSIData

try:  # pragma: no cover - optional dependency
    from ..models.base import BaseActivityModel  # type: ignore
except Exception:  # pragma: no cover
    BaseActivityModel = nn.Module  # type: ignore


class _GradientReverse(torch.autograd.Function):
    """Autograd layer that multiplies the gradient by ``-lambda``."""

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, lambd: float
    ) -> torch.Tensor:  # pragma: no cover - trivial
        """Return the input unchanged while storing the reversal factor.

        Args:
            ctx: Autograd context used to persist ``lambd`` for the backward pass.
            x: Input tensor passed through unchanged.
            lambd: Scaling factor applied to reversed gradients.

        Returns:
            View of ``x`` used by the custom autograd function.
        """
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:  # pragma: no cover - trivial
        """Reverse and scale gradients during backpropagation.

        Args:
            ctx: Autograd context populated during ``forward``.
            grad_output: Gradient flowing from subsequent layers.

        Returns:
            Tuple containing the reversed gradient for ``x`` and ``None`` for
            the non-differentiable ``lambd`` parameter.
        """
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    """Apply a gradient reversal operation."""
    return _GradientReverse.apply(x, lambd)


def _compute_covariance(x: torch.Tensor) -> torch.Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    return x.t().mm(x) / (x.size(0) - 1)


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute CORAL loss aligning source and target covariance."""
    d = source.size(1)
    cs = _compute_covariance(source)
    ct = _compute_covariance(target)
    return torch.mean((cs - ct) ** 2) / (4 * d * d)


def mmd_loss(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: Optional[float] = None,
) -> torch.Tensor:
    """Maximum Mean Discrepancy with RBF kernels."""
    n_s = source.size(0)
    n_t = target.size(0)
    total = torch.cat([source, target], dim=0)
    L2 = ((total.unsqueeze(0) - total.unsqueeze(1)) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2.data) / (n_s + n_t) ** 2
    bandwidth /= kernel_mul ** (kernel_num // 2)
    kernel_val = [
        torch.exp(-L2 / (bandwidth * (kernel_mul**i))) for i in range(kernel_num)
    ]
    kernels = sum(kernel_val)
    XX = kernels[:n_s, :n_s]
    YY = kernels[n_s:, n_s:]
    XY = kernels[:n_s, n_s:]
    YX = kernels[n_s:, :n_s]
    return torch.mean(XX + YY - XY - YX)


def wasserstein_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Approximate Wasserstein distance between feature batches."""
    return torch.cdist(source, target, p=1).mean()


class DomainAdversarialNetwork(nn.Module):
    """Domain Adversarial Neural Network (DANN)."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        class_classifier: nn.Module,
        feature_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        """Initialize the DANN model."""
        super().__init__()
        self.feature_extractor = feature_extractor
        self.class_classifier = class_classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self, x: torch.Tensor, lambd: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass returning class and domain logits."""
        feat = self.feature_extractor(x)
        feat = feat.view(feat.size(0), -1)
        class_logits = self.class_classifier(feat)
        domain_logits = self.domain_classifier(grad_reverse(feat, lambd))
        return class_logits, domain_logits

    def train_epoch(
        self,
        source_x: torch.Tensor,
        source_y: torch.Tensor,
        target_x: torch.Tensor,
        optimizer: optim.Optimizer,
        lambd: float = 1.0,
    ) -> float:
        """Train the DANN for one epoch on source and target batches."""
        self.train()
        x = torch.cat([source_x, target_x], dim=0)
        class_logits, domain_logits = self(x, lambd)
        class_loss = F.cross_entropy(class_logits[: source_x.size(0)], source_y)
        domain_labels = torch.cat(
            [torch.zeros(source_x.size(0)), torch.ones(target_x.size(0))], dim=0
        ).long()
        domain_loss = F.cross_entropy(domain_logits, domain_labels)
        loss = class_loss + domain_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach().cpu())


class DomainAdapter:
    """Adapt pre-trained models to new target environments."""

    def __init__(self, source_model: BaseActivityModel) -> None:
        """Initialize the domain adapter with a source model."""
        self.model = source_model
        self._mean: Optional[np.ndarray] = None

    def _extract_feature(self, csi: CSIData) -> np.ndarray:
        """Extract a simple feature vector from :class:`CSIData`."""
        return np.array([float(csi.amplitude.mean())], dtype=np.float32)

    def adapt_to_target(
        self, target_csi: Iterable[CSIData], method: str = "coral"
    ) -> None:
        """Estimate target-domain statistics for adaptation.

        Parameters
        ----------
        target_csi:
            Iterable of unlabeled :class:`CSIData` from the target domain.
        method:
            ``"coral"`` uses mean-centering and covariance alignment.
            ``"mmd"`` uses maximum mean discrepancy to align means.
            ``"wasserstein"`` aligns means using Wasserstein distance.
        """
        feats = [self._extract_feature(csi) for csi in target_csi]
        if not feats:
            raise ValueError("No target CSI data provided for adaptation")
        feat_mat = np.vstack(feats)
        self._mean = feat_mat.mean(axis=0)
        if method not in {"coral", "mmd", "wasserstein"}:
            raise ValueError(f"Unsupported adaptation method: {method}")

    def update_online(self, csi: CSIData, alpha: float = 0.1) -> None:
        """Update stored statistics with a new target-domain sample."""
        feat = self._extract_feature(csi)
        if self._mean is None:
            self._mean = feat
        else:
            self._mean = (1 - alpha) * self._mean + alpha * feat

    def predict(self, csi: CSIData) -> Tuple[int, float]:
        """Predict activity label and confidence for a CSI packet."""
        feat = self._extract_feature(csi)
        if self._mean is not None:
            feat = feat - self._mean
        tensor = torch.tensor(feat, dtype=torch.float32)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=0)
        conf, idx = torch.max(probs, dim=0)
        return int(idx), float(conf)

    def evaluate_adaptation(
        self, target_test: Sequence[Tuple[CSIData, int]]
    ) -> Dict[str, float]:
        """Evaluate adapted model on labeled target-domain data."""
        if not target_test:
            raise ValueError("Empty evaluation dataset")
        correct = 0
        for csi, label in target_test:
            pred, _ = self.predict(csi)
            correct += int(pred == label)
        accuracy = correct / len(target_test)
        return {"accuracy": accuracy}

    @staticmethod
    def match_subcarrier_count(csi: CSIData, target_subcarriers: int) -> CSIData:
        """Resample CSI data to a different number of subcarriers."""
        if csi.n_subcarriers == target_subcarriers:
            return csi
        old_axis = np.linspace(0, 1, csi.n_subcarriers)
        new_axis = np.linspace(0, 1, target_subcarriers)
        amp = np.array(
            [
                np.interp(new_axis, old_axis, row)
                for row in csi.amplitude.reshape(-1, csi.n_subcarriers)
            ]
        ).reshape(csi.n_rx, csi.n_tx, target_subcarriers)
        phase = np.array(
            [
                np.interp(new_axis, old_axis, row)
                for row in csi.phase.reshape(-1, csi.n_subcarriers)
            ]
        ).reshape(csi.n_rx, csi.n_tx, target_subcarriers)
        return CSIData(
            timestamp=csi.timestamp,
            amplitude=amp,
            phase=phase,
            frequency=csi.frequency,
            bandwidth=csi.bandwidth,
            n_tx=csi.n_tx,
            n_rx=csi.n_rx,
            n_subcarriers=target_subcarriers,
            rssi=csi.rssi,
            noise_floor=csi.noise_floor,
            metadata=csi.metadata,
        )

    @staticmethod
    def match_antenna_config(
        csi: CSIData, target_n_rx: int, target_n_tx: int
    ) -> CSIData:
        """Adjust CSI data to a new antenna configuration via padding."""
        amp = np.zeros(
            (target_n_rx, target_n_tx, csi.n_subcarriers), dtype=csi.amplitude.dtype
        )
        phase = np.zeros_like(amp)
        rx = min(csi.n_rx, target_n_rx)
        tx = min(csi.n_tx, target_n_tx)
        amp[:rx, :tx] = csi.amplitude[:rx, :tx]
        phase[:rx, :tx] = csi.phase[:rx, :tx]
        return CSIData(
            timestamp=csi.timestamp,
            amplitude=amp,
            phase=phase,
            frequency=csi.frequency,
            bandwidth=csi.bandwidth,
            n_tx=target_n_tx,
            n_rx=target_n_rx,
            n_subcarriers=csi.n_subcarriers,
            rssi=csi.rssi,
            noise_floor=csi.noise_floor,
            metadata=csi.metadata,
        )

    @staticmethod
    def match_frequency_band(
        csi: CSIData, target_frequency: float, target_bandwidth: Optional[float] = None
    ) -> CSIData:
        """Update CSI metadata to reflect a new frequency band."""
        return CSIData(
            timestamp=csi.timestamp,
            amplitude=csi.amplitude.copy(),
            phase=csi.phase.copy(),
            frequency=target_frequency,
            bandwidth=target_bandwidth or csi.bandwidth,
            n_tx=csi.n_tx,
            n_rx=csi.n_rx,
            n_subcarriers=csi.n_subcarriers,
            rssi=csi.rssi,
            noise_floor=csi.noise_floor,
            metadata=csi.metadata,
        )


__all__ = [
    "DomainAdapter",
    "DomainAdversarialNetwork",
    "coral_loss",
    "mmd_loss",
    "wasserstein_distance",
    "grad_reverse",
]

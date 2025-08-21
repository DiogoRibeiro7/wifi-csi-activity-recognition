"""Few-shot learning utilities for rapid activity adaptation."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.stateless import functional_call

from ..hardware.base import CSIData

try:  # pragma: no cover
    from ..models.base import BaseActivityModel  # type: ignore
except Exception:  # pragma: no cover
    BaseActivityModel = nn.Module  # type: ignore


class FewShotLearner:
    """Meta-learning helper using prototype-based classification."""

    def __init__(
        self, base_model: BaseActivityModel, novelty_threshold: float = 5.0
    ) -> None:
        """Initialize the learner with a base model and novelty threshold."""
        self.model = base_model
        self.prototypes: Dict[str, np.ndarray] = {}
        self.novelty_threshold = float(novelty_threshold)

    def _embed(self, csi: CSIData) -> np.ndarray:
        feat = np.array([float(csi.amplitude.mean())], dtype=np.float32)
        tensor = torch.tensor(feat, dtype=torch.float32)
        with torch.no_grad():
            emb = self.model(tensor)
        return emb.detach().cpu().numpy().reshape(-1)

    def learn_new_activity(
        self, support_set: Iterable[CSIData], activity_name: str
    ) -> None:
        """Create prototype from a small support set for a new activity."""
        feats = [self._embed(csi) for csi in support_set]
        if not feats:
            raise ValueError("Support set is empty")
        self.prototypes[activity_name] = np.mean(np.vstack(feats), axis=0)

    def predict_with_confidence(self, csi_data: CSIData) -> Tuple[str, float, bool]:
        """Predict activity with novelty detection."""
        feat = self._embed(csi_data)
        if not self.prototypes:
            return "unknown", 0.0, True
        dists = {
            name: float(np.linalg.norm(feat - proto))
            for name, proto in self.prototypes.items()
        }
        best_name, best_dist = min(dists.items(), key=lambda kv: kv[1])
        confidence = float(np.exp(-best_dist))
        is_novel = best_dist > self.novelty_threshold
        return best_name, confidence, is_novel


class PrototypicalNetwork:
    """Prototypical Networks for few-shot activity classification."""

    def __init__(self, encoder: BaseActivityModel) -> None:
        """Initialize with an embedding model."""
        self.encoder = encoder
        self.prototypes: Dict[int, torch.Tensor] = {}

    def _encode(self, csi: CSIData) -> torch.Tensor:
        feat = torch.tensor([[float(csi.amplitude.mean())]], dtype=torch.float32)
        with torch.no_grad():
            emb = self.encoder(feat)
        return emb.squeeze(0)

    def fit(self, support: Iterable[Tuple[CSIData, int]]) -> None:
        """Create class prototypes from a labelled support set."""
        by_class: Dict[int, List[torch.Tensor]] = {}
        for csi, label in support:
            by_class.setdefault(int(label), []).append(self._encode(csi))
        self.prototypes = {
            label: torch.stack(feats).mean(dim=0) for label, feats in by_class.items()
        }

    def predict(self, csi: CSIData) -> Tuple[int, float]:
        """Predict the class for ``csi`` returning label and confidence."""
        if not self.prototypes:
            return -1, 0.0
        query = self._encode(csi)
        dists = {
            label: torch.norm(query - proto).unsqueeze(0)
            for label, proto in self.prototypes.items()
        }
        labels, dist_vals = zip(*dists.items())
        probs = F.softmax(-torch.stack(dist_vals), dim=0)
        best_idx = int(torch.argmax(probs))
        return int(labels[best_idx]), float(probs[best_idx])


class RelationNetwork:
    """Relation Network for similarity-based few-shot learning."""

    def __init__(
        self,
        encoder: BaseActivityModel,
        relation_module: Optional[nn.Module] = None,
    ) -> None:
        """Initialize with an encoder and optional relation module."""
        self.encoder = encoder
        self.relation = relation_module

    def _encode(self, csi: CSIData) -> torch.Tensor:
        feat = torch.tensor([[float(csi.amplitude.mean())]], dtype=torch.float32)
        with torch.no_grad():
            emb = self.encoder(feat)
        return emb.squeeze(0)

    def predict(
        self, support: Iterable[Tuple[CSIData, int]], query: CSIData
    ) -> Tuple[int, float]:
        """Classify ``query`` using relation scores with ``support``."""
        q = self._encode(query)
        scores: Dict[int, List[torch.Tensor]] = {}
        for csi, label in support:
            s = self._encode(csi)
            if self.relation is not None:
                pair = torch.cat([s, q], dim=0).unsqueeze(0)
                score = self.relation(pair).squeeze(0)
            else:
                score = torch.exp(-torch.norm(s - q))
            scores.setdefault(int(label), []).append(score)
        avg_scores = {k: torch.stack(v).mean() for k, v in scores.items()}
        labels, score_vals = zip(*avg_scores.items())
        best_idx = int(torch.argmax(torch.stack(score_vals)))
        return int(labels[best_idx]), float(score_vals[best_idx])


def get_meta_optimizer(
    model: nn.Module, optimizer: str = "sgd", lr: float = 0.001
) -> torch.optim.Optimizer:
    """Return a meta-optimizer for the given model."""
    if optimizer.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    return torch.optim.SGD(model.parameters(), lr=lr)


class MAMLLearner:
    """Model-Agnostic Meta-Learner (MAML)."""

    def __init__(
        self,
        model: BaseActivityModel,
        inner_lr: float = 0.01,
        meta_optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        """Create a MAML learner around ``model``."""
        self.model = model
        self.inner_lr = inner_lr
        self.meta_optimizer = meta_optimizer or get_meta_optimizer(model)

    def _batch(
        self, data: Sequence[Tuple[CSIData, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = [[float(csi.amplitude.mean())] for csi, _ in data]
        labels = [int(label) for _, label in data]
        x = torch.tensor(feats, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        return x, y

    def adapt(
        self,
        support: Sequence[Tuple[CSIData, int]],
        query: Sequence[Tuple[CSIData, int]],
    ) -> float:
        """Perform one MAML update and return query accuracy."""
        if not support or not query:
            raise ValueError("Support and query sets must be non-empty")

        params = dict(self.model.named_parameters())
        xs, ys = self._batch(support)
        xq, yq = self._batch(query)

        def loss_fn(p: Dict[str, torch.Tensor]) -> torch.Tensor:
            logits = functional_call(self.model, p, (xs,))
            return F.cross_entropy(logits, ys)

        grads = torch.autograd.grad(loss_fn(params), params.values(), create_graph=True)
        adapted = {
            name: param - self.inner_lr * grad
            for (name, param), grad in zip(params.items(), grads)
        }

        logits_q = functional_call(self.model, adapted, (xq,))
        loss_q = F.cross_entropy(logits_q, yq)
        self.meta_optimizer.zero_grad()
        loss_q.backward()
        self.meta_optimizer.step()

        with torch.no_grad():
            preds = logits_q.argmax(dim=1)
        return float((preds == yq).float().mean().item())

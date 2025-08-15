"""Training utilities for WiFi activity recognition models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils.logging import setup_logging


@dataclass
class Trainer:
    """Simple training helper wrapping a PyTorch model and dataset.

    Parameters
    ----------
    model:
        Neural network to train.
    dataset:
        Dataset providing ``train`` and ``val`` splits of ``(data, labels)``.
    batch_size:
        Mini-batch size.
    learning_rate:
        Optimizer learning rate.
    device:
        Device string (``"cpu"`` or ``"cuda"``). If ``None`` the best available
        device is selected automatically.
    """

    model: nn.Module
    dataset: "Dataset"  # forward reference
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: Optional[str] = None

    def __post_init__(self) -> None:
        """Prepare model, optimizer, and data loaders."""
        self.logger = setup_logging(__name__)
        self._device = torch.device(
            self.device
            if self.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self._device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        self.metrics: Dict[str, float] = {
            "train_accuracy": 0.0,
            "val_accuracy": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "val_f1": 0.0,
        }

        train_data = torch.tensor(self.dataset.train[0], dtype=torch.float32)
        train_labels = torch.tensor(self.dataset.train[1], dtype=torch.long)
        val_data = torch.tensor(self.dataset.val[0], dtype=torch.float32)
        val_labels = torch.tensor(self.dataset.val[1], dtype=torch.long)

        self.train_loader = DataLoader(
            TensorDataset(train_data, train_labels),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(val_data, val_labels),
            batch_size=self.batch_size,
            shuffle=False,
        )

    # ------------------------------------------------------------------
    def train(
        self,
        epochs: int,
        progress_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> None:
        """Train the model for a number of epochs."""
        for epoch in range(1, epochs + 1):
            self.model.train()
            train_preds: list[int] = []
            train_targets: list[int] = []
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                preds = outputs.argmax(dim=1)
                train_preds.extend(preds.cpu().tolist())
                train_targets.extend(targets.cpu().tolist())

            train_metrics = self._compute_metrics(train_targets, train_preds)
            val_metrics = self._evaluate(self.val_loader)
            self.metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
            self.metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            if progress_callback:
                progress_callback(epoch, dict(self.metrics))

    # ------------------------------------------------------------------
    def _evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        preds: list[int] = []
        targets_all: list[int] = []
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self.model(inputs)
                preds.extend(outputs.argmax(dim=1).cpu().tolist())
                targets_all.extend(targets.cpu().tolist())
        return self._compute_metrics(targets_all, preds)

    # ------------------------------------------------------------------
    @staticmethod
    def _compute_metrics(targets: list[int], preds: list[int]) -> Dict[str, float]:
        accuracy = accuracy_score(targets, preds) if targets else 0.0
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, preds, average="macro", zero_division=0
        )
        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    # ------------------------------------------------------------------
    def save_model(self, path: str | Path) -> None:
        """Persist the trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, float]:
        """Return metrics collected during training."""
        return dict(self.metrics)

    # ------------------------------------------------------------------
    def cross_validate(self, folds: int = 5, epochs: int = 1) -> Dict[str, float]:
        """Perform k-fold cross-validation and return averaged metrics."""
        X, y = self.dataset.train
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        fold_metrics = []
        for train_idx, val_idx in kf.split(X):
            train_split = (X[train_idx], y[train_idx])
            val_split = (X[val_idx], y[val_idx])
            fold_dataset = self.dataset.__class__(
                train=train_split, val=val_split, test=self.dataset.test
            )
            model = copy.deepcopy(self.model)
            model.apply(
                lambda m: m.reset_parameters()
                if hasattr(m, "reset_parameters")
                else None
            )
            fold_trainer = Trainer(
                model=model,
                dataset=fold_dataset,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                device=str(self._device),
            )
            fold_trainer.train(epochs)
            fold_metrics.append(
                {
                    k: v
                    for k, v in fold_trainer.get_metrics().items()
                    if k.startswith("val_")
                }
            )
        summary = {
            key[4:]: float(np.mean([m[key] for m in fold_metrics]))
            for key in fold_metrics[0].keys()
        }
        self.metrics["cross_val"] = summary
        return summary

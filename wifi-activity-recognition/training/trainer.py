"""Training utilities for WiFi activity recognition models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import torch
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
        self.metrics: Dict[str, float] = {"train_accuracy": 0.0, "val_accuracy": 0.0}

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
            correct = 0
            total = 0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            train_acc = correct / total if total else 0.0
            val_acc = self._evaluate(self.val_loader)
            self.metrics["train_accuracy"] = train_acc
            self.metrics["val_accuracy"] = val_acc

            if progress_callback:
                progress_callback(epoch, dict(self.metrics))

    # ------------------------------------------------------------------
    def _evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return correct / total if total else 0.0

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

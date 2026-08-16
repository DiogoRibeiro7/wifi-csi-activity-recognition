"""Training utilities for WiFi activity recognition models."""

from __future__ import annotations

import copy
import typing as t
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..models import save_model_artifact
from ..utils.logging import setup_logging
from .callbacks import (
    Callback,
    EarlyStopping,
    LRScheduler,
    ModelCheckpoint,
    TensorBoardLogger,
)
from .metrics import classification_metrics

if t.TYPE_CHECKING:  # pragma: no cover
    from ..datasets import Dataset


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
    checkpoint_path:
        Optional path where the best model checkpoint will be stored.
    early_stopping_patience:
        If set, training stops after this many epochs without improvement in
        validation loss.
    early_stopping_delta:
        Minimum change in monitored metric to qualify as an improvement.
    """

    model: nn.Module
    dataset: "Dataset"  # forward reference
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: Optional[str] = None
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    callbacks: Optional[List[Callback]] = None
    use_data_parallel: bool = False
    checkpoint_path: Optional[str | Path] = None
    early_stopping_patience: Optional[int] = None
    early_stopping_delta: float = 0.0
    tensorboard_log_dir: Optional[str | Path] = None
    use_lr_scheduler: bool = False
    lr_scheduler_factor: float = 0.1
    lr_scheduler_patience: int = 2
    lr_scheduler_monitor: str = "val_loss"

    def __post_init__(self) -> None:
        """Prepare model, optimizer, and data loaders."""
        self.logger = setup_logging(__name__)
        self._device = torch.device(
            self.device
            if self.device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if (
            self.use_data_parallel
            and self._device.type == "cuda"
            and torch.cuda.device_count() > 1
        ):
            self.model = nn.DataParallel(self.model)
        self.model.to(self._device)
        self.criterion = self.loss_fn if self.loss_fn else nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        if self.use_lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience,
            )
            self.callbacks = (self.callbacks or []) + [
                LRScheduler(scheduler, monitor=self.lr_scheduler_monitor)
            ]
        self.metrics: Dict[str, Any] = {
            "train_accuracy": 0.0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "val_accuracy": 0.0,
            "val_precision": 0.0,
            "val_recall": 0.0,
            "val_f1": 0.0,
            "val_per_class_accuracy": [],
            "val_confusion_matrix": [],
        }
        self.callbacks = self.callbacks or []
        if self.early_stopping_patience is not None:
            self.callbacks.append(
                EarlyStopping(
                    patience=self.early_stopping_patience,
                    min_delta=self.early_stopping_delta,
                )
            )
        if self.checkpoint_path is not None:
            self.callbacks.append(ModelCheckpoint(self.checkpoint_path))
        if self.tensorboard_log_dir is not None:
            self.callbacks.append(TensorBoardLogger(self.tensorboard_log_dir))
        self.stop_training = False

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
        for cb in self.callbacks:
            cb.on_train_begin(self)
        for epoch in range(1, epochs + 1):
            if self.stop_training:
                break
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)
            self.model.train()
            train_preds: list[int] = []
            train_targets: list[int] = []
            train_loss = 0.0
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
                train_loss += loss.item() * len(targets)

            train_loss /= len(self.train_loader.dataset)
            train_metrics = classification_metrics(
                train_targets, train_preds, num_classes=len(self.dataset.classes)
            )
            val_metrics = self._evaluate(self.val_loader)
            val_loss = val_metrics.pop("loss")
            self.metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
            self.metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            self.metrics["train_loss"] = float(train_loss)
            self.metrics["val_loss"] = float(val_loss)

            logs = dict(self.metrics)
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs)
            if progress_callback:
                progress_callback(epoch, logs)
        for cb in self.callbacks:
            cb.on_train_end(self)

    # ------------------------------------------------------------------
    def _evaluate(self, loader: DataLoader) -> Dict[str, Any]:
        """Evaluate the model on ``loader`` and return metrics with loss."""
        self.model.eval()
        preds: list[int] = []
        targets_all: list[int] = []
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * len(targets)
                preds.extend(outputs.argmax(dim=1).cpu().tolist())
                targets_all.extend(targets.cpu().tolist())
        metrics = classification_metrics(
            targets_all, preds, num_classes=len(self.dataset.classes)
        )
        metrics["loss"] = (
            total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
        )
        return metrics

    # ------------------------------------------------------------------
    def evaluate(self, split: str = "test") -> Dict[str, Any]:
        """Evaluate the model on one split, returning report-friendly metrics."""
        split_name = split.lower()
        split_map = {
            "train": self.dataset.train,
            "val": self.dataset.val,
            "validation": self.dataset.val,
            "test": self.dataset.test,
        }
        try:
            data, labels = split_map[split_name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ValueError(
                f"Unknown evaluation split '{split}'. Expected train, val, or test."
            ) from exc

        eval_loader = DataLoader(
            TensorDataset(
                torch.tensor(data, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )
        metrics = self._evaluate(eval_loader)
        metrics["f1_score"] = metrics["f1"]
        metrics["split"] = "val" if split_name == "validation" else split_name
        metrics["num_samples"] = int(len(labels))
        return metrics

    # ------------------------------------------------------------------
    def save_model(self, path: str | Path) -> None:
        """Persist the trained model to disk."""
        save_model_artifact(self.model, path)

    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """Return metrics collected during training."""
        return dict(self.metrics)

    # ------------------------------------------------------------------
    def cross_validate(
        self,
        folds: int = 5,
        epochs: int = 1,
        groups: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """Run k-fold cross-validation and return averaged metrics.

        Parameters
        ----------
        folds:
            Number of folds.
        epochs:
            Training epochs per fold.
        groups:
            Optional group identifier per training sample -- subject, session,
            environment or device. When given, ``StratifiedGroupKFold`` keeps
            every sample from a group inside one fold, so the score measures
            generalisation to unseen groups.

            Without it, plain ``StratifiedKFold`` scatters windows from the
            same recording across folds. For WiFi sensing that is usually
            optimistic: the model can score well by recognising the room or the
            person rather than the activity.
        """
        X, y = self.dataset.train

        splitter: Any
        if groups is not None:
            groups = np.asarray(groups)
            if len(groups) != len(y):
                raise ValueError(
                    f"groups has length {len(groups)} but the training split "
                    f"has {len(y)} samples"
                )
            distinct = len(np.unique(groups))
            if distinct < folds:
                raise ValueError(
                    f"cannot build {folds} group-disjoint folds from "
                    f"{distinct} distinct groups"
                )
            splitter = StratifiedGroupKFold(n_splits=folds)
            split_iter = splitter.split(X, y, groups=groups)
        else:
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            split_iter = splitter.split(X, y)

        fold_metrics: List[Dict[str, Any]] = []
        for train_idx, val_idx in split_iter:
            train_split = (X[train_idx], y[train_idx])
            val_split = (X[val_idx], y[val_idx])
            fold_dataset = self.dataset.__class__(
                train=train_split, val=val_split, test=self.dataset.test
            )
            model = copy.deepcopy(self.model)
            model.apply(
                lambda m: (
                    m.reset_parameters() if hasattr(m, "reset_parameters") else None
                )
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
        summary: Dict[str, Any] = {}
        for key in fold_metrics[0].keys():
            values = [m[key] for m in fold_metrics]
            bare = key[4:]
            if isinstance(values[0], list):
                summary[bare] = np.mean(np.array(values), axis=0).tolist()
            else:
                summary[bare] = float(np.mean(values))
        self.metrics["cross_val"] = summary
        return summary

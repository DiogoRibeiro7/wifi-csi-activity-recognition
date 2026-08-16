"""Group-aware cross-validation in the Trainer.

``cross_validate`` used ``StratifiedKFold`` unconditionally, which scatters
windows from one recording across folds. Passing ``groups`` switches it to
``StratifiedGroupKFold`` so each fold measures generalisation to unseen
subjects, sessions or environments.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import nn

from wifi_activity_recognition.datasets import Dataset
from wifi_activity_recognition.training import Trainer

N_SUBJECTS = 6
WINDOWS_EACH = 12


class TinyNet(nn.Module):
    """Minimal classifier; the split behaviour is what is under test."""

    def __init__(self, num_classes: int = 2) -> None:
        """Build a single linear layer."""
        super().__init__()
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return logits."""
        return self.fc(x)


def _cohort():
    """Grouped data where column 0 identifies the subject."""
    total = N_SUBJECTS * WINDOWS_EACH
    subjects = np.repeat(np.arange(N_SUBJECTS), WINDOWS_EACH)
    labels = np.tile([0, 1], total // 2)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(total, 4)).astype(np.float32)
    data[:, 0] = subjects
    return data, labels, subjects


def _dataset(data, labels) -> Dataset:
    split = (data, labels)
    return Dataset(train=split, val=split, test=split)


# ---------------------------------------------------------------------------
# Fold composition -- checked directly against the splitters
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_ungrouped_folds_share_subjects() -> None:
    """Records why the groups argument exists.

    Plain StratifiedKFold is correct for IID data; this shows what it does to
    grouped data, so the contrast below is meaningful.
    """
    data, labels, subjects = _cohort()
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    train_idx, val_idx = next(splitter.split(data, labels))
    overlap = set(subjects[train_idx]) & set(subjects[val_idx])
    assert overlap, "expected subject overlap without grouping"


@pytest.mark.regression
def test_grouped_folds_never_share_a_subject() -> None:
    """Every fold must be group-disjoint."""
    data, labels, subjects = _cohort()
    splitter = StratifiedGroupKFold(n_splits=3)

    for train_idx, val_idx in splitter.split(data, labels, groups=subjects):
        overlap = set(subjects[train_idx]) & set(subjects[val_idx])
        assert not overlap, f"fold leaked subjects {overlap}"


# ---------------------------------------------------------------------------
# Trainer.cross_validate
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_cross_validate_accepts_groups_and_returns_metrics() -> None:
    """The grouped path must run and report averaged validation metrics."""
    data, labels, subjects = _cohort()
    trainer = Trainer(model=TinyNet(), dataset=_dataset(data, labels), batch_size=8)

    summary = trainer.cross_validate(folds=3, epochs=1, groups=subjects)

    # cross_validate strips the val_ prefix when averaging across folds.
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in summary, f"{key} missing from {sorted(summary)}"
        assert 0.0 <= summary[key] <= 1.0


def test_cross_validate_still_works_without_groups() -> None:
    """Omitting groups must keep the previous behaviour."""
    data, labels, _ = _cohort()
    trainer = Trainer(model=TinyNet(), dataset=_dataset(data, labels), batch_size=8)

    summary = trainer.cross_validate(folds=3, epochs=1)

    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in summary, f"{key} missing from {sorted(summary)}"


def test_cross_validate_rejects_groups_of_the_wrong_length() -> None:
    """A mismatched groups array is a caller error, not silent truncation."""
    data, labels, _ = _cohort()
    trainer = Trainer(model=TinyNet(), dataset=_dataset(data, labels), batch_size=8)

    with pytest.raises(ValueError, match="groups has length"):
        trainer.cross_validate(folds=3, epochs=1, groups=np.arange(len(labels) - 1))


def test_cross_validate_rejects_more_folds_than_groups() -> None:
    """Five group-disjoint folds cannot come from three groups."""
    data, labels, subjects = _cohort()
    subjects = subjects % 3  # collapse to three distinct groups
    trainer = Trainer(model=TinyNet(), dataset=_dataset(data, labels), batch_size=8)

    with pytest.raises(ValueError, match="cannot build 5 group-disjoint folds"):
        trainer.cross_validate(folds=5, epochs=1, groups=subjects)

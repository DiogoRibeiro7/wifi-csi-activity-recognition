"""Group-aware splitting for subject-independent evaluation.

Windows cut from one recording session share room geometry, body habitus,
antenna placement and hardware state. Splitting samples at random puts windows
from the same subject on both sides, so a model can score well by recognising
*the person or the room* rather than the activity. The reported number then
does not survive contact with a new subject.

These tests pin the distinction: they show the sample-wise splitter leaks, and
that the group-aware splitters do not.
"""

from __future__ import annotations

import numpy as np
import pytest

from wifi_activity_recognition.datasets.loaders import (
    leave_one_group_out,
    split_dataset,
    split_dataset_by_groups,
)

N_SUBJECTS = 6
WINDOWS_EACH = 20


def _cohort(n_subjects: int = N_SUBJECTS, windows: int = WINDOWS_EACH):
    """Return (data, labels, subjects) where each row is traceable to a subject.

    Column 0 carries the subject id so a split can be traced back without
    matching whole rows.
    """
    total = n_subjects * windows
    subjects = np.repeat(np.arange(n_subjects), windows)
    labels = np.tile([0, 1], total // 2)
    rng = np.random.default_rng(0)
    data = rng.normal(size=(total, 4))
    data[:, 0] = subjects
    return data, labels, subjects


def _subjects_in(split_data: np.ndarray) -> set[int]:
    """Recover the subject ids present in a split."""
    return {int(value) for value in split_data[:, 0]}


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_sample_wise_split_leaks_subjects_across_splits() -> None:
    """Documents why the group-aware splitters exist.

    This is not a bug in ``split_dataset`` -- it is correct for IID data. It is
    a demonstration that it is the wrong tool for grouped CSI, so that anyone
    changing these APIs can see what the alternative is for.
    """
    data, labels, _ = _cohort()
    (train_x, _), _, (test_x, _) = split_dataset(
        data, labels, val_ratio=0.2, test_ratio=0.2, random_state=0
    )

    overlap = _subjects_in(train_x) & _subjects_in(test_x)
    assert overlap, (
        "expected sample-wise splitting to place subjects on both sides; if "
        "this now passes, split_dataset has changed behaviour"
    )
    assert len(overlap) == N_SUBJECTS, "every subject should appear in both"


# ---------------------------------------------------------------------------
# split_dataset_by_groups
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_group_split_keeps_every_subject_on_one_side() -> None:
    """No subject may appear in more than one split."""
    data, labels, subjects = _cohort()
    (train_x, _), (val_x, _), (test_x, _) = split_dataset_by_groups(
        data, labels, subjects, val_ratio=0.2, test_ratio=0.2, random_state=0
    )

    train_s, val_s, test_s = map(_subjects_in, (train_x, val_x, test_x))
    assert not train_s & val_s, f"train/val share subjects {train_s & val_s}"
    assert not train_s & test_s, f"train/test share subjects {train_s & test_s}"
    assert not val_s & test_s, f"val/test share subjects {val_s & test_s}"


def test_group_split_uses_every_sample_exactly_once() -> None:
    """Splitting must not drop or duplicate data."""
    data, labels, subjects = _cohort()
    (train_x, _), (val_x, _), (test_x, _) = split_dataset_by_groups(
        data, labels, subjects, random_state=0
    )

    assert len(train_x) + len(val_x) + len(test_x) == len(data)
    seen = _subjects_in(train_x) | _subjects_in(val_x) | _subjects_in(test_x)
    assert seen == set(range(N_SUBJECTS))


def test_group_split_leaves_no_split_empty() -> None:
    """Every split must receive at least one group."""
    data, labels, subjects = _cohort(n_subjects=3, windows=10)
    splits = split_dataset_by_groups(data, labels, subjects, random_state=0)

    for name, (split_x, _) in zip(("train", "val", "test"), splits):
        assert len(split_x) > 0, f"{name} split is empty"


def test_group_split_is_reproducible_for_a_fixed_seed() -> None:
    """The same seed must produce the same group assignment."""
    data, labels, subjects = _cohort()
    first = split_dataset_by_groups(data, labels, subjects, random_state=42)
    second = split_dataset_by_groups(data, labels, subjects, random_state=42)

    for (a_x, a_y), (b_x, b_y) in zip(first, second):
        assert np.array_equal(a_x, b_x)
        assert np.array_equal(a_y, b_y)


def test_group_split_rejects_too_few_groups() -> None:
    """Three splits cannot be built from fewer than three groups."""
    data, labels, subjects = _cohort(n_subjects=2, windows=10)
    with pytest.raises(ValueError, match="at least 3 distinct groups"):
        split_dataset_by_groups(data, labels, subjects)


def test_group_split_rejects_mismatched_lengths() -> None:
    """A groups array of the wrong length is a caller error, not a silent crop."""
    data, labels, _ = _cohort()
    with pytest.raises(ValueError, match="same length"):
        split_dataset_by_groups(data, labels, np.arange(len(labels) - 1))


# ---------------------------------------------------------------------------
# leave_one_group_out
# ---------------------------------------------------------------------------


@pytest.mark.regression
def test_loso_holds_out_exactly_one_subject_per_fold() -> None:
    """The defining property of leave-one-subject-out."""
    data, labels, subjects = _cohort()
    folds = list(leave_one_group_out(data, labels, subjects))

    assert len(folds) == N_SUBJECTS, "one fold per subject"
    for (train_x, _), (test_x, _), held_out in folds:
        assert _subjects_in(test_x) == {int(held_out)}
        assert int(held_out) not in _subjects_in(train_x)


def test_loso_folds_partition_the_data(tmp_path) -> None:
    """Train and test must together account for every sample, every fold."""
    data, labels, subjects = _cohort()
    for (train_x, _), (test_x, _), _ in leave_one_group_out(data, labels, subjects):
        assert len(train_x) + len(test_x) == len(data)


def test_loso_tests_each_subject_exactly_once() -> None:
    """Across all folds every subject is held out once and only once."""
    data, labels, subjects = _cohort()
    held = [int(group) for _, _, group in leave_one_group_out(data, labels, subjects)]

    assert sorted(held) == list(range(N_SUBJECTS))


def test_loso_requires_at_least_two_groups() -> None:
    """One group cannot be both trained on and held out."""
    data, labels, _ = _cohort(n_subjects=1, windows=10)
    with pytest.raises(ValueError, match="at least 2 groups"):
        list(leave_one_group_out(data, labels, np.zeros(len(labels))))

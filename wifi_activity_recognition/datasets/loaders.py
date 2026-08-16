"""Dataset loading and splitting utilities.

Two families of splitter live here, and the difference matters more for WiFi
sensing than for ordinary classification.

:func:`split_dataset` divides individual samples at random. For IID data that
is correct. For CSI it usually is not: windows cut from one recording session
share room geometry, body habitus, antenna placement and hardware state.
Scattering them across train and test lets a model score well by recognising
*the room or the person*, not the activity, and the number it reports will not
survive contact with a new subject.

:func:`split_dataset_by_groups` and :func:`leave_one_group_out` keep every
sample from a group on one side of the split, which is what subject-,
session- or environment-independent evaluation requires.
"""

from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np

Split = Tuple[np.ndarray, np.ndarray]
ThreeSplits = Tuple[Split, Split, Split]


def load_dataset(
    directory: Path | str,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> ThreeSplits:
    """Load ``data.npy`` and ``labels.npy`` from ``directory`` and split them.

    The function expects two ``.npy`` files named ``data.npy`` and
    ``labels.npy`` inside ``directory``.  ``val_ratio`` and ``test_ratio``
    control how many samples are allocated to the respective splits.  ``shuffle``
    controls whether data are shuffled before splitting and ``random_state``
    makes the operation deterministic.
    """
    directory = Path(directory)
    data_file = directory / "data.npy"
    labels_file = directory / "labels.npy"
    if not data_file.exists() or not labels_file.exists():  # pragma: no cover -
        raise FileNotFoundError("data.npy and labels.npy must exist in directory")
    data = np.load(data_file)
    labels = np.load(labels_file)
    return split_dataset(data, labels, val_ratio, test_ratio, shuffle, random_state)


def split_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> ThreeSplits:
    """Split data and labels into train, validation, and test sets.

    ``val_ratio`` and ``test_ratio`` must sum to less than ``1.0``; the
    remaining samples are assigned to the training split.  If ``shuffle`` is
    ``True`` the data are shuffled deterministically using ``random_state``.

    .. warning::
       This splits individual samples. If several samples come from the same
       subject, session or environment, all of those appear on both sides of
       the split and the resulting scores are optimistic. Use
       :func:`split_dataset_by_groups` when group labels are available.
    """
    if len(data) != len(labels):  # pragma: no cover - safety check
        raise ValueError("Data and labels must have the same length")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    num_samples = len(data)
    indices = np.arange(num_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    test_size = int(num_samples * test_ratio)
    val_size = int(num_samples * val_ratio)

    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]

    train = (data[train_indices], labels[train_indices])
    val = (data[val_indices], labels[val_indices])
    test = (data[test_indices], labels[test_indices])
    return train, val, test


def split_dataset_by_groups(
    data: np.ndarray,
    labels: np.ndarray,
    groups: Sequence,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: Optional[int] = None,
) -> ThreeSplits:
    """Split so that no group appears in more than one split.

    ``groups`` labels each sample with the subject, session, environment or
    device it came from. Whole groups are assigned to a split, so evaluation
    measures generalisation to *unseen* groups rather than to unseen windows
    of groups the model has already learned.

    The ratios are approximate: groups are indivisible, so the achieved sizes
    depend on how many samples each group contributes.

    Parameters
    ----------
    data, labels:
        Arrays of equal length.
    groups:
        Group identifier per sample, same length as ``labels``.
    val_ratio, test_ratio:
        Approximate fraction of *samples* for each split.
    random_state:
        Seeds the group shuffle, making the assignment reproducible.

    Raises
    ------
    ValueError
        If lengths disagree, the ratios are impossible, or there are fewer
        than three distinct groups to divide between three splits.
    """
    groups = np.asarray(groups)
    if not len(data) == len(labels) == len(groups):
        raise ValueError(
            f"data ({len(data)}), labels ({len(labels)}) and groups "
            f"({len(groups)}) must have the same length"
        )
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be < 1")

    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError(
            f"need at least 3 distinct groups to build train/val/test splits, "
            f"got {len(unique)}: {list(unique)}. Use leave_one_group_out for "
            "small group counts."
        )

    rng = np.random.default_rng(random_state)
    order = rng.permutation(unique)

    total = len(labels)
    want_test = total * test_ratio
    want_val = total * val_ratio

    # Fill test, then validation, one whole group at a time; the rest trains.
    # Every split keeps at least one group, so none can come out empty.
    test_groups: list = []
    val_groups: list = []
    counted = {group: int((groups == group).sum()) for group in unique}

    remaining = list(order)
    filled = 0
    while remaining and (filled < want_test or not test_groups):
        group = remaining.pop(0)
        if len(remaining) < 2:  # keep one group each for val and train
            remaining.insert(0, group)
            break
        test_groups.append(group)
        filled += counted[group]

    filled = 0
    while remaining and (filled < want_val or not val_groups):
        group = remaining.pop(0)
        if not remaining:  # the last group must go to train
            remaining.insert(0, group)
            break
        val_groups.append(group)
        filled += counted[group]

    train_groups = remaining

    def _select(selected: Sequence) -> Split:
        mask = np.isin(groups, selected)
        return data[mask], labels[mask]

    return _select(train_groups), _select(val_groups), _select(test_groups)


def leave_one_group_out(
    data: np.ndarray,
    labels: np.ndarray,
    groups: Sequence,
) -> Iterator[Tuple[Split, Split, object]]:
    """Yield ``(train, test, held_out_group)`` once per distinct group.

    This is the leave-one-subject-out protocol when ``groups`` identifies
    subjects, and the leave-one-environment-out or leave-one-session-out
    protocol when it identifies those. Each iteration trains on every other
    group and tests on the one held out, so the reported score is what the
    model achieves on a person, room or session it has never seen.

    Groups are yielded in sorted order for reproducibility.

    Raises
    ------
    ValueError
        If lengths disagree or fewer than two groups are present.
    """
    groups = np.asarray(groups)
    if not len(data) == len(labels) == len(groups):
        raise ValueError(
            f"data ({len(data)}), labels ({len(labels)}) and groups "
            f"({len(groups)}) must have the same length"
        )

    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError(
            f"leave-one-group-out needs at least 2 groups, got {len(unique)}"
        )

    for held_out in unique:
        test_mask = groups == held_out
        train_mask = ~test_mask
        yield (
            (data[train_mask], labels[train_mask]),
            (data[test_mask], labels[test_mask]),
            held_out,
        )


__all__ = [
    "leave_one_group_out",
    "load_dataset",
    "split_dataset",
    "split_dataset_by_groups",
]

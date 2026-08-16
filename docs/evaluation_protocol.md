# Evaluation protocol

Addresses the leakage concern raised in the project review: for WiFi sensing,
splitting samples at random usually produces optimistic numbers that do not
survive contact with a new subject.

## The problem

Windows cut from one recording session share far more than their label. They
share room geometry, multipath structure, body habitus, antenna placement,
and whatever the hardware was doing that afternoon. Scattering them at random
across train and test means the model sees the same room and the same person on
both sides.

It can then score well by recognising *the person or the room* rather than the
activity — and nothing in the reported accuracy reveals that.

Measured on six subjects with twenty windows each:

```
current split_dataset (sample-wise):
  train subjects: [0, 1, 2, 3, 4, 5]
  test  subjects: [0, 1, 2, 3, 4, 5]
  OVERLAP       : [0, 1, 2, 3, 4, 5]

StratifiedGroupKFold:
  train subjects: [1, 2, 4, 5]
  test  subjects: [0, 3]
  overlap       : []
```

Every subject appeared on both sides. This is not a bug in `split_dataset` —
it is correct for IID data — but CSI windows are not IID.

## What to use

| Situation | Use |
|---|---|
| Samples genuinely independent | `split_dataset` |
| Samples grouped by subject, session, environment or device | `split_dataset_by_groups` |
| Reporting subject-independent performance | `leave_one_group_out` |
| Cross-validation over grouped data | `Trainer.cross_validate(groups=...)` |

### Group-disjoint train/val/test

```python
from wifi_activity_recognition.datasets import split_dataset_by_groups

train, val, test = split_dataset_by_groups(
    data, labels, groups=subject_ids, val_ratio=0.2, test_ratio=0.2, random_state=0
)
```

Whole groups are assigned to a split, so the ratios are approximate — groups
are indivisible. Requires at least three distinct groups.

### Leave-one-subject-out

```python
from wifi_activity_recognition.datasets import leave_one_group_out

for (train_x, train_y), (test_x, test_y), held_out in leave_one_group_out(
    data, labels, groups=subject_ids
):
    ...  # train on everyone else, test on `held_out`
```

One fold per group, each group held out exactly once. Substitute session or
environment identifiers for leave-one-session-out or
leave-one-environment-out; the mechanism is the same and only the meaning of
`groups` changes.

### Grouped cross-validation

```python
summary = trainer.cross_validate(folds=5, epochs=10, groups=subject_ids)
```

With `groups`, folds are built by `StratifiedGroupKFold`, which keeps class
balance while holding groups together. Without it the previous
`StratifiedKFold` behaviour is unchanged, so existing callers are unaffected.

## Reporting

For a result to be interpretable, state which protocol produced it. "94%
accuracy" means very different things under a random split and under LOSO, and
the gap between them is itself informative — a large drop indicates the model
was leaning on subject or environment identity.

Recommended minimum for a study:

- the protocol used, and what `groups` identified
- macro-F1 and balanced accuracy alongside accuracy, since activity classes are
  rarely balanced
- per-fold scores, not only the mean — LOSO variance across subjects is a
  result in its own right
- a confusion matrix

## Still to do

- `Dataset` does not yet carry group metadata; groups are passed alongside the
  arrays. Attaching subject, session and environment to the data structure is
  the natural next step, and belongs with the `CSISequence` work the review
  proposes.
- The Widar3 and SignFi loaders delegate to the generic NumPy loader and do not
  surface the subject and environment identifiers those datasets record, so
  group-aware evaluation on them is not yet possible.

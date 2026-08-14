# Test quality audit

Tracking issue: [#23](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/23).

Baseline: CI run on `main`, Python 3.10 — **3327 statements, 557 missed, 83% coverage**, 262 tests passing.

The headline number is not the interesting part. A module can sit at 40% because
half of it is an optional-dependency path that cannot run in CI, or at 85%
because its tests call it once and assert the output has the right shape. This
audit classifies the gaps by *why* they exist, because the remedy differs.

## Classification

### A — Optional-dependency paths (coverage artefact, not a gap)

TensorFlow is not installed in CI, so every `*TensorFlowModel` class body is
unreachable. This accounts for essentially all the shortfall in the model
package.

| Module | Cover | Uncovered region |
|---|---|---|
| `models/cnn2d.py` | 40% | 16-17, 70-114 — TF variant + import fallback |
| `models/cnn3d.py` | 40% | 15-16, 55-88 — TF variant |
| `models/advanced_cnn3d.py` | 47% | 18-19, 121-216 — TF variant |
| `models/resnet.py` | 62% | 19-20, 57-58, 74-96 — TF variant |

These four are among the five lowest-coverage modules in the package and are
**not** weakly tested: their PyTorch paths carry behaviour-level tests added
under #21. Reading the percentage alone would send effort exactly the wrong way.

*Action: none on test depth.* Optionally add a TensorFlow job to CI, or accept
the artefact and exclude these blocks from the coverage denominator so the
number reflects testable code.

### B — Guard-clause-only coverage (real gap, now closed)

The function is called only to trigger a validation error, so its body never
runs. Coverage registers the module as partly covered while the actual
transform is untested.

| Module | Finding |
|---|---|
| `preprocessing/advanced_filtering.py` | `median_filter` was exercised **only** by `test_median_filter_validation`, which asserts an even kernel raises. Lines 85-92, the filtering itself, never executed. |
| `preprocessing/advanced_filtering.py` | `morphological_filter` never executed at all (117-130), including its own guards. |

*Action: closed.* Behaviour tests added — impulse removal and edge preservation
for the median filter; anti-extensivity of opening and extensivity of closing
for the morphological filter.

### C — Tests that pass on a special case

The most dangerous category: a green test that constrains almost nothing.

`multirate_resample` had a passing test using `up=2, down=1` on 8 subcarriers.
That ratio is exact, so floor and ceiling division agree. Every inexact ratio
raised, because the declared subcarrier count was computed as `n * up // down`
while `resample_poly` returns `ceil(n * up / down)`:

```
n= 31 1/2: resample_poly->16  declared->15  ValueError from CSIData validation
n= 30 1/4: resample_poly-> 8  declared-> 7  ValueError
n= 33 1/2: resample_poly->17  declared->16  ValueError
n= 30 3/4: resample_poly->23  declared->22  ValueError
```

Four of five sampled ratios were broken. The same function also indexed
`fields[0]` despite typing `fields` as `Iterable`, which fails for a generator.

*Action: closed.* Length is now taken from the resampled array; `fields` is
materialised; partial field sets that would desync `amplitude` and `phase` are
rejected. Parametrised regression test covers all the ratios above.

### D — Genuinely untested behaviour (open)

Real logic with no test, ranked by how load-bearing it is.

| Module | Cover | Untested | Why it matters |
|---|---|---|---|
| `hardware/base.py` | 75% | 319-343 `validate_csi_data`; 364-377 `normalize_csi_amplitude` zscore/log branches | Validation that silently returns the wrong verdict is worse than none. Two of three normalization methods never run. |
| `models/serialization.py` | 72% | 224-240 legacy pickled-model and raw state-dict load paths | Checkpoint loading is where a release breaks user data. The pickle path is also the unsafe one (`weights_only=False`). |
| `inference/streaming_pipeline.py` | 76% | 51-52, 63-84, 135-153 | Error handling and shutdown paths in threaded code — exactly what is hard to get right and easy to leave untested. |
| `research/domain_adaptation.py` | 82% | 153-165 DANN training step | The one genuinely learned method in the module; the rest reduces packets to a mean. |
| `training/federated/privacy.py` | 72% | 33-36 | Privacy accounting that is wrong is a correctness *and* a claims problem. |
| `multimodal/fusion_strategies.py` | 77% | 20, 38, 41, 44, 47, 61, 64, 87, 91 | Scattered single lines: alternative fusion branches, each unexercised. |
| `cli.py` | 77% | 112 statements | Largest absolute gap. Mostly error-handling branches per command. |

## Plan

Ordered by value per unit of effort.

1. **`hardware/base.py`** — `validate_csi_data` and the two unrun normalization
   branches. Small, pure, high leverage: every driver depends on them.
2. **`models/serialization.py`** — round-trip tests per artefact format,
   including the legacy pickled path, before it is moved behind an explicitly
   unsafe API.
3. **`inference/streaming_pipeline.py`** — failure and shutdown paths. Poll to a
   deadline rather than sleeping a fixed interval; see the de-flake in #33.
4. **`multimodal/fusion_strategies.py`** — cheap, mechanical: one test per
   fusion branch.
5. **`cli.py`** — error branches per command, using `CliRunner` as the existing
   CLI tests do.
6. **`research/domain_adaptation.py`** — deferred. The module is scaffolding
   (`adapt_to_target` ignores its `method` argument entirely); test depth should
   follow the algorithmic work, not precede it.

## Method note

Percentages come from the coverage report already produced by the CI test job,
not from a separate local run. Classification came from reading the uncovered
line ranges — which is the step that distinguishes A from D, and the reason
this document exists rather than a coverage badge.

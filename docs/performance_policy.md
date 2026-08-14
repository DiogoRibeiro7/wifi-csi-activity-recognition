# Performance regression policy

Tracking issue: [#24](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/24).

The repository already had benchmark utilities for latency, memory and
accuracy, but nothing decided what a *regression* was or enforced it. This
document is that decision. The enforcement lives in
`tests/benchmarks/test_performance_regression.py`.

## Why the thresholds are not wall-clock numbers

The obvious policy — "inference must stay under N milliseconds" — does not
survive contact with shared CI runners. GitHub-hosted runners vary by roughly
an order of magnitude depending on neighbours, and this project's own
maintainer machine has been observed at 100% CPU across 22 cores from unrelated
work. A threshold loose enough never to produce a false failure is loose enough
to miss every real regression.

So the enforced checks are built from quantities that do **not** depend on
machine speed:

| Kind | What it measures | Why it is stable |
|---|---|---|
| **Complexity** | how cost grows as input grows | a property of the algorithm, not the host |
| **Memory** | `tracemalloc` allocation counts | deterministic for a fixed workload |
| **Relative** | two measurements, same host, same run | the host cancels out |
| **Invariant** | orderings any correct measurement must satisfy | always true or the harness is broken |

Absolute time ceilings exist only as backstops for order-of-magnitude
collapses, set ~100× above typical observations and marked `slow`.

## Baselines

### Complexity budget

Growing an input by **4×** costs about 4× if linear and about 16× if
quadratic. The budget is **8×** — between the two in log space. Verified to
separate them:

```
linear     4x input ->   4.0x cost   budget 8.0  -> PASS
quadratic  4x input ->  14.4x cost   budget 8.0  -> FAIL (caught)
```

Timings take the **minimum of five runs**, not the mean. Contention can only
make a run slower, so the fastest observation is the best estimate of true
cost and the most robust to a noisy host.

### Enforced checks

| Check | Path | Budget |
|---|---|---|
| Segmentation scales linearly in packet count | `preprocessing.segment_windows` | < 8× for 4× input |
| Packet construction scales linearly in array size | `CSIData.__post_init__` validation | < 8× for 4× input |
| Streaming peak memory scales with stream length | `measure_memory_usage` | < 8× for 4× input |
| Repeated processing does not accumulate | `profile_memory_usage` | peak < 4× mean |
| Leak detector clears a clean function | `detect_memory_leak` | must return False |
| Latency percentiles ordered | `measure_latency` | p50 ≤ p95 ≤ p99 ≤ max |
| A slower predictor measures slower | `measure_latency` | strict inequality |
| Trivial prediction under a ceiling (`slow`) | `measure_latency` | p95 < 5 ms |

## Where each check runs

| Tier | Runs | Contents |
|---|---|---|
| **CI, every push and PR** | `pytest -m regression` | everything above except the `slow` backstop |
| **CI, full suite** | `pytest` | adds the `slow` absolute ceiling |
| **Manual / scheduled** | `benchmarks/performance_report.py` | absolute latency, memory and accuracy figures for a named environment |

The `regression` marker is already a dedicated CI job, so these checks gate
merges without a new workflow.

Absolute numbers from `performance_report.py` are **reporting, not gating**.
They are only comparable against runs from the same machine, so they belong in
a scheduled or manual workflow with the environment recorded alongside.

## Environment assumptions

Results are interpretable only with these stated:

- **CPU-only.** CI installs the CPU build of PyTorch; no CUDA path is measured.
- **Python 3.10–3.12** on `ubuntu-latest`.
- **Cold caches.** No warm model or dataset cache is assumed; `measure_latency`
  takes an explicit `warmup` argument for exactly this reason.
- **Shared runner.** Absolute timings are not comparable across runs, which is
  the whole basis for the relative approach above.

## Hot paths most likely to regress

Ranked by how easily a change here degrades performance without failing a
correctness test:

1. **Streaming pipeline** — threaded, with buffering and drop accounting. Bugs
   here show up as latency or dropped packets, not wrong answers.
2. **Feature extraction** — per-window loops over subcarriers; an accidental
   Python-level loop replacing a vectorised call is invisible to correctness.
3. **Preprocessing filters** — `np.apply_along_axis` is convenient and slow;
   swapping a vectorised op for it would not change any result.
4. **Training loop utilities** — callbacks and metric accumulation run per
   batch.
5. **Deployment optimisation** — quantisation and pruning exist specifically to
   improve performance, so a regression defeats their purpose.

Checks currently cover 1–3 indirectly through segmentation, packet
construction and streaming memory. Extending to feature extraction and the
training loop is the natural next step.

## Adding a check

Prefer, in order:

1. a **complexity** assertion — cost growth against input growth
2. a **relative** comparison — two variants measured in the same run
3. an **invariant** — an ordering that must hold
4. an **absolute ceiling** — only as a last resort, marked `slow`, set ~100×
   above the typical observation

A new check must be run several times on a loaded machine before it is
committed. The checks here were validated over four consecutive runs at 100%
CPU. A performance test that fails intermittently gets ignored, and an ignored
test is worse than no test.

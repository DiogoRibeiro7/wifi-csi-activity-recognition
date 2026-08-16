# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-16

The first release built and verified end to end. 0.1.0 shipped a wheel that
contained three modules; this one is checked on every push.

### Fixed

- **Built wheels contained only the top-level package.** `pyproject.toml` listed
  a single package, so all eleven subpackages were omitted: a wheel held 3
  modules instead of 77. Editable installs hid it, and lazy `__getattr__` meant
  `import wifi_activity_recognition` succeeded against the broken wheel while
  every console script failed with `ModuleNotFoundError`.
- **`autotrain` crashed on save.** It called `torch.save` with no `torch` in
  scope, running the entire hyperparameter search before failing.
- **`requirements-dev.txt` was uninstallable.** `pdb++` is not a valid PEP 508
  name, and pip aborts on the first parse error, so the whole file failed. The
  distribution is `pdbpp`.
- **The Transformer ignored sequence order.** With no positional encoding it
  returned bit-identical logits for a sequence, its reverse and a shuffle of
  it — it could not distinguish sitting down from standing up.
- **Temporal filters had no time axis.** `butterworth_filter` took a cutoff in
  Hz but defaulted to the subcarrier axis, and filtering across packets was
  impossible. They now accept a packet sequence, where time is the packet index.
- **`multirate_resample` failed for most ratios.** It declared
  `n * up // down` while `resample_poly` returns `ceil(n * up / down)`; four of
  five sampled ratios raised.
- **Container images could not start.** `python -m wifi_activity_recognition`
  had no `__main__`; no stage installed the package; `opencv-python` needs
  `libGL`, absent from slim images; and `libatlas-base-dev` no longer exists in
  Debian trixie.
- **The Kubernetes manifest guaranteed CrashLoopBackOff** — it probed `/health`
  on a package that serves no HTTP.
- **ESP32 mock mode required hardware.** `connect()` opened a serial port
  regardless of mode, so no driver ran headless.
- **Advertised hardware support overstated reality.** `SUPPORTED_HARDWARE`
  listed Broadcom and MediaTek, which have no drivers.
- Four test modules failed at import, so the suite could never exit zero.

### Added

- **`wifi-har-quickstart`** — a complete train, evaluate and predict cycle in
  about ten seconds on synthetic data, with no hardware and no downloads.
- **Group-aware evaluation** — `split_dataset_by_groups` and
  `leave_one_group_out` for subject-, session- and environment-independent
  protocols, plus `Trainer.cross_validate(groups=...)` using
  `StratifiedGroupKFold`. Random splits put every subject on both sides.
- **Representation adapters** — each model family gets the tensor layout it
  expects. Four of seven registered architectures previously raised on the
  first forward pass through `ActivityRecognizer`.
- **Sinusoidal positional encoding** for the Transformer. Non-persistent
  buffer, so checkpoints written before it still load.
- Packet-sequence support in `butterworth_filter`, `moving_average_filter` and
  `kalman_filter`, with named axis constants.
- `scripts/verify_device.py` — nine hardware-in-the-loop checks for real
  devices, reporting observed values.
- `PLANNED_HARDWARE`, separating shipped from intended platforms.
- Documentation: evaluation protocol, hardware verification, deployment status,
  performance policy, test-quality audit, lint status.

### Changed

- **CI now runs and blocks.** Every run in the project's history had failed,
  all within seconds. There are now twelve jobs including a wheel build with a
  clean-install smoke test, a container build-and-run, and a blocking lint gate.
- Dependabot version updates enabled for pip and GitHub Actions.
- `scikit-learn>=1.0` (was `>=0.24.0`) — `StratifiedGroupKFold` needs 1.0.
- Test suite grew from 196 to over 350 tests; coverage 83%.
- `version.py` now derives `__version__` from its numeric components only. The
  module previously also assigned it as a literal and then overwrote it, so
  editing that literal had no effect.

### Known limitations

- `CSIData` represents a single packet; there is no first-class temporal type
  carrying timestamps, sampling frequency and subject metadata.
- The Widar3 and SignFi loaders delegate to the generic NumPy loader and do not
  surface the subject and environment identifiers those datasets record, so
  group-aware evaluation is not yet possible on them.
- Domain adaptation and few-shot modules operate on mean-amplitude summaries
  rather than learned CSI embeddings; treat as experimental.
- `mypy` reports 245 errors at the project's strict settings; the type-check
  job is non-blocking and the backlog is tracked in `docs/lint_status.md`.
- No hardware platform has a recorded real-device verification run.

## [0.1.0] - 2026-08-14

Initial release.

### Added

- Feature extraction utilities for time and frequency domains
- Transformer and advanced model architectures with factory registration
- Atheros AR9300 hardware driver
- Training framework, dataset utilities and preprocessing pipeline
- Deployment tooling for Docker, Kubernetes and edge devices
- Benchmarking suite for accuracy, latency and memory profiling
- Comprehensive documentation, example notebooks and community tooling

[0.2.0]: https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/releases/tag/v0.1.0

# Deployment status

Tracking issue: [#25](https://github.com/DiogoRibeiro7/wifi-csi-activity-recognition/issues/25).

This page records which deployment targets are **verified**, which are
**examples**, and what each would need to become real. The assets in
`deployment/` previously described a service this package does not implement,
so the distinction was not recoverable by reading them.

## Verified

| Target | How it is verified |
|---|---|
| **Docker `prod` image** | Built and run on every push and PR by the `deployment` CI job |
| **Docker `edge` image** | Built and run in the same job |
| **Kubernetes manifest validity** | Schema-checked with `kubeconform -strict` |

The CI job does more than build. It runs the image's default command, imports
the package inside the container, runs all three console scripts, checks that
imports resolve from outside the source tree, and executes the exact command
the Kubernetes liveness probe uses.

## Examples, not verified

| Target | Status |
|---|---|
| **Kubernetes Deployment** | The manifest is schema-valid and its probe command works, but no cluster deployment is exercised. It is marked `EXAMPLE MANIFEST` in the file itself. |
| **Raspberry Pi / ARM** | The `edge` stage builds and runs on `linux/amd64` in CI. No ARM build is performed and no device is tested. |
| **Long-running service** | Nothing. There is no server. See below. |

## There is no HTTP service

This is the central fact the previous assets obscured. The package ships a CLI
and a library. Nothing binds a socket, no web framework is a dependency, and
no `/health` endpoint exists anywhere in the codebase.

The assets nonetheless described one, and each claim was broken in a way that
would only surface on deployment:

| Asset | Claim | Consequence |
|---|---|---|
| `Dockerfile` (`prod`, `edge`) | `CMD python -m wifi_activity_recognition` | No `__main__` module existed. **Both containers exited immediately on start.** |
| `Dockerfile` (all stages) | package available | Only dependencies were installed. Imports worked by `WORKDIR` accident; console scripts were absent from the image. |
| `docker-compose.yml` | `ports: 8080:8080` on three services | Nothing listens. The mapping advertised a listener that does not exist. |
| `deployment.yaml` | `httpGet /health` liveness + readiness | Probes could never succeed. **Guaranteed `CrashLoopBackOff`** had the container started. |

All four are corrected. The Dockerfile installs the package and runs a module
that exists; compose no longer publishes ports; the manifest uses `exec` probes
that run the CLI. `tests/deployment/test_deployment_contracts.py` pins each one
so they cannot drift back.

## To make Kubernetes a verified target

Either of two routes, and the choice is a product decision rather than a
packaging one:

1. **Add a service interface.** A small HTTP server exposing `/health` and a
   prediction endpoint. The manifest's original shape then becomes correct as
   written, and probes can go back to `httpGet`.
2. **Run a real long-running workload.** `stream` is the natural candidate, but
   it currently needs a capture device: the ESP32 driver opens a serial port in
   `connect()` even in `mock` mode, so no driver runs headless today. Fixing
   that would make a device-free streaming container possible.

Until one of those lands, a `Deployment` has nothing durable to run, which is
why the manifest is labelled an example rather than quietly left looking
production-ready.

## Note on image size

`requirements.txt` names plain `torch`, which on `linux/amd64` resolves to the
CUDA build — several GB of GPU runtime no container here can use, and
especially wrong for a stage described as edge-optimised. The Dockerfile now
installs CPU-only wheels from the PyTorch index before the requirements pass,
so the pinned versions find torch already satisfied.

ARM builds are a separate matter: `linux/arm64` needs platform-appropriate
wheels, and no ARM image is built or tested today.

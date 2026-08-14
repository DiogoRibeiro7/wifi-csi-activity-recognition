# syntax=docker/dockerfile:1

# Base image
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
# Two substitutions, both container-specific:
#
# 1. CPU-only torch first, so the pinned requirements below find it satisfied.
#    requirements.txt names plain `torch`, which on linux/amd64 resolves to the
#    CUDA build: several GB of GPU runtime no container here can use, and
#    actively wrong for the edge stage.
# 2. opencv-python-headless in place of opencv-python. The GUI wheel links
#    against libGL, which python:*-slim does not ship, so `import cv2` fails at
#    runtime and the package's own dependency check aborts on start. The
#    headless wheel is the same library without the GUI bindings.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && grep -v '^opencv-python' requirements.txt > /tmp/requirements-container.txt \
    && pip install --no-cache-dir -r /tmp/requirements-container.txt \
        opencv-python-headless

# Create unprivileged user for runtime stages
RUN useradd -m appuser

# Development image with additional tooling
FROM base AS dev
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
# Editable so a bind-mounted source tree stays live, matching docker-compose.
RUN pip install --no-cache-dir --no-deps -e .
USER appuser
CMD ["bash"]

# Production image
FROM base AS prod
COPY . .
# Install the package rather than relying on the working directory. Without
# this the console scripts do not exist in the image and imports only resolve
# because WORKDIR happens to be the source tree.
RUN pip install --no-cache-dir --no-deps .
USER appuser
CMD ["python", "-m", "wifi_activity_recognition", "--help"]

# Edge-optimized image targeting ARM devices
FROM python:3.11-slim AS edge
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# libatlas-base-dev used to be installed here for BLAS. It was dropped from
# Debian in trixie, which python:3.11-slim now tracks, so this stage failed to
# build at all:
#
#     E: Package 'libatlas-base-dev' has no installation candidate
#
# No system BLAS is needed: the numpy and scipy wheels vendor OpenBLAS, and
# nothing in this image compiles against a BLAS, so the -dev headers were never
# used even when the package existed.
RUN useradd -m appuser
WORKDIR /app
COPY requirements.txt ./
# Same two substitutions as the base stage; see the comment there.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && grep -v '^opencv-python' requirements.txt > /tmp/requirements-container.txt \
    && pip install --no-cache-dir -r /tmp/requirements-container.txt \
        opencv-python-headless
COPY . .
RUN pip install --no-cache-dir --no-deps .
USER appuser
CMD ["python", "-m", "wifi_activity_recognition", "--help"]

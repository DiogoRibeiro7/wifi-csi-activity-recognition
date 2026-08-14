# syntax=docker/dockerfile:1

# Base image
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
# CPU-only torch first, so the pinned requirements below find it already
# satisfied. requirements.txt names plain `torch`, which on linux/amd64
# resolves to the CUDA build: several GB of GPU runtime that no container here
# can use, and actively wrong for the edge stage.
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

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
RUN apt-get update && apt-get install -y --no-install-recommends libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m appuser
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt
COPY . .
RUN pip install --no-cache-dir --no-deps .
USER appuser
CMD ["python", "-m", "wifi_activity_recognition", "--help"]

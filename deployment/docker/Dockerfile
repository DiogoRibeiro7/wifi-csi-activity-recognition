# syntax=docker/dockerfile:1

# Base image
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Create unprivileged user for runtime stages
RUN useradd -m appuser

# Development image with additional tooling
FROM base AS dev
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
USER appuser
CMD ["bash"]

# Production image
FROM base AS prod
COPY . .
USER appuser
CMD ["python", "-m", "wifi_activity_recognition"]

# Edge-optimized image targeting ARM devices
FROM python:3.11-slim AS edge
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m appuser
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
USER appuser
CMD ["python", "-m", "wifi_activity_recognition"]

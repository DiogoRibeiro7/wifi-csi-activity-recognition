# syntax=docker/dockerfile:1

# Base image
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Development image with additional tooling
FROM base AS dev
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY . .
CMD ["bash"]

# Production image
FROM base AS prod
COPY . .
CMD ["python", "-m", "wifi_activity_recognition"]

# Edge-optimized image targeting ARM devices
FROM python:3.11-slim AS edge
RUN apt-get update && apt-get install -y --no-install-recommends libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "wifi_activity_recognition"]

# Deployment Guide

This document outlines how to package and run models in various environments.

## Docker Images

Build the multi-stage Docker image:

```bash
docker build -f deployment/docker/Dockerfile -t wifi-ar:prod .
```

For local development:

```bash
docker-compose -f deployment/docker/docker-compose.yml up --build
```

## Kubernetes

Deploy the container to a cluster:

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

Use `kubectl logs` and `kubectl get pods` to monitor the service.

## Edge Deployment (Raspberry Pi)

Optimize and run a model on ARM devices:

```bash
python deployment/edge/raspberry_pi/optimize.py --model path/to/model.pt
```

Copy the resulting artifact to the device and execute your inference script
using the optimized model.

With these tools, the package can be deployed from local workstations to cloud
clusters and resource‑constrained edge devices.

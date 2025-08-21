# Deployment Guide

This document outlines how to package and run models in various environments
and provides resource guidelines and troubleshooting tips.

## Docker Images

Build the multi-stage Docker image:

```bash
docker build -f deployment/docker/Dockerfile -t wifi-ar:prod .
```

Run the container with volumes, ports, and environment variables:

```bash
docker run -it --rm \
  -e MODEL_PATH=/models/resnet.pt \
  -v $(pwd)/models:/models \
  -p 8000:8000 \
  wifi-ar:prod
```

Recommended resources: **2 vCPU**, **4 GB RAM**, optional **GPU** for inference.

For local development:

```bash
docker-compose -f deployment/docker/docker-compose.yml up --build
```

## Kubernetes

Deploy the container to a cluster:

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

Override environment variables and resource requests in the manifest:

```yaml
containers:
  - name: wifi-ar
    image: wifi-ar:prod
    env:
      - name: MODEL_PATH
        value: /models/resnet.pt
    resources:
      requests:
        cpu: "500m"
        memory: "1Gi"
      limits:
        cpu: "1"
        memory: "2Gi"
```

Use `kubectl logs` and `kubectl get pods` to monitor the service.

## Edge Deployment (Raspberry Pi)

Optimize and run a model on ARM devices:

```bash
python deployment/edge/raspberry_pi/optimize.py --model path/to/model.pt
```

Recommended resources: **4 ARM cores**, **1 GB RAM**. After optimization copy the
artifact to the device and execute your inference script using the optimized
model.

## Troubleshooting

- Container fails to start → check `docker logs` for missing environment
  variables.
- Model not found → ensure volume mounts map the model path correctly.
- Pod pending in Kubernetes → verify node resource availability and matching
  selectors.

With these tools, the package can be deployed from local workstations to cloud
clusters and resource‑constrained edge devices.

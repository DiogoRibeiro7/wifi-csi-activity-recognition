# Deployment Guide

Instructions for deploying models using Docker and edge tools.

1. Build Docker image with `docker build -f deployment/docker/Dockerfile .`.
2. Use `docker-compose` or Kubernetes manifests for scaling.
3. For Raspberry Pi, run the optimization script in `deployment/edge/raspberry_pi`.

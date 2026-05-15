#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "Building Docker images..."
docker build -t retail-support-backend:latest .
docker build -t retail-support-frontend:latest ./frontend

echo "Applying Kubernetes manifests..."
kubectl apply -f deployment/kubernetes/secrets.example.yaml
kubectl apply -f deployment/kubernetes/postgres.yaml
kubectl apply -f deployment/kubernetes/redis.yaml
kubectl apply -f deployment/kubernetes/backend-deployment.yaml
kubectl apply -f deployment/kubernetes/frontend-deployment.yaml
kubectl apply -f deployment/kubernetes/ingress.yaml

echo "Deployment complete. Check status with: kubectl get pods"

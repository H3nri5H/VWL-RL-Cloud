# Kubernetes Deployment Guide

## Prerequisites

1. **GKE Cluster** (via Terraform)
```bash
cd terraform
terraform init
terraform apply
```

2. **Configure kubectl**
```bash
gcloud container clusters get-credentials vwl-rl-cluster-dev --region europe-west1
```

3. **Build & Push Docker Images**
```bash
# Backend
gcloud builds submit --config backend/cloudbuild.yaml

# Frontend
gcloud builds submit --config frontend/cloudbuild.yaml
```

## Deploy to Kubernetes

### 1. Update PROJECT_ID
```bash
sed -i 's/PROJECT_ID/your-actual-project-id/g' k8s/*.yaml
```

### 2. Deploy Backend
```bash
kubectl apply -f k8s/backend-deployment.yaml
```

### 3. Deploy Frontend
```bash
kubectl apply -f k8s/frontend-deployment.yaml
```

### 4. Check Status
```bash
# Pods
kubectl get pods

# Services (get LoadBalancer IPs)
kubectl get services

# Logs
kubectl logs -f deployment/vwl-rl-backend
kubectl logs -f deployment/vwl-rl-frontend
```

## Access Application

```bash
# Get External IPs
kubectl get service vwl-rl-frontend-service
kubectl get service vwl-rl-backend-service

# Frontend: http://<FRONTEND-EXTERNAL-IP>
# Backend: http://<BACKEND-EXTERNAL-IP>
```

## Scale

```bash
# Scale backend to 5 replicas
kubectl scale deployment vwl-rl-backend --replicas=5

# Scale frontend to 10 replicas
kubectl scale deployment vwl-rl-frontend --replicas=10
```

## Update Deployment

```bash
# Rolling update with new image
kubectl set image deployment/vwl-rl-backend backend=gcr.io/PROJECT_ID/vwl-rl-backend:v2

# Check rollout status
kubectl rollout status deployment/vwl-rl-backend

# Rollback if needed
kubectl rollout undo deployment/vwl-rl-backend
```

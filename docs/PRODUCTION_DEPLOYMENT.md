# Production Deployment Guide

Complete guide for deploying Social Media Radar to production with high availability, security, and monitoring.

## Prerequisites

- Kubernetes cluster (1.25+)
- PostgreSQL 15+ with pgvector extension
- Redis 7+
- S3-compatible storage (AWS S3, MinIO, etc.)
- Domain name with SSL certificate
- OpenAI API key (or Anthropic API key)

## Architecture Overview

```
┌─────────────┐
│   Ingress   │ (HTTPS, Load Balancer)
└──────┬──────┘
       │
┌──────▼──────────────────────────────┐
│  API Pods (3-10 replicas)           │
│  - FastAPI application              │
│  - Health checks                    │
│  - Prometheus metrics               │
└──────┬──────────────────────────────┘
       │
┌──────▼──────────────────────────────┐
│  Celery Workers (5-20 replicas)     │
│  - Content ingestion                │
│  - Clustering & ranking             │
│  - Output generation                │
└──────┬──────────────────────────────┘
       │
┌──────▼──────────────────────────────┐
│  Infrastructure                     │
│  - PostgreSQL (pgvector)            │
│  - Redis (task queue)               │
│  - S3 (media storage)               │
└─────────────────────────────────────┘
```

## Step 1: Prepare Infrastructure

### 1.1 PostgreSQL Setup

```bash
# Install PostgreSQL with pgvector
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgresql bitnami/postgresql \
  --set auth.database=social_media_radar \
  --set auth.username=smr_user \
  --set auth.password=<SECURE_PASSWORD> \
  --set primary.persistence.size=100Gi

# Install pgvector extension
kubectl exec -it postgresql-0 -- psql -U smr_user -d social_media_radar
CREATE EXTENSION IF NOT EXISTS vector;
```

### 1.2 Redis Setup

```bash
# Install Redis
helm install redis bitnami/redis \
  --set auth.password=<SECURE_PASSWORD> \
  --set master.persistence.size=20Gi
```

### 1.3 S3 Storage

For AWS S3:
```bash
# Create S3 bucket
aws s3 mb s3://social-media-radar-storage
aws s3api put-bucket-versioning \
  --bucket social-media-radar-storage \
  --versioning-configuration Status=Enabled
```

For MinIO (self-hosted):
```bash
helm install minio bitnami/minio \
  --set auth.rootUser=admin \
  --set auth.rootPassword=<SECURE_PASSWORD> \
  --set persistence.size=500Gi
```

## Step 2: Configure Secrets

Create Kubernetes secrets:

```bash
# Create namespace
kubectl create namespace production

# Create secrets
kubectl create secret generic social-media-radar-secrets \
  --namespace=production \
  --from-literal=database-url="postgresql+asyncpg://smr_user:<PASSWORD>@postgresql:5432/social_media_radar" \
  --from-literal=redis-url="redis://:<PASSWORD>@redis-master:6379/0" \
  --from-literal=encryption-key="<GENERATE_32_CHAR_KEY>" \
  --from-literal=openai-api-key="<YOUR_OPENAI_KEY>" \
  --from-literal=anthropic-api-key="<YOUR_ANTHROPIC_KEY>" \
  --from-literal=s3-access-key="<S3_ACCESS_KEY>" \
  --from-literal=s3-secret-key="<S3_SECRET_KEY>" \
  --from-literal=sentry-dsn="<SENTRY_DSN>"

# Create config map
kubectl create configmap social-media-radar-config \
  --namespace=production \
  --from-literal=s3-endpoint="https://s3.amazonaws.com" \
  --from-literal=s3-bucket="social-media-radar-storage"
```

## Step 3: Build and Push Docker Image

```bash
# Build image
docker build -t your-registry/social-media-radar:latest .

# Push to registry
docker push your-registry/social-media-radar:latest
```

## Step 4: Deploy Application

```bash
# Apply Kubernetes manifests
kubectl apply -f infra/k8s/production/deployment.yaml
kubectl apply -f infra/k8s/production/celery-worker.yaml
kubectl apply -f infra/k8s/production/ingress.yaml

# Verify deployment
kubectl get pods -n production
kubectl logs -f deployment/social-media-radar-api -n production
```

## Step 5: Run Database Migrations

```bash
# Run migrations
kubectl exec -it deployment/social-media-radar-api -n production -- \
  alembic upgrade head
```

## Step 6: Set Up Monitoring

### 6.1 Prometheus

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace

# Apply custom rules
kubectl apply -f infra/monitoring/prometheus-rules.yaml
```

### 6.2 Grafana Dashboards

```bash
# Access Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Default credentials: admin / prom-operator
# Import dashboards from infra/monitoring/grafana-dashboards/
```

### 6.3 Sentry (Error Tracking)

Configure Sentry DSN in secrets (already done in Step 2).

## Step 7: Verify Deployment

```bash
# Check health
curl https://api.yourdomain.com/health

# Check readiness
curl https://api.yourdomain.com/health/ready

# Check metrics
curl https://api.yourdomain.com/metrics
```

## Scaling

### Horizontal Scaling

HorizontalPodAutoscaler is already configured:
- API: 3-10 replicas (based on CPU/memory)
- Celery Workers: 5-20 replicas (based on queue length)

### Vertical Scaling

Adjust resource requests/limits in deployment manifests.

## Security Checklist

- [x] Run as non-root user
- [x] Read-only root filesystem
- [x] Network policies configured
- [x] Secrets encrypted at rest
- [x] TLS/HTTPS enabled
- [x] Rate limiting configured
- [x] Input validation and sanitization
- [x] SQL injection prevention
- [x] XSS protection

## Monitoring and Alerts

Access monitoring dashboards:
- Prometheus: `kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090`
- Grafana: `kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80`

## Backup and Disaster Recovery

### Database Backups

Automated daily backups configured via CronJob.

### S3 Versioning

Enabled for all media files.

## Troubleshooting

See [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues and solutions.


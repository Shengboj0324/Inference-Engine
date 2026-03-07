# Social Media Radar - Production Deployment Guide

**Version**: 1.0.0  
**Date**: 2026-03-06  
**Status**: Production Ready ✅

---

## Prerequisites

### System Requirements
- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- 8GB+ RAM
- 50GB+ disk space

### Required Environment Variables

Create `.env` file with the following:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/social_media_radar

# Security
SECRET_KEY=<generate-with-secrets.token_urlsafe(32)>
ENCRYPTION_KEY=<generate-with-secrets.token_urlsafe(32)>

# OAuth Credentials
TWITTER_CLIENT_ID=your_twitter_client_id
TWITTER_CLIENT_SECRET=your_twitter_client_secret
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
LINKEDIN_CLIENT_ID=your_linkedin_client_id
LINKEDIN_CLIENT_SECRET=your_linkedin_client_secret

# API Configuration
API_BASE_URL=https://your-domain.com
CORS_ORIGINS=https://your-frontend.com

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM Configuration
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Monitoring
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO
```

---

## Installation

### 1. Clone and Setup

```bash
git clone <repository-url>
cd Social-Media-Radar

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Create database
createdb social_media_radar

# Run migrations
alembic upgrade head

# Verify
psql social_media_radar -c "\dt"
```

### 3. Redis Setup

```bash
# Start Redis
redis-server

# Verify
redis-cli ping  # Should return PONG
```

---

## Deployment

### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t social-media-radar:latest .

# Run with docker-compose
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Option 2: Manual Deployment

```bash
# Start application
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Or use gunicorn for production
gunicorn app.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Option 3: Systemd Service

Create `/etc/systemd/system/social-media-radar.service`:

```ini
[Unit]
Description=Social Media Radar API
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/social-media-radar
Environment="PATH=/opt/social-media-radar/venv/bin"
ExecStart=/opt/social-media-radar/venv/bin/gunicorn app.api.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable social-media-radar
sudo systemctl start social-media-radar
sudo systemctl status social-media-radar
```

---

## Training Pipeline

### Quick Test Training

```bash
# Validate configuration
python train.py --config configs/training/quick-test.yaml --validate-only

# Run quick test (5 epochs)
python train.py --config configs/training/quick-test.yaml
```

### Production Training

```bash
# Full training run
python train.py --config configs/training/default.yaml

# With custom parameters
python train.py --config configs/training/default.yaml \
    --override training.num_epochs=10 \
    --override training.learning_rate=0.0001
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir checkpoints/

# View logs
tail -f logs/training.log
```

---

## Health Checks

### API Health
```bash
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

### Database Connection
```bash
curl http://localhost:8000/health/db
# Expected: {"status": "connected"}
```

### Redis Connection
```bash
curl http://localhost:8000/health/redis
# Expected: {"status": "connected"}
```

---

## Monitoring

### Prometheus Metrics
- Endpoint: `http://localhost:8000/metrics`
- Grafana Dashboard: Import `deployment/grafana-dashboard.json`

### Key Metrics to Monitor
- Request latency (p50, p95, p99)
- Error rate
- Database connection pool
- Redis cache hit rate
- Memory usage
- Circuit breaker states

---

## Security Checklist

- [ ] All environment variables set
- [ ] SECRET_KEY and ENCRYPTION_KEY are strong random values
- [ ] OAuth credentials configured
- [ ] CORS origins restricted to your domains
- [ ] HTTPS enabled (use nginx/traefik as reverse proxy)
- [ ] Database credentials secured
- [ ] Redis password set (if exposed)
- [ ] Firewall rules configured
- [ ] Rate limiting enabled
- [ ] Logging configured (no sensitive data logged)

---

## Troubleshooting

### Application won't start
```bash
# Check logs
journalctl -u social-media-radar -n 100

# Verify environment
python -c "from app.core.config import settings; print(settings)"
```

### Database connection errors
```bash
# Test connection
psql $DATABASE_URL -c "SELECT 1"

# Check migrations
alembic current
alembic history
```

### High memory usage
```bash
# Check for memory leaks
# Memory cleanup runs automatically every 10 minutes
# Monitor: /metrics endpoint for memory_usage_bytes
```

---

## Support

For issues or questions:
1. Check logs: `logs/app.log`
2. Review metrics: `http://localhost:8000/metrics`
3. Run audit: `./scripts/production-audit.sh`

---

**Deployment Status**: ✅ Ready for Production


# Deployment Guide

This guide covers deploying Social Media Radar in various environments.

## Quick Start (Development)

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Poetry (for local development)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/social-media-radar.git
   cd social-media-radar
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

3. **Start services with Docker Compose**
   ```bash
   docker-compose up -d
   ```

4. **Run database migrations**
   ```bash
   docker-compose exec api alembic upgrade head
   ```

5. **Access the API**
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - MinIO Console: http://localhost:9001

## Local Development (Without Docker)

### Prerequisites
- PostgreSQL 15+ with pgvector extension
- Redis 7+
- Python 3.11+
- Poetry

### Steps

1. **Install dependencies**
   ```bash
   poetry install
   ```

2. **Start PostgreSQL and Redis**
   ```bash
   # Using Homebrew on macOS
   brew services start postgresql@15
   brew services start redis
   
   # Or using Docker
   docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=radar_password ankane/pgvector
   docker run -d -p 6379:6379 redis:7-alpine
   ```

3. **Create database**
   ```bash
   createdb social_radar
   psql social_radar -c "CREATE EXTENSION vector;"
   ```

4. **Run migrations**
   ```bash
   poetry run alembic upgrade head
   ```

5. **Start the API server**
   ```bash
   poetry run uvicorn app.api.main:app --reload
   ```

6. **Start Celery worker (in another terminal)**
   ```bash
   poetry run celery -A app.ingestion.celery_app worker --loglevel=info
   ```

7. **Start Celery beat (in another terminal)**
   ```bash
   poetry run celery -A app.ingestion.celery_app beat --loglevel=info
   ```

## Production Deployment

### Environment Variables

Critical environment variables for production:

```bash
# Security - MUST CHANGE
SECRET_KEY=<generate-strong-random-key>
ENCRYPTION_KEY=<generate-32-byte-base64-key>

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/dbname
DATABASE_SYNC_URL=postgresql://user:pass@host:5432/dbname

# Redis
REDIS_URL=redis://host:6379/0

# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Object Storage (S3)
S3_ENDPOINT=https://s3.amazonaws.com
S3_ACCESS_KEY=...
S3_SECRET_KEY=...
S3_BUCKET=radar-content

# API Settings
API_WORKERS=4
CORS_ORIGINS=["https://yourdomain.com"]

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Docker Compose (Production)

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  api:
    image: your-registry/social-media-radar:latest
    restart: always
    env_file: .env.production
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  celery-worker:
    image: your-registry/social-media-radar:latest
    restart: always
    command: celery -A app.ingestion.celery_app worker --loglevel=info
    env_file: .env.production
    depends_on:
      - postgres
      - redis

  celery-beat:
    image: your-registry/social-media-radar:latest
    restart: always
    command: celery -A app.ingestion.celery_app beat --loglevel=info
    env_file: .env.production
    depends_on:
      - postgres
      - redis
```

### Kubernetes Deployment

See `infra/k8s/` for Kubernetes manifests.

Key components:
- `deployment.yaml` - API and worker deployments
- `service.yaml` - Service definitions
- `ingress.yaml` - Ingress configuration
- `configmap.yaml` - Configuration
- `secrets.yaml` - Sensitive credentials

Deploy:
```bash
kubectl apply -f infra/k8s/
```

## Database Migrations

### Creating a new migration
```bash
alembic revision --autogenerate -m "Description of changes"
```

### Applying migrations
```bash
alembic upgrade head
```

### Rolling back
```bash
alembic downgrade -1
```

## Monitoring

### Health Checks
- API: `GET /health`
- Database: Check PostgreSQL connection
- Redis: Check Redis connection
- Celery: Monitor task queue length

### Logging
- Structured JSON logging in production
- Centralized logging with ELK/Loki recommended
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Metrics
- API request latency
- Task processing time
- Content fetch success rate
- LLM API usage and costs
- Database query performance

## Scaling

### Horizontal Scaling
- Run multiple API instances behind load balancer
- Run multiple Celery workers
- Use Redis Sentinel for HA
- Use PostgreSQL replication

### Vertical Scaling
- Increase API worker count
- Increase database resources
- Increase Redis memory
- Use faster storage for embeddings

## Security Checklist

- [ ] Change default passwords
- [ ] Use strong SECRET_KEY and ENCRYPTION_KEY
- [ ] Enable HTTPS/TLS
- [ ] Configure CORS properly
- [ ] Set up firewall rules
- [ ] Enable database encryption at rest
- [ ] Rotate API keys regularly
- [ ] Set up backup and disaster recovery
- [ ] Enable audit logging
- [ ] Implement rate limiting
- [ ] Use secrets management (Vault, AWS Secrets Manager)


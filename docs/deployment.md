# Deployment Reference

## System Requirements

| Component | Minimum | Notes |
|---|---|---|
| Python | 3.11 | `asyncpg>=0.28.0`, `numpy>=1.24` required for ARM64 wheels |
| PostgreSQL | 15 + pgvector | `CREATE EXTENSION IF NOT EXISTS vector;` before first migration |
| Redis | 7 | Celery broker and result backend |
| RAM | 8 GB | 16 GB required when running Ollama 7B+ locally |
| Docker Engine | 24.0 + Compose v2 | For the Docker path only |

---

## Option A — Docker Compose

Starts `postgres`, `redis`, `minio`, `api`, `celery-worker`, and `celery-beat`. Migrations run automatically via the `db-init` service.

```bash
git clone https://github.com/yourusername/social-media-radar.git
cd social-media-radar
cp .env.example .env

# Generate secrets
python3 -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(32))"
python3 -c "import secrets; print('ENCRYPTION_KEY=' + secrets.token_urlsafe(32))"
# Paste both into .env, then add OPENAI_API_KEY and/or ANTHROPIC_API_KEY

docker compose up

# After the stack is healthy, seed the calibrator
docker compose exec api python training/calibrate.py --epochs 5

# Verify
curl -s http://localhost:8000/health
# → {"status": "healthy", "database": "ok", "redis": "ok"}
```

**Service map:**

| Service | Address |
|---|---|
| FastAPI | `http://localhost:8000` |
| OpenAPI UI | `http://localhost:8000/docs` |
| MinIO console | `http://localhost:9001` (admin/minioadmin) |
| PostgreSQL | `localhost:5432` |
| Redis | `localhost:6379` |

---

## Option B — Bare Metal (macOS)

```bash
# System dependencies
brew install postgresql@15 pgvector redis minio/stable/minio python@3.11
echo 'export PATH="/opt/homebrew/opt/postgresql@15/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
brew services start postgresql@15 redis
xcode-select --install

# Project setup
git clone https://github.com/yourusername/social-media-radar.git && cd social-media-radar
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt

# Configure .env
cp .env.example .env
# DATABASE_URL=postgresql+asyncpg://<user>@localhost:5432/social_radar
# DATABASE_SYNC_URL=postgresql://<user>@localhost:5432/social_radar
# REDIS_URL=redis://localhost:6379/0
# SECRET_KEY=<token_urlsafe(32)>   ENCRYPTION_KEY=<token_urlsafe(32)>
# OPENAI_API_KEY=sk-...

# Database
createdb social_radar
python scripts/init_db.py      # enables pgvector extension
alembic upgrade head

# Calibration
python training/calibrate.py --epochs 5

# Start (three terminals)
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
celery -A app.ingestion.celery_app worker --loglevel=info
celery -A app.ingestion.celery_app beat --loglevel=info
```

---

## Option C — Bare Metal (Ubuntu 22.04 / WSL2)

```bash
sudo apt-get update && sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    postgresql-15 postgresql-15-pgvector redis-server libpq-dev gcc
sudo systemctl enable --now postgresql redis-server
sudo -u postgres psql -c "CREATE USER radar WITH PASSWORD 'radar_password';"
sudo -u postgres psql -c "CREATE DATABASE social_radar OWNER radar;"
```

Follow Steps 2–6 from Option B, setting `DATABASE_URL` to `postgresql+asyncpg://radar:radar_password@localhost:5432/social_radar`.

---

## Fully Offline (Ollama)

No LLM API keys required. Set in `.env`:

```bash
LOCAL_LLM_URL=http://localhost:11434
LOCAL_LLM_MODEL=llama3.1:8b
```

`LLMRouter` will prefer `LOCAL_LLM_URL` when set. All 18 signal types can route locally; the two-tier separation (frontier vs. fine-tuned) is preserved — risk types go to the configured primary model, which in this case is also the local model.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `SECRET_KEY` | ✅ | JWT signing key — generate with `secrets.token_urlsafe(32)` |
| `ENCRYPTION_KEY` | ✅ | Credential vault key — generate with `secrets.token_urlsafe(32)` |
| `DATABASE_URL` | ✅ | `postgresql+asyncpg://user:pass@host:5432/dbname` |
| `DATABASE_SYNC_URL` | ✅ | Same but `postgresql://` (sync driver for Alembic) |
| `REDIS_URL` | ✅ | `redis://host:6379/0` |
| `OPENAI_API_KEY` | ✴️ | Required unless `LOCAL_LLM_URL` is set |
| `ANTHROPIC_API_KEY` | ✴️ | Required if Anthropic routing is active |
| `LOCAL_LLM_URL` | — | Ollama endpoint; disables cloud LLM requirement |
| `LOCAL_LLM_MODEL` | — | e.g., `llama3.1:8b` |
| `S3_ENDPOINT` | — | MinIO or S3 URL for media storage |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | — | Object storage credentials |
| `CORS_ORIGINS` | — | JSON array of allowed origins, e.g., `["https://app.example.com"]` |
| `API_WORKERS` | — | Uvicorn worker count (default 4) |
| `LOG_LEVEL` | — | `DEBUG` / `INFO` / `WARNING` (default `INFO`) |

---

## Database Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Create a new migration after model changes
alembic revision --autogenerate -m "description"

# Roll back one step
alembic downgrade -1
```

The `pgvector` extension must exist before the first migration:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## Kubernetes

Manifests are in `deployment/kubernetes/`. The HPA (`hpa.yaml`) targets 70% CPU utilisation with `minReplicas=2` and `maxReplicas=10`.

```bash
kubectl apply -f deployment/kubernetes/llm-secrets.yaml
kubectl apply -f deployment/kubernetes/llm-deployment.yaml
kubectl apply -f deployment/kubernetes/hpa.yaml
```

---

## Monitoring

Prometheus scrapes `http://api:8000/metrics`. Grafana dashboard JSON is at `deployment/grafana/dashboards/llm-overview.json`. Import via:

```bash
curl -X POST http://localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @deployment/grafana/dashboards/llm-overview.json
```

Key metrics: `llm_requests_total`, `llm_request_duration_seconds`, `llm_cost_total`, `llm_circuit_breaker_state`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pg_isready: command not found` | Add PostgreSQL 15 bin to `PATH` |
| `ImportError: No module named 'asyncpg'` | `source .venv/bin/activate` |
| `FATAL: role "radar" does not exist` | `createuser radar` |
| `redis.exceptions.ConnectionError` | `brew services start redis` or `sudo systemctl start redis` |
| `InvalidToken` on credential decrypt | `python scripts/migrate_credentials.py` |
| `db-init exited with code 1` (Docker) | Increase postgres healthcheck `retries` in `docker-compose.yml` |
| `pgvector` type error on insert | `CREATE EXTENSION IF NOT EXISTS vector;` not run |
| Abstention rate > 20% on live traffic | Rerun `training/calibrate.py --epochs 5` or lower `confidence_required` |


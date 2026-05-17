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


---

## Desktop deployment (Phase 1\u20136 sidecar)

The desktop topology is a different shape from the Docker / k8s paths above.  No
Postgres, no Redis, no MinIO — a Tauri 2.0 shell launches a single FastAPI
sidecar bound to ``127.0.0.1`` and authenticated with a per-launch loopback
token.  All persistence is local SQLite files inside the OS user-data dir.

### Prerequisites

| Component | Version | Notes |
|---|---|---|
| Python | 3.11 | Same as server mode |
| Rust toolchain | 1.74+ | Required to build the Tauri shell |
| Node.js | 20 LTS | UI build (Vite + React) |
| `sqlite-vec` | 0.1.9+ | ``pip install sqlite-vec``; ANN search is the hot path |

### Bring-up

```bash
# 1. Install Python deps (incl. sqlite-vec for ANN)
pip install -r requirements.txt

# 2. Run the sidecar in desktop mode
DEPLOYMENT_MODE=desktop uvicorn app.api.main:app --host 127.0.0.1 --port 8765

# 3. (separate terminal) launch the Tauri shell
cd ui && npm install && npm run tauri dev
```

The shell calls ``GET /api/v1/desktop/manifest`` on launch to discover the
sidecar's contract version and ``GET /api/v1/desktop/ready`` to wait for
readiness.  Every other route is gated by ``require_desktop_mode`` and returns
404 in server mode.

### Operating the local RAG layer

| Action | Endpoint | Notes |
|---|---|---|
| Check status (indexed / stale counts, current model) | ``GET /api/v1/rag/status`` | ``stale_count`` = "would be re-embedded by next reindex" |
| Synchronous backfill | ``POST /api/v1/rag/reindex`` | Bounded by ``max_items``; UI polls until ``remaining=0`` or ``embedded==0 && skipped==scanned`` |
| Background reindex (full corpus) | ``POST /api/v1/rag/reindex/jobs`` \u2192 ``GET /api/v1/rag/reindex/jobs/{id}`` | Returns ``202``; poll status; ``DELETE`` to cancel |
| Search | ``POST /api/v1/rag/search`` | Snippets are PII-scrubbed; include ``base_score`` + ``personalization_bonus`` |
| Record explicit feedback | ``POST /api/v1/rag/feedback`` | ``score`` in ``[-1.0, 1.0]``; running total clamped at \u00b110 |
| Inspect personalization signals | ``GET /api/v1/rag/signals`` | For the "Why am I seeing this?" UI drawer |
| Reset personalization | ``DELETE /api/v1/rag/signals`` | Wipes citation + feedback counters; semantic ranking unaffected |

Citations are recorded automatically by the chat path whenever a
RAG-augmented reply is produced; explicit feedback is the only thing the UI
must POST.

### Mock user session

A scripted end-to-end walkthrough — fetch \u2192 embed \u2192 search \u2192 feedback \u2192
personalized re-rank \u2192 background reindex \u2192 model-swap stale detection — is
in ``scripts/mock_user_session.py``:

```bash
DEPLOYMENT_MODE=desktop python scripts/mock_user_session.py
```

Use it before any release to confirm the Phase 5/6 contracts still hold
end-to-end on the operator's machine.

---

## Verified-vs-attestation matrix (enterprise readiness)

The engineering substrate below is verified by the in-repo test suite.  The
"requires attestation" column lists the external evidence an enterprise
rollout still needs — these are organisational artefacts, not code, and no
test suite can produce them.

| Property | Verified in this repo | Requires external attestation |
|---|---|---|
| Zero network egress without ``NETWORK_FETCH`` permission | \u2713 ``app/local/permissions.py`` + ``tests/desktop/test_permissions*.py`` | Network-level pen-test on the target build |
| BYOK keys never on disk plaintext | \u2713 ``app/local/key_vault.py`` (OS keychain) + ``tests/desktop/test_key_vault.py`` | OS-keychain configuration audit per supported OS |
| PII scrubbed before any LLM / embedder call | \u2713 ``DataResidencyGuard`` + ``tests/desktop/test_phase2_safety.py``, ``tests/desktop/test_rag_phase5.py::TestRetrieverSoftFail`` | Locale-specific regex coverage review (DPO sign-off) |
| Sidecar reachable only via loopback + per-launch token | \u2713 ``require_desktop_mode`` + loopback-token middleware + contract surface test | Tauri shell signing + OS notarization (Apple/Microsoft) |
| Embedding model swap surfaces all prior rows as stale | \u2713 ``tests/desktop/test_rag_phase5.py::TestStaleEmbeddingSelector`` | None |
| RAG retrieval soft-fails on missing key / denied permission | \u2713 ``tests/desktop/test_rag_phase5.py::TestRetrieverSoftFail`` | None |
| Personalization bonus is bounded; cannot override semantic floor | \u2713 ``tests/desktop/test_rag_phase5.py::TestPersonalizationRerank`` | UX review of "Why am I seeing this?" disclosure |
| Background reindex is cancellable and progress-observable | \u2713 ``tests/desktop/test_rag_phase5.py::TestRAGRoutes::test_background_job_lifecycle`` | Soak test against operator-scale corpus (>=100k items) |
| Public API surface is contract-locked | \u2713 ``tests/contract/test_public_api_surface.py`` (78 routes pinned, ``CONTRACT_VERSION`` stamped) | Versioning + deprecation policy in customer contract |
| Multi-tenant isolation | \u2717 Not applicable \u2014 desktop is single-user-per-machine by construction | If you re-host the sidecar multi-tenant, the verified guarantees above DO NOT carry over |

### What this matrix is NOT

It is not a SOC 2 / ISO 27001 / GDPR attestation, not a load-test report,
and not a vendor sign-off.  Those are organisational deliverables.  This
matrix only asserts what the code in this repo currently enforces \u2014 every
line under "Verified" maps to a test that fails CI when the property
regresses.

### Pre-release checklist

Before promoting a desktop build to enterprise pilot:

1. ``python -m pytest --ignore=tests/llm/test_load.py`` \u2014 must be all-green.
2. ``DEPLOYMENT_MODE=desktop python scripts/mock_user_session.py`` \u2014 must
   exit ``0`` with the printed summary matching the documented contract.
3. Confirm ``API_CONTRACT_VERSION`` in ``app/api/main.py`` matches the version
   the shipped UI was built against.
4. Run the platform-specific code-signing pipeline for the Tauri bundle and
   the sidecar binary; record the signing certificate IDs in your release
   notes.
5. File the network-egress pen-test request with your security team; do not
   ship without a written confirmation that ``NETWORK_FETCH=deny`` actually
   prevents outbound traffic on the packaged build.


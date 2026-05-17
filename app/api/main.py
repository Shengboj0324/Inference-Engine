"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import time

from app.api.routes import (
    auth,
    chat,
    desktop,
    digest,
    ingest,
    keys,
    llm,
    multimodal,
    permissions,
    rag,
    search,
    signals,
    sources,
)
from app.core.config import settings
from app.core.errors import BaseAppException
from app.core.health import HealthChecker
from app.core.monitoring import MetricsCollector
from app.monitoring.health import HealthMonitor

logger = logging.getLogger(__name__)


#: Public API contract version.  Bumped whenever a route is added,
#: removed, or changes method/path.  The contract test
#: (``tests/contract/test_public_api_surface.py``) imports this constant
#: and the desktop shell reads it from ``/api/v1/desktop/manifest`` so
#: the UI can refuse to load against an incompatible sidecar.
API_CONTRACT_VERSION = "phase5.0"


async def _probe_redis() -> None:
    """Verify Redis connectivity on startup.

    Attempts a ``PING`` to the Redis instance configured in
    ``settings.redis_url``.  If the ping fails, a ``CRITICAL`` log entry is
    written and a :class:`RuntimeError` is raised so that container
    orchestrators (Kubernetes, ECS, Docker Compose ``healthcheck``) see the
    process exit with a non-zero status and withhold traffic until a
    subsequent restart succeeds.

    Raises:
        RuntimeError: If Redis is unreachable at startup.
    """
    import redis.asyncio as aioredis

    client: aioredis.Redis = aioredis.from_url(
        settings.redis_url,
        socket_connect_timeout=5,
        socket_timeout=5,
    )
    try:
        pong = await client.ping()
        if not pong:
            raise ConnectionError("Redis PING returned a falsy response.")
        logger.info("Redis connectivity confirmed (PING → PONG) at %s", settings.redis_url)
    except Exception as exc:
        logger.critical(
            "STARTUP FAILURE — cannot reach Redis at %s: %s. "
            "Ensure Redis is running and REDIS_URL is correctly configured.",
            settings.redis_url,
            exc,
        )
        raise RuntimeError(
            f"Redis unavailable at startup ({settings.redis_url}): {exc}"
        ) from exc
    finally:
        await client.aclose()


def _validate_capabilities() -> None:
    """Thin shim — delegates to ``app.core.startup_validation.validate_capabilities``.

    Keeping the implementation in ``startup_validation`` makes it importable
    without loading all FastAPI route modules (which have module-scope
    side-effects such as LLM client instantiation).
    """
    from app.core.startup_validation import validate_capabilities
    validate_capabilities()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info(
        "Starting Social Media Radar API (environment=%s strict=%s)...",
        settings.environment,
        settings.is_strict,
    )

    # 1. Capability validation — fails fast in strict/production mode if any
    #    enabled capability cannot resolve a real backend.
    _validate_capabilities()

    # 2. Redis connectivity probe — raises RuntimeError and aborts startup if
    #    Redis is unreachable, preventing orchestrators from routing traffic to
    #    an API instance that cannot honour its backlist contract.  Skipped in
    #    desktop deployment mode where the sidecar uses an in-process pub/sub
    #    backend and no external Redis service exists by design.
    if not settings.is_desktop:
        await _probe_redis()
    else:
        logger.info("Desktop deployment mode — skipping external Redis probe.")

    # 3. Flip the desktop readiness flag once all startup checks have
    #    passed.  ``/api/v1/desktop/ready`` returns 503 until this point
    #    so the Tauri shell can poll cleanly during boot.
    desktop.mark_ready()

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    desktop.mark_not_ready()
    logger.info("Shutting down Social Media Radar API...")


app = FastAPI(
    title="Social Media Radar API",
    description="Multi-channel intelligence aggregation system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(signals.router, prefix="/api/v1", tags=["Signals"])  # Primary product interface
app.include_router(sources.router, prefix="/api/v1/sources", tags=["Sources"])
app.include_router(digest.router, prefix="/api/v1/digest", tags=["Digest"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(llm.router, prefix="/api/llm", tags=["LLM"])
app.include_router(keys.router, prefix="/api/v1/keys", tags=["Keys"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(permissions.router, prefix="/api/v1/permissions", tags=["Permissions"])
app.include_router(ingest.router, prefix="/api/v1/ingest", tags=["Ingest"])
app.include_router(multimodal.router, prefix="/api/v1/multimodal", tags=["Multimodal"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(desktop.router, prefix="/api/v1/desktop", tags=["Desktop"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Social Media Radar API",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health")
async def health():
    """Basic health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/ready")
async def readiness():
    """Readiness probe endpoint."""
    health_checker = HealthChecker()
    system_health = await health_checker.check_all()

    if system_health.status.value == "unhealthy":
        return JSONResponse(
            status_code=503,
            content=system_health.model_dump(),
        )

    return system_health.model_dump()


@app.get("/health/live")
async def liveness():
    """Liveness probe endpoint."""
    return {"status": "alive"}


@app.get("/api/v1/health", tags=["Health"])
async def intelligence_health():
    """Full intelligence-layer health check.

    Pings every LLM provider circuit breaker, verifies HNSW index
    initialisation, and confirms database connectivity.

    Returns:
        HTTP 200 with a :class:`~app.monitoring.health.HealthReport` JSON body
        when all critical components are healthy.
        HTTP 503 when any critical component is unhealthy.
    """
    monitor = HealthMonitor()
    report = await monitor.check()
    if not report.healthy:
        return JSONResponse(
            status_code=503,
            content=report.model_dump(),
        )
    return report.model_dump()


# Exception handlers
@app.exception_handler(BaseAppException)
async def app_exception_handler(request: Request, exc: BaseAppException):
    """Handle application exceptions."""
    from app.core.monitoring import MetricsCollector

    MetricsCollector.record_error(exc.error_code.value, exc.severity.value)

    return JSONResponse(
        status_code=500 if exc.severity.value == "critical" else 400,
        content=exc.to_dict(),
    )


# ---------------------------------------------------------------------------
# Loopback token middleware (Tauri shell handshake)
# ---------------------------------------------------------------------------
#
# Opt-in: only enforced when ``SMR_SIDECAR_TOKEN`` is exported by the
# launcher *and* deployment_mode='desktop'.  Tests never set the env
# var, so they bypass enforcement completely.  The shell reads the
# token from ``sidecar.json`` (mode 0600 in the user-data dir) and
# sends it as ``X-Sidecar-Token`` on every API call.

import os as _os  # local alias — avoids shadowing the module-level import

#: Paths exempt from the token requirement.  The shell must be able to
#: hit these before it has a token (bootstrap) and operator probes
#: should keep working without the env var.
_TOKEN_EXEMPT_PATHS = frozenset({
    "/",
    "/health",
    "/health/ready",
    "/health/live",
    "/api/v1/health",
    "/api/v1/desktop/manifest",
    "/api/v1/desktop/ready",
})


@app.middleware("http")
async def enforce_loopback_token(request: Request, call_next):
    """Require ``X-Sidecar-Token`` when running under the launcher.

    Returns 401 JSON when the env-var token is set, the request is not
    on an exempt path, and the header is missing or wrong.  Header
    comparison is constant-time to neutralise timing oracles.
    """
    expected = _os.environ.get("SMR_SIDECAR_TOKEN", "")
    if expected and settings.is_desktop and request.url.path not in _TOKEN_EXEMPT_PATHS:
        provided = request.headers.get("x-sidecar-token", "")
        import hmac
        if not hmac.compare_digest(expected, provided):
            return JSONResponse(
                status_code=401,
                content={"detail": "missing or invalid X-Sidecar-Token header"},
            )
    return await call_next(request)


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    MetricsCollector.record_http_request(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
        duration=duration,
    )

    return response


# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


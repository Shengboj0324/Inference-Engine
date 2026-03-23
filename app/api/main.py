"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
import time

from app.api.routes import auth, digest, llm, search, signals, sources
from app.core.config import settings
from app.core.errors import BaseAppException
from app.core.health import HealthChecker
from app.core.monitoring import MetricsCollector
from app.monitoring.health import HealthMonitor

logger = logging.getLogger(__name__)


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # ── Startup ────────────────────────────────────────────────────────────
    logger.info("Starting Social Media Radar API...")

    # Redis connectivity probe — raises RuntimeError and aborts startup if
    # Redis is unreachable, preventing orchestrators from routing traffic to
    # an API instance that cannot honour its blacklist contract.
    await _probe_redis()

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
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


"""Health check and graceful degradation system."""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentType(str, Enum):
    """Types of system components."""

    DATABASE = "database"
    REDIS = "redis"
    S3 = "s3"
    LLM_PROVIDER = "llm_provider"
    CONNECTOR = "connector"
    SCRAPER = "scraper"


class ComponentHealth(BaseModel):
    """Health status of a component."""

    component: str
    component_type: ComponentType
    status: HealthStatus
    message: Optional[str] = None
    last_check: datetime = Field(default_factory=datetime.utcnow)
    response_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SystemHealth(BaseModel):
    """Overall system health."""

    status: HealthStatus
    components: List[ComponentHealth]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    degraded_features: List[str] = Field(default_factory=list)
    version: str = "1.0.0"


class HealthChecker:
    """Perform health checks on system components."""

    def __init__(self):
        """Initialize health checker."""
        self._component_status: Dict[str, ComponentHealth] = {}
        self._check_interval = 60  # seconds
        self._running = False

    async def check_database(self) -> ComponentHealth:
        """Check database health."""
        from app.core.db import async_engine

        start_time = asyncio.get_event_loop().time()

        try:
            async with async_engine.connect() as conn:
                await conn.execute("SELECT 1")

            response_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return ComponentHealth(
                component="postgresql",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return ComponentHealth(
                component="postgresql",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    async def check_redis(self) -> ComponentHealth:
        """Check Redis health."""
        import redis.asyncio as redis
        from app.core.config import settings

        start_time = asyncio.get_event_loop().time()

        try:
            client = redis.from_url(settings.redis_url)
            await client.ping()
            await client.close()

            response_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return ComponentHealth(
                component="redis",
                component_type=ComponentType.REDIS,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return ComponentHealth(
                component="redis",
                component_type=ComponentType.REDIS,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    async def check_s3(self) -> ComponentHealth:
        """Check S3/MinIO health."""
        import boto3
        from botocore.exceptions import ClientError
        from app.core.config import settings

        start_time = asyncio.get_event_loop().time()

        try:
            s3_client = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint,
                aws_access_key_id=settings.s3_access_key,
                aws_secret_access_key=settings.s3_secret_key,
            )

            # Try to list buckets
            s3_client.list_buckets()

            response_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return ComponentHealth(
                component="s3",
                component_type=ComponentType.S3,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            logger.error(f"S3 health check failed: {e}")
            return ComponentHealth(
                component="s3",
                component_type=ComponentType.S3,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    async def check_llm_provider(self) -> ComponentHealth:
        """Check LLM provider health."""
        from app.core.config import settings

        start_time = asyncio.get_event_loop().time()

        try:
            # Simple check - just verify API key is set
            if settings.openai_api_key or settings.anthropic_api_key:
                status = HealthStatus.HEALTHY
                message = "API key configured"
            else:
                status = HealthStatus.DEGRADED
                message = "No API key configured"

            response_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            return ComponentHealth(
                component="llm_provider",
                component_type=ComponentType.LLM_PROVIDER,
                status=status,
                message=message,
                response_time_ms=response_time_ms,
            )

        except Exception as e:
            logger.error(f"LLM provider health check failed: {e}")
            return ComponentHealth(
                component="llm_provider",
                component_type=ComponentType.LLM_PROVIDER,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    async def check_all(self) -> SystemHealth:
        """Check health of all components."""
        components = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_s3(),
            self.check_llm_provider(),
            return_exceptions=True,
        )

        # Filter out exceptions
        valid_components = [c for c in components if isinstance(c, ComponentHealth)]

        # Determine overall status
        unhealthy_count = sum(
            1 for c in valid_components if c.status == HealthStatus.UNHEALTHY
        )
        degraded_count = sum(
            1 for c in valid_components if c.status == HealthStatus.DEGRADED
        )

        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Identify degraded features
        degraded_features = []
        for component in valid_components:
            if component.status != HealthStatus.HEALTHY:
                degraded_features.append(component.component)

        return SystemHealth(
            status=overall_status,
            components=valid_components,
            degraded_features=degraded_features,
        )


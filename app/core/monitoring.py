"""Production monitoring, metrics, and observability."""

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Summary


# ---------------------------------------------------------------------------
# Helpers — safe to call on every module import (including uvicorn reloads)
# ---------------------------------------------------------------------------

def _lookup(name: str) -> object:
    """Return an already-registered collector by name.

    prometheus_client stores Counters under both ``<name>`` and ``<base>``
    (where base strips the ``_total`` suffix).  Try both so this works for
    Counter, Histogram and Gauge alike.
    """
    c = REGISTRY._names_to_collectors  # type: ignore[attr-defined]
    if name in c:
        return c[name]
    base = name[: -len("_total")] if name.endswith("_total") else name
    if base in c:
        return c[base]
    raise KeyError(f"prometheus metric {name!r} not found in registry")


def _counter(name: str, doc: str, labels: tuple = ()) -> Counter:
    try:
        return Counter(name, doc, list(labels))
    except ValueError:
        return _lookup(name)  # type: ignore[return-value]


def _histogram(name: str, doc: str, labels: tuple = ()) -> Histogram:
    try:
        return Histogram(name, doc, list(labels))
    except ValueError:
        return _lookup(name)  # type: ignore[return-value]


def _gauge(name: str, doc: str) -> Gauge:
    try:
        return Gauge(name, doc)
    except ValueError:
        return _lookup(name)  # type: ignore[return-value]

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# ---------------------------------------------------------------------------
# Prometheus metrics — all defined via safe helpers so that uvicorn --reload
# re-imports of this module do not crash with "Duplicated timeseries".
# ---------------------------------------------------------------------------

# Request metrics
http_requests_total = _counter(
    "http_requests_total", "Total HTTP requests", ("method", "endpoint", "status")
)
http_request_duration_seconds = _histogram(
    "http_request_duration_seconds", "HTTP request duration in seconds", ("method", "endpoint")
)

# Connector metrics
connector_requests_total = _counter(
    "connector_requests_total", "Total connector requests", ("platform", "status")
)
connector_request_duration_seconds = _histogram(
    "connector_request_duration_seconds", "Connector request duration in seconds", ("platform",)
)
connector_items_fetched = _counter(
    "connector_items_fetched", "Total items fetched by connectors", ("platform",)
)

# Scraping metrics
scraping_requests_total = _counter(
    "scraping_requests_total", "Total scraping requests", ("domain", "status")
)
scraping_duration_seconds = _histogram(
    "scraping_duration_seconds", "Scraping duration in seconds", ("domain",)
)

# LLM metrics
llm_requests_total = _counter(
    "llm_requests_total", "Total LLM requests", ("provider", "model", "status")
)
llm_request_duration_seconds = _histogram(
    "llm_request_duration_seconds", "LLM request duration in seconds", ("provider", "model")
)
llm_tokens_used = _counter(
    "llm_tokens_used", "Total LLM tokens used", ("provider", "model", "type")
)

# Output generation metrics
output_generation_total = _counter(
    "output_generation_total", "Total output generations", ("format", "status")
)
output_generation_duration_seconds = _histogram(
    "output_generation_duration_seconds", "Output generation duration in seconds", ("format",)
)
output_quality_score = _histogram(
    "output_quality_score", "Output quality scores", ("format",)
)

# Database metrics
database_queries_total = _counter(
    "database_queries_total", "Total database queries", ("operation", "status")
)
database_query_duration_seconds = _histogram(
    "database_query_duration_seconds", "Database query duration in seconds", ("operation",)
)

# System metrics
active_users = _gauge("active_users", "Number of active users")
content_items_total = _gauge("content_items_total", "Total content items in database")
clusters_generated_total = _counter("clusters_generated_total", "Total clusters generated")

# Error metrics
errors_total = _counter("errors_total", "Total errors", ("error_code", "severity"))


class MetricsCollector:
    """Collect and export application metrics."""

    @staticmethod
    def record_http_request(method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
            duration
        )

    @staticmethod
    def record_connector_request(
        platform: str, status: str, duration: float, items_count: int = 0
    ):
        """Record connector request metrics."""
        connector_requests_total.labels(platform=platform, status=status).inc()
        connector_request_duration_seconds.labels(platform=platform).observe(duration)
        if items_count > 0:
            connector_items_fetched.labels(platform=platform).inc(items_count)

    @staticmethod
    def record_scraping_request(domain: str, status: str, duration: float):
        """Record scraping request metrics."""
        scraping_requests_total.labels(domain=domain, status=status).inc()
        scraping_duration_seconds.labels(domain=domain).observe(duration)

    @staticmethod
    def record_llm_request(
        provider: str,
        model: str,
        status: str,
        duration: float,
        tokens_used: Optional[Dict[str, int]] = None,
    ):
        """Record LLM request metrics."""
        llm_requests_total.labels(provider=provider, model=model, status=status).inc()
        llm_request_duration_seconds.labels(provider=provider, model=model).observe(
            duration
        )

        if tokens_used:
            for token_type, count in tokens_used.items():
                llm_tokens_used.labels(
                    provider=provider, model=model, type=token_type
                ).inc(count)

    @staticmethod
    def record_output_generation(
        format: str, status: str, duration: float, quality_score: Optional[float] = None
    ):
        """Record output generation metrics."""
        output_generation_total.labels(format=format, status=status).inc()
        output_generation_duration_seconds.labels(format=format).observe(duration)

        if quality_score is not None:
            output_quality_score.labels(format=format).observe(quality_score)

    @staticmethod
    def record_database_query(operation: str, status: str, duration: float):
        """Record database query metrics."""
        database_queries_total.labels(operation=operation, status=status).inc()
        database_query_duration_seconds.labels(operation=operation).observe(duration)

    @staticmethod
    def record_error(error_code: str, severity: str):
        """Record error metrics."""
        errors_total.labels(error_code=error_code, severity=severity).inc()

    @staticmethod
    def update_active_users(count: int):
        """Update active users gauge."""
        active_users.set(count)

    @staticmethod
    def update_content_items_total(count: int):
        """Update content items total gauge."""
        content_items_total.set(count)

    @staticmethod
    def increment_clusters_generated():
        """Increment clusters generated counter."""
        clusters_generated_total.inc()


@contextmanager
def track_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Context manager to track execution time.

    Args:
        metric_name: Name of the metric
        labels: Metric labels

    Yields:
        Start time
    """
    start_time = time.time()
    try:
        yield start_time
    finally:
        duration = time.time() - start_time
        logger.debug(f"{metric_name} took {duration:.3f}s", extra=labels or {})


@asynccontextmanager
async def track_async_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Async context manager to track execution time.

    Args:
        metric_name: Name of the metric
        labels: Metric labels

    Yields:
        Start time
    """
    start_time = time.time()
    try:
        yield start_time
    finally:
        duration = time.time() - start_time
        logger.debug(f"{metric_name} took {duration:.3f}s", extra=labels or {})


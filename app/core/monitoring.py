"""Production monitoring, metrics, and observability."""

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


# Define Prometheus metrics
# Request metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

# Connector metrics
connector_requests_total = Counter(
    "connector_requests_total",
    "Total connector requests",
    ["platform", "status"],
)

connector_request_duration_seconds = Histogram(
    "connector_request_duration_seconds",
    "Connector request duration in seconds",
    ["platform"],
)

connector_items_fetched = Counter(
    "connector_items_fetched",
    "Total items fetched by connectors",
    ["platform"],
)

# Scraping metrics
scraping_requests_total = Counter(
    "scraping_requests_total",
    "Total scraping requests",
    ["domain", "status"],
)

scraping_duration_seconds = Histogram(
    "scraping_duration_seconds",
    "Scraping duration in seconds",
    ["domain"],
)

# LLM metrics
llm_requests_total = Counter(
    "llm_requests_total",
    "Total LLM requests",
    ["provider", "model", "status"],
)

llm_request_duration_seconds = Histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ["provider", "model"],
)

llm_tokens_used = Counter(
    "llm_tokens_used",
    "Total LLM tokens used",
    ["provider", "model", "type"],
)

# Output generation metrics
output_generation_total = Counter(
    "output_generation_total",
    "Total output generations",
    ["format", "status"],
)

output_generation_duration_seconds = Histogram(
    "output_generation_duration_seconds",
    "Output generation duration in seconds",
    ["format"],
)

output_quality_score = Histogram(
    "output_quality_score",
    "Output quality scores",
    ["format"],
)

# Database metrics
database_queries_total = Counter(
    "database_queries_total",
    "Total database queries",
    ["operation", "status"],
)

database_query_duration_seconds = Histogram(
    "database_query_duration_seconds",
    "Database query duration in seconds",
    ["operation"],
)

# System metrics
active_users = Gauge(
    "active_users",
    "Number of active users",
)

content_items_total = Gauge(
    "content_items_total",
    "Total content items in database",
)

clusters_generated_total = Counter(
    "clusters_generated_total",
    "Total clusters generated",
)

# Error metrics
errors_total = Counter(
    "errors_total",
    "Total errors",
    ["error_code", "severity"],
)


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


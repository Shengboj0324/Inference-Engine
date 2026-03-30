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

# ---------------------------------------------------------------------------
# Industrial-grade observability metrics (Phase 2)
# ---------------------------------------------------------------------------

# Ingestion latency per source platform — measures end-to-end time from
# connector fetch to ContentItem stored in Postgres, segmented by platform.
ingestion_latency_seconds = _histogram(
    "ingestion_latency_seconds",
    "End-to-end content ingestion latency per source platform (seconds)",
    ("platform",),
)

# Token-to-signal efficiency — how many LLM tokens are consumed per actionable
# signal produced.  High ratios flag costly classification paths.
token_signal_ratio = _histogram(
    "token_signal_ratio",
    "LLM tokens consumed per actionable signal generated",
    ("signal_type",),
)

# HDBSCAN clustering density — distribution of items per cluster.
# Sparse clusters (size 2–3) may indicate over-sensitive similarity thresholds.
clustering_density = _histogram(
    "clustering_density_items",
    "Number of content items per HDBSCAN cluster",
    (),
)

# WebSocket signal-stream connections — tracks concurrent real-time subscribers.
websocket_connections_active = _gauge(
    "websocket_connections_active",
    "Number of active WebSocket signal-stream connections",
)

# Deep Research steps — counts recursive LLM adjudication rounds.
deep_research_steps_total = _counter(
    "deep_research_steps_total",
    "Total recursive LLM steps executed in Deep Research mode",
    ("signal_type",),
)

# PII scrubbing — counts entity instances removed during normalization,
# segmented by platform and PII category (email, phone, name, …).
pii_entities_scrubbed_total = _counter(
    "pii_entities_scrubbed_total",
    "PII entity instances scrubbed from content during normalization",
    ("platform", "entity_type"),
)

# Sentiment drift detections — counts observations where measured sentiment
# diverges materially from the author/channel rolling baseline.
sentiment_drift_detected_total = _counter(
    "sentiment_drift_detected_total",
    "Observations where sentiment drifted beyond threshold vs baseline",
    ("platform", "direction"),  # direction: positive | negative
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

    # ------------------------------------------------------------------
    # Industrial-grade observability (Phase 2)
    # ------------------------------------------------------------------

    @staticmethod
    def record_ingestion_latency(platform: str, latency_seconds: float) -> None:
        """Record end-to-end ingestion latency for a single content item.

        Args:
            platform: Source platform name (e.g. 'reddit', 'rss').
            latency_seconds: Elapsed seconds from fetch to DB write.
        """
        ingestion_latency_seconds.labels(platform=platform).observe(latency_seconds)

    @staticmethod
    def record_token_signal_ratio(signal_type: str, tokens: int) -> None:
        """Record LLM token cost per actionable signal produced.

        Args:
            signal_type: Signal type value string.
            tokens: Total tokens (prompt + completion) used to produce the signal.
        """
        token_signal_ratio.labels(signal_type=signal_type).observe(tokens)

    @staticmethod
    def record_clustering_density(items_per_cluster: int) -> None:
        """Record the size (item count) of a single HDBSCAN cluster.

        Args:
            items_per_cluster: Number of content items grouped into the cluster.
        """
        clustering_density.observe(items_per_cluster)

    @staticmethod
    def record_websocket_connection(delta: int) -> None:
        """Adjust the active WebSocket connection gauge.

        Args:
            delta: +1 when a client connects, -1 when it disconnects.
        """
        websocket_connections_active.inc(delta)

    @staticmethod
    def record_deep_research_step(signal_type: str) -> None:
        """Increment the Deep Research step counter for a signal type.

        Args:
            signal_type: Signal type value string for the research session.
        """
        deep_research_steps_total.labels(signal_type=signal_type).inc()

    @staticmethod
    def record_pii_scrub(platform: str, entity_type: str, count: int = 1) -> None:
        """Record PII entity instances scrubbed during normalization.

        Args:
            platform: Source platform of the content being scrubbed.
            entity_type: PII category (e.g. 'email', 'phone', 'name').
            count: Number of entity instances removed (default 1).
        """
        pii_entities_scrubbed_total.labels(platform=platform, entity_type=entity_type).inc(count)

    @staticmethod
    def record_sentiment_drift(platform: str, direction: str) -> None:
        """Record a sentiment drift detection event.

        Args:
            platform: Source platform of the drifting observation.
            direction: 'positive' or 'negative' relative to baseline.
        """
        sentiment_drift_detected_total.labels(platform=platform, direction=direction).inc()


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


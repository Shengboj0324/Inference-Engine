"""Monitoring and observability for LLM infrastructure.

This module provides industrial-grade monitoring:
- Prometheus metrics for requests, tokens, cost, latency
- Cost tracking and budget alerts
- Quality monitoring and regression detection
- Circuit breaker state tracking
- Provider health monitoring
"""

import logging
from typing import Any, Dict, Optional

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Info

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe helpers — idempotent against module reloads and duplicate imports
# ---------------------------------------------------------------------------

def _lookup(name: str) -> object:
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


def _histogram(name: str, doc: str, labels: tuple = (), buckets: tuple = Histogram.DEFAULT_BUCKETS) -> Histogram:
    try:
        return Histogram(name, doc, list(labels), buckets=buckets)
    except ValueError:
        return _lookup(name)  # type: ignore[return-value]


def _gauge(name: str, doc: str, labels: tuple = ()) -> Gauge:
    try:
        return Gauge(name, doc, list(labels))
    except ValueError:
        return _lookup(name)  # type: ignore[return-value]


def _info(name: str, doc: str) -> Info:
    try:
        return Info(name, doc)
    except ValueError:
        return _lookup(name)  # type: ignore[return-value]


# ============================================================================
# REQUEST METRICS
# ============================================================================

llm_requests_total = _counter(
    "llm_requests_total",
    "Total number of LLM requests",
    ("provider", "model", "status"),
)

llm_request_duration_seconds = _histogram(
    "llm_request_duration_seconds",
    "LLM request duration in seconds",
    ("provider", "model"),
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

llm_time_to_first_token_seconds = _histogram(
    "llm_time_to_first_token_seconds",
    "Time to first token in seconds",
    ("provider", "model"),
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
)


# ============================================================================
# TOKEN METRICS
# ============================================================================

llm_tokens_total = _counter(
    "llm_tokens_total",
    "Total number of tokens processed",
    ("provider", "model", "token_type"),
)

llm_tokens_per_second = _gauge(
    "llm_tokens_per_second",
    "Tokens generated per second",
    ("provider", "model"),
)


# ============================================================================
# COST METRICS
# ============================================================================

llm_cost_total = _counter(
    "llm_cost_total",
    "Total cost in USD",
    ("provider", "model"),
)

llm_cost_per_request = _histogram(
    "llm_cost_per_request",
    "Cost per request in USD",
    ("provider", "model"),
    buckets=(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
)

llm_daily_cost = _gauge("llm_daily_cost", "Daily cost in USD", ("provider",))
llm_monthly_cost = _gauge("llm_monthly_cost", "Monthly cost in USD", ("provider",))


# ============================================================================
# ERROR METRICS
# ============================================================================

llm_errors_total = _counter(
    "llm_errors_total", "Total number of errors", ("provider", "model", "error_type")
)
llm_rate_limit_errors = _counter(
    "llm_rate_limit_errors", "Number of rate limit errors", ("provider", "model")
)
llm_timeout_errors = _counter(
    "llm_timeout_errors", "Number of timeout errors", ("provider", "model")
)


# ============================================================================
# CIRCUIT BREAKER METRICS
# ============================================================================

llm_circuit_breaker_state = _gauge(
    "llm_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ("provider", "model"),
)
llm_circuit_breaker_failures = _counter(
    "llm_circuit_breaker_failures", "Circuit breaker failure count", ("provider", "model")
)


# ============================================================================
# QUALITY METRICS
# ============================================================================

llm_response_length = _histogram(
    "llm_response_length",
    "Response length in characters",
    ("provider", "model"),
    buckets=(10, 50, 100, 500, 1000, 5000, 10000),
)
llm_quality_score = _histogram(
    "llm_quality_score",
    "Quality score (0.0-1.0)",
    ("provider", "model"),
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


# ============================================================================
# ROUTER METRICS
# ============================================================================

llm_router_decisions = _counter(
    "llm_router_decisions", "Router decision count", ("strategy", "selected_model")
)
llm_router_fallbacks = _counter(
    "llm_router_fallbacks", "Router fallback count", ("primary_model", "fallback_model")
)


# ============================================================================
# PROVIDER HEALTH METRICS
# ============================================================================

llm_provider_health = _gauge(
    "llm_provider_health",
    "Provider health status (0=unhealthy, 1=healthy)",
    ("provider", "model"),
)
llm_provider_info = _info("llm_provider_info", "Provider information")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def record_request(
    provider: str,
    model: str,
    status: str,
    duration_seconds: float,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    response_length: int,
    ttft_seconds: Optional[float] = None,
    tokens_per_second: Optional[float] = None,
    quality_score: Optional[float] = None,
) -> None:
    """Record metrics for an LLM request.

    Args:
        provider: Provider name
        model: Model name
        status: Request status (success, error)
        duration_seconds: Request duration
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        cost: Request cost in USD
        response_length: Response length in characters
        ttft_seconds: Time to first token
        tokens_per_second: Tokens per second
        quality_score: Quality score (0.0-1.0)
    """
    # Request metrics
    llm_requests_total.labels(provider=provider, model=model, status=status).inc()
    llm_request_duration_seconds.labels(provider=provider, model=model).observe(duration_seconds)

    # Token metrics
    llm_tokens_total.labels(provider=provider, model=model, token_type="prompt").inc(prompt_tokens)
    llm_tokens_total.labels(provider=provider, model=model, token_type="completion").inc(completion_tokens)

    if tokens_per_second is not None:
        llm_tokens_per_second.labels(provider=provider, model=model).set(tokens_per_second)

    # TTFT
    if ttft_seconds is not None:
        llm_time_to_first_token_seconds.labels(provider=provider, model=model).observe(ttft_seconds)

    # Cost metrics
    llm_cost_total.labels(provider=provider, model=model).inc(cost)
    llm_cost_per_request.labels(provider=provider, model=model).observe(cost)

    # Quality metrics
    llm_response_length.labels(provider=provider, model=model).observe(response_length)

    if quality_score is not None:
        llm_quality_score.labels(provider=provider, model=model).observe(quality_score)


def record_error(
    provider: str,
    model: str,
    error_type: str,
) -> None:
    """Record an error.

    Args:
        provider: Provider name
        model: Model name
        error_type: Error type
    """
    llm_errors_total.labels(provider=provider, model=model, error_type=error_type).inc()

    # Specific error counters
    if "rate_limit" in error_type.lower():
        llm_rate_limit_errors.labels(provider=provider, model=model).inc()
    elif "timeout" in error_type.lower():
        llm_timeout_errors.labels(provider=provider, model=model).inc()


def record_circuit_breaker_state(
    provider: str,
    model: str,
    state: str,  # "closed", "open", "half_open"
) -> None:
    """Record circuit breaker state.

    Args:
        provider: Provider name
        model: Model name
        state: Circuit breaker state
    """
    state_map = {"closed": 0, "open": 1, "half_open": 2}
    state_value = state_map.get(state.lower(), 0)

    llm_circuit_breaker_state.labels(provider=provider, model=model).set(state_value)


def record_circuit_breaker_failure(
    provider: str,
    model: str,
) -> None:
    """Record circuit breaker failure.

    Args:
        provider: Provider name
        model: Model name
    """
    llm_circuit_breaker_failures.labels(provider=provider, model=model).inc()


def record_router_decision(
    strategy: str,
    selected_model: str,
) -> None:
    """Record router decision.

    Args:
        strategy: Routing strategy
        selected_model: Selected model
    """
    llm_router_decisions.labels(strategy=strategy, selected_model=selected_model).inc()


def record_router_fallback(
    primary_model: str,
    fallback_model: str,
) -> None:
    """Record router fallback.

    Args:
        primary_model: Primary model that failed
        fallback_model: Fallback model used
    """
    llm_router_fallbacks.labels(primary_model=primary_model, fallback_model=fallback_model).inc()


def record_provider_health(
    provider: str,
    model: str,
    healthy: bool,
) -> None:
    """Record provider health status.

    Args:
        provider: Provider name
        model: Model name
        healthy: Health status
    """
    llm_provider_health.labels(provider=provider, model=model).set(1 if healthy else 0)


def update_daily_cost(provider: str, cost: float) -> None:
    """Update daily cost.

    Args:
        provider: Provider name
        cost: Daily cost
    """
    llm_daily_cost.labels(provider=provider).set(cost)


def update_monthly_cost(provider: str, cost: float) -> None:
    """Update monthly cost.

    Args:
        provider: Provider name
        cost: Monthly cost
    """
    llm_monthly_cost.labels(provider=provider).set(cost)


class LLMMetricsCollector:
    """Collector for LLM metrics with aggregation."""

    def __init__(self):
        """Initialize metrics collector."""
        self.daily_costs: Dict[str, float] = {}
        self.monthly_costs: Dict[str, float] = {}

        logger.info("Initialized LLM metrics collector")

    def add_cost(self, provider: str, cost: float) -> None:
        """Add cost to daily and monthly totals.

        Args:
            provider: Provider name
            cost: Cost to add
        """
        # Update daily cost
        self.daily_costs[provider] = self.daily_costs.get(provider, 0.0) + cost
        update_daily_cost(provider, self.daily_costs[provider])

        # Update monthly cost
        self.monthly_costs[provider] = self.monthly_costs.get(provider, 0.0) + cost
        update_monthly_cost(provider, self.monthly_costs[provider])

    def reset_daily_costs(self) -> None:
        """Reset daily costs (call at midnight)."""
        self.daily_costs = {}
        for provider in ["openai", "anthropic", "vllm"]:
            update_daily_cost(provider, 0.0)

        logger.info("Reset daily costs")

    def reset_monthly_costs(self) -> None:
        """Reset monthly costs (call at month start)."""
        self.monthly_costs = {}
        for provider in ["openai", "anthropic", "vllm"]:
            update_monthly_cost(provider, 0.0)

        logger.info("Reset monthly costs")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "daily_costs": self.daily_costs,
            "monthly_costs": self.monthly_costs,
            "total_daily_cost": sum(self.daily_costs.values()),
            "total_monthly_cost": sum(self.monthly_costs.values()),
        }


# Global metrics collector
_metrics_collector: Optional[LLMMetricsCollector] = None


def get_metrics_collector() -> LLMMetricsCollector:
    """Get global metrics collector.

    Returns:
        Metrics collector
    """
    global _metrics_collector

    if _metrics_collector is None:
        _metrics_collector = LLMMetricsCollector()

    return _metrics_collector


"""Operational health monitoring and SLO tracking for the inference pipeline.

``PipelineHealthMonitor`` is the single source of truth for *"is the system
healthy right now?"*  It collects observations from the various pipeline stages
and aggregates them into a structured ``HealthReport`` that downstream alerting,
dashboards, and the CI retrieval gate can consume.

Tracked signals
---------------
+-------------------------+--------------------------------------------------+
| Signal                  | Source                                           |
+=========================+==================================================+
| Stage latencies (p50,   | One ``LatencyObservation`` per pipeline stage    |
| p95, p99)               | call (route, chunk, cluster, dedup, rank)        |
+-------------------------+--------------------------------------------------+
| Error rates             | Counts from ``IndexingStats.route_errors``       |
+-------------------------+--------------------------------------------------+
| Chunk store growth rate | ``ChunkStore.count()`` sampled at each report    |
+-------------------------+--------------------------------------------------+
| ECE drift               | ``ArtifactRecord.ece`` from ModelArtifactRegistry|
+-------------------------+--------------------------------------------------+
| Connector freshness     | Seconds since last successful connector fetch    |
+-------------------------+--------------------------------------------------+

SLO definitions (all configurable)
-----------------------------------
- Routing p95 latency   ≤ ``route_p95_slo_s``   (default 30 s)
- Chunk store error rate ≤ ``error_rate_slo``    (default 5 %)
- ECE                   ≤ ``ece_slo``            (default 0.10)
- Connector freshness   ≤ ``freshness_slo_h``    (default 24 h)

Usage::

    monitor = PipelineHealthMonitor()

    # Record one routing observation
    monitor.record_latency("route", stage_s=1.23)
    monitor.record_error("route")

    # Record ECE from registry
    monitor.record_ece(0.0103)

    # Sample chunk store
    monitor.record_chunk_count(store.count())

    # Mark connector as refreshed
    monitor.record_connector_refresh("github_releases")

    report = monitor.health_report()
    print(report.overall_status)   # SLOStatus.GREEN
"""

from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SLOStatus(str, Enum):
    GREEN  = "green"   # All SLOs met
    YELLOW = "yellow"  # One or more SLOs at risk (within 20 % of threshold)
    RED    = "red"     # One or more SLOs violated


# ---------------------------------------------------------------------------
# Raw observation types
# ---------------------------------------------------------------------------

@dataclass
class LatencyObservation:
    stage:      str
    latency_s:  float
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SLOViolation:
    """A single SLO threshold breach detected at report time."""
    component: str
    metric:    str
    value:     float
    threshold: float
    status:    SLOStatus
    message:   str


# ---------------------------------------------------------------------------
# Per-stage latency stats (computed on demand)
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Percentile summary for one pipeline stage."""
    stage:   str
    count:   int
    p50_s:   float
    p95_s:   float
    p99_s:   float
    mean_s:  float


# ---------------------------------------------------------------------------
# Full health report
# ---------------------------------------------------------------------------

@dataclass
class HealthReport:
    """Snapshot of overall pipeline health.

    Attributes
    ----------
    overall_status:    Worst SLO status across all components.
    generated_at:      UTC timestamp.
    latency_stats:     Per-stage latency percentiles.
    error_rates:       Per-stage error rate (errors / total calls).
    current_ece:       Most recently recorded ECE (``None`` if not recorded).
    chunk_store_size:  Most recently sampled chunk count.
    chunk_growth_rate: Chunks-per-hour computed from last two samples.
    connector_ages_h:  ``{connector_id: hours since last refresh}``.
    violations:        All detected SLO violations.
    summary:           Human-readable one-line status.
    """
    overall_status:    SLOStatus
    generated_at:      datetime
    latency_stats:     List[LatencyStats]
    error_rates:       Dict[str, float]
    current_ece:       Optional[float]
    chunk_store_size:  int
    chunk_growth_rate: float               # chunks per hour
    connector_ages_h:  Dict[str, float]
    violations:        List[SLOViolation]
    summary:           str


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class PipelineHealthMonitor:
    """Collects pipeline observations and produces ``HealthReport`` on demand.

    All public methods are thread-safe.

    Args:
        window_size:       Maximum number of latency observations retained
                           per stage (oldest are evicted).
        route_p95_slo_s:   SLO for routing p95 latency (seconds).
        error_rate_slo:    SLO for error rate (0–1).
        ece_slo:           SLO for Expected Calibration Error.
        freshness_slo_h:   SLO for connector freshness (hours).
        slo_warning_pct:   Fraction of threshold at which YELLOW is raised
                           before the threshold is breached (default 0.80,
                           meaning YELLOW fires at 80 % of the limit).
    """

    def __init__(
        self,
        window_size:       int   = 1_000,
        route_p95_slo_s:   float = 30.0,
        error_rate_slo:    float = 0.05,
        ece_slo:           float = 0.10,
        freshness_slo_h:   float = 24.0,
        slo_warning_pct:   float = 0.80,
        cb_open_threshold: int   = 5,
        slo_tracker:       Optional[Any] = None,
        slo_tenant_id:     str = "system",
    ) -> None:
        if slo_tracker is not None and not callable(
            getattr(slo_tracker, "record_observation", None)
        ):
            raise TypeError(
                "'slo_tracker' must have a callable 'record_observation' method; "
                f"got {type(slo_tracker)!r}"
            )
        if not isinstance(slo_tenant_id, str) or not slo_tenant_id.strip():
            raise ValueError(
                f"'slo_tenant_id' must be a non-empty string; got {slo_tenant_id!r}"
            )

        self._lock              = threading.Lock()
        self._window            = window_size
        self._route_p95_slo     = route_p95_slo_s
        self._err_slo           = error_rate_slo
        self._ece_slo           = ece_slo
        self._fresh_slo_h       = freshness_slo_h
        self._warn_pct          = slo_warning_pct
        self._cb_open_threshold = cb_open_threshold
        self._slo_tracker       = slo_tracker
        self._slo_tenant_id     = slo_tenant_id.strip()

        # Stage → deque of LatencyObservation
        self._latencies:     Dict[str, Deque[LatencyObservation]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        # Stage → (total_calls, error_calls)
        self._call_counts:   Dict[str, List[int]]                 = defaultdict(lambda: [0, 0])
        # ECE history
        self._ece_history:   Deque[Tuple[datetime, float]]        = deque(maxlen=200)
        # Chunk count history: list of (observed_at, count)
        self._chunk_samples: Deque[Tuple[datetime, int]]          = deque(maxlen=500)
        # Connector → last refresh timestamp
        self._connectors:    Dict[str, datetime]                  = {}
        # Most recently recorded watchlist gap count (None = never recorded)
        self._watchlist_gap_count: Optional[int]                  = None
        # Per-connector circuit-breaker state:
        #   connector_id → consecutive_failures (int)
        # Circuit is considered OPEN when count >= _cb_open_threshold
        self._cb_failures:  Dict[str, int]                        = {}
        # Connectors whose circuit is currently OPEN
        self._cb_open:      set                                    = set()

    # ------------------------------------------------------------------
    # SLOTracker properties
    # ------------------------------------------------------------------

    @property
    def slo_tracker(self) -> Optional[Any]:
        """The enterprise ``SLOTracker`` injected at construction, or ``None``."""
        return self._slo_tracker

    @property
    def slo_tenant_id(self) -> str:
        """Tenant identifier used when emitting to the enterprise ``SLOTracker``."""
        return self._slo_tenant_id

    def _emit(self, metric_name: str, value: float) -> None:
        """Forward one metric observation to the enterprise ``SLOTracker``.

        Errors are caught and logged so they never disrupt the main recording path.
        """
        if self._slo_tracker is None:
            return
        try:
            self._slo_tracker.record_observation(
                self._slo_tenant_id, metric_name, value
            )
        except Exception as exc:
            logger.warning(
                "PipelineHealthMonitor: SLOTracker emission failed "
                "tenant=%r metric=%r value=%.4g: %s",
                self._slo_tenant_id, metric_name, value, exc,
            )

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def record_latency(self, stage: str, stage_s: float) -> None:
        """Record a latency sample for *stage*.

        Args:
            stage:   Pipeline stage name (e.g. ``"route"``, ``"cluster"``).
            stage_s: Wall-clock seconds for that stage invocation.

        Raises:
            ValueError: If ``stage_s`` is negative.
        """
        if stage_s < 0:
            raise ValueError(f"stage_s must be ≥ 0; got {stage_s!r}")
        obs = LatencyObservation(stage=stage, latency_s=stage_s)
        with self._lock:
            self._latencies[stage].append(obs)
            self._call_counts[stage][0] += 1
        self._emit(f"latency.{stage}", stage_s)

    def record_error(self, stage: str) -> None:
        """Increment the error counter for *stage* (also counts as one call)."""
        with self._lock:
            counts = self._call_counts[stage]
            counts[0] += 1
            counts[1] += 1
        self._emit(f"error.{stage}", 1.0)

    def record_indexing_stats(self, stats: Any) -> None:
        """Ingest an ``IndexingStats`` object from a completed ``IndexingPipeline`` run.

        Records one error observation per ``stats.route_errors`` and one
        latency sample for the full pipeline.
        """
        self.record_latency("indexing_pipeline", stats.wall_s)
        for _ in range(stats.route_errors):
            self.record_error("route")
        for _ in range(stats.routed_ok):
            with self._lock:
                self._call_counts["route"][0] += 1

    def record_ece(self, ece: float) -> None:
        """Record a calibration ECE observation.

        Args:
            ece: Expected Calibration Error (0–1).

        Raises:
            ValueError: If ``ece`` is outside [0, 1].
        """
        if not 0.0 <= ece <= 1.0:
            raise ValueError(f"ECE must be in [0, 1]; got {ece!r}")
        with self._lock:
            self._ece_history.append((datetime.now(timezone.utc), ece))
        self._emit("ece", ece)

    def record_chunk_count(self, count: int) -> None:
        """Sample the current ``ChunkStore`` size for growth-rate computation.

        Args:
            count: Current total number of chunks in the store.

        Raises:
            ValueError: If ``count`` is negative.
        """
        if count < 0:
            raise ValueError(f"count must be ≥ 0; got {count!r}")
        with self._lock:
            self._chunk_samples.append((datetime.now(timezone.utc), count))
        self._emit("chunk_count", float(count))

    def record_connector_refresh(self, connector_id: str) -> None:
        """Mark *connector_id* as having successfully fetched data right now."""
        with self._lock:
            self._connectors[connector_id] = datetime.now(timezone.utc)
        self._emit(f"freshness.{connector_id}", 0.0)

    def record_connector_failure(self, connector_id: str) -> None:
        """Record one consecutive failure for *connector_id*.

        If the consecutive-failure count reaches ``cb_open_threshold`` (default 5),
        the circuit is opened and a ``RED`` SLO violation is recorded.  The circuit
        remains open until :meth:`record_connector_success` is called.

        Args:
            connector_id: Connector identifier (e.g. ``"github_releases"``).
        """
        with self._lock:
            count = self._cb_failures.get(connector_id, 0) + 1
            self._cb_failures[connector_id] = count
            if count >= self._cb_open_threshold:
                self._cb_open.add(connector_id)
        self._emit(f"connector.failures.{connector_id}", float(count))

    def record_connector_success(self, connector_id: str) -> None:
        """Record a successful fetch for *connector_id*, resetting its failure counter.

        If the circuit was open, it is closed.

        Args:
            connector_id: Connector identifier.
        """
        with self._lock:
            self._cb_failures[connector_id] = 0
            self._cb_open.discard(connector_id)
            # Also record as a successful refresh for the freshness SLO
            self._connectors[connector_id] = datetime.now(timezone.utc)
        self._emit(f"connector.failures.{connector_id}", 0.0)
        self._emit(f"freshness.{connector_id}", 0.0)

    def is_circuit_open(self, connector_id: str) -> bool:
        """Return ``True`` if the circuit breaker for *connector_id* is OPEN."""
        with self._lock:
            return connector_id in self._cb_open

    def record_watchlist_gap_count(self, n: int) -> None:
        """Record the number of watched nodes that currently have zero coverage.

        SLO thresholds
        --------------
        - ``n < 3``   → no violation (GREEN contribution)
        - ``3 ≤ n < 10`` → YELLOW violation recorded
        - ``n ≥ 10``  → RED violation recorded

        This method is called automatically by ``IndexingPipeline.process_batch()``
        after each batch when a ``WatchlistGraph`` is injected.  It can also be
        called manually after calling ``graph.coverage_report()``.

        Args:
            n: Number of watched nodes with zero coverage (``nodes_at_risk`` from
               ``WatchlistCoverageReport``).

        Raises:
            ValueError: If ``n`` is negative.
        """
        if n < 0:
            raise ValueError(f"watchlist gap count must be ≥ 0; got {n!r}")
        with self._lock:
            self._watchlist_gap_count = n
        self._emit("watchlist_gap_count", float(n))

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def health_report(self) -> HealthReport:
        """Compute and return a ``HealthReport`` from current observations.

        This is an O(N log N) operation (sorting latency samples per stage)
        and is expected to be called at most once per minute.
        """
        with self._lock:
            lat_snap  = {k: list(v) for k, v in self._latencies.items()}
            call_snap = {k: list(v) for k, v in self._call_counts.items()}
            ece_snap  = list(self._ece_history)
            chunk_snap = list(self._chunk_samples)
            conn_snap  = dict(self._connectors)
            gap_snap   = self._watchlist_gap_count
            cb_open_snap   = set(self._cb_open)
            cb_counts_snap = dict(self._cb_failures)

        now = datetime.now(timezone.utc)
        violations: List[SLOViolation] = []

        # ── Latency stats ───────────────────────────────────────────
        latency_stats: List[LatencyStats] = []
        for stage, obs in lat_snap.items():
            if not obs:
                continue
            lats = sorted(o.latency_s for o in obs)
            stats = LatencyStats(
                stage=stage,
                count=len(lats),
                p50_s=_percentile(lats, 50),
                p95_s=_percentile(lats, 95),
                p99_s=_percentile(lats, 99),
                mean_s=sum(lats) / len(lats),
            )
            latency_stats.append(stats)
            # Check routing SLO
            if stage == "route":
                v = self._check_slo(
                    component="routing", metric="p95_latency_s",
                    value=stats.p95_s, threshold=self._route_p95_slo,
                    higher_is_worse=True,
                )
                if v:
                    violations.append(v)

        # ── Error rates ─────────────────────────────────────────────
        error_rates: Dict[str, float] = {}
        for stage, (total, errors) in call_snap.items():
            rate = errors / total if total > 0 else 0.0
            error_rates[stage] = rate
            v = self._check_slo(
                component=stage, metric="error_rate",
                value=rate, threshold=self._err_slo,
                higher_is_worse=True,
            )
            if v:
                violations.append(v)

        # ── ECE ─────────────────────────────────────────────────────
        current_ece: Optional[float] = None
        if ece_snap:
            current_ece = ece_snap[-1][1]
            v = self._check_slo(
                component="model", metric="ece",
                value=current_ece, threshold=self._ece_slo,
                higher_is_worse=True,
            )
            if v:
                violations.append(v)

        # ── Chunk growth rate ────────────────────────────────────────
        chunk_size = chunk_snap[-1][1] if chunk_snap else 0
        growth_rate = 0.0
        if len(chunk_snap) >= 2:
            t1, c1 = chunk_snap[0]
            t2, c2 = chunk_snap[-1]
            elapsed_h = (t2 - t1).total_seconds() / 3600.0
            if elapsed_h > 0:
                growth_rate = (c2 - c1) / elapsed_h

        # ── Connector freshness ──────────────────────────────────────
        connector_ages: Dict[str, float] = {}
        for cid, last_refresh in conn_snap.items():
            age_h = (now - last_refresh).total_seconds() / 3600.0
            connector_ages[cid] = round(age_h, 2)
            v = self._check_slo(
                component=f"connector:{cid}", metric="freshness_h",
                value=age_h, threshold=self._fresh_slo_h,
                higher_is_worse=True,
            )
            if v:
                violations.append(v)

        # ── Circuit-breaker violations ───────────────────────────────
        for cid in cb_open_snap:
            count = cb_counts_snap.get(cid, 0)
            violations.append(SLOViolation(
                component=f"connector:{cid}",
                metric="circuit_breaker",
                value=float(count),
                threshold=float(self._cb_open_threshold),
                status=SLOStatus.RED,
                message=(
                    f"Connector '{cid}' circuit is OPEN after "
                    f"{count} consecutive failure(s) "
                    f"(threshold: {self._cb_open_threshold})"
                ),
            ))

        # ── Watchlist gap count ──────────────────────────────────────
        if gap_snap is not None:
            if gap_snap >= 10:
                violations.append(SLOViolation(
                    component="watchlist", metric="zero_coverage_nodes",
                    value=float(gap_snap), threshold=10.0,
                    status=SLOStatus.RED,
                    message=(
                        f"{gap_snap} watched node(s) have zero coverage "
                        f"(RED threshold: ≥ 10)"
                    ),
                ))
            elif gap_snap >= 3:
                violations.append(SLOViolation(
                    component="watchlist", metric="zero_coverage_nodes",
                    value=float(gap_snap), threshold=3.0,
                    status=SLOStatus.YELLOW,
                    message=(
                        f"{gap_snap} watched node(s) have zero coverage "
                        f"(YELLOW threshold: ≥ 3)"
                    ),
                ))

        # ── Overall status ───────────────────────────────────────────
        if any(v.status == SLOStatus.RED for v in violations):
            overall = SLOStatus.RED
        elif any(v.status == SLOStatus.YELLOW for v in violations):
            overall = SLOStatus.YELLOW
        else:
            overall = SLOStatus.GREEN

        n_red    = sum(1 for v in violations if v.status == SLOStatus.RED)
        n_yellow = sum(1 for v in violations if v.status == SLOStatus.YELLOW)
        if overall == SLOStatus.GREEN:
            summary = "All SLOs met — pipeline is healthy."
        else:
            summary = (
                f"{n_red} SLO violation(s), {n_yellow} SLO warning(s) — "
                f"see violations for details."
            )

        return HealthReport(
            overall_status=overall,
            generated_at=now,
            latency_stats=latency_stats,
            error_rates=error_rates,
            current_ece=current_ece,
            chunk_store_size=chunk_size,
            chunk_growth_rate=round(growth_rate, 2),
            connector_ages_h=connector_ages,
            violations=violations,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Convenience resets (useful in tests)
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all observations (irreversible — primarily for testing)."""
        with self._lock:
            self._latencies.clear()
            self._call_counts.clear()
            self._ece_history.clear()
            self._chunk_samples.clear()
            self._connectors.clear()
            self._watchlist_gap_count = None
            self._cb_failures.clear()
            self._cb_open.clear()

    # ------------------------------------------------------------------
    # Internal SLO check
    # ------------------------------------------------------------------

    def _check_slo(
        self,
        component:       str,
        metric:          str,
        value:           float,
        threshold:       float,
        higher_is_worse: bool = True,
    ) -> Optional[SLOViolation]:
        """Return a ``SLOViolation`` when *value* breaches or approaches *threshold*."""
        if higher_is_worse:
            ratio = value / threshold if threshold > 0 else float("inf")
            breached = value > threshold
            at_risk  = value > threshold * self._warn_pct
        else:
            ratio = threshold / value if value > 0 else float("inf")
            breached = value < threshold
            at_risk  = value < threshold / self._warn_pct

        if breached:
            status = SLOStatus.RED
            msg    = (
                f"{component}.{metric} = {value:.4f} violates SLO "
                f"({'≤' if higher_is_worse else '≥'} {threshold})"
            )
        elif at_risk:
            status = SLOStatus.YELLOW
            msg    = (
                f"{component}.{metric} = {value:.4f} is approaching SLO "
                f"({'≤' if higher_is_worse else '≥'} {threshold})"
            )
        else:
            return None

        return SLOViolation(
            component=component,
            metric=metric,
            value=value,
            threshold=threshold,
            status=status,
            message=msg,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_values: List[float], pct: float) -> float:
    """Return the *pct*-th percentile from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    idx = (pct / 100.0) * (n - 1)
    lo  = int(math.floor(idx))
    hi  = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


from typing import Any  # noqa: E402


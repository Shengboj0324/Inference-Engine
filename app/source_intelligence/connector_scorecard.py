"""ConnectorScorecard — per-source fetch/parse quality tracking.

A ``ConnectorScorecard`` records the operational health of one source
connector on five independent axes:

* **fetch_success_rate**  — HTTP-level success (2xx or cached hit).
* **parse_success_rate**  — structured extraction without exception or
  empty-body fallback.
* **parse_completeness**  — fraction of expected fields populated on
  successfully parsed items (0.0–1.0).
* **duplicate_rate**      — fraction of fetched items matching an existing
  BloomFilter entry (indicates source is stale or over-crawled).
* **evidence_yield**      — mean number of evidence spans produced per item
  that reached the indexing pipeline.
* **latency_ms_p95**      — 95th-percentile fetch-to-parse wall time in ms.

``ScorecardRegistry`` is a thread-safe store of scorecards keyed by
``source_id``.  ``AcquisitionScheduler`` consults it through
``ScorecardRegistry.slo_violations()`` to deprioritise connectors that
fail their SLOs before pulling a new batch.

Usage::

    registry = ScorecardRegistry()
    registry.record_fetch("arxiv-cs", success=True, latency_ms=340.0)
    registry.record_parse("arxiv-cs", success=True, completeness=0.92,
                          evidence_yield=3.1, duplicate=False)
    card = registry.get("arxiv-cs")
    # card.fetch_success_rate  → float
    # card.parse_success_rate  → float
    # card.slo_ok(min_fetch=0.9, min_parse=0.85)  → bool
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── SLO defaults (overridable per-call) ──────────────────────────────────────
DEFAULT_MIN_FETCH_SUCCESS = 0.90
DEFAULT_MIN_PARSE_SUCCESS = 0.85
DEFAULT_MAX_DUPLICATE_RATE = 0.40
DEFAULT_MAX_LATENCY_P95_MS = 5_000.0
DEFAULT_MIN_EVIDENCE_YIELD = 0.5
_WINDOW = 200          # rolling window for all rate calculations
_LATENCY_WINDOW = 100  # rolling window for p95 latency


@dataclass
class SLOViolation:
    source_id: str
    axis: str           # e.g. "fetch_success_rate", "latency_ms_p95"
    measured: float
    threshold: float
    message: str


class ConnectorScorecard:
    """Operational quality scorecard for a single source connector.

    All computations are over a rolling window of ``_WINDOW`` observations
    so that recent regressions surface quickly without permanently penalising
    a connector for a past incident.
    """

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._lock = threading.Lock()

        # Rolling windows
        self._fetches: Deque[bool] = deque(maxlen=_WINDOW)
        self._parses: Deque[bool] = deque(maxlen=_WINDOW)
        self._completeness: Deque[float] = deque(maxlen=_WINDOW)
        self._evidence_yields: Deque[float] = deque(maxlen=_WINDOW)
        self._duplicates: Deque[bool] = deque(maxlen=_WINDOW)
        self._latencies_ms: Deque[float] = deque(maxlen=_LATENCY_WINDOW)

    # ── Recording ─────────────────────────────────────────────────────────

    def record_fetch(self, success: bool, latency_ms: float = 0.0) -> None:
        with self._lock:
            self._fetches.append(success)
            if latency_ms > 0:
                self._latencies_ms.append(latency_ms)

    def record_parse(
        self,
        success: bool,
        completeness: float = 1.0,
        evidence_yield: float = 0.0,
        duplicate: bool = False,
    ) -> None:
        with self._lock:
            self._parses.append(success)
            if success:
                self._completeness.append(max(0.0, min(1.0, completeness)))
                self._evidence_yields.append(max(0.0, evidence_yield))
            self._duplicates.append(duplicate)

    # ── Metrics ───────────────────────────────────────────────────────────

    @property
    def fetch_success_rate(self) -> float:
        with self._lock:
            return _mean_bool(self._fetches)

    @property
    def parse_success_rate(self) -> float:
        with self._lock:
            return _mean_bool(self._parses)

    @property
    def parse_completeness(self) -> float:
        with self._lock:
            return _mean_float(self._completeness)

    @property
    def duplicate_rate(self) -> float:
        with self._lock:
            return _mean_bool(self._duplicates)

    @property
    def evidence_yield(self) -> float:
        with self._lock:
            return _mean_float(self._evidence_yields)

    @property
    def latency_ms_p95(self) -> float:
        with self._lock:
            return _percentile(list(self._latencies_ms), 95)

    @property
    def sample_count(self) -> int:
        with self._lock:
            return len(self._fetches)

    def slo_ok(
        self,
        min_fetch: float = DEFAULT_MIN_FETCH_SUCCESS,
        min_parse: float = DEFAULT_MIN_PARSE_SUCCESS,
        max_duplicate: float = DEFAULT_MAX_DUPLICATE_RATE,
        max_latency_p95_ms: float = DEFAULT_MAX_LATENCY_P95_MS,
        min_evidence: float = DEFAULT_MIN_EVIDENCE_YIELD,
    ) -> bool:
        """True iff all SLO axes are within acceptable bounds."""
        return (
            self.fetch_success_rate >= min_fetch
            and self.parse_success_rate >= min_parse
            and self.duplicate_rate <= max_duplicate
            and (self.latency_ms_p95 <= max_latency_p95_ms or self.sample_count < 5)
            and self.evidence_yield >= min_evidence
        )

    def violations(
        self,
        min_fetch: float = DEFAULT_MIN_FETCH_SUCCESS,
        min_parse: float = DEFAULT_MIN_PARSE_SUCCESS,
        max_duplicate: float = DEFAULT_MAX_DUPLICATE_RATE,
        max_latency_p95_ms: float = DEFAULT_MAX_LATENCY_P95_MS,
        min_evidence: float = DEFAULT_MIN_EVIDENCE_YIELD,
    ) -> List[SLOViolation]:
        """Return a list of all SLO axes currently violated."""
        v: List[SLOViolation] = []
        checks = [
            ("fetch_success_rate", self.fetch_success_rate, min_fetch, True),
            ("parse_success_rate", self.parse_success_rate, min_parse, True),
            ("duplicate_rate", self.duplicate_rate, max_duplicate, False),
            ("latency_ms_p95", self.latency_ms_p95, max_latency_p95_ms, False),
            ("evidence_yield", self.evidence_yield, min_evidence, True),
        ]
        for axis, measured, threshold, higher_is_better in checks:
            failed = (measured < threshold) if higher_is_better else (measured > threshold)
            if failed:
                direction = "below" if higher_is_better else "above"
                v.append(SLOViolation(
                    source_id=self.source_id,
                    axis=axis,
                    measured=measured,
                    threshold=threshold,
                    message=(
                        f"source '{self.source_id}' {axis}={measured:.3f} "
                        f"is {direction} SLO threshold {threshold:.3f}"
                    ),
                ))
        return v

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "fetch_success_rate": round(self.fetch_success_rate, 4),
            "parse_success_rate": round(self.parse_success_rate, 4),
            "parse_completeness": round(self.parse_completeness, 4),
            "duplicate_rate": round(self.duplicate_rate, 4),
            "evidence_yield": round(self.evidence_yield, 4),
            "latency_ms_p95": round(self.latency_ms_p95, 1),
            "sample_count": self.sample_count,
        }


class ScorecardRegistry:
    """Thread-safe registry of ``ConnectorScorecard`` objects keyed by ``source_id``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cards: Dict[str, ConnectorScorecard] = {}

    def _get_or_create(self, source_id: str) -> ConnectorScorecard:
        if source_id not in self._cards:
            self._cards[source_id] = ConnectorScorecard(source_id)
        return self._cards[source_id]

    def record_fetch(
        self, source_id: str, success: bool, latency_ms: float = 0.0
    ) -> None:
        with self._lock:
            self._get_or_create(source_id).record_fetch(success, latency_ms)

    def record_parse(
        self,
        source_id: str,
        success: bool,
        completeness: float = 1.0,
        evidence_yield: float = 0.0,
        duplicate: bool = False,
    ) -> None:
        with self._lock:
            self._get_or_create(source_id).record_parse(
                success, completeness, evidence_yield, duplicate
            )

    def get(self, source_id: str) -> Optional[ConnectorScorecard]:
        with self._lock:
            return self._cards.get(source_id)

    def all_source_ids(self) -> List[str]:
        with self._lock:
            return list(self._cards.keys())

    def slo_violations(self, **slo_kwargs) -> Dict[str, List[SLOViolation]]:
        """Return {source_id: [SLOViolation, ...]} for all sources with violations."""
        result: Dict[str, List[SLOViolation]] = {}
        with self._lock:
            for sid, card in self._cards.items():
                vv = card.violations(**slo_kwargs)
                if vv:
                    result[sid] = vv
        return result

    def summary(self) -> List[dict]:
        with self._lock:
            return [c.to_dict() for c in self._cards.values()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean_bool(dq: Deque[bool]) -> float:
    return sum(dq) / len(dq) if dq else 0.0


def _mean_float(dq: Deque[float]) -> float:
    return sum(dq) / len(dq) if dq else 0.0


def _percentile(data: List[float], pct: int) -> float:
    if not data:
        return 0.0
    data_sorted = sorted(data)
    idx = math.ceil((pct / 100) * len(data_sorted)) - 1
    return data_sorted[max(0, idx)]


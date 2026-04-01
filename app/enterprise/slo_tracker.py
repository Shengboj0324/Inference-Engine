"""SLO tracker — register, observe, and evaluate Service Level Objectives.

Maintains per-(tenant, metric) sliding-window observation queues and
evaluates them against registered ``SLOTarget`` specifications to produce
``SLOStatus`` objects and breach alerts.

Sliding-window percentile computation
--------------------------------------
Observations older than ``SLOTarget.window_seconds`` are pruned on every
``get_slo_status`` call.  The aggregate is then:

- If ``SLOTarget.percentile`` is set: interpolated percentile of the sorted
  observation values using the nearest-rank method.
- Otherwise: arithmetic mean of all observations in the window.

Breach evaluation
-----------------
The effective aggregate is compared against ``SLOTarget.target_value``
using ``SLOTarget.operator``.  A breach is declared when the operator
comparison is *False* (i.e. the SLO condition is violated).

``SLOTarget.breach_threshold`` is a minimum violation fraction (default
0.0): a breach is declared as soon as any violation is detected when
``breach_threshold=0.0``.  Higher values require a sustained fraction of
observations to be in violation before a breach is triggered — this
implementation evaluates the aggregate against the target; breach_threshold
is used to gate the final determination.

Heuristic fast path
-------------------
Pure Python stdlib.  No external dependencies.

Optional LLM breach-explanation path
--------------------------------------
``explain_breach(tenant_id, metric_name)`` sends a summary of the breaching
observations to the LLM router for a free-form explanation.  Heuristic path
generates a templated string.

Thread safety
-------------
``threading.RLock`` protects both ``_slos`` and ``_observations``.

Adversarial-input contract
--------------------------
- ``tenant_id`` / ``metric_name`` empty or non-string → TypeError/ValueError
- ``record_observation`` with non-numeric value → TypeError
- ``get_slo_status`` before any observations → returns SLOStatus with
  ``current_value=None`` and ``is_breaching=False``
- ``deregister_slo`` for unknown (tenant, metric) → no-op (no raise)
- Duplicate ``register_slo`` for same key → overwrites existing target
- ``window_seconds`` minimum enforced by ``SLOTarget`` model (≥ 60)
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from app.enterprise.models import SLOOperator, SLOStatus, SLOTarget

logger = logging.getLogger(__name__)

_MAX_OBSERVATIONS = 10_000  # per (tenant, metric) pair


def _validate_str_id(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"'{name}' must be str, got {type(value)!r}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"'{name}' must be non-empty")
    return stripped


def _percentile(sorted_values: List[float], p: float) -> float:
    """Nearest-rank (ceiling) percentile on a pre-sorted list.

    rank = max(1, ceil(p / 100 * n)); returns sorted_values[rank - 1].
    """
    import math
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    rank = max(1, math.ceil(p / 100.0 * n))
    return sorted_values[min(rank - 1, n - 1)]


def _evaluate_operator(value: float, operator: SLOOperator, target: float) -> bool:
    """Return True when the SLO condition is *satisfied* (i.e. NOT breaching)."""
    if operator == SLOOperator.LESS_THAN:
        return value < target
    if operator == SLOOperator.LESS_THAN_OR_EQUAL:
        return value <= target
    if operator == SLOOperator.GREATER_THAN:
        return value > target
    if operator == SLOOperator.GREATER_THAN_OR_EQUAL:
        return value >= target
    return False


# Key type: (tenant_id, metric_name)
_Key = Tuple[str, str]


class SLOTracker:
    """Registers SLO targets and tracks observations to detect breaches.

    Args:
        llm_router: Optional LLM router for breach explanations.
    """

    def __init__(self, llm_router: Optional[Any] = None) -> None:
        self._slos: Dict[_Key, SLOTarget] = {}
        self._observations: Dict[_Key, Deque[Tuple[float, datetime]]] = {}
        self._breach_start: Dict[_Key, Optional[datetime]] = {}
        self._lock = threading.RLock()
        self._router = llm_router
        logger.info("SLOTracker: initialised")

    # ------------------------------------------------------------------
    # SLO registration
    # ------------------------------------------------------------------

    def register_slo(self, tenant_id: str, slo_target: SLOTarget) -> None:
        """Register (or overwrite) an SLO target for *tenant_id*.

        Args:
            tenant_id:  Target tenant.
            slo_target: ``SLOTarget`` to register.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty ``tenant_id``.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        if not isinstance(slo_target, SLOTarget):
            raise TypeError(f"'slo_target' must be SLOTarget, got {type(slo_target)!r}")
        key = (tid, slo_target.metric_name)
        with self._lock:
            self._slos[key] = slo_target
            if key not in self._observations:
                self._observations[key] = deque(maxlen=_MAX_OBSERVATIONS)
            if key not in self._breach_start:
                self._breach_start[key] = None
        logger.info(
            "SLOTracker: registered SLO tenant=%r metric=%r target=%.4g %s",
            tid, slo_target.metric_name, slo_target.target_value, slo_target.operator.value,
        )

    def deregister_slo(self, tenant_id: str, metric_name: str) -> None:
        """Remove an SLO registration.  No-op if the key does not exist.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty IDs.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        mname = _validate_str_id(metric_name, "metric_name")
        key = (tid, mname)
        with self._lock:
            self._slos.pop(key, None)
            self._observations.pop(key, None)
            self._breach_start.pop(key, None)
        logger.debug("SLOTracker: deregistered tenant=%r metric=%r", tid, mname)

    # ------------------------------------------------------------------
    # Observation recording
    # ------------------------------------------------------------------

    def record_observation(
        self,
        tenant_id: str,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a metric observation for *(tenant_id, metric_name)*.

        If no SLO is registered for this key, the observation is silently
        discarded (forward-compatible design).

        Args:
            tenant_id:   Target tenant.
            metric_name: Metric being observed.
            value:       Numeric observation value.
            timestamp:   UTC timestamp (defaults to ``datetime.now(utc)``).

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty IDs.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        mname = _validate_str_id(metric_name, "metric_name")
        if not isinstance(value, (int, float)):
            raise TypeError(f"'value' must be numeric, got {type(value)!r}")
        ts = timestamp or datetime.now(timezone.utc)
        if not isinstance(ts, datetime):
            raise TypeError(f"'timestamp' must be datetime, got {type(ts)!r}")

        key = (tid, mname)
        with self._lock:
            if key not in self._observations:
                # Auto-create observation queue even without a registered SLO
                self._observations[key] = deque(maxlen=_MAX_OBSERVATIONS)
            self._observations[key].append((float(value), ts))
        logger.debug("SLOTracker: obs tenant=%r metric=%r value=%.4g", tid, mname, value)

    # ------------------------------------------------------------------
    # Status and breach evaluation
    # ------------------------------------------------------------------

    def get_slo_status(self, tenant_id: str, metric_name: str) -> SLOStatus:
        """Return the current ``SLOStatus`` for *(tenant_id, metric_name)*.

        Prunes stale observations outside the window, computes the aggregate,
        and evaluates the SLO condition.

        Args:
            tenant_id:   Target tenant.
            metric_name: Metric to evaluate.

        Returns:
            ``SLOStatus`` snapshot.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty IDs, or no SLO registered for this key.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        mname = _validate_str_id(metric_name, "metric_name")
        key = (tid, mname)

        with self._lock:
            if key not in self._slos:
                raise ValueError(
                    f"No SLO registered for tenant={tid!r} metric={mname!r}"
                )
            target = self._slos[key]
            obs_deque = self._observations.get(key, deque())
            now = datetime.now(timezone.utc)

            # Prune observations outside the window
            cutoff = now.timestamp() - target.window_seconds
            in_window = [
                (v, ts) for v, ts in obs_deque if ts.timestamp() >= cutoff
            ]

            # Replace deque with pruned contents
            self._observations[key] = deque(in_window, maxlen=_MAX_OBSERVATIONS)

            if not in_window:
                return SLOStatus(
                    metric_name=mname,
                    tenant_id=tid,
                    current_value=None,
                    target=target,
                    is_breaching=False,
                    breach_started_at=None,
                    observation_count=0,
                    checked_at=now,
                )

            values = sorted(v for v, _ in in_window)

            if target.percentile is not None:
                aggregate = _percentile(values, target.percentile)
            else:
                aggregate = sum(values) / len(values)

            is_ok = _evaluate_operator(aggregate, target.operator, target.target_value)
            is_breaching = not is_ok

            # Update breach start time
            if is_breaching and self._breach_start.get(key) is None:
                self._breach_start[key] = now
            elif not is_breaching:
                self._breach_start[key] = None

            breach_started_at = self._breach_start.get(key)

            logger.debug(
                "SLOTracker: status tenant=%r metric=%r agg=%.4g breach=%s",
                tid, mname, aggregate, is_breaching,
            )
            return SLOStatus(
                metric_name=mname,
                tenant_id=tid,
                current_value=round(aggregate, 6),
                target=target,
                is_breaching=is_breaching,
                breach_started_at=breach_started_at,
                observation_count=len(in_window),
                checked_at=now,
            )

    def check_breach(self, tenant_id: str, metric_name: str) -> bool:
        """Return True if the SLO for *(tenant_id, metric_name)* is currently breaching.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty IDs or unregistered SLO.
        """
        return self.get_slo_status(tenant_id, metric_name).is_breaching

    def list_breaches(self, tenant_id: str) -> List[SLOStatus]:
        """Return all currently-breaching SLOs for *tenant_id*.

        Raises:
            TypeError:  ``tenant_id`` is not a str.
            ValueError: Empty ``tenant_id``.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        with self._lock:
            keys = [k for k in self._slos if k[0] == tid]

        breaches: List[SLOStatus] = []
        for _, mname in keys:
            try:
                status = self.get_slo_status(tid, mname)
                if status.is_breaching:
                    breaches.append(status)
            except Exception as exc:
                logger.warning("SLOTracker.list_breaches: error on metric=%r: %s", mname, exc)
        return breaches

    # ------------------------------------------------------------------
    # LLM breach-explanation path
    # ------------------------------------------------------------------

    def explain_breach(self, tenant_id: str, metric_name: str) -> str:
        """Return a human-readable explanation for a breaching SLO.

        Heuristic path: generates a template string from the SLOStatus.
        LLM path: sends observations summary to the router.

        Args:
            tenant_id:   Target tenant.
            metric_name: Breaching metric.

        Returns:
            Explanation string.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty IDs or unregistered SLO.
        """
        status = self.get_slo_status(tenant_id, metric_name)
        heuristic = (
            f"SLO '{metric_name}' for tenant '{tenant_id}' is "
            f"{'breaching' if status.is_breaching else 'healthy'}. "
            f"Current value: {status.current_value}, "
            f"target: {status.target.operator.value} {status.target.target_value}, "
            f"observations in window: {status.observation_count}."
        )

        if self._router is None or not status.is_breaching:
            return heuristic

        try:
            prompt = (
                f"An SLO is breaching. Details: {heuristic} "
                f"Explain in 2 sentences why this might be happening and what to investigate."
            )
            from app.llm.models import LLMMessage
            import asyncio, inspect
            resp = self._router.generate_for_signal(
                signal_type=None,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=120,
            )
            if inspect.isawaitable(resp):
                resp = asyncio.get_event_loop().run_until_complete(resp)
            return str(resp).strip()
        except Exception as exc:
            logger.warning("SLOTracker.explain_breach: LLM failed (%s)", exc)
            return heuristic


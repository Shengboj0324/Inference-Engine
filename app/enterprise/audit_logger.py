"""Tamper-evident audit logger.

Maintains a per-tenant in-memory deque of ``AuditEntry`` objects linked by a
SHA-256 hash chain.  Each new entry's ``chain_hash`` is computed as:

    chain_hash = sha256(prev_chain_hash + canonical_entry_json)

The first entry in a tenant chain uses ``"GENESIS"`` as the previous hash.
This makes offline tampering of any historical entry detectable via
``verify_chain()``.

Heuristic fast path
-------------------
All storage is in-memory (``collections.deque``).  SHA-256 is provided by
the standard-library ``hashlib`` module.  No external dependencies.

Optional LLM anomaly-detection path
------------------------------------
``detect_suspicious_patterns(tenant_id)`` accepts an optional ``llm_router``.
The heuristic path checks for: high frequency of ``PERMISSION_CHANGE`` events,
multiple CRITICAL-risk events in the last window, unusual actor diversity.
If the LLM router is provided and available, it additionally sends a compact
audit summary to the router for free-form anomaly commentary.

Thread safety
-------------
A single ``threading.RLock`` protects both ``_chains`` and ``_last_hashes``.
All public methods acquire the lock.

Adversarial-input contract
--------------------------
- ``log_event`` with a non-``AuditEntry`` → ``TypeError``
- ``query_events`` with unknown ``tenant_id`` → returns empty list (no raise)
- ``verify_chain`` on a tenant with no events → returns ``True`` (vacuously valid)
- ``export_events`` on unknown tenant → returns ``"[]"``
- Negative ``limit`` in ``query_events`` → ``ValueError``
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.enterprise.models import AuditEntry, AuditEventType, RiskLevel

logger = logging.getLogger(__name__)

_GENESIS_HASH = "GENESIS"
_DEFAULT_MAX_ENTRIES_PER_TENANT = 100_000
_SUSPICIOUS_PERMISSION_CHANGE_COUNT = 5   # per query window
_SUSPICIOUS_CRITICAL_EVENT_COUNT = 3


def _canonical_json(entry: AuditEntry) -> str:
    """Deterministic JSON representation of *entry* for hash computation."""
    payload = {
        "entry_id": entry.entry_id,
        "tenant_id": entry.tenant_id,
        "event_type": entry.event_type.value,
        "actor_id": entry.actor_id,
        "resource_id": entry.resource_id,
        "risk_level": entry.risk_level.value,
        "occurred_at": entry.occurred_at.isoformat(),
        "ip_address": entry.ip_address,
        "success": entry.success,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _compute_chain_hash(prev_hash: str, entry: AuditEntry) -> str:
    data = prev_hash + _canonical_json(entry)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class AuditLogger:
    """Tamper-evident, per-tenant audit event store.

    Args:
        max_entries_per_tenant: Maximum events kept per tenant (oldest pruned).
        llm_router:             Optional router for anomaly commentary.
    """

    def __init__(
        self,
        max_entries_per_tenant: int = _DEFAULT_MAX_ENTRIES_PER_TENANT,
        llm_router: Optional[Any] = None,
    ) -> None:
        if max_entries_per_tenant < 1:
            raise ValueError(
                f"'max_entries_per_tenant' must be ≥ 1, got {max_entries_per_tenant!r}"
            )
        self._max_per_tenant = max_entries_per_tenant
        self._router = llm_router
        self._chains: Dict[str, deque] = {}   # tenant_id → deque[AuditEntry]
        self._last_hashes: Dict[str, str] = {}  # tenant_id → current chain tip hash
        self._lock = threading.RLock()
        logger.info("AuditLogger: initialised (max_per_tenant=%d)", max_entries_per_tenant)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(self, entry: AuditEntry) -> AuditEntry:
        """Append *entry* to its tenant's audit chain.

        The entry's ``chain_hash`` is recomputed and a new frozen
        ``AuditEntry`` (with the computed hash) is stored and returned.

        Args:
            entry: ``AuditEntry`` to record.

        Returns:
            The stored ``AuditEntry`` with ``chain_hash`` set.

        Raises:
            TypeError: *entry* is not an ``AuditEntry``.
        """
        if not isinstance(entry, AuditEntry):
            raise TypeError(f"'entry' must be AuditEntry, got {type(entry)!r}")

        with self._lock:
            tid = entry.tenant_id
            if tid not in self._chains:
                self._chains[tid] = deque(maxlen=self._max_per_tenant)
                self._last_hashes[tid] = _GENESIS_HASH

            prev_hash = self._last_hashes[tid]
            computed_hash = _compute_chain_hash(prev_hash, entry)
            stored = entry.model_copy(update={"chain_hash": computed_hash})
            self._chains[tid].append(stored)
            self._last_hashes[tid] = computed_hash
            logger.debug(
                "AuditLogger: logged %s for tenant=%r (hash=%s…)",
                entry.event_type.value, tid, computed_hash[:12],
            )
            return stored

    def query_events(
        self,
        tenant_id: str,
        event_type: Optional[AuditEventType] = None,
        actor_id: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
        success_only: Optional[bool] = None,
    ) -> List[AuditEntry]:
        """Filter and return audit entries for *tenant_id*.

        Args:
            tenant_id:   Target tenant.
            event_type:  If set, filter by this event type.
            actor_id:    If set, filter by actor.
            risk_level:  If set, filter by risk level.
            start:       Inclusive start of time range.
            end:         Inclusive end of time range.
            limit:       Maximum entries to return (≥ 1).
            success_only: If True/False, filter by success flag.

        Returns:
            Matching ``AuditEntry`` list, newest-first.

        Raises:
            TypeError:  ``tenant_id`` is not a str.
            ValueError: ``limit`` < 1 or ``tenant_id`` is empty.
        """
        if not isinstance(tenant_id, str):
            raise TypeError(f"'tenant_id' must be str, got {type(tenant_id)!r}")
        if not tenant_id.strip():
            raise ValueError("'tenant_id' must be non-empty")
        if limit < 1:
            raise ValueError(f"'limit' must be ≥ 1, got {limit!r}")

        with self._lock:
            chain = list(self._chains.get(tenant_id.strip(), []))

        # Apply filters (newest-first)
        result = []
        for entry in reversed(chain):
            if event_type is not None and entry.event_type != event_type:
                continue
            if actor_id is not None and entry.actor_id != actor_id:
                continue
            if risk_level is not None and entry.risk_level != risk_level:
                continue
            if start is not None and entry.occurred_at < start:
                continue
            if end is not None and entry.occurred_at > end:
                continue
            if success_only is not None and entry.success != success_only:
                continue
            result.append(entry)
            if len(result) >= limit:
                break
        return result

    def verify_chain(self, tenant_id: str) -> bool:
        """Verify that the hash chain for *tenant_id* has not been tampered with.

        Recomputes every hash from GENESIS and checks it matches the stored
        ``chain_hash`` on each entry.

        Args:
            tenant_id: Target tenant.

        Returns:
            ``True`` if the chain is intact (or the tenant has no events).

        Raises:
            TypeError:  ``tenant_id`` is not a str.
            ValueError: ``tenant_id`` is empty.
        """
        if not isinstance(tenant_id, str):
            raise TypeError(f"'tenant_id' must be str, got {type(tenant_id)!r}")
        if not tenant_id.strip():
            raise ValueError("'tenant_id' must be non-empty")

        with self._lock:
            chain = list(self._chains.get(tenant_id.strip(), []))

        if not chain:
            return True

        prev_hash = _GENESIS_HASH
        for entry in chain:
            expected = _compute_chain_hash(prev_hash, entry)
            if entry.chain_hash != expected:
                logger.warning(
                    "AuditLogger.verify_chain: tamper detected at entry_id=%r", entry.entry_id
                )
                return False
            prev_hash = entry.chain_hash
        return True

    def export_events(self, tenant_id: str) -> str:
        """Return all events for *tenant_id* as a JSON array string.

        Args:
            tenant_id: Target tenant.

        Returns:
            JSON string (``"[]"`` if the tenant has no events).

        Raises:
            TypeError:  ``tenant_id`` is not a str.
            ValueError: Empty ``tenant_id``.
        """
        if not isinstance(tenant_id, str):
            raise TypeError(f"'tenant_id' must be str, got {type(tenant_id)!r}")
        if not tenant_id.strip():
            raise ValueError("'tenant_id' must be non-empty")

        with self._lock:
            chain = list(self._chains.get(tenant_id.strip(), []))

        records = [e.model_dump(mode="json") for e in chain]
        return json.dumps(records, default=str)

    def get_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Return event counts grouped by type and risk level for *tenant_id*.

        Args:
            tenant_id: Target tenant.

        Returns:
            Dict with keys: ``total_events``, ``by_event_type``,
            ``by_risk_level``, ``failed_events``.

        Raises:
            TypeError:  ``tenant_id`` is not a str.
            ValueError: Empty ``tenant_id``.
        """
        if not isinstance(tenant_id, str):
            raise TypeError(f"'tenant_id' must be str, got {type(tenant_id)!r}")
        if not tenant_id.strip():
            raise ValueError("'tenant_id' must be non-empty")

        with self._lock:
            chain = list(self._chains.get(tenant_id.strip(), []))

        by_type: Dict[str, int] = {}
        by_risk: Dict[str, int] = {}
        failed = 0
        for entry in chain:
            by_type[entry.event_type.value] = by_type.get(entry.event_type.value, 0) + 1
            by_risk[entry.risk_level.value] = by_risk.get(entry.risk_level.value, 0) + 1
            if not entry.success:
                failed += 1

        return {
            "total_events": len(chain),
            "by_event_type": by_type,
            "by_risk_level": by_risk,
            "failed_events": failed,
        }

    def detect_suspicious_patterns(
        self, tenant_id: str, window_size: int = 50
    ) -> Dict[str, Any]:
        """Identify suspicious patterns in the most recent *window_size* events.

        Heuristic checks:
        - High ``PERMISSION_CHANGE`` frequency (≥ threshold).
        - Multiple CRITICAL-risk events.
        - High actor diversity (many different actors in a small window).

        LLM path (optional): asks the router to comment on the pattern summary.

        Args:
            tenant_id:   Target tenant.
            window_size: Number of recent events to inspect (≥ 1).

        Returns:
            Dict with keys: ``suspicious``, ``reasons`` (list), ``llm_comment``.

        Raises:
            TypeError:  ``tenant_id`` is not a str.
            ValueError: Empty ``tenant_id`` or ``window_size`` < 1.
        """
        if not isinstance(tenant_id, str):
            raise TypeError(f"'tenant_id' must be str, got {type(tenant_id)!r}")
        if not tenant_id.strip():
            raise ValueError("'tenant_id' must be non-empty")
        if window_size < 1:
            raise ValueError(f"'window_size' must be ≥ 1, got {window_size!r}")

        recent = self.query_events(tenant_id, limit=window_size)
        reasons: List[str] = []

        perm_changes = sum(1 for e in recent if e.event_type == AuditEventType.PERMISSION_CHANGE)
        if perm_changes >= _SUSPICIOUS_PERMISSION_CHANGE_COUNT:
            reasons.append(f"High permission-change frequency: {perm_changes} in last {window_size}")

        critical_count = sum(1 for e in recent if e.risk_level == RiskLevel.CRITICAL)
        if critical_count >= _SUSPICIOUS_CRITICAL_EVENT_COUNT:
            reasons.append(f"Multiple CRITICAL-risk events: {critical_count}")

        unique_actors = len({e.actor_id for e in recent if e.actor_id})
        if recent and unique_actors > max(3, window_size // 5):
            reasons.append(f"Unusual actor diversity: {unique_actors} distinct actors")

        suspicious = bool(reasons)
        llm_comment = ""

        if self._router is not None and recent:
            try:
                llm_comment = self._llm_anomaly_comment(tenant_id, recent[:10], reasons)
            except Exception as exc:
                logger.warning("AuditLogger: LLM anomaly comment failed (%s)", exc)

        logger.debug(
            "AuditLogger.detect_suspicious_patterns: tenant=%r suspicious=%s reasons=%d",
            tenant_id, suspicious, len(reasons),
        )
        return {"suspicious": suspicious, "reasons": reasons, "llm_comment": llm_comment}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _llm_anomaly_comment(
        self,
        tenant_id: str,
        recent_entries: List[AuditEntry],
        heuristic_reasons: List[str],
    ) -> str:
        """Ask the LLM router for free-form anomaly commentary."""
        summary = (
            f"Tenant: {tenant_id}. "
            f"Recent events: {[e.event_type.value for e in recent_entries]}. "
            f"Heuristic flags: {heuristic_reasons}."
        )
        prompt = (
            f"You are a security analyst. Review the following audit summary and "
            f"comment briefly on any suspicious patterns:\n{summary}\n"
            f"Reply in at most 2 sentences."
        )
        from app.llm.models import LLMMessage
        import asyncio, inspect
        resp = self._router.generate_for_signal(
            signal_type=None,
            messages=[LLMMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=100,
        )
        if inspect.isawaitable(resp):
            resp = asyncio.get_event_loop().run_until_complete(resp)
        return str(resp).strip()


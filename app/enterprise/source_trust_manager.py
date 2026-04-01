"""Source trust manager — per-tenant trust-score policy enforcement.

Applies a ``SourceTrustPolicy`` to compute an *effective* trust score for any
(tenant, source) pair, taking into account:

1. Dynamic block/allowlist (set at runtime, supplements the stored policy).
2. Per-source score multipliers from the policy.
3. A global floor/ceiling clamp applied last.

Effective trust computation
---------------------------
For source ``source_id`` with base score ``base_score``:

    if source_id in blocklist (policy OR dynamic):
        effective = 0.0
    elif source_id in allowlist:
        effective = policy.global_trust_ceiling
    else:
        multiplier = policy.source_multipliers.get(source_id, 1.0)
        effective = clamp(base_score * multiplier,
                          policy.global_trust_floor,
                          policy.global_trust_ceiling)

If no policy is set for a tenant, base_score is returned unchanged (pass-through).

Heuristic fast path
-------------------
All policy data lives in in-memory dicts.  No external dependencies.

Optional LLM policy-recommendation path
-----------------------------------------
``recommend_policy_adjustment(tenant_id, source_metadata)`` uses the LLM
router to suggest trust-policy parameter changes.  The heuristic path uses
simple metadata-based rules (domain age, official flag, error_rate).

Thread safety
-------------
``threading.RLock`` protects ``_policies`` and ``_dynamic_blocklists``.

Adversarial-input contract
--------------------------
- ``tenant_id`` empty/non-string → TypeError/ValueError
- ``source_id`` empty/non-string → TypeError/ValueError
- ``base_score`` outside [0, 1] → ValueError
- ``set_policy`` with mismatched ``tenant_id`` → ValueError
- ``get_policy`` for unknown tenant → returns None (no raise)
- ``unblock_source`` for source not in blocklist → no-op (no raise)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional, Set

from app.enterprise.models import SourceTrustPolicy

logger = logging.getLogger(__name__)

_DEFAULT_PASS_THROUGH = True  # return base_score unchanged when no policy exists


def _validate_str_id(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"'{name}' must be a str, got {type(value)!r}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"'{name}' must be a non-empty, non-whitespace string")
    return stripped


class SourceTrustManager:
    """Applies per-tenant trust policies to source trust scores.

    Args:
        llm_router:    Optional LLM router for policy recommendations.
        pass_through:  When True and no policy exists for a tenant, return
                       ``base_score`` unchanged.  When False, return 0.0.
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        pass_through: bool = _DEFAULT_PASS_THROUGH,
    ) -> None:
        self._policies: Dict[str, SourceTrustPolicy] = {}
        self._dynamic_blocklists: Dict[str, Set[str]] = {}
        self._lock = threading.RLock()
        self._router = llm_router
        self._pass_through = pass_through
        logger.info("SourceTrustManager: initialised (pass_through=%s)", pass_through)

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def set_policy(self, tenant_id: str, policy: SourceTrustPolicy) -> None:
        """Store a trust policy for *tenant_id*.

        Args:
            tenant_id: Target tenant.
            policy:    ``SourceTrustPolicy`` instance.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: ``tenant_id`` is empty, or ``policy.tenant_id`` differs.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        if not isinstance(policy, SourceTrustPolicy):
            raise TypeError(f"'policy' must be SourceTrustPolicy, got {type(policy)!r}")
        if policy.tenant_id != tid:
            raise ValueError(
                f"policy.tenant_id={policy.tenant_id!r} does not match tenant_id={tid!r}"
            )
        with self._lock:
            self._policies[tid] = policy
            logger.info("SourceTrustManager: policy set for tenant_id=%r", tid)

    def get_policy(self, tenant_id: str) -> Optional[SourceTrustPolicy]:
        """Return the ``SourceTrustPolicy`` for *tenant_id*, or None.

        Raises:
            TypeError:  ``tenant_id`` is not a str.
            ValueError: ``tenant_id`` is empty.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        with self._lock:
            return self._policies.get(tid)

    def reset_policy(self, tenant_id: str) -> None:
        """Remove the trust policy for *tenant_id* (resets to pass-through).

        Raises:
            TypeError:  ``tenant_id`` is not a str.
            ValueError: ``tenant_id`` is empty.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        with self._lock:
            self._policies.pop(tid, None)
            logger.info("SourceTrustManager: policy reset for tenant_id=%r", tid)

    # ------------------------------------------------------------------
    # Effective trust computation
    # ------------------------------------------------------------------

    def get_effective_trust(
        self,
        tenant_id: str,
        source_id: str,
        base_score: float,
    ) -> float:
        """Compute effective trust score for *(tenant_id, source_id)*.

        Args:
            tenant_id:  Target tenant.
            source_id:  Source being evaluated.
            base_score: Raw trust score from the source connector [0, 1].

        Returns:
            Effective trust score in [0, 1].

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty IDs, or ``base_score`` outside [0, 1].
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        sid = _validate_str_id(source_id, "source_id")
        if not isinstance(base_score, (int, float)):
            raise TypeError(f"'base_score' must be a float, got {type(base_score)!r}")
        if not (0.0 <= base_score <= 1.0):
            raise ValueError(f"'base_score' must be in [0, 1], got {base_score!r}")

        with self._lock:
            policy = self._policies.get(tid)
            dynamic_block = self._dynamic_blocklists.get(tid, set())

        # Dynamic blocklist is always checked, regardless of whether a policy exists.
        if sid in dynamic_block:
            logger.debug("SourceTrustManager: source %r dynamically blocked for tenant %r", sid, tid)
            return 0.0

        if policy is None:
            return base_score if self._pass_through else 0.0

        # Policy blocklist?
        if sid in policy.blocklisted_source_ids:
            logger.debug("SourceTrustManager: source %r policy-blocked for tenant %r", sid, tid)
            return 0.0

        # Allowlisted?
        if sid in policy.allowlisted_source_ids:
            return policy.global_trust_ceiling

        # Apply multiplier then clamp
        multiplier = policy.source_multipliers.get(sid, 1.0)
        adjusted = base_score * multiplier
        effective = min(policy.global_trust_ceiling, max(policy.global_trust_floor, adjusted))
        logger.debug(
            "SourceTrustManager: tenant=%r source=%r base=%.3f → effective=%.3f",
            tid, sid, base_score, effective,
        )
        return round(effective, 6)

    # ------------------------------------------------------------------
    # Dynamic blocklist management
    # ------------------------------------------------------------------

    def block_source(self, tenant_id: str, source_id: str, reason: str = "") -> None:
        """Add *source_id* to the dynamic blocklist for *tenant_id*.

        Args:
            tenant_id: Target tenant.
            source_id: Source to block.
            reason:    Optional human-readable reason for audit logs.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty IDs.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        sid = _validate_str_id(source_id, "source_id")
        with self._lock:
            if tid not in self._dynamic_blocklists:
                self._dynamic_blocklists[tid] = set()
            self._dynamic_blocklists[tid].add(sid)
        logger.info(
            "SourceTrustManager: blocked source=%r for tenant=%r reason=%r", sid, tid, reason
        )

    def unblock_source(self, tenant_id: str, source_id: str) -> None:
        """Remove *source_id* from the dynamic blocklist (no-op if not present).

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty IDs.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        sid = _validate_str_id(source_id, "source_id")
        with self._lock:
            if tid in self._dynamic_blocklists:
                self._dynamic_blocklists[tid].discard(sid)
        logger.debug("SourceTrustManager: unblocked source=%r for tenant=%r", sid, tid)

    def is_source_blocked(self, tenant_id: str, source_id: str) -> bool:
        """Return True if *source_id* is blocked (policy OR dynamic) for *tenant_id*.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty IDs.
        """
        tid = _validate_str_id(tenant_id, "tenant_id")
        sid = _validate_str_id(source_id, "source_id")
        with self._lock:
            policy = self._policies.get(tid)
            dynamic = self._dynamic_blocklists.get(tid, set())
        in_policy = policy is not None and sid in policy.blocklisted_source_ids
        return in_policy or sid in dynamic

    # ------------------------------------------------------------------
    # LLM policy-recommendation path
    # ------------------------------------------------------------------

    def recommend_policy_adjustment(
        self,
        tenant_id: str,
        source_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Recommend trust-policy adjustments for *tenant_id* based on metadata.

        Heuristic fast path: uses ``official`` flag and ``error_rate`` field
        from *source_metadata* to suggest floor/ceiling adjustments.
        LLM path: sends metadata summary to router for recommendations.

        Args:
            tenant_id:       Target tenant.
            source_metadata: Dict with optional keys: ``official`` (bool),
                             ``error_rate`` (float 0–1), ``domain_age_days`` (int).

        Returns:
            Dict with ``suggested_floor``, ``suggested_ceiling``,
            ``rationale`` (str), ``llm_enhanced`` (bool).

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty ``tenant_id``.
        """
        _validate_str_id(tenant_id, "tenant_id")
        if not isinstance(source_metadata, dict):
            raise TypeError(f"'source_metadata' must be a dict, got {type(source_metadata)!r}")

        result = self._heuristic_recommendation(source_metadata)
        result["llm_enhanced"] = False

        if self._router is not None:
            try:
                import json
                prompt = (
                    f"Given this source metadata: {json.dumps(source_metadata)}, "
                    f"suggest trust floor and ceiling values (0.0–1.0) for a trust policy. "
                    f"Reply as JSON: {{\"floor\": float, \"ceiling\": float, \"rationale\": str}}"
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
                data = json.loads(resp)
                result["suggested_floor"] = float(data.get("floor", result["suggested_floor"]))
                result["suggested_ceiling"] = float(data.get("ceiling", result["suggested_ceiling"]))
                result["rationale"] = str(data.get("rationale", result["rationale"]))
                result["llm_enhanced"] = True
            except Exception as exc:
                logger.warning("SourceTrustManager: LLM recommendation failed (%s)", exc)

        return result

    @staticmethod
    def _heuristic_recommendation(meta: Dict[str, Any]) -> Dict[str, Any]:
        official = bool(meta.get("official", False))
        error_rate = float(meta.get("error_rate", 0.0))
        # domain_age_days is only applied when explicitly provided in meta
        domain_age = meta.get("domain_age_days", None)

        floor = 0.0
        ceiling = 1.0
        notes = []

        if official:
            floor = max(floor, 0.5)
            notes.append("official source → floor raised to 0.5")
        if error_rate > 0.2:
            ceiling = min(ceiling, 0.6)
            notes.append(f"high error_rate ({error_rate:.0%}) → ceiling lowered to 0.6")
        if domain_age is not None and int(domain_age) < 30:
            ceiling = min(ceiling, 0.7)
            notes.append(f"new domain ({int(domain_age)}d) → ceiling lowered to 0.7")

        return {
            "suggested_floor": round(floor, 3),
            "suggested_ceiling": round(ceiling, 3),
            "rationale": "; ".join(notes) or "no adjustments suggested",
        }


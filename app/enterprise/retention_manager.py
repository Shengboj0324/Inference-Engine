"""Retention manager — per-tenant data lifecycle policy enforcement.

Computes whether individual data records have exceeded their configured
retention period and produces ``PurgeResult`` objects that indicate which
records are eligible for deletion.  Actual deletion is the caller's
responsibility; this module is intentionally side-effect-free with respect
to any external store.

Retention-period resolution order
-----------------------------------
1. ``RetentionPolicy.per_class_retention[data_class]``  (per-class override)
2. ``RetentionPolicy.default_retention_days``            (tenant-level default)
3. Global fallback ``_DEFAULT_RETENTION_DAYS``           (no policy set)

Legal hold
----------
When ``RetentionPolicy.legal_hold`` is True, ``purge_eligible`` returns a
``PurgeResult`` with ``records_purged=0`` and ``legal_hold_active=True``
regardless of record age.  ``check_expired`` still returns the
age-based boolean so callers can reason about what *would* be purged.

Heuristic fast path
-------------------
Pure ``datetime`` arithmetic.  No external dependencies.

Optional LLM policy-suggestion path
-------------------------------------
``suggest_policy(tenant_id, data_class, context)`` uses the LLM router to
recommend retention durations.  Heuristic path: static lookup table keyed by
``DataClass``.

Thread safety
-------------
``threading.RLock`` protects ``_policies``.

Adversarial-input contract
--------------------------
- ``tenant_id`` empty/non-str → TypeError/ValueError
- ``data_class`` not a ``DataClass`` → TypeError
- ``created_at`` not timezone-aware → ValueError
- ``purge_eligible`` with non-list records → TypeError
- ``records`` elements not 3-tuples → ValueError
- ``get_policy`` for unknown tenant → raises ``KeyError``
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.enterprise.models import DataClass, PurgeResult, RetentionPolicy

logger = logging.getLogger(__name__)

_DEFAULT_RETENTION_DAYS: int = 365

# Heuristic default retention periods per data class (days)
_CLASS_DEFAULTS: Dict[DataClass, int] = {
    DataClass.AUDIT_LOG:        2555,   # 7 years (regulatory)
    DataClass.CREDENTIAL:       90,
    DataClass.EMBEDDING:        180,
    DataClass.SUMMARY:          365,
    DataClass.CONTENT_ITEM:     730,
    DataClass.USER_FEEDBACK:    365,
    DataClass.SOURCE_METADATA:  365,
}


def _validate_tenant_id(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"'tenant_id' must be str, got {type(value)!r}")
    stripped = value.strip()
    if not stripped:
        raise ValueError("'tenant_id' must be non-empty")
    return stripped


class RetentionManager:
    """Evaluates and enforces per-tenant data-retention policies.

    Args:
        llm_router:               Optional router for policy suggestions.
        default_retention_days:   Global fallback when no policy exists (≥ 1).
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        default_retention_days: int = _DEFAULT_RETENTION_DAYS,
    ) -> None:
        if default_retention_days < 1:
            raise ValueError(
                f"'default_retention_days' must be ≥ 1, got {default_retention_days!r}"
            )
        self._policies: Dict[str, RetentionPolicy] = {}
        self._lock = threading.RLock()
        self._router = llm_router
        self._default_days = default_retention_days
        logger.info("RetentionManager: initialised (default=%d days)", default_retention_days)

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def set_policy(self, tenant_id: str, policy: RetentionPolicy) -> None:
        """Store a retention policy for *tenant_id*.

        Args:
            tenant_id: Target tenant.
            policy:    ``RetentionPolicy`` instance.

        Raises:
            TypeError:  Wrong types.
            ValueError: Empty ``tenant_id``, or ``policy.tenant_id`` mismatch.
        """
        tid = _validate_tenant_id(tenant_id)
        if not isinstance(policy, RetentionPolicy):
            raise TypeError(f"'policy' must be RetentionPolicy, got {type(policy)!r}")
        if policy.tenant_id != tid:
            raise ValueError(
                f"policy.tenant_id={policy.tenant_id!r} does not match tenant_id={tid!r}"
            )
        with self._lock:
            self._policies[tid] = policy
        logger.info("RetentionManager: policy set for tenant_id=%r (default=%dd)", tid,
                    policy.default_retention_days)

    def get_policy(self, tenant_id: str) -> RetentionPolicy:
        """Return the ``RetentionPolicy`` for *tenant_id*.

        Raises:
            TypeError:  Wrong type.
            ValueError: Empty ``tenant_id``.
            KeyError:   No policy configured for *tenant_id*.
        """
        tid = _validate_tenant_id(tenant_id)
        with self._lock:
            if tid not in self._policies:
                raise KeyError(f"No retention policy found for tenant '{tid}'")
            return self._policies[tid]

    def get_retention_days(self, tenant_id: str, data_class: DataClass) -> int:
        """Return effective retention days for *(tenant_id, data_class)*.

        Resolution: per-class override → tenant default → global fallback.

        Raises:
            TypeError:  Wrong types.
            ValueError: Empty ``tenant_id``.
        """
        tid = _validate_tenant_id(tenant_id)
        if not isinstance(data_class, DataClass):
            raise TypeError(f"'data_class' must be DataClass, got {type(data_class)!r}")
        with self._lock:
            policy = self._policies.get(tid)
        if policy is None:
            # No policy → use the constructor default (caller's explicit intent).
            # _CLASS_DEFAULTS are used only as suggestions in suggest_policy().
            return self._default_days
        override = policy.per_class_retention.get(data_class.value)
        return override if override is not None else policy.default_retention_days

    # ------------------------------------------------------------------
    # Expiry checking
    # ------------------------------------------------------------------

    def check_expired(
        self, tenant_id: str, data_class: DataClass, created_at: datetime
    ) -> bool:
        """Return True if a record created at *created_at* has exceeded its retention.

        Args:
            tenant_id:   Target tenant.
            data_class:  Data class of the record.
            created_at:  UTC-aware creation timestamp of the record.

        Returns:
            True if the record is past its retention date.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty ``tenant_id``, or ``created_at`` is not timezone-aware.
        """
        tid = _validate_tenant_id(tenant_id)
        if not isinstance(data_class, DataClass):
            raise TypeError(f"'data_class' must be DataClass, got {type(data_class)!r}")
        if not isinstance(created_at, datetime):
            raise TypeError(f"'created_at' must be a datetime, got {type(created_at)!r}")
        if created_at.tzinfo is None:
            raise ValueError("'created_at' must be timezone-aware (tzinfo must not be None)")

        retention_days = self.get_retention_days(tid, data_class)
        now = datetime.now(timezone.utc)
        age_days = (now - created_at).total_seconds() / 86_400
        expired = age_days > retention_days
        logger.debug(
            "check_expired: tenant=%r class=%s age=%.1fd retention=%dd → %s",
            tid, data_class.value, age_days, retention_days, expired,
        )
        return expired

    def purge_eligible(
        self,
        tenant_id: str,
        records: List[Tuple[str, DataClass, datetime]],
    ) -> PurgeResult:
        """Identify which records in *records* are eligible for purging.

        Each element of *records* is a 3-tuple:
            (record_id: str, data_class: DataClass, created_at: datetime)

        Legal hold trumps all: if the tenant policy has ``legal_hold=True``,
        zero records are purged regardless of age.

        Args:
            tenant_id: Target tenant.
            records:   List of record descriptors.

        Returns:
            ``PurgeResult`` with counts and metadata.

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty ``tenant_id``, or malformed record tuples.
        """
        tid = _validate_tenant_id(tenant_id)
        if not isinstance(records, list):
            raise TypeError(f"'records' must be a list, got {type(records)!r}")

        with self._lock:
            policy = self._policies.get(tid)

        legal_hold = policy is not None and policy.legal_hold
        policy_id = policy.policy_id if policy else ""

        if legal_hold:
            logger.info(
                "RetentionManager.purge_eligible: legal hold active for tenant=%r", tid
            )
            return PurgeResult(
                tenant_id=tid,
                data_class=DataClass.CONTENT_ITEM,   # generic placeholder
                records_checked=len(records),
                records_purged=0,
                legal_hold_active=True,
                policy_id=policy_id,
            )

        purged = 0
        data_class_used: Optional[DataClass] = None
        for idx, rec in enumerate(records):
            if not (isinstance(rec, (tuple, list)) and len(rec) == 3):
                raise ValueError(
                    f"records[{idx}] must be a 3-tuple (record_id, DataClass, datetime), "
                    f"got {rec!r}"
                )
            _, dc, created_at = rec
            if not isinstance(dc, DataClass):
                raise TypeError(
                    f"records[{idx}][1] must be DataClass, got {type(dc)!r}"
                )
            if data_class_used is None:
                data_class_used = dc
            if self.check_expired(tid, dc, created_at):
                purged += 1

        return PurgeResult(
            tenant_id=tid,
            data_class=data_class_used or DataClass.CONTENT_ITEM,
            records_checked=len(records),
            records_purged=purged,
            legal_hold_active=False,
            policy_id=policy_id,
        )

    # ------------------------------------------------------------------
    # LLM policy-suggestion path
    # ------------------------------------------------------------------

    def suggest_policy(
        self,
        tenant_id: str,
        data_class: DataClass,
        context: str = "",
    ) -> Dict[str, Any]:
        """Suggest a retention duration for *(tenant_id, data_class)*.

        Heuristic: returns the ``_CLASS_DEFAULTS`` value.
        LLM path: sends context to router for a recommendation.

        Args:
            tenant_id:  Target tenant (validated but not required to have a policy).
            data_class: Data class to recommend for.
            context:    Optional free-text context (e.g. regulatory requirement).

        Returns:
            Dict with ``suggested_days`` (int) and ``rationale`` (str).

        Raises:
            TypeError:  Wrong argument types.
            ValueError: Empty ``tenant_id``.
        """
        _validate_tenant_id(tenant_id)
        if not isinstance(data_class, DataClass):
            raise TypeError(f"'data_class' must be DataClass, got {type(data_class)!r}")

        default_days = _CLASS_DEFAULTS.get(data_class, self._default_days)
        result: Dict[str, Any] = {
            "suggested_days": default_days,
            "rationale": f"Heuristic default for {data_class.value}",
            "llm_enhanced": False,
        }

        if self._router is not None:
            try:
                import json
                prompt = (
                    f"What is an appropriate data retention period in days for "
                    f"'{data_class.value}' data? Context: {context or 'general SaaS'}. "
                    f"Reply ONLY with JSON: {{\"days\": int, \"rationale\": str}}"
                )
                from app.llm.models import LLMMessage
                import asyncio, inspect
                resp = self._router.generate_for_signal(
                    signal_type=None,
                    messages=[LLMMessage(role="user", content=prompt)],
                    temperature=0.0,
                    max_tokens=80,
                )
                if inspect.isawaitable(resp):
                    resp = asyncio.get_event_loop().run_until_complete(resp)
                data = json.loads(resp)
                result["suggested_days"] = int(data.get("days", default_days))
                result["rationale"] = str(data.get("rationale", result["rationale"]))
                result["llm_enhanced"] = True
            except Exception as exc:
                logger.warning("RetentionManager.suggest_policy: LLM failed (%s)", exc)

        return result


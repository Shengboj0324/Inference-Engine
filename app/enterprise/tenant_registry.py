"""Tenant registry — provision, deprovision, and inspect tenants.

Provides in-memory, thread-safe tenant lifecycle management with
per-tier resource caps and an optional LLM router path for tier suggestion.

Heuristic fast path
-------------------
All state is stored in a ``dict[str, TenantConfig]`` protected by a
``threading.RLock``.  Tier-based resource defaults are applied automatically
on provisioning; callers may override any field after provisioning via
``update_tenant``.

Optional LLM tier-suggestion path
----------------------------------
``suggest_tier(usage_stats, llm_router)`` accepts an optional LLM router.
When provided it sends a compact usage summary prompt and parses the response
as a ``TenantTier`` value.  If the router is absent or the call fails, a
deterministic heuristic (daily_api_calls thresholds) is used instead.

Thread safety
-------------
Every mutating operation acquires ``self._lock`` (``threading.RLock``).
Read-only operations also acquire the lock to avoid torn reads.

Adversarial-input contract
--------------------------
- ``tenant_id`` that is empty or whitespace-only → ``ValueError``
- ``tenant_id`` that is not a ``str`` → ``TypeError``
- Provisioning a ``tenant_id`` that already exists → ``ValueError``
- ``get_tenant`` / ``deprovision_tenant`` on unknown ID → ``KeyError``
- ``update_tenant`` with unknown field names silently ignores them to
  remain forward-compatible with future ``TenantConfig`` expansions.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from app.enterprise.models import DataClass, TenantConfig, TenantTier

logger = logging.getLogger(__name__)

# Per-tier default resource caps
_TIER_DEFAULTS: Dict[TenantTier, Dict[str, Any]] = {
    TenantTier.FREE: {
        "max_sources": 10,
        "max_daily_api_calls": 1_000,
        "allowed_data_classes": [DataClass.CONTENT_ITEM, DataClass.SUMMARY],
    },
    TenantTier.STANDARD: {
        "max_sources": 100,
        "max_daily_api_calls": 50_000,
        "allowed_data_classes": [
            DataClass.CONTENT_ITEM, DataClass.SUMMARY,
            DataClass.USER_FEEDBACK, DataClass.EMBEDDING,
        ],
    },
    TenantTier.ENTERPRISE: {
        "max_sources": 1_000,
        "max_daily_api_calls": 500_000,
        "allowed_data_classes": list(DataClass),
    },
    TenantTier.PLATFORM: {
        "max_sources": 100_000,
        "max_daily_api_calls": 10_000_000,
        "allowed_data_classes": list(DataClass),
    },
}

# LLM tier-suggestion thresholds (heuristic fallback)
_TIER_CALL_THRESHOLDS = [
    (500_000, TenantTier.PLATFORM),
    (50_000,  TenantTier.ENTERPRISE),
    (1_000,   TenantTier.STANDARD),
    (0,       TenantTier.FREE),
]


def _validate_tenant_id(tenant_id: Any) -> str:
    """Validate and normalise *tenant_id*; raise on bad input."""
    if not isinstance(tenant_id, str):
        raise TypeError(f"'tenant_id' must be a str, got {type(tenant_id)!r}")
    stripped = tenant_id.strip()
    if not stripped:
        raise ValueError("'tenant_id' must be a non-empty, non-whitespace string")
    return stripped


class TenantRegistry:
    """Lifecycle manager for tenant configurations.

    Args:
        llm_router: Optional LLM router for ``suggest_tier`` calls.
        max_tenants: Maximum number of tenants (0 = unlimited).
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        max_tenants: int = 0,
    ) -> None:
        if max_tenants < 0:
            raise ValueError(f"'max_tenants' must be ≥ 0, got {max_tenants!r}")
        self._tenants: Dict[str, TenantConfig] = {}
        self._lock = threading.RLock()
        self._router = llm_router
        self._max_tenants = max_tenants

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def provision_tenant(
        self,
        tenant_id: str,
        tier: TenantTier = TenantTier.FREE,
        display_name: str = "",
        data_residency_region: str = "us-east-1",
        feature_flags: Optional[Dict[str, bool]] = None,
        **overrides: Any,
    ) -> TenantConfig:
        """Provision a new tenant with tier-appropriate defaults.

        Args:
            tenant_id:             Unique tenant identifier.
            tier:                  Service tier.
            display_name:          Human-readable name.
            data_residency_region: Cloud region for data residency.
            feature_flags:         Per-tenant feature toggles.
            **overrides:           Field overrides (e.g. ``max_sources=500``).

        Returns:
            The newly created ``TenantConfig``.

        Raises:
            TypeError:  ``tenant_id`` is not a string.
            ValueError: ``tenant_id`` is empty, whitespace, or already exists;
                        or ``max_tenants`` limit would be exceeded.
        """
        tid = _validate_tenant_id(tenant_id)
        if not isinstance(tier, TenantTier):
            raise TypeError(f"'tier' must be a TenantTier, got {type(tier)!r}")

        with self._lock:
            if tid in self._tenants:
                raise ValueError(f"Tenant '{tid}' is already provisioned")
            if self._max_tenants > 0 and len(self._tenants) >= self._max_tenants:
                raise ValueError(
                    f"Maximum tenant limit ({self._max_tenants}) reached"
                )
            defaults = dict(_TIER_DEFAULTS[tier])
            defaults.update(overrides)
            config = TenantConfig(
                tenant_id=tid,
                tier=tier,
                display_name=display_name or tid,
                data_residency_region=data_residency_region,
                feature_flags=feature_flags or {},
                **defaults,
            )
            self._tenants[tid] = config
            logger.info("TenantRegistry: provisioned tenant_id=%r tier=%s", tid, tier.value)
            return config

    def deprovision_tenant(self, tenant_id: str) -> TenantConfig:
        """Soft-deactivate a tenant (marks ``is_active=False``).

        Args:
            tenant_id: Tenant to deprovision.

        Returns:
            Updated (inactive) ``TenantConfig``.

        Raises:
            TypeError:  Wrong type.
            ValueError: Empty/whitespace ID.
            KeyError:   Tenant not found.
        """
        tid = _validate_tenant_id(tenant_id)
        with self._lock:
            if tid not in self._tenants:
                raise KeyError(f"Tenant '{tid}' not found")
            updated = self._tenants[tid].model_copy(update={"is_active": False})
            self._tenants[tid] = updated
            logger.info("TenantRegistry: deprovisioned tenant_id=%r", tid)
            return updated

    def get_tenant(self, tenant_id: str) -> TenantConfig:
        """Return the ``TenantConfig`` for *tenant_id*.

        Raises:
            TypeError: Wrong type.
            ValueError: Empty ID.
            KeyError:  Tenant not found.
        """
        tid = _validate_tenant_id(tenant_id)
        with self._lock:
            if tid not in self._tenants:
                raise KeyError(f"Tenant '{tid}' not found")
            return self._tenants[tid]

    def list_tenants(self, active_only: bool = False) -> List[TenantConfig]:
        """Return all tenant configurations.

        Args:
            active_only: If True, return only active tenants.

        Returns:
            List of ``TenantConfig`` sorted by ``tenant_id``.
        """
        with self._lock:
            configs = list(self._tenants.values())
        if active_only:
            configs = [c for c in configs if c.is_active]
        return sorted(configs, key=lambda c: c.tenant_id)

    def update_tenant(self, tenant_id: str, **updates: Any) -> TenantConfig:
        """Apply field updates to an existing tenant configuration.

        Only fields present on ``TenantConfig`` are applied; unknown keys are
        ignored so callers remain forward-compatible.

        Args:
            tenant_id: Target tenant.
            **updates: Field-value pairs to update.

        Returns:
            Updated ``TenantConfig``.

        Raises:
            TypeError:  Wrong type for *tenant_id*.
            ValueError: Empty ID.
            KeyError:   Tenant not found.
        """
        tid = _validate_tenant_id(tenant_id)
        with self._lock:
            if tid not in self._tenants:
                raise KeyError(f"Tenant '{tid}' not found")
            valid_fields = TenantConfig.model_fields.keys()
            clean_updates = {k: v for k, v in updates.items() if k in valid_fields}
            updated = self._tenants[tid].model_copy(update=clean_updates)
            self._tenants[tid] = updated
            logger.debug("TenantRegistry: updated tenant_id=%r fields=%s", tid, list(clean_updates))
            return updated

    def suggest_tier(
        self,
        tenant_id: str,
        usage_stats: Dict[str, Any],
    ) -> TenantTier:
        """Return the recommended ``TenantTier`` for *tenant_id* based on usage.

        Heuristic fast path: compares ``usage_stats["daily_api_calls"]``
        against hard-coded thresholds.  LLM path: asks the router to classify
        the usage into one of the four tiers; falls back to heuristic on error.

        Args:
            tenant_id:   Tenant identifier (for logging; need not exist).
            usage_stats: Dict with at least ``"daily_api_calls": int``.

        Returns:
            Recommended ``TenantTier``.

        Raises:
            TypeError:  ``tenant_id`` not a str, or ``usage_stats`` not a dict.
            ValueError: ``tenant_id`` is empty.
        """
        _validate_tenant_id(tenant_id)
        if not isinstance(usage_stats, dict):
            raise TypeError(f"'usage_stats' must be a dict, got {type(usage_stats)!r}")

        if self._router is not None:
            try:
                return self._llm_suggest_tier(tenant_id, usage_stats)
            except Exception as exc:
                logger.warning("TenantRegistry.suggest_tier: LLM failed (%s), using heuristic", exc)

        return self._heuristic_suggest_tier(usage_stats)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_suggest_tier(usage_stats: Dict[str, Any]) -> TenantTier:
        daily = int(usage_stats.get("daily_api_calls", 0))
        for threshold, tier in _TIER_CALL_THRESHOLDS:
            if daily >= threshold:
                return tier
        return TenantTier.FREE

    def _llm_suggest_tier(
        self, tenant_id: str, usage_stats: Dict[str, Any]
    ) -> TenantTier:
        import json
        prompt = (
            f"A tenant has the following usage stats: {json.dumps(usage_stats)}. "
            f"Which service tier is most appropriate? "
            f"Reply with exactly one word: free, standard, enterprise, or platform."
        )
        try:
            from app.llm.models import LLMMessage
            import asyncio, inspect
            resp = self._router.generate_for_signal(
                signal_type=None,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=10,
            )
            if inspect.isawaitable(resp):
                resp = asyncio.get_event_loop().run_until_complete(resp)
            return TenantTier(resp.strip().lower())
        except Exception as exc:
            raise RuntimeError(f"LLM tier suggestion failed: {exc}") from exc


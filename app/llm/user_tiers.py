"""User-facing model tier abstraction.

This module exposes the three end-user product tiers (``lite``, ``medium``,
``turbo``) and resolves each one to a concrete OpenRouter model identifier
read from ``app.core.config.settings``.  It is the single source of truth
for tier <-> model mapping; all routing decisions, API endpoints, and tests
consult :func:`resolve_tier_model` so that operators only need to flip the
``USER_TIER_*_MODEL`` environment variables to swap an entire tier.

Design constraints:
* No side-effects at import time beyond enum construction.
* All public helpers are pure and thread-safe.
* The active tier may be mutated at runtime via :func:`set_active_tier`
  (used by ``POST /api/llm/tier``) without re-instantiating the router.
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Dict, List, Optional

from app.core.config import settings
from app.llm.config import MODEL_REGISTRY, ensure_openrouter_model_registered

logger = logging.getLogger(__name__)


class UserTier(str, Enum):
    """Product tiers offered to end users."""

    LITE = "lite"
    MEDIUM = "medium"
    TURBO = "turbo"

    @classmethod
    def parse(cls, value: Optional[str]) -> Optional["UserTier"]:
        """Parse a free-form string into a ``UserTier``.

        Accepts ``None`` / empty string (returns ``None``) and any case
        variant of the three tier names.  Unknown values raise ``ValueError``.
        """
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"tier must be a string, got {type(value).__name__}")
        v = value.strip().lower()
        if not v:
            return None
        try:
            return cls(v)
        except ValueError as exc:
            valid = ", ".join(t.value for t in cls)
            raise ValueError(f"unknown tier {value!r}; expected one of: {valid}") from exc


# Human-readable descriptors surfaced via GET /api/llm/tiers so the UI can
# render its tier-picker without hard-coding strings client-side.
TIER_DESCRIPTIONS: Dict[UserTier, Dict[str, str]] = {
    UserTier.LITE: {
        "label": "Lite",
        "summary": "Light, very cheap, low-performance model — ideal for casual use.",
    },
    UserTier.MEDIUM: {
        "label": "Medium",
        "summary": "Efficient and more powerful — recommended default for most users.",
    },
    UserTier.TURBO: {
        "label": "Turbo",
        "summary": "Top-tier model for power users with the heaviest workloads.",
    },
}


_active_tier_lock = threading.Lock()
_active_tier_override: Optional[UserTier] = None


def _tier_model_setting(tier: UserTier) -> str:
    """Return the OpenRouter model slug currently bound to ``tier``."""
    mapping = {
        UserTier.LITE: settings.user_tier_lite_model,
        UserTier.MEDIUM: settings.user_tier_medium_model,
        UserTier.TURBO: settings.user_tier_turbo_model,
    }
    return mapping[tier]


def resolve_tier_model(tier: UserTier) -> str:
    """Resolve a ``UserTier`` to a concrete model id, registering it if needed.

    Calls :func:`ensure_openrouter_model_registered` so that custom slugs set
    via ``USER_TIER_*_MODEL`` env vars are auto-registered with sensible
    quality/latency defaults derived from the tier itself.
    """
    if not isinstance(tier, UserTier):
        raise TypeError(f"tier must be a UserTier, got {type(tier).__name__}")
    model_id = _tier_model_setting(tier)
    if not model_id or not isinstance(model_id, str):
        raise RuntimeError(f"tier {tier.value!r} has no model id configured")
    quality_tier = {UserTier.LITE: 4, UserTier.MEDIUM: 2, UserTier.TURBO: 1}[tier]
    latency_tier = {UserTier.LITE: 1, UserTier.MEDIUM: 2, UserTier.TURBO: 2}[tier]
    ensure_openrouter_model_registered(
        model_id, quality_tier=quality_tier, latency_tier=latency_tier
    )
    return model_id


def get_active_tier() -> Optional[UserTier]:
    """Return the tier currently active for the process.

    Resolution order:
      1. Runtime override set via :func:`set_active_tier` (e.g. POST /tier).
      2. The ``USER_TIER`` env var captured in ``settings.user_tier``.
      3. ``None`` (tier mode disabled — legacy routing).
    """
    with _active_tier_lock:
        if _active_tier_override is not None:
            return _active_tier_override
    return UserTier.parse(settings.user_tier)


def set_active_tier(tier: Optional[UserTier]) -> Optional[UserTier]:
    """Override the process-wide active tier.  Pass ``None`` to clear."""
    if tier is not None and not isinstance(tier, UserTier):
        raise TypeError(f"tier must be a UserTier or None, got {type(tier).__name__}")
    global _active_tier_override
    with _active_tier_lock:
        _active_tier_override = tier
    logger.info("Active user tier set to %s", tier.value if tier else "<disabled>")
    return tier


def is_tier_mode_enabled() -> bool:
    """True when both a tier is active and an OpenRouter API key is configured."""
    return bool(get_active_tier() is not None and settings.openrouter_api_key)


def list_tiers() -> List[Dict[str, str]]:
    """Return UI-ready descriptors for every tier (id, label, summary, model)."""
    out: List[Dict[str, str]] = []
    for t in UserTier:
        meta = TIER_DESCRIPTIONS[t]
        out.append({
            "id": t.value,
            "label": meta["label"],
            "summary": meta["summary"],
            "model": _tier_model_setting(t),
        })
    return out

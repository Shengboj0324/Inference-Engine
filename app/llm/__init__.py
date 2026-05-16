"""LLM and embedding integration layer."""

from app.llm.router import LLMRouter, RoutingStrategy, get_router
from app.llm.models import LLMMessage, LLMResponse, LLMProvider
from app.llm.user_tiers import (
    UserTier,
    get_active_tier,
    set_active_tier,
    list_tiers,
    is_tier_mode_enabled,
    resolve_tier_model,
)

__all__ = [
    "LLMRouter",
    "RoutingStrategy",
    "get_router",
    "LLMMessage",
    "LLMResponse",
    "LLMProvider",
    "UserTier",
    "get_active_tier",
    "set_active_tier",
    "list_tiers",
    "is_tier_mode_enabled",
    "resolve_tier_model",
]

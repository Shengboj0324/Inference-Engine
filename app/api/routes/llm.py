"""LLM API endpoints for text generation and chat.

This module provides REST API endpoints for:
- Simple text generation
- Chat-based generation
- Cost-optimized routing
- Quality-optimized routing
- Health checks
- Statistics
"""

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.llm.exceptions import LLMError
from app.llm.models import LLMMessage, MessageRole
from app.llm.router import get_router, RoutingStrategy
from app.llm.user_tiers import (
    UserTier,
    get_active_tier,
    is_tier_mode_enabled,
    list_tiers,
    set_active_tier,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class GenerateRequest(BaseModel):
    """Request for simple text generation."""

    prompt: str = Field(..., min_length=1, max_length=100_000)
    max_tokens: Optional[int] = Field(None, ge=1, le=32_000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    strategy: Optional[str] = Field("balanced", description="Routing strategy")
    enable_fallback: bool = Field(True, description="Enable automatic fallback")
    tier: Optional[str] = Field(
        None,
        description="Per-request user tier override: 'lite' | 'medium' | 'turbo'.",
    )


class ChatMessage(BaseModel):
    """Chat message."""

    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    """Request for chat-based generation."""

    messages: List[ChatMessage] = Field(..., min_items=1)
    max_tokens: Optional[int] = Field(None, ge=1, le=32_000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    strategy: Optional[str] = Field("balanced", description="Routing strategy")
    enable_fallback: bool = Field(True, description="Enable automatic fallback")
    tier: Optional[str] = Field(
        None,
        description="Per-request user tier override: 'lite' | 'medium' | 'turbo'.",
    )


class TierUpdateRequest(BaseModel):
    """Request body for POST /tier — activate or clear the user tier."""

    tier: Optional[str] = Field(
        None,
        description="Tier id ('lite' | 'medium' | 'turbo') or null to clear.",
    )


def _parse_tier_or_400(value: Optional[str]) -> Optional[UserTier]:
    """Parse a tier string, raising HTTP 400 on invalid input."""
    try:
        return UserTier.parse(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


class GenerateResponse(BaseModel):
    """Response for text generation."""

    content: str
    model: str
    provider: str
    tokens_used: int
    cost: float
    latency_ms: int
    cached: bool = False


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a prompt.

    Args:
        request: Generation request with prompt and parameters

    Returns:
        Generated text with metadata

    Raises:
        HTTPException: On generation failure
    """
    try:
        # Parse routing strategy
        try:
            strategy = RoutingStrategy(request.strategy.upper())
        except ValueError:
            strategy = RoutingStrategy.BALANCED

        # Get router
        router_instance = get_router()

        # Build message list and delegate to generate() so we receive a full
        # LLMResponse object (with model, provider, usage, latency fields).
        # generate_simple() only returns a bare str and cannot populate the
        # GenerateResponse metadata fields.
        messages = [LLMMessage(role=MessageRole.USER, content=request.prompt)]
        response = await router_instance.generate(
            messages=messages,
            strategy=strategy,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            enable_fallback=request.enable_fallback,
            user_tier=_parse_tier_or_400(request.tier),
        )

        return GenerateResponse(
            content=response.content,
            model=response.model,
            provider=response.provider.value,
            tokens_used=response.tokens_used,
            cost=response.cost,
            latency_ms=response.latency_ms,
            cached=response.cached,
        )

    except LLMError as e:
        logger.error(f"LLM error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM generation failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}",
        )


@router.post("/chat", response_model=GenerateResponse)
async def chat(request: ChatRequest):
    """Generate chat response from messages.

    Args:
        request: Chat request with messages and parameters

    Returns:
        Generated response with metadata

    Raises:
        HTTPException: On generation failure
    """
    try:
        # Convert messages
        messages = [
            LLMMessage(
                role=MessageRole(msg.role),
                content=msg.content,
            )
            for msg in request.messages
        ]

        # Parse routing strategy
        try:
            strategy = RoutingStrategy(request.strategy.upper())
        except ValueError:
            strategy = RoutingStrategy.BALANCED

        # Get router
        router_instance = get_router()

        # Generate
        response = await router_instance.generate(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            strategy=strategy,
            enable_fallback=request.enable_fallback,
            user_tier=_parse_tier_or_400(request.tier),
        )

        return GenerateResponse(
            content=response.content,
            model=response.model,
            provider=response.provider.value,
            tokens_used=response.tokens_used,
            cost=response.cost,
            latency_ms=response.latency_ms,
            cached=response.cached,
        )

    except LLMError as e:
        logger.error(f"LLM error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat generation failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}",
        )


@router.get("/health")
async def health():
    """Check health of all LLM providers.

    Returns:
        Health status for each configured model
    """
    try:
        router_instance = get_router()
        health_status = await router_instance.health_check()
        return {
            "status": "healthy" if all(health_status.values()) else "degraded",
            "models": health_status,
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/stats")
async def stats():
    """Get LLM usage statistics.

    Returns:
        Usage statistics including requests, tokens, and cost
    """
    try:
        router_instance = get_router()
        statistics = router_instance.get_statistics()
        return statistics
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}",
        )



# ============================================================================
# USER-TIER ENDPOINTS (Lite / Medium / Turbo) — OpenRouter-backed
# ============================================================================


@router.get("/tiers")
async def list_available_tiers():
    """List every selectable user tier with UI-friendly metadata.

    Returns a JSON document containing the tier id, human label, summary
    text suitable for rendering in a tier-picker, and the OpenRouter model
    id that the tier is currently bound to.  Also reports the currently
    active tier (if any) and whether OpenRouter is configured.
    """
    active = get_active_tier()
    return {
        "tiers": list_tiers(),
        "active_tier": active.value if active else None,
        "tier_mode_enabled": is_tier_mode_enabled(),
    }


@router.get("/tier")
async def get_current_tier():
    """Return the currently active user tier (or ``null`` if disabled)."""
    active = get_active_tier()
    return {
        "active_tier": active.value if active else None,
        "tier_mode_enabled": is_tier_mode_enabled(),
    }


@router.post("/tier")
async def update_current_tier(request: TierUpdateRequest):
    """Switch the active user tier (or clear it with ``tier: null``).

    Validates the tier string and, when non-null, applies it as the
    process-wide override so subsequent ``/generate`` and ``/chat`` calls
    route through the OpenRouter model bound to that tier.
    """
    tier = _parse_tier_or_400(request.tier)
    set_active_tier(tier)
    return {
        "active_tier": tier.value if tier else None,
        "tier_mode_enabled": is_tier_mode_enabled(),
    }

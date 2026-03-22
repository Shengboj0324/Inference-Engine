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


"""Enhanced LLM models with industrial-grade validation and tracking.

This module provides comprehensive data models for LLM interactions with:
- Strict validation
- Cost tracking
- Performance metrics
- Error handling
- Type safety
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class MessageRole(str, Enum):
    """Valid message roles with strict typing."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"  # For function calling


class FinishReason(str, Enum):
    """Completion finish reasons."""

    STOP = "stop"  # Natural completion
    LENGTH = "length"  # Max tokens reached
    CONTENT_FILTER = "content_filter"  # Content policy violation
    FUNCTION_CALL = "function_call"  # Function call triggered
    ERROR = "error"  # Error occurred


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"  # Local models via vLLM
    OLLAMA = "ollama"  # Local models via Ollama
    TOGETHER = "together"  # Together AI
    REPLICATE = "replicate"  # Replicate


class LLMMessage(BaseModel):
    """Message in LLM conversation with validation."""

    role: MessageRole
    content: str = Field(..., min_length=1, max_length=1_000_000)
    name: Optional[str] = Field(None, max_length=64)  # For function calls
    function_call: Optional[Dict[str, Any]] = None  # Function call data

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty or whitespace only")
        return v.strip()

    class Config:
        """Pydantic config."""

        use_enum_values = True


class TokenUsage(BaseModel):
    """Token usage tracking with cost calculation."""

    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)

    # Cost tracking (USD)
    prompt_cost: float = Field(0.0, ge=0.0)
    completion_cost: float = Field(0.0, ge=0.0)
    total_cost: float = Field(0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_totals(self) -> "TokenUsage":
        """Validate token and cost totals."""
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("Total tokens must equal prompt + completion tokens")

        if abs(self.total_cost - (self.prompt_cost + self.completion_cost)) > 0.0001:
            raise ValueError("Total cost must equal prompt + completion cost")

        return self


class PerformanceMetrics(BaseModel):
    """Performance metrics for LLM calls."""

    latency_ms: int = Field(..., ge=0)  # Total latency
    time_to_first_token_ms: Optional[int] = Field(None, ge=0)  # TTFT for streaming
    tokens_per_second: Optional[float] = Field(None, ge=0.0)  # Generation speed

    # Timestamps
    request_time: datetime = Field(default_factory=datetime.utcnow)
    response_time: Optional[datetime] = None

    @model_validator(mode="after")
    def calculate_derived_metrics(self) -> "PerformanceMetrics":
        """Calculate derived metrics."""
        if self.response_time and self.request_time:
            delta = (self.response_time - self.request_time).total_seconds() * 1000
            if abs(delta - self.latency_ms) > 100:  # Allow 100ms tolerance
                self.latency_ms = int(delta)
        return self


class LLMResponse(BaseModel):
    """Enhanced LLM response with comprehensive tracking."""

    content: str
    model: str
    provider: LLMProvider

    # Token usage and cost
    usage: TokenUsage

    # Performance metrics
    metrics: PerformanceMetrics

    # Completion metadata
    finish_reason: FinishReason
    function_call: Optional[Dict[str, Any]] = None

    # Request metadata
    request_id: Optional[str] = None
    cached: bool = False  # Whether response was cached

    @property
    def tokens_used(self) -> int:
        """Backward compatibility property."""
        return self.usage.total_tokens

    @property
    def cost(self) -> float:
        """Total cost in USD."""
        return self.usage.total_cost

    @property
    def latency_ms(self) -> int:
        """Latency in milliseconds."""
        return self.metrics.latency_ms


class EmbeddingResponse(BaseModel):
    """Enhanced embedding response with tracking."""

    embedding: List[float] = Field(..., min_length=1)
    model: str
    provider: LLMProvider

    # Token usage and cost
    usage: TokenUsage

    # Metadata
    request_id: Optional[str] = None
    cached: bool = False

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        """Validate embedding dimensions."""
        if not v:
            raise ValueError("Embedding cannot be empty")
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding must contain only numbers")
        return v

    @property
    def tokens_used(self) -> int:
        """Backward compatibility property."""
        return self.usage.total_tokens

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return len(self.embedding)


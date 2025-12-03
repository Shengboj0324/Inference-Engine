"""LLM configuration with comprehensive model registry and cost tracking.

This module provides industrial-grade configuration for LLM operations:
- Model registry with pricing
- Provider configuration
- Cost optimization
- Quality tiers
- Fallback strategies
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.llm.models import LLMProvider


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    input_cost_per_1m: float  # USD per 1M input tokens
    output_cost_per_1m: float  # USD per 1M output tokens
    context_window: int  # Maximum context window
    supports_streaming: bool = True
    supports_function_calling: bool = False
    supports_vision: bool = False
    supports_json_mode: bool = False


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    provider: LLMProvider
    pricing: ModelPricing
    quality_tier: int  # 1 (best) to 5 (basic)
    latency_tier: int  # 1 (fastest) to 5 (slowest)
    max_retries: int = 3
    timeout_seconds: int = 60
    enabled: bool = True


# Model Registry with latest pricing (as of Dec 2024)
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # OpenAI Models
    "gpt-4-turbo-2024-04-09": ModelConfig(
        name="gpt-4-turbo-2024-04-09",
        provider=LLMProvider.OPENAI,
        pricing=ModelPricing(
            input_cost_per_1m=10.0,
            output_cost_per_1m=30.0,
            context_window=128_000,
            supports_function_calling=True,
            supports_json_mode=True,
        ),
        quality_tier=1,
        latency_tier=3,
    ),
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider=LLMProvider.OPENAI,
        pricing=ModelPricing(
            input_cost_per_1m=5.0,
            output_cost_per_1m=15.0,
            context_window=128_000,
            supports_function_calling=True,
            supports_vision=True,
            supports_json_mode=True,
        ),
        quality_tier=1,
        latency_tier=2,
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider=LLMProvider.OPENAI,
        pricing=ModelPricing(
            input_cost_per_1m=0.15,
            output_cost_per_1m=0.60,
            context_window=128_000,
            supports_function_calling=True,
            supports_json_mode=True,
        ),
        quality_tier=2,
        latency_tier=1,
    ),
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        provider=LLMProvider.OPENAI,
        pricing=ModelPricing(
            input_cost_per_1m=0.50,
            output_cost_per_1m=1.50,
            context_window=16_385,
            supports_function_calling=True,
            supports_json_mode=True,
        ),
        quality_tier=3,
        latency_tier=1,
    ),
    # Anthropic Models
    "claude-3-5-sonnet-20241022": ModelConfig(
        name="claude-3-5-sonnet-20241022",
        provider=LLMProvider.ANTHROPIC,
        pricing=ModelPricing(
            input_cost_per_1m=3.0,
            output_cost_per_1m=15.0,
            context_window=200_000,
            supports_vision=True,
        ),
        quality_tier=1,
        latency_tier=2,
    ),
    "claude-3-5-haiku-20241022": ModelConfig(
        name="claude-3-5-haiku-20241022",
        provider=LLMProvider.ANTHROPIC,
        pricing=ModelPricing(
            input_cost_per_1m=0.80,
            output_cost_per_1m=4.0,
            context_window=200_000,
        ),
        quality_tier=2,
        latency_tier=1,
    ),
    "claude-3-opus-20240229": ModelConfig(
        name="claude-3-opus-20240229",
        provider=LLMProvider.ANTHROPIC,
        pricing=ModelPricing(
            input_cost_per_1m=15.0,
            output_cost_per_1m=75.0,
            context_window=200_000,
            supports_vision=True,
        ),
        quality_tier=1,
        latency_tier=4,
    ),
    # Open-Source Models (via vLLM) - Infrastructure costs only
    "meta-llama/Meta-Llama-3.1-405B-Instruct": ModelConfig(
        name="meta-llama/Meta-Llama-3.1-405B-Instruct",
        provider=LLMProvider.VLLM,
        pricing=ModelPricing(
            input_cost_per_1m=2.0,  # Estimated infrastructure cost
            output_cost_per_1m=2.0,
            context_window=128_000,
        ),
        quality_tier=2,
        latency_tier=3,
    ),
    "mistralai/Mixtral-8x22B-Instruct-v0.1": ModelConfig(
        name="mistralai/Mixtral-8x22B-Instruct-v0.1",
        provider=LLMProvider.VLLM,
        pricing=ModelPricing(
            input_cost_per_1m=1.0,
            output_cost_per_1m=1.0,
            context_window=64_000,
        ),
        quality_tier=2,
        latency_tier=2,
    ),
    "Qwen/Qwen2.5-72B-Instruct": ModelConfig(
        name="Qwen/Qwen2.5-72B-Instruct",
        provider=LLMProvider.VLLM,
        pricing=ModelPricing(
            input_cost_per_1m=0.80,
            output_cost_per_1m=0.80,
            context_window=128_000,
        ),
        quality_tier=2,
        latency_tier=2,
    ),
}


@dataclass
class LLMServiceConfig:
    """Configuration for LLM service layer."""

    # Primary model selection
    primary_model: str = "gpt-4o"
    fallback_models: List[str] = field(
        default_factory=lambda: ["claude-3-5-sonnet-20241022", "gpt-4o-mini"]
    )

    # Cost optimization
    enable_cost_optimization: bool = True
    max_cost_per_request: float = 0.10  # USD
    monthly_budget: Optional[float] = None  # USD

    # Quality requirements
    min_quality_tier: int = 2  # 1-5, lower is better
    enable_quality_validation: bool = True

    # Performance requirements
    max_latency_ms: int = 10_000  # 10 seconds
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour

    # Retry and error handling
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5  # Failures before opening

    # Monitoring
    enable_metrics: bool = True
    enable_logging: bool = True
    log_prompts: bool = False  # Privacy: don't log prompts by default

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a model."""
        return MODEL_REGISTRY.get(model_name)

    def get_enabled_models(self) -> List[ModelConfig]:
        """Get all enabled models."""
        return [m for m in MODEL_REGISTRY.values() if m.enabled]

    def get_models_by_quality_tier(self, tier: int) -> List[ModelConfig]:
        """Get models by quality tier."""
        return [m for m in MODEL_REGISTRY.values() if m.quality_tier == tier and m.enabled]


# Default configuration
DEFAULT_LLM_CONFIG = LLMServiceConfig()


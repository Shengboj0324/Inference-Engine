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
    # ---------------------------------------------------------------------------
    # Ollama local models (competitive_analysis.md §5.2 — Offline LLM Provider)
    # Cost is ~$0 for on-device compute; pricing set to symbolic $0.01 for internal
    # cost accounting so budget logic still functions correctly.
    # ---------------------------------------------------------------------------
    "llama3.1:8b": ModelConfig(
        name="llama3.1:8b",
        provider=LLMProvider.OLLAMA,
        pricing=ModelPricing(
            input_cost_per_1m=0.01,
            output_cost_per_1m=0.01,
            context_window=128_000,
            supports_json_mode=False,
        ),
        quality_tier=3,
        latency_tier=3,
    ),
    "llama3.1:70b": ModelConfig(
        name="llama3.1:70b",
        provider=LLMProvider.OLLAMA,
        pricing=ModelPricing(
            input_cost_per_1m=0.01,
            output_cost_per_1m=0.01,
            context_window=128_000,
        ),
        quality_tier=2,
        latency_tier=4,
    ),
    "mistral:7b": ModelConfig(
        name="mistral:7b",
        provider=LLMProvider.OLLAMA,
        pricing=ModelPricing(
            input_cost_per_1m=0.01,
            output_cost_per_1m=0.01,
            context_window=32_000,
        ),
        quality_tier=3,
        latency_tier=2,
    ),
    # ---------------------------------------------------------------------------
    # OpenRouter tier models — exposed to end-users via the Lite/Medium/Turbo
    # tier selector.  Identifiers MUST match OpenRouter's canonical model slugs
    # (https://openrouter.ai/models).  Pricing values are conservative upper
    # bounds used for cost estimation only; the real per-call cost is computed
    # by the provider from the upstream response's usage block.
    # ---------------------------------------------------------------------------
    "meta-llama/llama-3.2-3b-instruct": ModelConfig(
        name="meta-llama/llama-3.2-3b-instruct",
        provider=LLMProvider.OPENROUTER,
        pricing=ModelPricing(
            input_cost_per_1m=0.03,
            output_cost_per_1m=0.05,
            context_window=131_072,
            supports_json_mode=False,
        ),
        quality_tier=4,
        latency_tier=1,
    ),
    "openai/gpt-4o-mini": ModelConfig(
        name="openai/gpt-4o-mini",
        provider=LLMProvider.OPENROUTER,
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
    "anthropic/claude-3.5-sonnet": ModelConfig(
        name="anthropic/claude-3.5-sonnet",
        provider=LLMProvider.OPENROUTER,
        pricing=ModelPricing(
            input_cost_per_1m=3.0,
            output_cost_per_1m=15.0,
            context_window=200_000,
            supports_vision=True,
        ),
        quality_tier=1,
        latency_tier=2,
    ),
}


# Heuristic quality/latency tiers assigned to OpenRouter models that are not
# explicitly registered above (e.g. when an operator overrides one of the
# ``user_tier_*_model`` settings with a different OpenRouter slug).
_DEFAULT_OPENROUTER_PRICING = ModelPricing(
    input_cost_per_1m=1.0,
    output_cost_per_1m=3.0,
    context_window=128_000,
)


def ensure_openrouter_model_registered(
    model_id: str,
    *,
    quality_tier: int = 2,
    latency_tier: int = 2,
    pricing: Optional[ModelPricing] = None,
) -> ModelConfig:
    """Idempotently register an OpenRouter model in ``MODEL_REGISTRY``.

    Lets operators point a tier at any OpenRouter model slug via the
    ``USER_TIER_*_MODEL`` env vars without having to edit this file.  When
    ``model_id`` is already registered the existing entry is returned
    unchanged so explicit configurations always win over defaults.

    Args:
        model_id: OpenRouter model slug, e.g. ``"openai/gpt-4o"``.
        quality_tier: Fallback quality tier (1=best … 5=basic).
        latency_tier: Fallback latency tier (1=fastest … 5=slowest).
        pricing: Optional explicit pricing override.

    Returns:
        The resolved ``ModelConfig`` (either pre-existing or freshly created).

    Raises:
        ValueError: If ``model_id`` is not a non-empty string.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]
    cfg = ModelConfig(
        name=model_id,
        provider=LLMProvider.OPENROUTER,
        pricing=pricing or _DEFAULT_OPENROUTER_PRICING,
        quality_tier=quality_tier,
        latency_tier=latency_tier,
    )
    MODEL_REGISTRY[model_id] = cfg
    return cfg


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


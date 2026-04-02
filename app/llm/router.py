"""Intelligent LLM router with cost/quality/latency optimization.

This module provides industrial-grade routing for LLM requests:
- Cost-based routing (minimize cost while meeting quality requirements)
- Quality-based selection (select best model for task)
- Latency-based routing (optimize for speed)
- Automatic fallback (handle provider failures)
- A/B testing framework (compare models)
- Load balancing (distribute load across providers)
"""

import asyncio
import logging
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from app.core.config import settings
from app.domain.inference_models import SignalType
from app.llm.base_client import EnhancedBaseLLMClient
from app.llm.config import DEFAULT_LLM_CONFIG, LLMServiceConfig, MODEL_REGISTRY, ModelConfig
from app.llm.exceptions import LLMCircuitBreakerError, LLMError
from app.llm.models import LLMMessage, LLMProvider, LLMResponse
from app.llm.monitoring import record_router_decision, record_router_fallback

logger = logging.getLogger(__name__)

# Import providers conditionally
try:
    from app.llm.providers import AnthropicLLMClient, OpenAILLMClient, VLLMClient
    from app.llm.providers.ollama_provider import OllamaProvider
    PROVIDERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some LLM providers not available: {e}")
    PROVIDERS_AVAILABLE = False
    # Create placeholder classes
    AnthropicLLMClient = None  # type: ignore[assignment,misc]
    OpenAILLMClient = None  # type: ignore[assignment,misc]
    VLLMClient = None  # type: ignore[assignment,misc]
    OllamaProvider = None  # type: ignore[assignment,misc]


class RoutingStrategy(str, Enum):
    """Routing strategies for LLM selection."""

    COST_OPTIMIZED = "cost_optimized"  # Minimize cost
    QUALITY_OPTIMIZED = "quality_optimized"  # Maximize quality
    LATENCY_OPTIMIZED = "latency_optimized"  # Minimize latency
    BALANCED = "balanced"  # Balance cost, quality, latency
    FALLBACK = "fallback"  # Try in order until success
    ROUND_ROBIN = "round_robin"  # Distribute load evenly
    A_B_TEST = "a_b_test"  # A/B testing between models


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    model_name: str
    provider: LLMProvider
    reason: str
    estimated_cost: float
    quality_tier: int
    latency_tier: int


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    model_a: str
    model_b: str
    traffic_split: float = 0.5  # Percentage to model A (0.0-1.0)
    enabled: bool = True


# ---------------------------------------------------------------------------
# Signal-type tier dispatch constants (competitive_analysis.md §5.3)
# Defined at module level so they can be referenced in generator expressions
# and by test code without instantiating LLMRouter.
# ---------------------------------------------------------------------------

#: High-stakes signal types that ALWAYS route to the frontier model.
_FRONTIER_SIGNAL_TYPES = frozenset({
    SignalType.CHURN_RISK,
    SignalType.LEGAL_RISK,
    SignalType.SECURITY_CONCERN,
    SignalType.REPUTATION_RISK,
})

#: All other signal types — route to fine-tuned model when configured.
_FINE_TUNED_SIGNAL_TYPES = frozenset(
    st for st in SignalType if st not in _FRONTIER_SIGNAL_TYPES
)


class LLMRouter:
    """Intelligent router for LLM requests.

    The router selects the optimal model based on:
    - Cost constraints
    - Quality requirements
    - Latency requirements
    - Provider availability
    - A/B testing configuration
    - Signal-type tier (frontier vs. fine-tuned) per competitive_analysis.md §5.3
    """

    # Expose module-level constants as class attributes for external access
    _FRONTIER_SIGNALS: frozenset = _FRONTIER_SIGNAL_TYPES
    _FINE_TUNED_SIGNALS: frozenset = _FINE_TUNED_SIGNAL_TYPES

    def __init__(
        self,
        service_config: Optional[LLMServiceConfig] = None,
        ab_test_config: Optional[ABTestConfig] = None,
    ):
        """Initialize LLM router.

        Args:
            service_config: Service configuration
            ab_test_config: A/B testing configuration
        """
        self.service_config = service_config or DEFAULT_LLM_CONFIG
        self.ab_test_config = ab_test_config

        # Client cache
        self._clients: Dict[str, EnhancedBaseLLMClient] = {}

        # Round-robin state
        self._round_robin_index = 0

        # Statistics
        self._total_requests = 0
        self._requests_by_model: Dict[str, int] = {}
        self._total_cost = 0.0

        logger.info(
            f"Initialized LLM router: strategy=dynamic, "
            f"primary={self.service_config.primary_model}, "
            f"fallbacks={self.service_config.fallback_models}"
        )

    def _get_client(self, model_name: str) -> EnhancedBaseLLMClient:
        """Get or create client for model.

        Args:
            model_name: Model name

        Returns:
            LLM client
        """
        if model_name in self._clients:
            return self._clients[model_name]

        # Get model config
        model_config = MODEL_REGISTRY.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model: {model_name}")

        # Create client based on provider
        if model_config.provider == LLMProvider.OPENAI:
            client = OpenAILLMClient(
                model_name=model_name,
                service_config=self.service_config,
            )
        elif model_config.provider == LLMProvider.ANTHROPIC:
            client = AnthropicLLMClient(
                model_name=model_name,
                service_config=self.service_config,
            )
        elif model_config.provider == LLMProvider.VLLM:
            client = VLLMClient(
                model_name=model_name,
                service_config=self.service_config,
            )
        elif model_config.provider == LLMProvider.OLLAMA:
            if OllamaProvider is None:
                raise ValueError("OllamaProvider is not available (import error)")
            client = OllamaProvider(
                model_name=model_name,
                service_config=self.service_config,
            )
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")

        self._clients[model_name] = client
        return client

    def _estimate_cost(
        self,
        model_config: ModelConfig,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
    ) -> float:
        """Estimate cost for a request.

        Args:
            model_config: Model configuration
            messages: Messages
            max_tokens: Maximum tokens to generate

        Returns:
            Estimated cost in USD
        """
        # Rough token estimation (4 chars per token)
        prompt_chars = sum(len(msg.content) for msg in messages)
        prompt_tokens = prompt_chars // 4

        completion_tokens = max_tokens or 500  # Default estimate

        # Calculate cost
        prompt_cost = (prompt_tokens / 1_000_000) * model_config.pricing.input_cost_per_1m
        completion_cost = (completion_tokens / 1_000_000) * model_config.pricing.output_cost_per_1m

        return prompt_cost + completion_cost

    def select_model(
        self,
        messages: List[LLMMessage],
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        max_tokens: Optional[int] = None,
        quality_requirement: Optional[int] = None,
        cost_limit: Optional[float] = None,
        latency_limit: Optional[int] = None,
    ) -> RoutingDecision:
        """Select optimal model based on strategy and constraints.

        Args:
            messages: Messages to process
            strategy: Routing strategy
            max_tokens: Maximum tokens to generate
            quality_requirement: Minimum quality tier (1-5, lower is better)
            cost_limit: Maximum cost in USD
            latency_limit: Maximum latency in ms

        Returns:
            Routing decision
        """
        # A/B testing override
        if strategy == RoutingStrategy.A_B_TEST and self.ab_test_config and self.ab_test_config.enabled:
            if random.random() < self.ab_test_config.traffic_split:
                model_name = self.ab_test_config.model_a
            else:
                model_name = self.ab_test_config.model_b

            model_config = MODEL_REGISTRY[model_name]
            return RoutingDecision(
                model_name=model_name,
                provider=model_config.provider,
                reason=f"A/B test: {model_name}",
                estimated_cost=self._estimate_cost(model_config, messages, max_tokens),
                quality_tier=model_config.quality_tier,
                latency_tier=model_config.latency_tier,
            )

        # Get candidate models
        candidates = [self.service_config.primary_model] + self.service_config.fallback_models
        candidate_configs = [MODEL_REGISTRY[m] for m in candidates if m in MODEL_REGISTRY]

        # Filter by constraints
        filtered = []
        for config in candidate_configs:
            # Check if enabled
            if not config.enabled:
                continue

            # Check quality requirement
            if quality_requirement and config.quality_tier > quality_requirement:
                continue

            # Check cost limit
            if cost_limit:
                estimated_cost = self._estimate_cost(config, messages, max_tokens)
                if estimated_cost > cost_limit:
                    continue

            # Check latency limit (approximate)
            if latency_limit and config.latency_tier > 3:  # Tier 4-5 are slow
                continue

            filtered.append(config)

        if not filtered:
            # Fallback to primary if no candidates match
            filtered = [MODEL_REGISTRY[self.service_config.primary_model]]

        # Select based on strategy
        if strategy == RoutingStrategy.COST_OPTIMIZED:
            # Select cheapest model
            selected = min(
                filtered,
                key=lambda c: self._estimate_cost(c, messages, max_tokens),
            )
            reason = "Cost optimized"

        elif strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            # Select highest quality
            selected = min(filtered, key=lambda c: c.quality_tier)
            reason = "Quality optimized"

        elif strategy == RoutingStrategy.LATENCY_OPTIMIZED:
            # Select fastest
            selected = min(filtered, key=lambda c: c.latency_tier)
            reason = "Latency optimized"

        elif strategy == RoutingStrategy.ROUND_ROBIN:
            # Round-robin distribution
            selected = filtered[self._round_robin_index % len(filtered)]
            self._round_robin_index += 1
            reason = "Round-robin"

        else:  # BALANCED
            # Balance cost, quality, and latency
            def score(c: ModelConfig) -> float:
                cost = self._estimate_cost(c, messages, max_tokens)
                # Normalize and combine (lower is better)
                return (cost * 100) + (c.quality_tier * 10) + (c.latency_tier * 5)

            selected = min(filtered, key=score)
            reason = "Balanced optimization"

        return RoutingDecision(
            model_name=selected.name,
            provider=selected.provider,
            reason=reason,
            estimated_cost=self._estimate_cost(selected, messages, max_tokens),
            quality_tier=selected.quality_tier,
            latency_tier=selected.latency_tier,
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        enable_fallback: bool = True,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion with intelligent routing.

        Args:
            messages: Conversation messages
            strategy: Routing strategy
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            enable_fallback: Enable automatic fallback on failure
            **kwargs: Additional parameters

        Returns:
            LLM response

        Raises:
            LLMError: If all attempts fail
        """
        self._total_requests += 1

        # Select model
        decision = self.select_model(
            messages=messages,
            strategy=strategy,
            max_tokens=max_tokens,
        )

        logger.info(
            f"Router decision: model={decision.model_name}, "
            f"reason={decision.reason}, "
            f"estimated_cost=${decision.estimated_cost:.6f}"
        )

        # Record router decision
        record_router_decision(
            strategy=strategy.value,
            selected_model=decision.model_name,
        )

        # Try primary model
        try:
            client = self._get_client(decision.model_name)
            response = await client.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Update statistics
            self._requests_by_model[decision.model_name] = (
                self._requests_by_model.get(decision.model_name, 0) + 1
            )
            self._total_cost += response.usage.total_cost

            return response

        except (LLMError, LLMCircuitBreakerError) as e:
            logger.warning(
                f"Primary model {decision.model_name} failed: {str(e)}. "
                f"Fallback enabled: {enable_fallback}"
            )

            if not enable_fallback:
                raise

            # Try fallback models
            for fallback_model in self.service_config.fallback_models:
                if fallback_model == decision.model_name:
                    continue  # Skip already tried model

                try:
                    logger.info(f"Trying fallback model: {fallback_model}")

                    # Record fallback attempt
                    record_router_fallback(
                        primary_model=decision.model_name,
                        fallback_model=fallback_model,
                    )

                    client = self._get_client(fallback_model)
                    response = await client.generate(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )

                    # Update statistics
                    self._requests_by_model[fallback_model] = (
                        self._requests_by_model.get(fallback_model, 0) + 1
                    )
                    self._total_cost += response.usage.total_cost

                    logger.info(f"Fallback successful: {fallback_model}")
                    return response

                except (LLMError, LLMCircuitBreakerError) as fallback_error:
                    logger.warning(f"Fallback model {fallback_model} failed: {str(fallback_error)}")
                    continue

            # All models failed
            raise LLMError(
                f"All models failed. Primary: {decision.model_name}, "
                f"Fallbacks: {self.service_config.fallback_models}",
                retryable=False,
            )

    async def generate_simple(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Simple generation with routing.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            strategy: Routing strategy
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))

        response = await self.generate(
            messages=messages,
            strategy=strategy,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return response.content

    async def complete(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        **kwargs,
    ) -> str:
        """Thin completion alias used by audio-intelligence and document-intelligence modules.

        This method is intentionally a transparent delegate to ``generate_simple``
        so that callers do not need to import ``RoutingStrategy`` directly.

        Args:
            prompt: User-facing prompt text.
            system_prompt: Optional system instruction.
            max_tokens: Upper bound on generated tokens (``None`` = model default).
            temperature: Sampling temperature in ``[0.0, 2.0]``.
            strategy: Routing strategy (default: ``BALANCED``).
            **kwargs: Forwarded verbatim to the underlying provider client.

        Returns:
            Generated text as a plain string.

        Raises:
            TypeError: If *prompt* is not a ``str``.
            ValueError: If *prompt* is empty or *temperature* is outside ``[0.0, 2.0]``.
            LLMError: If all configured models fail.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a str, got {type(prompt).__name__!r}")
        if not prompt.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        if not isinstance(temperature, (int, float)):
            raise TypeError(f"temperature must be a float, got {type(temperature).__name__!r}")
        if not (0.0 <= float(temperature) <= 2.0):
            raise ValueError(f"temperature must be in [0.0, 2.0], got {temperature!r}")

        logger.debug(
            "LLMRouter.complete: max_tokens=%s temperature=%s strategy=%s",
            max_tokens, temperature, strategy.value,
        )
        return await self.generate_simple(
            prompt=prompt,
            system_prompt=system_prompt,
            strategy=strategy,
            temperature=float(temperature),
            max_tokens=max_tokens,
            **kwargs,
        )

    def get_stats(self) -> dict:
        """Get router statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_requests": self._total_requests,
            "total_cost": self._total_cost,
            "average_cost_per_request": (
                self._total_cost / self._total_requests if self._total_requests > 0 else 0.0
            ),
            "requests_by_model": self._requests_by_model,
            "primary_model": self.service_config.primary_model,
            "fallback_models": self.service_config.fallback_models,
            "ab_test_enabled": self.ab_test_config.enabled if self.ab_test_config else False,
        }

    def get_client_stats(self) -> Dict[str, dict]:
        """Get statistics for all clients.

        Returns:
            Dictionary mapping model name to client stats
        """
        return {name: client.get_stats() for name, client in self._clients.items()}

    async def health_check(self) -> Dict[str, bool]:
        """Check health of all configured models.

        Returns:
            Dictionary mapping model name to health status
        """
        health = {}

        for model_name in [self.service_config.primary_model] + self.service_config.fallback_models:
            try:
                client = self._get_client(model_name)
                # Simple health check with minimal prompt
                await client.generate_simple(
                    prompt="Hello",
                    max_tokens=5,
                )
                health[model_name] = True
            except Exception as e:
                logger.error(f"Health check failed for {model_name}: {str(e)}")
                health[model_name] = False

        return health

    # ------------------------------------------------------------------
    # Tiered signal-type routing (competitive_analysis.md §5.3)
    # ------------------------------------------------------------------

    def _model_for_signal_type(self, signal_type: Optional[SignalType]) -> str:
        """Return the model name to use for a given signal type.

        High-stakes signals (CHURN_RISK, LEGAL_RISK, SECURITY_CONCERN,
        REPUTATION_RISK) always route to the frontier model.  All other
        types route to ``settings.fine_tuned_model_id`` when configured,
        falling back to the frontier model otherwise.

        Args:
            signal_type: The top-candidate signal type from retrieval.
                Use ``None`` to unconditionally select the frontier model.

        Returns:
            Model name string suitable for passing to ``_get_client()``.
        """
        if signal_type is None or signal_type in self._FRONTIER_SIGNALS:
            return self.service_config.primary_model

        # Non-frontier signal — prefer fine-tuned if configured
        fine_tuned = getattr(settings, "fine_tuned_model_id", None)
        if fine_tuned:
            return fine_tuned

        return self.service_config.primary_model

    async def generate_for_signal(
        self,
        signal_type: Optional[SignalType],
        messages: List[LLMMessage],
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate text using the model tier appropriate for ``signal_type``.

        This is the primary entry point used by ``LLMAdjudicator.adjudicate()``.
        It selects either the frontier model or the configured fine-tuned model
        based on the signal type tier, then delegates to the selected client's
        ``generate_simple()`` method.

        High-stakes signal types (CHURN_RISK, LEGAL_RISK, SECURITY_CONCERN,
        REPUTATION_RISK) always route to the frontier model, regardless of
        whether a fine-tuned model is configured.

        Args:
            signal_type: Top-candidate signal type from retrieval stage.
                Pass ``None`` to unconditionally route to the frontier model.
            messages: Conversation turns to pass to the LLM.
            temperature: Sampling temperature.  Defaults to 0.3 for
                consistent classification output.
            max_tokens: Optional maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Returns:
            Raw string content from the LLM response.

        Raises:
            LLMError: Any LLM provider error propagated from the client.
        """
        model_name = self._model_for_signal_type(signal_type)
        tier = (
            "frontier"
            if signal_type is None or signal_type in self._FRONTIER_SIGNALS
            else "fine_tuned"
        )
        logger.debug(
            "generate_for_signal: signal_type=%s tier=%s model=%s",
            signal_type,
            tier,
            model_name,
        )
        client = self._get_client(model_name)
        return await client.generate_simple(
            prompt=messages[-1].content if messages else "",
            messages=messages[:-1] if len(messages) > 1 else None,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


# Global router instance
_global_router: Optional[LLMRouter] = None


def get_router(
    service_config: Optional[LLMServiceConfig] = None,
    ab_test_config: Optional[ABTestConfig] = None,
) -> LLMRouter:
    """Get global router instance.

    Args:
        service_config: Service configuration
        ab_test_config: A/B testing configuration

    Returns:
        LLM router
    """
    global _global_router

    if _global_router is None:
        _global_router = LLMRouter(
            service_config=service_config,
            ab_test_config=ab_test_config,
        )

    return _global_router


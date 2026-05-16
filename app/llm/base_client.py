"""Enhanced base client with industrial-grade reliability features.

This module provides the foundation for all LLM clients with:
- Retry logic with exponential backoff
- Circuit breaker pattern
- Rate limiting
- Timeout management
- Comprehensive error handling
- Cost tracking
- Performance metrics
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, List, Optional
from uuid import uuid4

from app.llm.cache import LLMCacheManager, get_llm_cache_manager
from app.llm.circuit_breaker import CircuitBreaker
from app.llm.config import DEFAULT_LLM_CONFIG, LLMServiceConfig, ModelConfig
from app.llm.exceptions import (
    LLMError,
    LLMTimeoutError,
)
from app.llm.models import (
    EmbeddingResponse,
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    PerformanceMetrics,
    TokenUsage,
)
from app.llm.monitoring import (
    get_metrics_collector,
    record_error,
    record_request,
)
from app.llm.rate_limiter import RateLimiter
from app.llm.retry import RetryConfig, retry_async
from app.llm.token_counter import get_token_counter

logger = logging.getLogger(__name__)


class EnhancedBaseLLMClient(ABC):
    """Enhanced base class for LLM providers with reliability features.

    This class provides:
    - Automatic retry with exponential backoff
    - Circuit breaker for failure protection
    - Rate limiting for API compliance
    - Timeout management
    - Cost tracking
    - Performance metrics
    """

    def __init__(
        self,
        provider: LLMProvider,
        model_config: ModelConfig,
        service_config: Optional[LLMServiceConfig] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize enhanced LLM client.

        Args:
            provider: LLM provider
            model_config: Model configuration
            service_config: Service configuration
            api_key: API key for provider
        """
        self.provider = provider
        self.model_config = model_config
        self.service_config = service_config or DEFAULT_LLM_CONFIG
        self.api_key = api_key

        # Reliability features
        self.circuit_breaker = CircuitBreaker(
            name=f"{provider.value}:{model_config.name}",
            failure_threshold=self.service_config.circuit_breaker_threshold,
            recovery_timeout=60.0,
        )

        self.rate_limiter = self._create_rate_limiter()

        self.retry_config = RetryConfig(
            max_retries=model_config.max_retries,
            initial_delay=self.service_config.retry_delay_seconds,
        )

        # Metrics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_errors = 0

        # Caching (if enabled)
        self._cache_manager: Optional[LLMCacheManager] = None
        if self.service_config.enable_caching:
            self._cache_manager = get_llm_cache_manager()

        # Token counter
        self._token_counter = get_token_counter()

    def _create_rate_limiter(self) -> RateLimiter:
        """Create rate limiter based on provider.

        Returns:
            Configured rate limiter
        """
        # Provider-specific rate limits
        if self.provider == LLMProvider.OPENAI:
            return RateLimiter(
                requests_per_second=50.0,  # Conservative limit
                burst_size=100.0,
                max_requests_per_minute=3000,
            )
        elif self.provider == LLMProvider.ANTHROPIC:
            return RateLimiter(
                requests_per_second=40.0,
                burst_size=80.0,
                max_requests_per_minute=2000,
            )
        elif self.provider == LLMProvider.OPENROUTER:
            return RateLimiter(
                requests_per_second=20.0,
                burst_size=40.0,
                max_requests_per_minute=1200,
            )
        else:
            # Default for local/other providers
            return RateLimiter(
                requests_per_second=10.0,
                burst_size=20.0,
            )

    @abstractmethod
    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> LLMResponse:
        """Provider-specific generation implementation.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    async def _generate_stream_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncIterator[str]:
        """Provider-specific streaming implementation.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            Content chunks
        """
        pass

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion with full reliability stack.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters

        Returns:
            LLM response with metrics

        Raises:
            LLMError: On generation failure
        """
        request_id = str(uuid4())
        start_time = time.time()
        timeout = timeout or self.model_config.timeout_seconds

        logger.info(
            f"LLM request {request_id}: provider={self.provider.value}, "
            f"model={self.model_config.name}, messages={len(messages)}"
        )

        try:
            # Check cache first (if enabled)
            if self._cache_manager and temperature < 0.3:  # Only cache deterministic requests
                cached_response = await self._cache_manager.get_llm_response(
                    messages=messages,
                    model=self.model_config.name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                if cached_response:
                    logger.info(f"LLM cache hit for request {request_id}")
                    cached_response.request_id = request_id
                    return cached_response

            # Validate context window
            if not self._token_counter.validate_context_window(messages, self.model_config.name, max_tokens):
                prompt_tokens = self._token_counter.count_messages_tokens(messages, self.model_config.name)
                context_window = self._token_counter.get_context_window(self.model_config.name)
                raise LLMError(
                    f"Request exceeds context window: {prompt_tokens} + {max_tokens or 500} > {context_window}",
                    provider=self.provider.value,
                    model=self.model_config.name,
                )

            # Rate limiting
            await self.rate_limiter.acquire(timeout=timeout)

            # Circuit breaker + retry wrapper
            async def _execute():
                return await asyncio.wait_for(
                    self._generate_impl(messages, temperature, max_tokens, **kwargs),
                    timeout=timeout,
                )

            response = await self.circuit_breaker.call(
                retry_async,
                _execute,
                config=self.retry_config,
            )

            # Cache response (if enabled and deterministic)
            if self._cache_manager and temperature < 0.3:
                await self._cache_manager.set_llm_response(
                    messages=messages,
                    model=self.model_config.name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response=response,
                    ttl=self.service_config.cache_ttl_seconds,
                    **kwargs,
                )

            # Update metrics
            self._total_requests += 1
            self._total_tokens += response.usage.total_tokens
            self._total_cost += response.usage.total_cost

            # Add request metadata
            response.request_id = request_id

            # Record Prometheus metrics
            duration_seconds = (time.time() - start_time)
            record_request(
                provider=self.provider.value,
                model=self.model_config.name,
                status="success",
                duration_seconds=duration_seconds,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cost=response.usage.total_cost,
                response_length=len(response.content),
                ttft_seconds=(
                    response.metrics.time_to_first_token_ms / 1000.0
                    if response.metrics.time_to_first_token_ms
                    else None
                ),
                tokens_per_second=response.metrics.tokens_per_second,
            )

            # Update cost tracking
            get_metrics_collector().add_cost(self.provider.value, response.usage.total_cost)

            logger.info(
                f"LLM request {request_id} completed: "
                f"tokens={response.usage.total_tokens}, "
                f"cost=${response.usage.total_cost:.6f}, "
                f"latency={response.metrics.latency_ms}ms"
            )

            return response

        except asyncio.TimeoutError:
            self._total_errors += 1

            # Record error metrics
            record_error(
                provider=self.provider.value,
                model=self.model_config.name,
                error_type="timeout",
            )
            record_request(
                provider=self.provider.value,
                model=self.model_config.name,
                status="error",
                duration_seconds=(time.time() - start_time),
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                response_length=0,
            )

            elapsed_ms = int((time.time() - start_time) * 1000)
            raise LLMTimeoutError(
                f"Request timeout after {elapsed_ms}ms",
                timeout_seconds=timeout,
                provider=self.provider.value,
                model=self.model_config.name,
            )

        except Exception as e:
            self._total_errors += 1

            # Record error metrics
            error_type = type(e).__name__
            record_error(
                provider=self.provider.value,
                model=self.model_config.name,
                error_type=error_type,
            )
            record_request(
                provider=self.provider.value,
                model=self.model_config.name,
                status="error",
                duration_seconds=(time.time() - start_time),
                prompt_tokens=0,
                completion_tokens=0,
                cost=0.0,
                response_length=0,
            )

            logger.error(f"LLM request {request_id} failed: {str(e)}", exc_info=True)
            raise

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Generate completion with streaming.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters

        Yields:
            Content chunks

        Raises:
            LLMError: On generation failure
        """
        request_id = str(uuid4())
        timeout = timeout or self.model_config.timeout_seconds

        logger.info(
            f"LLM streaming request {request_id}: provider={self.provider.value}, "
            f"model={self.model_config.name}"
        )

        try:
            # Rate limiting
            await self.rate_limiter.acquire(timeout=timeout)

            # Circuit breaker wrapper
            async def _execute():
                async for chunk in self._generate_stream_impl(
                    messages, temperature, max_tokens, **kwargs
                ):
                    yield chunk

            async for chunk in self.circuit_breaker.call(_execute):
                yield chunk

            self._total_requests += 1

        except Exception as e:
            self._total_errors += 1
            logger.error(f"LLM streaming request {request_id} failed: {str(e)}", exc_info=True)
            raise

    async def generate_simple(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Simple generation from a single prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated text content
        """
        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))

        response = await self.generate(messages, temperature, max_tokens, **kwargs)
        return response.content

    def calculate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> TokenUsage:
        """Calculate token usage and cost.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Token usage with cost
        """
        pricing = self.model_config.pricing

        prompt_cost = (prompt_tokens / 1_000_000) * pricing.input_cost_per_1m
        completion_cost = (completion_tokens / 1_000_000) * pricing.output_cost_per_1m

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=prompt_cost + completion_cost,
        )

    def get_stats(self) -> dict:
        """Get client statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "provider": self.provider.value,
            "model": self.model_config.name,
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_cost": self._total_cost,
            "total_errors": self._total_errors,
            "error_rate": (
                self._total_errors / self._total_requests if self._total_requests > 0 else 0.0
            ),
            "circuit_breaker": self.circuit_breaker.get_stats(),
        }


class EnhancedBaseEmbeddingClient(ABC):
    """Enhanced base class for embedding providers."""

    def __init__(
        self,
        provider: LLMProvider,
        model_name: str,
        api_key: Optional[str] = None,
    ):
        """Initialize enhanced embedding client.

        Args:
            provider: Provider name
            model_name: Model name
            api_key: API key
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key

        # Reliability features
        self.circuit_breaker = CircuitBreaker(
            name=f"{provider.value}:embeddings",
            failure_threshold=5,
        )

        self.rate_limiter = RateLimiter(
            requests_per_second=50.0,
            burst_size=100.0,
        )

        # Metrics
        self._total_requests = 0
        self._total_tokens = 0

        # Caching (embeddings are deterministic, always cache)
        self._cache_manager = get_llm_cache_manager()

    @abstractmethod
    async def _embed_impl(self, texts: List[str]) -> List[EmbeddingResponse]:
        """Provider-specific embedding implementation.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding responses
        """
        pass

    async def embed_text(self, text: str, timeout: float = 30.0) -> EmbeddingResponse:
        """Embed a single text.

        Args:
            text: Text to embed
            timeout: Request timeout

        Returns:
            Embedding response
        """
        # Check cache first
        cached_embedding = await self._cache_manager.get_embedding(text, self.model_name)
        if cached_embedding:
            logger.debug(f"Embedding cache hit for model {self.model_name}")
            return cached_embedding

        # Generate embedding
        results = await self.embed_batch([text], timeout=timeout)
        embedding = results[0]

        # Cache result
        await self._cache_manager.set_embedding(text, self.model_name, embedding)

        return embedding

    async def embed_batch(
        self,
        texts: List[str],
        timeout: float = 60.0,
    ) -> List[EmbeddingResponse]:
        """Embed multiple texts.

        Args:
            texts: Texts to embed
            timeout: Request timeout

        Returns:
            List of embedding responses
        """
        # Rate limiting
        await self.rate_limiter.acquire(timeout=timeout)

        # Circuit breaker + retry
        async def _execute():
            return await asyncio.wait_for(
                self._embed_impl(texts),
                timeout=timeout,
            )

        results = await self.circuit_breaker.call(_execute)

        self._total_requests += 1
        self._total_tokens += sum(r.usage.total_tokens for r in results)

        return results


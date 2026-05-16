"""OpenRouter LLM provider — powers the Lite/Medium/Turbo user-tier feature.

OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible REST API,
so the implementation reuses the official ``openai`` async client with a
custom ``base_url``.  All retry / circuit-breaker / rate-limiting / cost
tracking infrastructure inherited from :class:`EnhancedBaseLLMClient`
therefore applies unchanged.
"""

import logging
import time
from datetime import datetime
from typing import AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from app.core.config import settings
from app.llm.base_client import EnhancedBaseLLMClient
from app.llm.config import MODEL_REGISTRY, LLMServiceConfig, ensure_openrouter_model_registered
from app.llm.exceptions import LLMAuthenticationError
from app.llm.models import (
    FinishReason,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    PerformanceMetrics,
)
from app.llm.providers.openai_provider import map_openai_error

logger = logging.getLogger(__name__)


class OpenRouterLLMClient(EnhancedBaseLLMClient):
    """LLM client that proxies requests through OpenRouter."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        service_config: Optional[LLMServiceConfig] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the OpenRouter client.

        Args:
            model_name: OpenRouter model slug (e.g. ``openai/gpt-4o-mini``).
            api_key: Override for ``settings.openrouter_api_key``.
            service_config: Shared service configuration.
            base_url: Override for ``settings.openrouter_base_url``.

        Raises:
            ValueError: If ``model_name`` is unknown and cannot be auto-registered.
            LLMAuthenticationError: If no OpenRouter API key is available.
        """
        model_config = MODEL_REGISTRY.get(model_name)
        if model_config is None:
            model_config = ensure_openrouter_model_registered(model_name)
        if model_config.provider != LLMProvider.OPENROUTER:
            raise ValueError(
                f"Model {model_name!r} is registered under provider "
                f"{model_config.provider.value!r}, not 'openrouter'."
            )

        super().__init__(
            provider=LLMProvider.OPENROUTER,
            model_config=model_config,
            service_config=service_config,
            api_key=api_key,
        )

        resolved_key = api_key or settings.openrouter_api_key
        if not resolved_key:
            raise LLMAuthenticationError(
                "OpenRouter API key missing: set OPENROUTER_API_KEY or pass api_key.",
                provider=LLMProvider.OPENROUTER.value,
                model=model_name,
            )

        # OpenRouter recommends sending these headers for proper attribution.
        default_headers: Dict[str, str] = {}
        if settings.openrouter_app_url:
            default_headers["HTTP-Referer"] = settings.openrouter_app_url
        if settings.openrouter_app_title:
            default_headers["X-Title"] = settings.openrouter_app_title

        self.client = AsyncOpenAI(
            api_key=resolved_key,
            base_url=base_url or settings.openrouter_base_url,
            default_headers=default_headers or None,
        )

    @staticmethod
    def _convert_messages(messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Translate internal ``LLMMessage`` objects to OpenAI wire format."""
        return [
            {
                "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                "content": msg.content,
            }
            for msg in messages
        ]

    async def _generate_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> LLMResponse:
        """OpenRouter-specific generation implementation."""
        start_time = time.time()
        request_time = datetime.utcnow()
        try:
            response = await self.client.chat.completions.create(
                model=self.model_config.name,
                messages=self._convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            response_time = datetime.utcnow()
            latency_ms = int((time.time() - start_time) * 1000)

            content = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "stop"
            usage = self.calculate_cost(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            metrics = PerformanceMetrics(
                latency_ms=latency_ms,
                request_time=request_time,
                response_time=response_time,
            )
            try:
                fr = FinishReason(finish_reason)
            except ValueError:
                # OpenRouter occasionally returns provider-specific reasons —
                # collapse anything unknown into STOP rather than failing the
                # whole request.
                fr = FinishReason.STOP
            return LLMResponse(
                content=content,
                model=response.model or self.model_config.name,
                provider=self.provider,
                usage=usage,
                metrics=metrics,
                finish_reason=fr,
            )
        except Exception as e:
            raise map_openai_error(e, self.provider.value, self.model_config.name)

    async def _generate_stream_impl(
        self,
        messages: List[LLMMessage],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs,
    ) -> AsyncIterator[str]:
        """OpenRouter-specific streaming implementation."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_config.name,
                messages=self._convert_messages(messages),
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise map_openai_error(e, self.provider.value, self.model_config.name)

"""OpenAI implementation of LLM and embedding clients."""

from typing import List, Optional

import openai

from app.core.config import settings
from app.llm.client_base import (
    BaseEmbeddingClient,
    BaseLLMClient,
    EmbeddingResponse,
    LLMMessage,
    LLMResponse,
)


class OpenAISyncEmbeddingClient:
    """Synchronous OpenAI embedding client for use in Celery tasks."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize synchronous OpenAI embedding client.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.client = openai.OpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.embedding_model

    def embed_text(self, text: str) -> EmbeddingResponse:
        """Generate embedding for a single text (synchronous)."""
        response = self.client.embeddings.create(input=text, model=self.model)

        return EmbeddingResponse(
            embedding=response.data[0].embedding,
            model=response.model,
            tokens_used=response.usage.total_tokens,
        )

    def embed_batch(self, texts: List[str]) -> List[EmbeddingResponse]:
        """Generate embeddings for multiple texts (synchronous)."""
        response = self.client.embeddings.create(input=texts, model=self.model)

        return [
            EmbeddingResponse(
                embedding=item.embedding,
                model=response.model,
                tokens_used=response.usage.total_tokens // len(texts),
            )
            for item in response.data
        ]


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI embedding client."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize OpenAI embedding client.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.embedding_model

    async def embed_text(self, text: str) -> EmbeddingResponse:
        """Generate embedding for a single text."""
        response = await self.client.embeddings.create(input=text, model=self.model)

        return EmbeddingResponse(
            embedding=response.data[0].embedding,
            model=response.model,
            tokens_used=response.usage.total_tokens,
        )

    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResponse]:
        """Generate embeddings for multiple texts."""
        response = await self.client.embeddings.create(input=texts, model=self.model)

        return [
            EmbeddingResponse(
                embedding=item.embedding,
                model=response.model,
                tokens_used=response.usage.total_tokens // len(texts),
            )
            for item in response.data
        ]


class OpenAILLMClient(BaseLLMClient):
    """OpenAI LLM client."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize OpenAI LLM client.

        Args:
            api_key: OpenAI API key (defaults to settings)
            model: Model name (defaults to settings)
        """
        self.client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)
        self.model = model or settings.llm_model

    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion from messages."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            tokens_used=response.usage.total_tokens if response.usage else 0,
            finish_reason=response.choices[0].finish_reason,
        )

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Generate completion with streaming."""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


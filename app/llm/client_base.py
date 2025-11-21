"""Base classes for LLM and embedding clients."""

from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel


class EmbeddingResponse(BaseModel):
    """Response from embedding generation."""

    embedding: List[float]
    model: str
    tokens_used: int


class LLMMessage(BaseModel):
    """Message in LLM conversation."""

    role: str  # system, user, assistant
    content: str


class LLMResponse(BaseModel):
    """Response from LLM generation."""

    content: str
    model: str
    tokens_used: int
    finish_reason: str


class BaseEmbeddingClient(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResponse:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            EmbeddingResponse with embedding vector
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> List[EmbeddingResponse]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of EmbeddingResponse objects
        """
        pass


class BaseLLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Generate completion from messages.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Generate completion with streaming.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Yields:
            Chunks of generated content
        """
        pass

    async def generate_simple(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Simple generation from a single prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text content
        """
        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))

        response = await self.generate(messages, temperature, max_tokens)
        return response.content


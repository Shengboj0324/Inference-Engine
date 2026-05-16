"""LLM provider implementations.

This package contains all LLM provider implementations:
- OpenAI (GPT-4o, GPT-4 Turbo, GPT-4o-mini, GPT-3.5 Turbo)
- Anthropic (Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus)
- vLLM (Llama 3.1, Mixtral, Qwen, any HuggingFace model)
- Ollama (local offline inference — competitive_analysis.md §5.2)
"""

import logging

logger = logging.getLogger(__name__)

# Import providers conditionally based on available dependencies
__all__ = []

# OpenAI provider (requires openai package)
try:
    from app.llm.providers.openai_provider import (
        OpenAIEmbeddingClient,
        OpenAILLMClient,
        OpenAISyncEmbeddingClient,
    )
    __all__.extend(["OpenAILLMClient", "OpenAIEmbeddingClient", "OpenAISyncEmbeddingClient"])
except ImportError as e:
    logger.warning(f"OpenAI provider not available: {e}")
    OpenAILLMClient = None
    OpenAIEmbeddingClient = None
    OpenAISyncEmbeddingClient = None

# Anthropic provider (requires anthropic package)
try:
    from app.llm.providers.anthropic_provider import AnthropicLLMClient
    __all__.append("AnthropicLLMClient")
except ImportError as e:
    logger.warning(f"Anthropic provider not available: {e}")
    AnthropicLLMClient = None

# vLLM provider (requires aiohttp for API calls)
try:
    from app.llm.providers.vllm_provider import VLLMClient
    __all__.append("VLLMClient")
except ImportError as e:
    logger.warning(f"vLLM provider not available: {e}")
    VLLMClient = None

# Ollama provider — local offline LLM (requires aiohttp)
try:
    from app.llm.providers.ollama_provider import OllamaProvider
    __all__.append("OllamaProvider")
except ImportError as e:
    logger.warning(f"Ollama provider not available: {e}")
    OllamaProvider = None

# OpenRouter provider — powers the Lite/Medium/Turbo user-tier feature
try:
    from app.llm.providers.openrouter_provider import OpenRouterLLMClient
    __all__.append("OpenRouterLLMClient")
except ImportError as e:
    logger.warning(f"OpenRouter provider not available: {e}")
    OpenRouterLLMClient = None


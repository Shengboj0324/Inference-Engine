"""Pluggable chunk-source abstraction for the SSE chat endpoint.

The actual LLM call lives behind a :class:`ChunkSource` protocol so:

* The default :class:`OpenAIStreamSource` uses the key the user stored
  in the local vault (BYOK) instead of any environment variable.
* An :class:`OfflineEchoSource` provides a deterministic stream when
  no API key is configured, which is how the desktop shell renders a
  usable demo state on a fresh install.
* Tests inject :class:`StaticChunkSource` to exercise the SSE
  framing, PII scrubbing, and error paths without network calls.

Every chunk is funnelled through :func:`scrub_chunk`, which delegates
to :meth:`DataResidencyGuard.scrub_text`.  This honours the project's
zero-egress mandate even though the streamed text is leaving an LLM
provider rather than entering one — defence in depth.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Iterable, List, Optional, Protocol, Sequence

from app.core.data_residency import DataResidencyGuard
from app.local.key_vault import KeyVault, get_key_vault

logger = logging.getLogger(__name__)

_GUARD = DataResidencyGuard()


class StreamMessage:
    """Provider-agnostic chat message handed to a :class:`ChunkSource`."""

    __slots__ = ("role", "content")

    def __init__(self, role: str, content: str) -> None:
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"invalid role: {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")
        self.role = role
        self.content = content


class ChunkSource(Protocol):
    """Yield successive text chunks for the assistant reply."""

    name: str

    async def stream(
        self,
        messages: Sequence[StreamMessage],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]: ...


def scrub_chunk(chunk: str) -> str:
    """Run a single streamed chunk through the data-residency guard."""
    if not chunk:
        return chunk
    scrubbed, _ = _GUARD.scrub_text(chunk)
    return scrubbed


class StaticChunkSource:
    """Yield a fixed list of chunks — used in tests."""

    name = "static"

    def __init__(self, chunks: Iterable[str], *, delay: float = 0.0) -> None:
        self._chunks: List[str] = list(chunks)
        self._delay = delay

    async def stream(
        self,
        messages: Sequence[StreamMessage],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        for c in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield c


class OfflineEchoSource:
    """Deterministic offline source for first-run UX without any key.

    Emits a short, clearly-labelled response so the user immediately
    knows the assistant is running in offline mode.  The reply is
    chunked word-by-word to exercise the SSE framing path.
    """

    name = "offline"

    PREFIX = "[offline mode] No API key is configured. Echo: "

    async def stream(
        self,
        messages: Sequence[StreamMessage],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        reply = self.PREFIX + last_user
        # Re-emit one word at a time so the UI receives a real stream.
        for tok in reply.split(" "):
            yield tok + " "


class OpenAIStreamSource:
    """Stream chunks from OpenAI using the key stored in the local vault.

    Falls back to :class:`OfflineEchoSource` semantics at call time if
    the vault has no key for the configured provider — callers should
    use :func:`pick_default_source` to choose, not instantiate this
    directly.
    """

    name = "openai"

    def __init__(self, *, vault: Optional[KeyVault] = None,
                 model: str = "gpt-4o-mini") -> None:
        self._vault = vault or get_key_vault()
        self._model = model

    async def stream(
        self,
        messages: Sequence[StreamMessage],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        key = self._vault.get("openai")
        if not key:
            async for c in OfflineEchoSource().stream(
                messages, temperature=temperature, max_tokens=max_tokens
            ):
                yield c
            return
        import openai  # local import keeps it optional at module load
        client = openai.AsyncOpenAI(api_key=key)
        stream = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


def pick_default_source() -> ChunkSource:
    """Choose the best available chunk source for the current environment."""
    try:
        vault = get_key_vault()
        if vault.get("openai"):
            return OpenAIStreamSource(vault=vault)
    except Exception as exc:
        logger.debug("vault lookup failed (%s); using offline source", exc)
    return OfflineEchoSource()

"""Tests for ``app.local.chat_streamer``."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from app.local import chat_streamer
from app.local.chat_streamer import (
    OfflineEchoSource,
    OpenAIStreamSource,
    StaticChunkSource,
    StreamMessage,
    pick_default_source,
    scrub_chunk,
)


# ---------------------------------------------------------------------------
# Message validation
# ---------------------------------------------------------------------------


class TestStreamMessage:
    def test_valid_roles_accepted(self) -> None:
        for role in ("system", "user", "assistant"):
            StreamMessage(role, "hello")

    def test_invalid_role_rejected(self) -> None:
        with pytest.raises(ValueError):
            StreamMessage("root", "hi")

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValueError):
            StreamMessage("user", "")


# ---------------------------------------------------------------------------
# Scrub guard
# ---------------------------------------------------------------------------


class TestScrubChunk:
    def test_passthrough_when_no_pii(self) -> None:
        assert scrub_chunk("hello world") == "hello world"

    def test_redacts_email(self) -> None:
        out = scrub_chunk("contact alice@example.com")
        assert "alice@example.com" not in out
        assert "<email_redacted>" in out

    def test_handles_empty(self) -> None:
        assert scrub_chunk("") == ""


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class TestStaticChunkSource:
    @pytest.mark.asyncio
    async def test_yields_in_order(self) -> None:
        src = StaticChunkSource(["a", "b", "c"])
        out: List[str] = []
        async for c in src.stream([StreamMessage("user", "hi")]):
            out.append(c)
        assert out == ["a", "b", "c"]


class TestOfflineEchoSource:
    @pytest.mark.asyncio
    async def test_echo_includes_last_user_message(self) -> None:
        msgs = [
            StreamMessage("system", "be helpful"),
            StreamMessage("user", "hello"),
        ]
        chunks: List[str] = []
        async for c in OfflineEchoSource().stream(msgs):
            chunks.append(c)
        joined = "".join(chunks)
        assert "offline mode" in joined.lower()
        assert "hello" in joined

    @pytest.mark.asyncio
    async def test_emits_multiple_chunks(self) -> None:
        chunks: List[str] = []
        async for c in OfflineEchoSource().stream(
            [StreamMessage("user", "alpha beta gamma")]
        ):
            chunks.append(c)
        assert len(chunks) > 1


class TestOpenAIStreamSourceFallback:
    @pytest.mark.asyncio
    async def test_falls_back_to_echo_when_no_key(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.local import key_vault as kv
        monkeypatch.setenv("SMR_KEY_VAULT_BACKEND", "file")
        kv.reset_key_vault()
        src = OpenAIStreamSource()
        chunks: List[str] = []
        async for c in src.stream([StreamMessage("user", "ping")]):
            chunks.append(c)
        assert "offline mode" in "".join(chunks).lower()
        kv.reset_key_vault()


class TestPickDefaultSource:
    def test_offline_when_vault_empty(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.local import key_vault as kv
        monkeypatch.setenv("SMR_KEY_VAULT_BACKEND", "file")
        kv.reset_key_vault()
        src = pick_default_source()
        assert isinstance(src, OfflineEchoSource)
        kv.reset_key_vault()

    def test_openai_when_vault_has_key(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.local import key_vault as kv
        monkeypatch.setenv("SMR_KEY_VAULT_BACKEND", "file")
        kv.reset_key_vault()
        kv.get_key_vault().set("openai", "sk-" + "a" * 40)
        src = pick_default_source()
        assert isinstance(src, OpenAIStreamSource)
        kv.reset_key_vault()

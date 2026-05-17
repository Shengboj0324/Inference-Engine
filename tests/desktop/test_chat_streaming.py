"""SSE streaming tests for ``POST /api/v1/chat/sessions/{id}/stream``."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, List

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.api.routes import chat as chat_route
from app.core.config import settings
from app.local import chat_store, chat_streamer, key_vault


@pytest.fixture
def desktop_client(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "deployment_mode", "desktop")
    monkeypatch.setenv("SMR_KEY_VAULT_BACKEND", "file")
    chat_store.reset_chat_store()
    key_vault.reset_key_vault()
    with TestClient(app) as client:
        yield client
    chat_store.reset_chat_store()
    key_vault.reset_key_vault()


def _parse_sse(raw: str) -> List[dict]:
    """Decode an SSE response body into a list of ``{event, data}`` dicts.

    Tolerant of the bare ``data: [DONE]`` sentinel emitted at end-of-stream.
    """
    frames: List[dict] = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = "message"
        data_lines: List[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if not data_lines:
            continue
        raw_data = "\n".join(data_lines)
        if raw_data == "[DONE]":
            frames.append({"event": "done", "data": None})
            continue
        try:
            frames.append({"event": event, "data": json.loads(raw_data)})
        except json.JSONDecodeError:
            frames.append({"event": event, "data": raw_data})
    return frames


class TestStreamHappyPath:
    def test_offline_stream_emits_deltas_and_completes(
        self, desktop_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = chat_streamer.StaticChunkSource(["hello ", "world", "!"])
        monkeypatch.setattr(chat_streamer, "pick_default_source", lambda: src)

        sid = desktop_client.post(
            "/api/v1/chat/sessions", json={"title": "x"}
        ).json()["id"]
        with desktop_client.stream(
            "POST",
            f"/api/v1/chat/sessions/{sid}/stream",
            json={"content": "go"},
        ) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())

        frames = _parse_sse(body)
        events = [f["event"] for f in frames]
        assert events[0] == "user_message"
        deltas = [f["data"]["text"] for f in frames if f["event"] == "delta"]
        assert deltas == ["hello ", "world", "!"]
        assert any(f["event"] == "assistant_message" for f in frames)
        assert frames[-1]["event"] == "done"

    def test_assistant_message_persisted_after_stream(
        self, desktop_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = chat_streamer.StaticChunkSource(["abc", "def"])
        monkeypatch.setattr(chat_streamer, "pick_default_source", lambda: src)

        sid = desktop_client.post(
            "/api/v1/chat/sessions", json={"title": "x"}
        ).json()["id"]
        with desktop_client.stream(
            "POST",
            f"/api/v1/chat/sessions/{sid}/stream",
            json={"content": "go"},
        ) as r:
            list(r.iter_text())

        detail = desktop_client.get(f"/api/v1/chat/sessions/{sid}").json()
        roles = [m["role"] for m in detail["messages"]]
        assert roles == ["user", "assistant"]
        assert detail["messages"][1]["content"] == "abcdef"


class TestStreamPIIScrubbing:
    def test_email_scrubbed_in_delta_frames(
        self, desktop_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = chat_streamer.StaticChunkSource(
            ["my email is ", "bob@example.com", " thanks"]
        )
        monkeypatch.setattr(chat_streamer, "pick_default_source", lambda: src)

        sid = desktop_client.post(
            "/api/v1/chat/sessions", json={"title": "x"}
        ).json()["id"]
        with desktop_client.stream(
            "POST",
            f"/api/v1/chat/sessions/{sid}/stream",
            json={"content": "go"},
        ) as r:
            body = "".join(r.iter_text())

        assert "bob@example.com" not in body
        assert "<email_redacted>" in body


class TestStreamErrors:
    def test_stream_for_missing_session_404(self, desktop_client: TestClient) -> None:
        r = desktop_client.post(
            "/api/v1/chat/sessions/nope/stream", json={"content": "go"}
        )
        assert r.status_code == 404

    def test_source_failure_yields_error_frame(
        self, desktop_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Boom:
            name = "boom"

            async def stream(self, messages, *, temperature=0.7, max_tokens=None):
                yield "first "
                raise RuntimeError("planned failure")

        monkeypatch.setattr(chat_streamer, "pick_default_source", lambda: _Boom())

        sid = desktop_client.post(
            "/api/v1/chat/sessions", json={"title": "x"}
        ).json()["id"]
        with desktop_client.stream(
            "POST",
            f"/api/v1/chat/sessions/{sid}/stream",
            json={"content": "go"},
        ) as r:
            body = "".join(r.iter_text())
        frames = _parse_sse(body)
        events = [f["event"] for f in frames]
        assert "error" in events
        # No assistant_message was committed on failure.
        assert "assistant_message" not in events

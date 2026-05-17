"""Integration tests for ``/api/v1/chat`` (sessions + non-streaming reply)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
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


class TestSessionsCRUD:
    def test_create_then_list(self, desktop_client: TestClient) -> None:
        r = desktop_client.post("/api/v1/chat/sessions", json={"title": "first"})
        assert r.status_code == 201
        sid = r.json()["id"]

        r2 = desktop_client.get("/api/v1/chat/sessions")
        assert r2.status_code == 200
        assert any(s["id"] == sid for s in r2.json())

    def test_get_session_returns_messages(self, desktop_client: TestClient) -> None:
        sid = desktop_client.post(
            "/api/v1/chat/sessions", json={"title": "x"}
        ).json()["id"]
        r = desktop_client.get(f"/api/v1/chat/sessions/{sid}")
        assert r.status_code == 200
        body = r.json()
        assert body["session"]["id"] == sid
        assert body["messages"] == []

    def test_get_missing_session_404(self, desktop_client: TestClient) -> None:
        r = desktop_client.get("/api/v1/chat/sessions/does-not-exist")
        assert r.status_code == 404

    def test_rename_session(self, desktop_client: TestClient) -> None:
        sid = desktop_client.post(
            "/api/v1/chat/sessions", json={"title": "old"}
        ).json()["id"]
        r = desktop_client.patch(
            f"/api/v1/chat/sessions/{sid}", json={"title": "renamed"}
        )
        assert r.status_code == 200
        assert r.json()["title"] == "renamed"

    def test_delete_session(self, desktop_client: TestClient) -> None:
        sid = desktop_client.post(
            "/api/v1/chat/sessions", json={"title": "x"}
        ).json()["id"]
        r = desktop_client.delete(f"/api/v1/chat/sessions/{sid}")
        assert r.status_code == 200
        # Now 404 on follow-up GET.
        r2 = desktop_client.get(f"/api/v1/chat/sessions/{sid}")
        assert r2.status_code == 404


class TestNonStreamingReply:
    def test_offline_reply_echoes_user(self, desktop_client: TestClient) -> None:
        sid = desktop_client.post(
            "/api/v1/chat/sessions", json={"title": "x"}
        ).json()["id"]
        r = desktop_client.post(
            f"/api/v1/chat/sessions/{sid}/messages", json={"content": "ping"}
        )
        assert r.status_code == 201
        body = r.json()
        assert body["source"] == "offline"
        assert body["user_message"]["content"] == "ping"
        assert "ping" in body["assistant_message"]["content"]
        assert body["assistant_message"]["role"] == "assistant"

    def test_reply_persists_in_session(self, desktop_client: TestClient) -> None:
        sid = desktop_client.post(
            "/api/v1/chat/sessions", json={"title": "x"}
        ).json()["id"]
        desktop_client.post(
            f"/api/v1/chat/sessions/{sid}/messages", json={"content": "hello"}
        )
        detail = desktop_client.get(f"/api/v1/chat/sessions/{sid}").json()
        assert len(detail["messages"]) == 2
        assert detail["messages"][0]["role"] == "user"
        assert detail["messages"][1]["role"] == "assistant"

    def test_post_to_missing_session_404(self, desktop_client: TestClient) -> None:
        r = desktop_client.post(
            "/api/v1/chat/sessions/nope/messages", json={"content": "hi"}
        )
        assert r.status_code == 404

    def test_pii_scrubbed_in_reply(
        self, desktop_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Inject a static source that emits a chunk containing an email.
        src = chat_streamer.StaticChunkSource(["please contact alice@example.com now"])
        monkeypatch.setattr(chat_streamer, "pick_default_source", lambda: src)
        sid = desktop_client.post(
            "/api/v1/chat/sessions", json={"title": "x"}
        ).json()["id"]
        r = desktop_client.post(
            f"/api/v1/chat/sessions/{sid}/messages", json={"content": "go"}
        )
        assert r.status_code == 201
        content = r.json()["assistant_message"]["content"]
        assert "alice@example.com" not in content
        assert "<email_redacted>" in content


class TestServerModeGating:
    def test_chat_routes_404_in_server_mode(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "deployment_mode", "server")
        client = TestClient(app)
        assert client.get("/api/v1/chat/sessions").status_code == 404
        assert client.post(
            "/api/v1/chat/sessions", json={"title": "x"}
        ).status_code == 404

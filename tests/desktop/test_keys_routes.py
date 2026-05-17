"""Integration tests for ``/api/v1/keys`` (desktop-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.config import settings
from app.local import key_vault as kv


@pytest.fixture
def desktop_client(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "deployment_mode", "desktop")
    # Force the file backend for deterministic, hermetic tests.
    monkeypatch.setenv("SMR_KEY_VAULT_BACKEND", "file")
    kv.reset_key_vault()
    with TestClient(app) as client:
        yield client
    kv.reset_key_vault()


@pytest.fixture
def server_client(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "deployment_mode", "server")
    monkeypatch.setenv("SMR_KEY_VAULT_BACKEND", "file")
    kv.reset_key_vault()
    # We cannot enter the lifespan in server mode (it would probe Redis),
    # so use TestClient without the context manager.
    yield TestClient(app)
    kv.reset_key_vault()


class TestKeysListing:
    def test_lists_all_providers_initially_empty(self, desktop_client: TestClient) -> None:
        r = desktop_client.get("/api/v1/keys")
        assert r.status_code == 200
        body = r.json()
        ids = {entry["id"] for entry in body}
        assert ids == {"openai", "anthropic", "openrouter"}
        assert all(entry["has_key"] is False for entry in body)
        assert all(entry["masked"] is None for entry in body)


class TestKeysUpsert:
    def test_put_then_list_shows_masked(self, desktop_client: TestClient) -> None:
        secret = "sk-" + "a" * 40
        r = desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": secret}
        )
        assert r.status_code == 200
        assert r.json()["masked"] == "sk-a…aaaa"
        assert "a" * 10 not in r.text  # raw key never echoed

        r2 = desktop_client.get("/api/v1/keys")
        openai = next(e for e in r2.json() if e["id"] == "openai")
        assert openai["has_key"] is True
        assert openai["masked"] == "sk-a…aaaa"

    def test_put_rejects_invalid_key(self, desktop_client: TestClient) -> None:
        r = desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": "short"}
        )
        assert r.status_code == 400
        assert "short" in r.json()["detail"]

    def test_put_rejects_unknown_provider(self, desktop_client: TestClient) -> None:
        r = desktop_client.put(
            "/api/v1/keys/cohere", json={"api_key": "sk-" + "a" * 40}
        )
        assert r.status_code == 400

    def test_put_overwrites(self, desktop_client: TestClient) -> None:
        desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": "sk-" + "a" * 40}
        )
        r = desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": "sk-" + "b" * 40}
        )
        assert r.status_code == 200
        assert r.json()["masked"] == "sk-b…bbbb"


class TestKeysDelete:
    def test_delete_existing(self, desktop_client: TestClient) -> None:
        desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": "sk-" + "a" * 40}
        )
        r = desktop_client.delete("/api/v1/keys/openai")
        assert r.status_code == 200
        assert r.json() == {"provider": "openai", "removed": True}

    def test_delete_missing(self, desktop_client: TestClient) -> None:
        r = desktop_client.delete("/api/v1/keys/openai")
        assert r.status_code == 200
        assert r.json() == {"provider": "openai", "removed": False}


class TestKeysTest:
    def test_test_endpoint_reports_missing(self, desktop_client: TestClient) -> None:
        r = desktop_client.post("/api/v1/keys/openai/test")
        assert r.status_code == 200
        assert r.json() == {"provider": "openai", "ok": False, "detail": "no key stored"}

    def test_test_endpoint_reports_ok(self, desktop_client: TestClient) -> None:
        desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": "sk-" + "a" * 40}
        )
        r = desktop_client.post("/api/v1/keys/openai/test")
        assert r.status_code == 200
        assert r.json()["ok"] is True


class TestServerModeGating:
    def test_routes_return_404_in_server_mode(self, server_client: TestClient) -> None:
        for path, method in (
            ("/api/v1/keys", "get"),
            ("/api/v1/keys/openai", "delete"),
            ("/api/v1/keys/openai/test", "post"),
        ):
            r = getattr(server_client, method)(path)
            assert r.status_code == 404, (path, method, r.text)

    def test_put_returns_404_in_server_mode(self, server_client: TestClient) -> None:
        r = server_client.put(
            "/api/v1/keys/openai", json={"api_key": "sk-" + "a" * 40}
        )
        assert r.status_code == 404

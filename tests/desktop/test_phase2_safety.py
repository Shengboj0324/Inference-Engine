"""Cross-cutting safety guarantees for the Phase 2 desktop surface.

These tests do not exercise functional happy-paths (which live in the
per-pillar suites); they assert the *negative space* that the desktop
sidecar must never violate:

* Raw key material never appears in any response body or log line on
  any keys-route surface.
* The Phase 2 router set is unreachable when the API is started in
  server (multi-tenant) deployment mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.config import settings
from app.local import key_vault, permissions as perm_mod, chat_store


_RAW_OPENAI_KEY = "sk-test-" + "a" * 40  # >= 20 chars, base64-ish


@pytest.fixture
def desktop_client(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "deployment_mode", "desktop")
    monkeypatch.setenv("SMR_KEY_VAULT_BACKEND", "file")
    key_vault.reset_key_vault()
    perm_mod.reset_permission_manager()
    chat_store.reset_chat_store()
    with TestClient(app) as client:
        yield client
    key_vault.reset_key_vault()
    perm_mod.reset_permission_manager()
    chat_store.reset_chat_store()


class TestNoKeyLeakInResponses:
    def test_upsert_response_does_not_contain_raw_key(
        self, desktop_client: TestClient
    ) -> None:
        r = desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": _RAW_OPENAI_KEY}
        )
        assert r.status_code == 200
        assert _RAW_OPENAI_KEY not in r.text

    def test_list_response_does_not_contain_raw_key(
        self, desktop_client: TestClient
    ) -> None:
        desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": _RAW_OPENAI_KEY}
        )
        r = desktop_client.get("/api/v1/keys")
        assert r.status_code == 200
        assert _RAW_OPENAI_KEY not in r.text

    def test_test_response_does_not_contain_raw_key(
        self, desktop_client: TestClient
    ) -> None:
        desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": _RAW_OPENAI_KEY}
        )
        r = desktop_client.post("/api/v1/keys/openai/test")
        assert r.status_code == 200
        assert _RAW_OPENAI_KEY not in r.text

    def test_validation_error_does_not_echo_rejected_key(
        self, desktop_client: TestClient
    ) -> None:
        bad = "sk-bad-" + "@" * 30  # min_length OK; characters fail
        r = desktop_client.put("/api/v1/keys/openai", json={"api_key": bad})
        assert r.status_code == 400
        assert bad not in r.text


class TestNoKeyLeakInLogs:
    def test_upsert_does_not_log_raw_key(
        self, desktop_client: TestClient, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.DEBUG)
        r = desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": _RAW_OPENAI_KEY}
        )
        assert r.status_code == 200
        for record in caplog.records:
            assert _RAW_OPENAI_KEY not in record.getMessage()
            for arg in (record.args or ()) if isinstance(record.args, tuple) else ():
                assert _RAW_OPENAI_KEY not in str(arg)


class TestServerModeBlocksPhase2:
    def test_keys_routes_404_in_server_mode(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "deployment_mode", "server")
        client = TestClient(app)
        assert client.get("/api/v1/keys").status_code == 404
        assert client.put(
            "/api/v1/keys/openai", json={"api_key": _RAW_OPENAI_KEY}
        ).status_code == 404
        assert client.post("/api/v1/keys/openai/test").status_code == 404

    def test_chat_routes_404_in_server_mode(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "deployment_mode", "server")
        client = TestClient(app)
        assert client.get("/api/v1/chat/sessions").status_code == 404

    def test_permissions_routes_404_in_server_mode(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "deployment_mode", "server")
        client = TestClient(app)
        assert client.get("/api/v1/permissions").status_code == 404


class TestContractIncludesPhase2:
    def test_frozen_routes_contains_phase2_endpoints(self) -> None:
        from tests.contract.test_public_api_surface import (
            FROZEN_ROUTES,
            CONTRACT_VERSION,
        )

        assert CONTRACT_VERSION == "phase2.0"
        # Spot-check one route from each new router.
        assert ("APIRoute", "/api/v1/keys", ("GET",)) in FROZEN_ROUTES
        assert (
            "APIRoute",
            "/api/v1/chat/sessions/{session_id}/stream",
            ("POST",),
        ) in FROZEN_ROUTES
        assert ("APIRoute", "/api/v1/permissions", ("GET",)) in FROZEN_ROUTES

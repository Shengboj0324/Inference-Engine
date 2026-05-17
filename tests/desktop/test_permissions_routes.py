"""Integration tests for ``/api/v1/permissions``."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.config import settings
from app.local import permissions as perm_mod


@pytest.fixture
def desktop_client(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "deployment_mode", "desktop")
    perm_mod.reset_permission_manager()
    with TestClient(app) as client:
        yield client
    perm_mod.reset_permission_manager()


class TestListAndGet:
    def test_list_returns_all_permissions(self, desktop_client: TestClient) -> None:
        r = desktop_client.get("/api/v1/permissions")
        assert r.status_code == 200
        names = {g["permission"] for g in r.json()}
        assert names == {p.value for p in perm_mod.Permission}

    def test_default_decision_is_ask(self, desktop_client: TestClient) -> None:
        r = desktop_client.get("/api/v1/permissions/read_files")
        assert r.status_code == 200
        assert r.json()["decision"] == "ask"

    def test_unknown_permission_returns_400(self, desktop_client: TestClient) -> None:
        r = desktop_client.get("/api/v1/permissions/does_not_exist")
        assert r.status_code == 400


class TestGrantRevoke:
    def test_grant_then_get_shows_grant(self, desktop_client: TestClient) -> None:
        r = desktop_client.post(
            "/api/v1/permissions/read_files/grant",
            json={"note": "user clicked allow"},
        )
        assert r.status_code == 200
        assert r.json()["decision"] == "grant"
        assert r.json()["note"] == "user clicked allow"
        assert r.json()["expires_at"] is None

        r2 = desktop_client.get("/api/v1/permissions/read_files")
        assert r2.json()["decision"] == "grant"

    def test_grant_with_ttl_sets_expiry(self, desktop_client: TestClient) -> None:
        r = desktop_client.post(
            "/api/v1/permissions/network_fetch/grant",
            json={"ttl_seconds": 120},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["decision"] == "grant"
        assert body["expires_at"] is not None
        assert body["granted_at"] is not None
        assert body["expires_at"] > body["granted_at"]

    def test_grant_rejects_non_positive_ttl(self, desktop_client: TestClient) -> None:
        r = desktop_client.post(
            "/api/v1/permissions/network_fetch/grant",
            json={"ttl_seconds": 0},
        )
        assert r.status_code == 422

    def test_revoke_returns_ask(self, desktop_client: TestClient) -> None:
        desktop_client.post("/api/v1/permissions/write_files/grant", json={})
        r = desktop_client.post(
            "/api/v1/permissions/write_files/revoke", json={"note": "changed mind"}
        )
        assert r.status_code == 200
        assert r.json()["decision"] == "ask"

    def test_deny_blocks_future_grants_until_revoked(
        self, desktop_client: TestClient
    ) -> None:
        r = desktop_client.post(
            "/api/v1/permissions/execute_shell/deny",
            json={"note": "never"},
        )
        assert r.status_code == 200
        assert r.json()["decision"] == "deny"
        # Re-fetch confirms persistence.
        r2 = desktop_client.get("/api/v1/permissions/execute_shell")
        assert r2.json()["decision"] == "deny"


class TestServerModeGating:
    def test_routes_404_in_server_mode(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "deployment_mode", "server")
        client = TestClient(app)
        assert client.get("/api/v1/permissions").status_code == 404
        assert client.post(
            "/api/v1/permissions/read_files/grant", json={}
        ).status_code == 404

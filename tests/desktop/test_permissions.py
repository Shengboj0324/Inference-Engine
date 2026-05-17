"""Tests for ``app.local.permissions``."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.local.permissions import (
    Decision,
    Permission,
    PermissionError,
    PermissionManager,
)


@pytest.fixture
def mgr(tmp_path: Path) -> PermissionManager:
    return PermissionManager(path=tmp_path / "permissions.json")


class TestDefaults:
    def test_get_unknown_returns_ask(self, mgr: PermissionManager) -> None:
        g = mgr.get(Permission.READ_FILES)
        assert g.decision is Decision.ASK
        assert g.granted_at is None
        assert g.expires_at is None

    def test_list_grants_includes_every_permission(self, mgr: PermissionManager) -> None:
        grants = mgr.list_grants()
        names = {g.permission for g in grants}
        assert names == set(Permission)

    def test_is_allowed_false_by_default(self, mgr: PermissionManager) -> None:
        for p in Permission:
            assert mgr.is_allowed(p) is False


class TestGrantAndRevoke:
    def test_grant_then_is_allowed(self, mgr: PermissionManager) -> None:
        mgr.grant(Permission.READ_FILES)
        assert mgr.is_allowed(Permission.READ_FILES) is True

    def test_revoke_reverts_to_ask(self, mgr: PermissionManager) -> None:
        mgr.grant(Permission.READ_FILES)
        mgr.revoke(Permission.READ_FILES)
        assert mgr.get(Permission.READ_FILES).decision is Decision.ASK
        assert mgr.is_allowed(Permission.READ_FILES) is False

    def test_deny_blocks(self, mgr: PermissionManager) -> None:
        mgr.deny(Permission.EXECUTE_SHELL, note="dangerous")
        g = mgr.get(Permission.EXECUTE_SHELL)
        assert g.decision is Decision.DENY
        assert g.note == "dangerous"
        assert mgr.is_allowed(Permission.EXECUTE_SHELL) is False

    def test_require_raises_when_not_granted(self, mgr: PermissionManager) -> None:
        with pytest.raises(PermissionError):
            mgr.require(Permission.NETWORK_FETCH)

    def test_require_silent_when_granted(self, mgr: PermissionManager) -> None:
        mgr.grant(Permission.NETWORK_FETCH)
        mgr.require(Permission.NETWORK_FETCH)


class TestTTL:
    def test_grant_with_zero_or_negative_ttl_rejected(self, mgr: PermissionManager) -> None:
        with pytest.raises(ValueError):
            mgr.grant(Permission.READ_FILES, ttl_seconds=0)
        with pytest.raises(ValueError):
            mgr.grant(Permission.READ_FILES, ttl_seconds=-1)

    def test_grant_with_ttl_records_expires_at(self, mgr: PermissionManager) -> None:
        before = time.time()
        g = mgr.grant(Permission.READ_FILES, ttl_seconds=60)
        assert g.expires_at is not None
        assert g.expires_at >= before + 59

    def test_expired_grant_degrades_to_ask(self, mgr: PermissionManager) -> None:
        g = mgr.grant(Permission.READ_FILES, ttl_seconds=0.05)
        assert g.is_active() is True
        time.sleep(0.1)
        after = mgr.get(Permission.READ_FILES)
        assert after.decision is Decision.ASK
        assert mgr.is_allowed(Permission.READ_FILES) is False

    def test_expired_grant_degradation_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "permissions.json"
        m1 = PermissionManager(path=path)
        m1.grant(Permission.READ_FILES, ttl_seconds=0.05)
        time.sleep(0.1)
        m1.get(Permission.READ_FILES)  # triggers persisted degradation
        m2 = PermissionManager(path=path)
        assert m2.get(Permission.READ_FILES).decision is Decision.ASK


class TestPersistence:
    def test_grant_survives_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "permissions.json"
        PermissionManager(path=path).grant(Permission.WRITE_FILES, note="ok")
        m2 = PermissionManager(path=path)
        g = m2.get(Permission.WRITE_FILES)
        assert g.decision is Decision.GRANT
        assert g.note == "ok"

    def test_corrupted_file_starts_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "permissions.json"
        path.write_text("{not valid json", encoding="utf-8")
        m = PermissionManager(path=path)
        for p in Permission:
            assert m.get(p).decision is Decision.ASK

    def test_unknown_permission_in_file_ignored(self, tmp_path: Path) -> None:
        path = tmp_path / "permissions.json"
        path.write_text(
            '{"grants":[{"permission":"made_up","decision":"grant"},'
            '{"permission":"read_files","decision":"grant"}]}',
            encoding="utf-8",
        )
        m = PermissionManager(path=path)
        assert m.get(Permission.READ_FILES).decision is Decision.GRANT


class TestThreadSafety:
    def test_concurrent_grant_revoke(self, mgr: PermissionManager) -> None:
        import threading

        def hammer() -> None:
            for _ in range(50):
                mgr.grant(Permission.READ_FILES)
                mgr.revoke(Permission.READ_FILES)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # End state is well-defined (revoke last) and file is intact.
        assert mgr.get(Permission.READ_FILES).decision is Decision.ASK

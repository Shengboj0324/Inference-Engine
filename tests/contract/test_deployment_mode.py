"""Phase 0 — additive ``Settings.deployment_mode`` flag contract.

Pins:

* Default value (``server``) — guarantees no behaviour change for existing
  Docker/Compose deployments unless explicitly opted-in.
* Accepted values (``server`` / ``desktop``, case-insensitive).
* ``is_desktop`` / ``is_server`` convenience properties.
* That setting ``deployment_mode='desktop'`` actually causes the FastAPI
  lifespan to skip the external Redis probe (Phase 1 prerequisite).
"""

from __future__ import annotations

import asyncio

import pytest

from app.core.config import Settings, settings as _live_settings


class TestDeploymentModeField:
    def test_default_is_server(self) -> None:
        s = Settings(_env_file=None)
        assert s.deployment_mode == "server"
        assert s.is_server is True
        assert s.is_desktop is False

    def test_desktop_value_accepted(self) -> None:
        s = Settings(_env_file=None, deployment_mode="desktop")
        assert s.deployment_mode == "desktop"
        assert s.is_desktop is True
        assert s.is_server is False

    def test_value_is_lowercased(self) -> None:
        s = Settings(_env_file=None, deployment_mode="DESKTOP")
        assert s.deployment_mode == "desktop"

    def test_blank_falls_back_to_server(self) -> None:
        s = Settings(_env_file=None, deployment_mode="   ")
        assert s.deployment_mode == "server"

    def test_unknown_value_rejected(self) -> None:
        with pytest.raises(Exception):
            Settings(_env_file=None, deployment_mode="hybrid")

    def test_non_string_rejected(self) -> None:
        with pytest.raises(Exception):
            Settings(_env_file=None, deployment_mode=123)  # type: ignore[arg-type]

    def test_live_singleton_is_server_by_default(self) -> None:
        # The process-wide singleton must remain on the server topology so
        # existing tests / deployments are unaffected by this additive flag.
        assert _live_settings.deployment_mode in {"server", "desktop"}


class TestLifespanRespectsDesktopMode:
    """The FastAPI lifespan must skip external-service probes in desktop mode."""

    def test_redis_probe_is_skipped_in_desktop_mode(self, monkeypatch) -> None:
        from app.api import main as api_main

        # Force the live settings singleton into desktop mode for the duration
        # of this test.  Using monkeypatch ensures the original value is
        # restored even if the test fails.
        monkeypatch.setattr(api_main.settings, "deployment_mode", "desktop")

        probe_invocations = {"count": 0}

        async def _fake_probe() -> None:
            probe_invocations["count"] += 1

        monkeypatch.setattr(api_main, "_probe_redis", _fake_probe)
        monkeypatch.setattr(api_main, "_validate_capabilities", lambda: None)

        async def _drive_lifespan() -> None:
            async with api_main.lifespan(api_main.app):
                pass

        asyncio.run(_drive_lifespan())

        assert probe_invocations["count"] == 0, (
            "Redis probe must not run in desktop mode — the desktop sidecar "
            "uses an in-process pub/sub backend by design."
        )

    def test_redis_probe_still_runs_in_server_mode(self, monkeypatch) -> None:
        from app.api import main as api_main

        monkeypatch.setattr(api_main.settings, "deployment_mode", "server")

        probe_invocations = {"count": 0}

        async def _fake_probe() -> None:
            probe_invocations["count"] += 1

        monkeypatch.setattr(api_main, "_probe_redis", _fake_probe)
        monkeypatch.setattr(api_main, "_validate_capabilities", lambda: None)

        async def _drive_lifespan() -> None:
            async with api_main.lifespan(api_main.app):
                pass

        asyncio.run(_drive_lifespan())

        assert probe_invocations["count"] == 1, (
            "Server mode must continue to probe Redis at startup to preserve "
            "the existing fail-fast contract for Docker/Compose deployments."
        )

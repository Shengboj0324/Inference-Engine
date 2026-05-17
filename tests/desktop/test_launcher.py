"""Tests for ``app.desktop.launcher`` and the ``python -m app.desktop`` entry."""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from app.desktop import launcher


class TestFindFreePort:
    def test_returns_bindable_loopback_port(self) -> None:
        port = launcher.find_free_port()
        assert 1024 <= port <= 65535
        # The returned port must be immediately bindable on 127.0.0.1.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", port))

    def test_two_calls_return_different_ports_under_load(self) -> None:
        seen = {launcher.find_free_port() for _ in range(20)}
        # The OS may legitimately reuse a port across rapid close→reopen, but
        # over 20 calls we expect at least a handful of distinct values.
        assert len(seen) >= 3


class TestWritePortFile:
    def test_writes_port_and_is_atomic(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "sidecar.port"
        written = launcher.write_port_file(54321, target=target)
        assert written == target
        assert target.read_text(encoding="utf-8") == "54321"
        # No stray temp files left behind.
        tmps = list(target.parent.glob("sidecar.port.tmp.*"))
        assert tmps == []

    def test_default_target_is_user_data_dir(
        self, tmp_data_dir: Path
    ) -> None:
        written = launcher.write_port_file(42)
        assert written == tmp_data_dir / "data" / "sidecar.port"
        assert written.read_text(encoding="utf-8") == "42"

    def test_overwrite_replaces_content(self, tmp_path: Path) -> None:
        target = tmp_path / "p"
        launcher.write_port_file(1111, target=target)
        launcher.write_port_file(2222, target=target)
        assert target.read_text(encoding="utf-8") == "2222"


class TestBuildConfig:
    def test_auto_port_when_zero(self, tmp_path: Path) -> None:
        cfg = launcher.build_config(
            port=0, port_file=tmp_path / "p",
        )
        assert cfg.port != 0
        assert cfg.host == "127.0.0.1"

    def test_respects_explicit_port(self, tmp_path: Path) -> None:
        cfg = launcher.build_config(
            port=51234, port_file=tmp_path / "p",
        )
        assert cfg.port == 51234

    def test_host_defaults_to_loopback(self, tmp_path: Path) -> None:
        cfg = launcher.build_config(port=12345, port_file=tmp_path / "p")
        assert cfg.host == "127.0.0.1"


class TestModuleEntryPoint:
    """The ``python -m app.desktop`` shim must remain import-side-effect free."""

    def test_main_module_importable(self) -> None:
        import importlib
        mod = importlib.import_module("app.desktop.__main__")
        assert hasattr(mod, "main")
        assert callable(mod.main)

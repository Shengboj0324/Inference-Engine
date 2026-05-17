"""Tests for ``app.local.user_data_dir``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from app.local import user_data_dir as udd


class TestUserDataDir:
    def test_env_override_wins_verbatim(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SMR_DATA_DIR", str(tmp_path / "x" / "y"))
        resolved = udd.get_user_data_dir(ensure=True)
        assert resolved == tmp_path / "x" / "y"
        assert resolved.is_dir()

    def test_ensure_creates_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "fresh"
        monkeypatch.setenv("SMR_DATA_DIR", str(target))
        assert not target.exists()
        resolved = udd.get_user_data_dir(ensure=True)
        assert resolved.is_dir()

    def test_no_ensure_does_not_create(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "never"
        monkeypatch.setenv("SMR_DATA_DIR", str(target))
        udd.get_user_data_dir(ensure=False)
        assert not target.exists()

    def test_cache_and_log_use_separate_envs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SMR_DATA_DIR", str(tmp_path / "d"))
        monkeypatch.setenv("SMR_CACHE_DIR", str(tmp_path / "c"))
        monkeypatch.setenv("SMR_LOG_DIR", str(tmp_path / "l"))
        assert udd.get_user_data_dir() == tmp_path / "d"
        assert udd.get_user_cache_dir() == tmp_path / "c"
        assert udd.get_user_log_dir() == tmp_path / "l"

    def test_unset_env_uses_platform_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SMR_DATA_DIR", raising=False)
        resolved = udd.get_user_data_dir(ensure=False)
        # The path must end in the canonical app suffix on every platform.
        assert resolved.name == "SocialMediaRadar"
        assert resolved.is_absolute()

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific path shape")
    def test_macos_default_path_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SMR_DATA_DIR", raising=False)
        resolved = udd.get_user_data_dir(ensure=False)
        assert "Library/Application Support/SocialMediaRadar" in str(resolved)

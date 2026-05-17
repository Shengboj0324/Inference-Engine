"""Cross-platform user data directory resolution.

Resolves the canonical per-user directory the desktop sidecar uses for
SQLite databases, object blobs, scheduler state, and rolling log files.

Paths
-----
* **macOS**   — ``~/Library/Application Support/SocialMediaRadar``
* **Windows** — ``%APPDATA%\\SocialMediaRadar``
* **Linux**   — ``$XDG_DATA_HOME/SocialMediaRadar`` (defaults to
  ``~/.local/share/SocialMediaRadar``).

A ``SMR_DATA_DIR`` environment variable overrides the resolved path for
tests, ephemeral installs, and operator-driven relocation.  When set the
directory is used verbatim — no platform-specific suffix is appended.

All resolution functions are pure (no side effects) except for the
optional ``mkdir`` call that creates the directory on first use; callers
opt in by passing ``ensure=True``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_APP_NAME = "SocialMediaRadar"
_OVERRIDE_ENV = "SMR_DATA_DIR"


def _platform_root() -> Path:
    """Return the platform-canonical root before the app-name suffix."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    if sys.platform.startswith("win"):
        # %APPDATA% is per-user roaming; fall back to ~ on exotic Windows envs.
        appdata = os.environ.get("APPDATA")
        return Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    # Linux / *BSD — honour XDG.
    xdg = os.environ.get("XDG_DATA_HOME")
    return Path(xdg) if xdg else Path.home() / ".local" / "share"


def _cache_platform_root() -> Path:
    """Platform-canonical cache root (separate from data dir on Linux/macOS)."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches"
    if sys.platform.startswith("win"):
        appdata = os.environ.get("LOCALAPPDATA")
        return Path(appdata) if appdata else Path.home() / "AppData" / "Local"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg) if xdg else Path.home() / ".cache"


def _log_platform_root() -> Path:
    """Platform-canonical logs root."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs"
    if sys.platform.startswith("win"):
        appdata = os.environ.get("LOCALAPPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Local"
        return base
    xdg = os.environ.get("XDG_STATE_HOME")
    return Path(xdg) if xdg else Path.home() / ".local" / "state"


def _resolve(override_env: str, fallback: Path, ensure: bool) -> Path:
    """Honour the ``SMR_DATA_DIR`` override; otherwise return ``fallback``."""
    override = os.environ.get(override_env)
    path = Path(override) if override else fallback
    if ensure:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_user_data_dir(ensure: bool = False) -> Path:
    """Return the persistent user-data directory (SQLite, blobs, settings).

    Args:
        ensure: When ``True`` the directory is created if missing.

    Returns:
        Absolute path to the data directory.
    """
    return _resolve(_OVERRIDE_ENV, _platform_root() / _APP_NAME, ensure)


def get_user_cache_dir(ensure: bool = False) -> Path:
    """Return the user-cache directory (rebuildable artefacts)."""
    return _resolve(
        "SMR_CACHE_DIR",
        _cache_platform_root() / _APP_NAME,
        ensure,
    )


def get_user_log_dir(ensure: bool = False) -> Path:
    """Return the rolling-log directory."""
    return _resolve(
        "SMR_LOG_DIR",
        _log_platform_root() / _APP_NAME,
        ensure,
    )

"""Tauri-sidecar launcher for the FastAPI backend.

Responsibilities
----------------
1. Force ``settings.deployment_mode = 'desktop'`` so external-service
   probes are skipped (see Phase 0 lifespan gate).
2. Resolve a free TCP port on ``127.0.0.1`` (loopback-only — the sidecar
   must never be reachable from another machine).
3. Atomically write that port to ``<user-data-dir>/sidecar.port`` so the
   Tauri shell can read it.
4. Run uvicorn programmatically with graceful shutdown semantics.

The module is import-side-effect free; everything happens in
:func:`run_sidecar` which is the only function ``__main__`` calls.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import socket
import stat
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from app.local.user_data_dir import get_user_data_dir

logger = logging.getLogger(__name__)

_PORT_FILENAME = "sidecar.port"
_MANIFEST_FILENAME = "sidecar.json"
_LOOPBACK = "127.0.0.1"

#: Environment variable the launcher sets before booting uvicorn so the
#: loopback-token middleware in ``app.api.main`` can enforce the per-
#: launch shared secret.  Tests never set this, so they are unaffected.
_TOKEN_ENV = "SMR_SIDECAR_TOKEN"


@dataclass
class SidecarConfig:
    """Resolved configuration for a single launcher invocation."""

    host: str
    port: int
    port_file: Path
    manifest_file: Path
    token: str
    log_level: str = "info"


def find_free_port(host: str = _LOOPBACK) -> int:
    """Return an OS-assigned free TCP port on ``host``.

    The socket is bound with ``SO_REUSEADDR`` and closed before returning
    so the port is immediately reusable by uvicorn.  Standard caveat: in
    the (nanosecond) window between close and bind another process could
    in theory grab the port — uvicorn will surface that as a clear
    ``OSError`` on startup, which is the safest behaviour for a single-
    user desktop app.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        return sock.getsockname()[1]


def write_port_file(port: int, *, target: Optional[Path] = None) -> Path:
    """Atomically write ``port`` to the sidecar port file.

    Writes to a sibling ``.tmp`` file and ``os.replace`` it into position
    so a partially-written file is never observable by the shell.
    """
    if target is None:
        target = get_user_data_dir(ensure=True) / _PORT_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    tmp.write_text(str(int(port)), encoding="utf-8")
    os.replace(tmp, target)
    return target


def generate_token() -> str:
    """Return a fresh URL-safe loopback-shared-secret for this launch."""
    return secrets.token_urlsafe(32)


def build_manifest_payload(cfg: "SidecarConfig") -> Dict[str, Any]:
    """Compose the on-disk ``sidecar.json`` payload.

    Shape matches ``GET /api/v1/desktop/manifest`` plus the
    transport-level fields the shell needs to dial back in (``host``,
    ``port``, ``token``).  Computed via
    :func:`app.api.desktop_manifest.build_capability_snapshot` so there
    is exactly one source of truth for the capability map.
    """
    # Local import: keeps the launcher importable in environments where
    # the FastAPI app is not on sys.path (e.g. packaging scripts).
    from app.api.desktop_manifest import build_capability_snapshot

    snap = build_capability_snapshot()
    snap.update({
        "host": cfg.host,
        "port": cfg.port,
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "token": cfg.token,
    })
    return snap


def write_manifest_file(
    payload: Dict[str, Any], *, target: Optional[Path] = None,
) -> Path:
    """Atomically write ``payload`` to the sidecar manifest file.

    The file is created with mode ``0o600`` so other local users on a
    shared box cannot read the per-launch token.  Atomic via
    ``os.replace`` for the same reason as :func:`write_port_file`.
    """
    if target is None:
        target = get_user_data_dir(ensure=True) / _MANIFEST_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:  # Windows or restricted FS — best-effort, not fatal
        pass
    os.replace(tmp, target)
    return target


def build_config(
    *,
    port: Optional[int] = None,
    host: str = _LOOPBACK,
    port_file: Optional[Path] = None,
    manifest_file: Optional[Path] = None,
    token: Optional[str] = None,
    log_level: str = "info",
) -> SidecarConfig:
    """Resolve the launcher config without starting uvicorn.

    Splitting the resolution out makes the launcher unit-testable without
    blocking on a running server.
    """
    resolved_port = port if port and port > 0 else find_free_port(host)
    data_dir = get_user_data_dir(ensure=True)
    pf = port_file or (data_dir / _PORT_FILENAME)
    mf = manifest_file or (data_dir / _MANIFEST_FILENAME)
    tok = token if token else generate_token()
    return SidecarConfig(
        host=host,
        port=resolved_port,
        port_file=pf,
        manifest_file=mf,
        token=tok,
        log_level=log_level,
    )


def run_sidecar(
    *,
    port: Optional[int] = None,
    host: str = _LOOPBACK,
    port_file: Optional[Path] = None,
    manifest_file: Optional[Path] = None,
    token: Optional[str] = None,
    log_level: str = "info",
) -> None:
    """Boot the FastAPI app under uvicorn on a loopback-only port.

    Blocks until uvicorn exits.  Safe to call from ``__main__``.
    """
    # Force desktop deployment mode *before* importing app.api.main so the
    # lifespan honours the Phase 0 ``is_desktop`` gate.
    os.environ.setdefault("DEPLOYMENT_MODE", "desktop")

    cfg = build_config(
        port=port, host=host, port_file=port_file,
        manifest_file=manifest_file, token=token, log_level=log_level,
    )

    # Export the per-launch token so the loopback-token middleware can
    # enforce it inside the uvicorn worker process.  Done *before* the
    # FastAPI app is imported so the middleware sees a populated env
    # var at construction time.
    os.environ[_TOKEN_ENV] = cfg.token

    # Refresh the live settings singleton so anything cached during import
    # picks up the override.  The Settings class is a pydantic-settings
    # BaseSettings which reads env at construction time.
    from app.core.config import settings as _live_settings
    if _live_settings.deployment_mode != "desktop":
        object.__setattr__(_live_settings, "deployment_mode", "desktop")

    written = write_port_file(cfg.port, target=cfg.port_file)
    manifest_written = write_manifest_file(
        build_manifest_payload(cfg), target=cfg.manifest_file,
    )
    logger.info(
        "Desktop sidecar port=%d -> %s | manifest -> %s",
        cfg.port, written, manifest_written,
    )

    import uvicorn  # local import: keeps unit tests independent of uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
        access_log=False,
        workers=1,
    )

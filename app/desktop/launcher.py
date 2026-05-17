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

import logging
import os
import socket
import uuid
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.local.user_data_dir import get_user_data_dir

logger = logging.getLogger(__name__)

_PORT_FILENAME = "sidecar.port"
_LOOPBACK = "127.0.0.1"


@dataclass
class SidecarConfig:
    """Resolved configuration for a single launcher invocation."""

    host: str
    port: int
    port_file: Path
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


def build_config(
    *,
    port: Optional[int] = None,
    host: str = _LOOPBACK,
    port_file: Optional[Path] = None,
    log_level: str = "info",
) -> SidecarConfig:
    """Resolve the launcher config without starting uvicorn.

    Splitting the resolution out makes the launcher unit-testable without
    blocking on a running server.
    """
    resolved_port = port if port and port > 0 else find_free_port(host)
    pf = port_file or (get_user_data_dir(ensure=True) / _PORT_FILENAME)
    return SidecarConfig(
        host=host,
        port=resolved_port,
        port_file=pf,
        log_level=log_level,
    )


def run_sidecar(
    *,
    port: Optional[int] = None,
    host: str = _LOOPBACK,
    port_file: Optional[Path] = None,
    log_level: str = "info",
) -> None:
    """Boot the FastAPI app under uvicorn on a loopback-only port.

    Blocks until uvicorn exits.  Safe to call from ``__main__``.
    """
    # Force desktop deployment mode *before* importing app.api.main so the
    # lifespan honours the Phase 0 ``is_desktop`` gate.
    os.environ.setdefault("DEPLOYMENT_MODE", "desktop")

    cfg = build_config(
        port=port, host=host, port_file=port_file, log_level=log_level,
    )

    # Refresh the live settings singleton so anything cached during import
    # picks up the override.  The Settings class is a pydantic-settings
    # BaseSettings which reads env at construction time.
    from app.core.config import settings as _live_settings
    if _live_settings.deployment_mode != "desktop":
        object.__setattr__(_live_settings, "deployment_mode", "desktop")

    written = write_port_file(cfg.port, target=cfg.port_file)
    logger.info("Desktop sidecar port=%d -> %s", cfg.port, written)

    import uvicorn  # local import: keeps unit tests independent of uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
        access_log=False,
        workers=1,
    )

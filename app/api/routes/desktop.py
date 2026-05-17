"""Shell-handshake endpoints for the Tauri desktop sidecar.

Two routes are exposed here, both gated by
:func:`require_desktop_mode` so the multi-tenant server deployment is
unaffected (they return 404 there):

* ``GET /api/v1/desktop/manifest`` — capability snapshot the shell uses
  to decide which UI panels to render.  Matches the on-disk
  ``sidecar.json`` payload written by :mod:`app.desktop.launcher`.
* ``GET /api/v1/desktop/ready``    — returns 200 once the sidecar
  lifespan has completed startup; 503 until then.  Lets the shell poll
  during boot without false-starting against an app that hasn't yet
  finished its capability validation.

Both routes are intentionally exempt from the loopback-token middleware
(see ``app.api.main``): the shell must be able to fetch them *before* it
has the manifest, and they reveal no secrets.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from app.api.deps import require_desktop_mode
from app.api.desktop_manifest import build_capability_snapshot

router = APIRouter(dependencies=[Depends(require_desktop_mode)])


# ---------------------------------------------------------------------------
# Lifespan-readiness flag
# ---------------------------------------------------------------------------
#
# A simple module-level boolean toggled by the FastAPI lifespan in
# ``app.api.main``.  Using a plain bool (not asyncio.Event) keeps the
# readiness check synchronous and lock-free, which is exactly what a
# readiness probe wants.
_READY: bool = False


def mark_ready() -> None:
    """Flip the readiness flag to True.  Called from the app lifespan."""
    global _READY
    _READY = True


def mark_not_ready() -> None:
    """Flip the readiness flag to False (shutdown / test reset)."""
    global _READY
    _READY = False


def is_ready() -> bool:
    return _READY


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ManifestResponse(BaseModel):
    schema_version: str
    contract_version: str
    app_version: str
    deployment_mode: str
    pid: int = Field(..., description="OS process id of the running sidecar")
    started_at: str = Field(..., description="ISO-8601 UTC startup timestamp")
    capabilities: Dict[str, bool]


class ReadyResponse(BaseModel):
    ready: bool
    pid: int


# ---------------------------------------------------------------------------
# Module-level start time, captured at first import so /manifest reflects
# the actual process lifetime rather than the request time.
# ---------------------------------------------------------------------------

_STARTED_AT: str = datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/manifest", response_model=ManifestResponse)
async def get_manifest() -> Dict[str, Any]:
    """Return the live capability snapshot for the running sidecar."""
    snap = build_capability_snapshot()
    snap["pid"] = os.getpid()
    snap["started_at"] = _STARTED_AT
    return snap


@router.get("/ready", response_model=ReadyResponse)
async def get_ready(response: Response) -> ReadyResponse:
    """Return 200 once the lifespan has completed; 503 while starting up.

    The shell polls this with a short interval after spawning the
    sidecar; the legacy ``/health/ready`` route may succeed before the
    sidecar lifespan finishes, which is too early for the UI.
    """
    ready = is_ready()
    if not ready:
        response.status_code = 503
    return ReadyResponse(ready=ready, pid=os.getpid())

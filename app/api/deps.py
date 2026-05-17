"""Shared FastAPI dependencies for the desktop sidecar.

These dependencies gate access to local-first endpoints (keys, chat,
permissions) that only make sense when the API is running as an
embedded sidecar bound to the loopback interface.  In ``server`` mode
they short-circuit with HTTP 404 so the external API surface mirrors
the legacy multi-tenant deployment.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from app.core.config import settings


def require_desktop_mode() -> None:
    """Return 404 unless the process is running in desktop mode.

    The desktop-only routes assume a single trusted local user and bind
    to ``127.0.0.1`` via the sidecar launcher.  Exposing them on a
    multi-tenant server deployment without further authentication would
    be a security regression, so we hide them entirely instead.
    """
    if not settings.is_desktop:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="endpoint available in desktop deployment mode only",
        )

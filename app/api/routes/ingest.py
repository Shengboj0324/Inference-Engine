"""Local ingestion control endpoints for the desktop sidecar.

Exposes a minimal CRUD surface over :class:`SourcesStore` plus three
control actions backed by :class:`IngestRuntime`:

* ``GET  /status``                 — snapshot of runtime stats + dedup
* ``POST /fetch`` (manual run)     — enqueue one job per enabled source
* ``POST /cleanup``                — run the retention purge immediately
* ``GET  /sources``                — list configured sources
* ``PUT  /sources/{platform}``     — upsert (credentials + settings)
* ``DELETE /sources/{platform}``   — remove a source

All endpoints are gated by :func:`require_desktop_mode`; in server mode
they return 404 to keep the multi-tenant surface unchanged.  Credentials
sent on ``PUT`` are never echoed back — list/get responses report only
that they exist, never their contents.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import require_desktop_mode
from app.local.ingest_runtime import IngestRuntime, get_ingest_runtime
from app.local.sources_store import SourceConfig, SourcesStore, get_sources_store

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_desktop_mode)])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SourceUpsertRequest(BaseModel):
    credentials: Dict[str, Any] = Field(default_factory=dict)
    settings: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class SourceResponse(BaseModel):
    """Source view *without* credential material — only presence is shown."""

    platform: str
    enabled: bool
    has_credentials: bool
    settings: Dict[str, Any]
    last_fetch_at: Optional[float]
    last_status: Optional[str]
    last_error: Optional[str]
    created_at: float
    updated_at: float

    @classmethod
    def from_orm_like(cls, c: SourceConfig) -> "SourceResponse":
        return cls(
            platform=c.platform,
            enabled=c.enabled,
            has_credentials=bool(c.credentials),
            settings=dict(c.settings),
            last_fetch_at=c.last_fetch_at,
            last_status=c.last_status,
            last_error=c.last_error,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )


class StatusResponse(BaseModel):
    stats: Dict[str, Any]
    sources: int
    sources_enabled: int


class FetchResponse(BaseModel):
    enqueued: int


class CleanupResponse(BaseModel):
    purged: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/status", response_model=StatusResponse)
async def get_status(
    runtime: IngestRuntime = Depends(get_ingest_runtime),
    sources: SourcesStore = Depends(get_sources_store),
) -> StatusResponse:
    all_sources = sources.list_all()
    return StatusResponse(
        stats=runtime.snapshot_stats(),
        sources=len(all_sources),
        sources_enabled=sum(1 for s in all_sources if s.enabled),
    )


@router.post("/fetch", response_model=FetchResponse)
async def trigger_fetch(
    runtime: IngestRuntime = Depends(get_ingest_runtime),
) -> FetchResponse:
    """Manually enqueue a fetch for every enabled source.

    Honours :data:`Permission.NETWORK_FETCH` — when the user has not
    granted network access, returns ``enqueued = 0`` and the runtime
    records the denial in its stats counter.
    """
    enq = await runtime.enqueue_fetch_all()
    return FetchResponse(enqueued=enq)


@router.post("/cleanup", response_model=CleanupResponse)
async def trigger_cleanup(
    runtime: IngestRuntime = Depends(get_ingest_runtime),
) -> CleanupResponse:
    purged = await runtime.run_cleanup()
    return CleanupResponse(purged=purged)


@router.get("/sources", response_model=List[SourceResponse])
async def list_sources(
    sources: SourcesStore = Depends(get_sources_store),
) -> List[SourceResponse]:
    return [SourceResponse.from_orm_like(s) for s in sources.list_all()]


@router.put("/sources/{platform}", response_model=SourceResponse)
async def upsert_source(
    platform: str,
    body: SourceUpsertRequest,
    sources: SourcesStore = Depends(get_sources_store),
) -> SourceResponse:
    try:
        cfg = sources.upsert(
            platform,
            credentials=body.credentials,
            settings=body.settings,
            enabled=body.enabled,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc),
        ) from exc
    return SourceResponse.from_orm_like(cfg)


@router.delete("/sources/{platform}")
async def delete_source(
    platform: str,
    sources: SourcesStore = Depends(get_sources_store),
) -> dict:
    if not sources.delete(platform):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no source configured for platform {platform!r}",
        )
    return {"platform": platform, "removed": True}

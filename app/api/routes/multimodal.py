"""Local multimodal analysis control endpoints for the desktop sidecar.

Exposes a minimal probe surface around :class:`LocalMultimodalAnalyzer`
so the UI can show capability badges and trigger a one-shot analysis of
a user-selected file (e.g. for the "Test my models" panel).

* ``GET  /status``  — availability of CLIP / ASR runtimes + thresholds
* ``POST /probe``   — analyse a single local file (image or audio) and
                      return the structured result

Both endpoints are gated by :func:`require_desktop_mode`; in server mode
they return 404 to keep the multi-tenant surface unchanged.  The probe
endpoint requires a *local* filesystem path — it never reaches into the
network and never accepts uploads, in keeping with the zero-egress
contract.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import require_desktop_mode
from app.core.models import ContentItem, MediaType, SourcePlatform
from app.local.ingest_runtime import LOCAL_USER_ID, get_ingest_runtime
from app.local.multimodal_analyzer import LocalMultimodalAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_desktop_mode)])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class MultimodalStatusResponse(BaseModel):
    available: bool
    clip_available: bool
    clip_permitted: bool
    asr_available: bool
    asr_backend: str


class ProbeRequest(BaseModel):
    path: str = Field(..., min_length=1, description="Absolute local file path")
    media_type: str = Field(..., description="image | audio | video | mixed")


class ProbeResponse(BaseModel):
    status: str
    media_type: str
    low_confidence: bool
    caption: str = ""
    transcript: str = ""
    transcript_backend: str = ""
    asr_confidence: Optional[float] = None
    asr_duration_s: float = 0.0
    image_labels: Dict[str, float] = Field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_analyzer() -> LocalMultimodalAnalyzer:
    analyzer = get_ingest_runtime().analyzer
    if analyzer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="multimodal analyzer is not configured on this runtime",
        )
    return analyzer


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/status", response_model=MultimodalStatusResponse)
async def get_status() -> MultimodalStatusResponse:
    analyzer = _get_analyzer()
    clip = analyzer._clip
    asr = analyzer._asr
    return MultimodalStatusResponse(
        available=analyzer.is_available(),
        clip_available=clip.is_available(),
        clip_permitted=clip.is_permitted(),
        asr_available=asr.is_available(),
        asr_backend=asr.resolved_backend(),
    )


@router.post("/probe", response_model=ProbeResponse)
async def probe(req: ProbeRequest) -> ProbeResponse:
    try:
        mt = MediaType(req.media_type)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"invalid media_type {req.media_type!r}",
        ) from exc
    p = Path(req.path)
    if not p.is_absolute() or not p.exists() or not p.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="path must be an absolute path to an existing local file",
        )
    analyzer = _get_analyzer()
    # Build a throwaway ContentItem; the analyzer reads media_type +
    # media_urls only, so the synthetic identity fields don't matter.
    item = ContentItem(
        id=uuid4(), user_id=LOCAL_USER_ID,
        source_platform=SourcePlatform.RSS, source_id="probe",
        source_url="file://" + str(p), title="probe",
        media_type=mt, media_urls=[str(p)],
        published_at=datetime.now(timezone.utc),
    )
    result = await analyzer.analyze(item, media_path=str(p))
    return ProbeResponse(
        status=result.status,
        media_type=result.media_type,
        low_confidence=result.low_confidence,
        caption=result.caption,
        transcript=result.transcript,
        transcript_backend=result.transcript_backend,
        asr_confidence=result.asr_confidence,
        asr_duration_s=result.asr_duration_s,
        image_labels=result.image_labels,
        reason=result.reason,
    )

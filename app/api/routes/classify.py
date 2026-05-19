"""Grounded classification endpoints backed by ``SituationEngine``.

These routes are the public surface for the post-migration intelligence
engine documented in ``docs/intelligence_migration.md``. They accept a
list of :class:`~app.evals.scenario_loader.Observation` records and
return a :class:`~app.intelligence.situation_report.SituationReport`
whose every claim is grounded in a verified citation span.

Two endpoints are exposed:

* ``POST /api/v1/classify`` — synchronous, returns the parsed report.
* ``POST /api/v1/classify/stream`` — SSE stream emitting ``start``,
  ``report``, ``error`` and ``done`` frames so UIs can render progress
  without holding a long-lived request.

A frontier-model fallback is wired in transparently: when the primary
generator returns ``calibrated_confidence`` below
``settings.situation_engine_min_confidence``, the engine re-runs once on
the routed frontier model before responding.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.routes.auth import get_current_user
from app.core.db_models import User
from app.evals.scenario_loader import Observation
from app.intelligence.citation_verifier import CitationGroundingError
from app.intelligence.situation_engine import (
    AsyncSituationEngine,
    EngineResult,
    GenerationError,
    OutputParseError,
    build_router_generator,
)
from app.intelligence.situation_report import SituationReport
from app.llm.router import get_router

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["Classify"])


class ClassifyRequest(BaseModel):
    """Input payload for the grounded classification endpoints."""

    observations: List[Observation] = Field(..., min_length=1, max_length=64)
    min_confidence: float = Field(0.0, ge=0.0, le=1.0)
    primary_model: Optional[str] = Field(None, max_length=200)
    fallback_model: Optional[str] = Field(None, max_length=200)


class ClassifyResponse(BaseModel):
    """Successful response: report plus engine-side provenance."""

    report: SituationReport
    used_fallback: bool
    redaction_count: int


def _build_engine(req: ClassifyRequest) -> AsyncSituationEngine:
    router_instance = get_router()
    primary = build_router_generator(router_instance, model=req.primary_model)
    fallback = None
    if req.fallback_model:
        fallback = build_router_generator(
            router_instance, model=req.fallback_model
        )
    return AsyncSituationEngine(
        generate=primary,
        fallback_generate=fallback,
        min_confidence=req.min_confidence,
    )


def _engine_failure_to_http(exc: Exception) -> HTTPException:
    if isinstance(exc, CitationGroundingError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "citation_grounding_failed", "message": str(exc)},
        )
    if isinstance(exc, OutputParseError):
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "output_parse_failed", "message": str(exc)},
        )
    if isinstance(exc, GenerationError):
        return HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "generator_failed", "message": str(exc)},
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"error": "engine_error", "message": str(exc)},
    )


@router.post("", response_model=ClassifyResponse)
async def classify(
    request: ClassifyRequest,
    current_user: User = Depends(get_current_user),
) -> ClassifyResponse:
    """Run the grounded classification engine and return a SituationReport."""
    engine = _build_engine(request)
    try:
        result: EngineResult = await engine.analyze(request.observations)
    except Exception as exc:
        logger.warning("classify.engine_failure", extra={"exc": str(exc)})
        raise _engine_failure_to_http(exc)
    return ClassifyResponse(
        report=result.report,
        used_fallback=result.used_fallback,
        redaction_count=result.redaction_count,
    )


@router.post("/stream")
async def classify_stream(
    request: ClassifyRequest,
    http_request: Request,
    current_user: User = Depends(get_current_user),
) -> StreamingResponse:
    """SSE wrapper around :func:`classify` for incremental UI rendering."""

    engine = _build_engine(request)

    async def _stream():
        def _sse(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        if await http_request.is_disconnected():
            return
        yield _sse({"event": "start", "n_observations": len(request.observations)})
        try:
            result = await engine.analyze(request.observations)
        except Exception as exc:
            logger.warning(
                "classify_stream.engine_failure", extra={"exc": str(exc)}
            )
            if not await http_request.is_disconnected():
                yield _sse({
                    "event": "error",
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                })
        else:
            if not await http_request.is_disconnected():
                yield _sse({
                    "event": "report",
                    "used_fallback": result.used_fallback,
                    "redaction_count": result.redaction_count,
                    "report": result.report.model_dump(mode="json"),
                })
        if not await http_request.is_disconnected():
            yield _sse({"event": "done"})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

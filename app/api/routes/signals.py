"""Signal queue API endpoints - core product interface.

This module provides the primary user-facing API for the signal-to-action workflow.
It replaces the digest-first approach with a queue-first approach.

Key endpoints:
- GET /queue: Get prioritized signal queue
- GET /{signal_id}: Get signal details
- POST /{signal_id}/act: Mark signal as acted upon
- POST /{signal_id}/dismiss: Dismiss signal
- POST /{signal_id}/assign: Assign signal to team member
- GET /stats: Get signal queue statistics

Design principles:
- Fast queue retrieval with proper indexing
- Flexible filtering for different views
- Outcome tracking for learning loop
- Team collaboration support
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.routes.auth import get_current_user
from app.core.db import get_db
from app.core.db_models import ActionableSignalDB, User
from app.core.models import TeamRole
from app.core.signal_models import (
    ActionableSignal,
    SignalFilter,
    SignalStatus,
    SignalSummary,
    SignalType,
    TeamDigest,
)
from app.domain.raw_models import RawObservation
from app.intelligence.inference_pipeline import InferencePipeline

logger = logging.getLogger(__name__)

# Absolute path to calibration state — resolved at import time so it is
# correct regardless of the process working directory.
_CALIBRATION_STATE_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent / "training" / "calibration_state.json"
)

router = APIRouter(prefix="/signals", tags=["signals"])


class ActRequest(BaseModel):
    """Request to mark signal as acted upon."""

    outcome: Optional[dict] = Field(
        None,
        description="Outcome data for learning loop",
    )
    notes: Optional[str] = Field(
        None,
        max_length=1000,
        description="Optional notes about the action taken",
    )


class DismissRequest(BaseModel):
    """Request to dismiss signal."""

    reason: Optional[str] = Field(
        None,
        max_length=500,
        description="Reason for dismissal",
    )


class AssignRequest(BaseModel):
    """Request to assign signal to a team member.

    The requesting user must have ``TeamRole.MANAGER`` (or higher) access.
    Enforced by ``POST /{signal_id}/assign`` — returns HTTP 403 otherwise.
    """

    user_id: UUID = Field(..., description="User ID to assign to")
    team_id: Optional[UUID] = Field(None, description="Team scope for this assignment")
    requester_role: TeamRole = Field(
        ...,
        description=(
            "Role of the user making the request. "
            "Must be MANAGER or higher. "
            "Clients should obtain this from their session/JWT."
        ),
    )


class FeedbackRequest(BaseModel):
    """Request body for ``POST /{signal_id}/feedback``.

    The caller supplies the correct signal type for a previously classified
    signal.  Requires ``TeamRole.ANALYST`` or higher.
    """

    true_signal_type: str = Field(
        ...,
        description="The correct SignalType value string (e.g. 'churn_risk').",
    )
    requester_role: TeamRole = Field(
        ...,
        description="Role of the requesting user.  Must be ANALYST or higher.",
    )


class FeedbackResponse(BaseModel):
    """Response body for ``POST /{signal_id}/feedback``."""

    feedback_id: UUID = Field(..., description="UUID of the created feedback record.")
    signal_id: UUID = Field(..., description="UUID of the signal that was corrected.")
    predicted_type: str
    true_type: str
    predicted_confidence: float


class SignalStats(BaseModel):
    """Signal queue statistics."""

    total_signals: int
    new_signals: int
    queued_signals: int
    in_progress_signals: int
    acted_signals: int
    dismissed_signals: int
    expired_signals: int
    avg_action_score: float
    avg_time_to_act_hours: Optional[float]


@router.get("/queue", response_model=List[SignalSummary])
async def get_signal_queue(
    signal_types: Optional[List[SignalType]] = Query(None),
    min_action_score: float = Query(default=0.5, ge=0.0, le=1.0),
    max_action_score: float = Query(default=1.0, ge=0.0, le=1.0),
    statuses: Optional[List[SignalStatus]] = Query(None),
    platforms: Optional[List[str]] = Query(None),
    include_expired: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get prioritized signal queue.

    This is the primary endpoint for the signal-to-action workflow.
    Returns signals sorted by action_score (highest first).

    Args:
        signal_types: Filter by signal types
        min_action_score: Minimum action score threshold
        max_action_score: Maximum action score threshold
        statuses: Filter by status (defaults to NEW and QUEUED)
        platforms: Filter by source platform
        include_expired: Include expired signals
        limit: Maximum number of signals to return
        offset: Pagination offset
        current_user: Authenticated user
        db: Database session

    Returns:
        List of signal summaries, sorted by action_score descending
    """
    try:
        # Build query
        query = select(ActionableSignalDB).where(
            ActionableSignalDB.user_id == current_user.id
        )

        # Apply filters
        if signal_types:
            # Pass string values explicitly so the DB driver receives the correct
            # scalar type regardless of which SQLAlchemy backend / dialect is in use.
            query = query.where(
                ActionableSignalDB.signal_type.in_([s.value for s in signal_types])
            )

        query = query.where(
            and_(
                ActionableSignalDB.action_score >= min_action_score,
                ActionableSignalDB.action_score <= max_action_score,
            )
        )

        if statuses:
            query = query.where(ActionableSignalDB.status.in_(statuses))
        else:
            # Default to active signals
            query = query.where(
                ActionableSignalDB.status.in_([
                    SignalStatus.NEW,
                    SignalStatus.QUEUED,
                    SignalStatus.IN_PROGRESS,
                ])
            )

        # Sort by action_score (highest first)
        query = query.order_by(ActionableSignalDB.action_score.desc())

        # Apply pagination
        query = query.limit(limit).offset(offset)

        # Execute query
        result = await db.execute(query)
        signals = result.scalars().all()

        # Convert to summaries
        summaries = [
            SignalSummary(
                id=signal.id,
                signal_type=signal.signal_type,
                title=signal.title,
                urgency_score=signal.urgency_score,
                impact_score=signal.impact_score,
                action_score=signal.action_score,
                status=signal.status,
                created_at=signal.created_at,
                expires_at=signal.expires_at,
                source_platform=signal.source_platform,
                source_author=signal.source_author,
            )
            for signal in signals
        ]

        logger.info(
            f"Retrieved {len(summaries)} signals for user {current_user.id} "
            f"(limit={limit}, offset={offset})"
        )

        return summaries

    except Exception as e:
        logger.error(f"Failed to retrieve signal queue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve signal queue",
        )


@router.get("/{signal_id}", response_model=ActionableSignal)
async def get_signal(
    signal_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get full signal details.

    Args:
        signal_id: Signal ID
        current_user: Authenticated user
        db: Database session

    Returns:
        Complete signal with all metadata and generated assets
    """
    try:
        result = await db.execute(
            select(ActionableSignalDB).where(
                and_(
                    ActionableSignalDB.id == signal_id,
                    ActionableSignalDB.user_id == current_user.id,
                )
            )
        )
        signal_db = result.scalar_one_or_none()

        if not signal_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Signal not found",
            )

        # Convert to Pydantic model
        signal = ActionableSignal(
            id=signal_db.id,
            user_id=signal_db.user_id,
            signal_type=signal_db.signal_type,
            source_item_ids=signal_db.source_item_ids,
            source_platform=signal_db.source_platform,
            source_url=signal_db.source_url,
            source_author=signal_db.source_author,
            title=signal_db.title,
            description=signal_db.description,
            context=signal_db.context,
            urgency_score=signal_db.urgency_score,
            impact_score=signal_db.impact_score,
            confidence_score=signal_db.confidence_score,
            action_score=signal_db.action_score,
            recommended_action=signal_db.recommended_action,
            suggested_channel=signal_db.suggested_channel,
            suggested_tone=signal_db.suggested_tone,
            draft_response=signal_db.draft_response,
            draft_post=signal_db.draft_post,
            draft_dm=signal_db.draft_dm,
            positioning_angle=signal_db.positioning_angle,
            status=signal_db.status,
            assigned_to=signal_db.assigned_to,
            created_at=signal_db.created_at,
            expires_at=signal_db.expires_at,
            acted_at=signal_db.acted_at,
            outcome_feedback=signal_db.outcome_feedback,
            metadata=signal_db.metadata_,
        )

        return signal

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve signal {signal_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve signal",
        )


@router.post("/{signal_id}/act", response_model=ActionableSignal)
async def mark_signal_acted(
    signal_id: UUID,
    request: ActRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Mark signal as acted upon.

    This records that the user took action on the signal and optionally
    captures outcome data for the learning loop.

    Args:
        signal_id: Signal ID
        request: Act request with optional outcome data
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated signal
    """
    try:
        result = await db.execute(
            select(ActionableSignalDB).where(
                and_(
                    ActionableSignalDB.id == signal_id,
                    ActionableSignalDB.user_id == current_user.id,
                )
            )
        )
        signal_db = result.scalar_one_or_none()

        if not signal_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Signal not found",
            )

        # Update signal
        signal_db.status = SignalStatus.ACTED
        signal_db.acted_at = datetime.utcnow()

        # Store outcome feedback
        if request.outcome:
            signal_db.outcome_feedback = request.outcome

        # Add notes to metadata
        if request.notes:
            if signal_db.metadata_ is None:
                signal_db.metadata_ = {}
            signal_db.metadata_['action_notes'] = request.notes

        await db.commit()
        await db.refresh(signal_db)

        logger.info(f"Signal {signal_id} marked as acted by user {current_user.id}")

        # Convert to Pydantic model
        signal = ActionableSignal(
            id=signal_db.id,
            user_id=signal_db.user_id,
            signal_type=signal_db.signal_type,
            source_item_ids=signal_db.source_item_ids,
            source_platform=signal_db.source_platform,
            source_url=signal_db.source_url,
            source_author=signal_db.source_author,
            title=signal_db.title,
            description=signal_db.description,
            context=signal_db.context,
            urgency_score=signal_db.urgency_score,
            impact_score=signal_db.impact_score,
            confidence_score=signal_db.confidence_score,
            action_score=signal_db.action_score,
            recommended_action=signal_db.recommended_action,
            suggested_channel=signal_db.suggested_channel,
            suggested_tone=signal_db.suggested_tone,
            draft_response=signal_db.draft_response,
            draft_post=signal_db.draft_post,
            draft_dm=signal_db.draft_dm,
            positioning_angle=signal_db.positioning_angle,
            status=signal_db.status,
            assigned_to=signal_db.assigned_to,
            created_at=signal_db.created_at,
            expires_at=signal_db.expires_at,
            acted_at=signal_db.acted_at,
            outcome_feedback=signal_db.outcome_feedback,
            metadata=signal_db.metadata_,
        )

        return signal

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark signal {signal_id} as acted: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update signal",
        )


@router.post("/{signal_id}/dismiss", response_model=ActionableSignal)
async def dismiss_signal(
    signal_id: UUID,
    request: DismissRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Dismiss signal.

    Args:
        signal_id: Signal ID
        request: Dismiss request with optional reason
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated signal
    """
    try:
        result = await db.execute(
            select(ActionableSignalDB).where(
                and_(
                    ActionableSignalDB.id == signal_id,
                    ActionableSignalDB.user_id == current_user.id,
                )
            )
        )
        signal_db = result.scalar_one_or_none()

        if not signal_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Signal not found",
            )

        # Update signal
        signal_db.status = SignalStatus.DISMISSED

        # Store dismissal reason
        if request.reason:
            if signal_db.metadata_ is None:
                signal_db.metadata_ = {}
            signal_db.metadata_['dismissal_reason'] = request.reason

        await db.commit()
        await db.refresh(signal_db)

        logger.info(f"Signal {signal_id} dismissed by user {current_user.id}")

        # Convert to Pydantic model
        signal = ActionableSignal(
            id=signal_db.id,
            user_id=signal_db.user_id,
            signal_type=signal_db.signal_type,
            source_item_ids=signal_db.source_item_ids,
            source_platform=signal_db.source_platform,
            source_url=signal_db.source_url,
            source_author=signal_db.source_author,
            title=signal_db.title,
            description=signal_db.description,
            context=signal_db.context,
            urgency_score=signal_db.urgency_score,
            impact_score=signal_db.impact_score,
            confidence_score=signal_db.confidence_score,
            action_score=signal_db.action_score,
            recommended_action=signal_db.recommended_action,
            suggested_channel=signal_db.suggested_channel,
            suggested_tone=signal_db.suggested_tone,
            draft_response=signal_db.draft_response,
            draft_post=signal_db.draft_post,
            draft_dm=signal_db.draft_dm,
            positioning_angle=signal_db.positioning_angle,
            status=signal_db.status,
            assigned_to=signal_db.assigned_to,
            created_at=signal_db.created_at,
            expires_at=signal_db.expires_at,
            acted_at=signal_db.acted_at,
            outcome_feedback=signal_db.outcome_feedback,
            metadata=signal_db.metadata_,
        )

        return signal

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to dismiss signal {signal_id}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update signal",
        )


@router.get("/stats", response_model=SignalStats)
async def get_signal_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get signal queue statistics.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        Signal queue statistics
    """
    try:
        # Count signals by status
        result = await db.execute(
            select(
                func.count(ActionableSignalDB.id).label('total'),
                func.count(ActionableSignalDB.id).filter(
                    ActionableSignalDB.status == SignalStatus.NEW
                ).label('new'),
                func.count(ActionableSignalDB.id).filter(
                    ActionableSignalDB.status == SignalStatus.QUEUED
                ).label('queued'),
                func.count(ActionableSignalDB.id).filter(
                    ActionableSignalDB.status == SignalStatus.IN_PROGRESS
                ).label('in_progress'),
                func.count(ActionableSignalDB.id).filter(
                    ActionableSignalDB.status == SignalStatus.ACTED
                ).label('acted'),
                func.count(ActionableSignalDB.id).filter(
                    ActionableSignalDB.status == SignalStatus.DISMISSED
                ).label('dismissed'),
                func.count(ActionableSignalDB.id).filter(
                    and_(
                        ActionableSignalDB.expires_at.isnot(None),
                        ActionableSignalDB.expires_at < datetime.utcnow(),
                    )
                ).label('expired'),
                func.avg(ActionableSignalDB.action_score).label('avg_score'),
            ).where(ActionableSignalDB.user_id == current_user.id)
        )

        row = result.one()

        # Calculate average time to act
        acted_signals = await db.execute(
            select(
                ActionableSignalDB.created_at,
                ActionableSignalDB.acted_at,
            ).where(
                and_(
                    ActionableSignalDB.user_id == current_user.id,
                    ActionableSignalDB.status == SignalStatus.ACTED,
                    ActionableSignalDB.acted_at.isnot(None),
                )
            )
        )

        time_diffs = []
        for created, acted in acted_signals:
            if created and acted:
                diff_hours = (acted - created).total_seconds() / 3600
                time_diffs.append(diff_hours)

        avg_time_to_act = sum(time_diffs) / len(time_diffs) if time_diffs else None

        stats = SignalStats(
            total_signals=row.total or 0,
            new_signals=row.new or 0,
            queued_signals=row.queued or 0,
            in_progress_signals=row.in_progress or 0,
            acted_signals=row.acted or 0,
            dismissed_signals=row.dismissed or 0,
            expired_signals=row.expired or 0,
            avg_action_score=float(row.avg_score) if row.avg_score else 0.0,
            avg_time_to_act_hours=avg_time_to_act,
        )

        return stats

    except Exception as e:
        logger.error(f"Failed to retrieve signal stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics",
        )



# ---------------------------------------------------------------------------
# SSE Streaming Inference Endpoint
# ---------------------------------------------------------------------------

async def _sse_inference_generator(
    raw_observation: RawObservation,
    pipeline: InferencePipeline,
    request: Request,
) -> AsyncGenerator[str, None]:
    """Async generator that streams SSE events for a single inference run.

    Yields structured ``data:`` lines in Server-Sent Events format.
    Each event is a JSON object.  Three event types are emitted:

    * ``{"event": "start", "observation_id": "..."}`` — pipeline started.
    * ``{"event": "result", "normalized": {...}, "inference": {...}}`` — success.
    * ``{"event": "error", "detail": "..."}`` — pipeline failure.
    * ``{"event": "done"}`` — always emitted last.

    Backpressure: the generator checks ``await request.is_disconnected()``
    before emitting each event and exits early on client disconnect.

    Args:
        raw_observation: The raw observation to classify.
        pipeline: Pre-built InferencePipeline instance.
        request: FastAPI Request object (used for disconnect detection).

    Yields:
        SSE-formatted strings (``data: <json>\\n\\n``).
    """

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    # Announce pipeline start
    if await request.is_disconnected():
        return
    yield _sse({"event": "start", "observation_id": str(raw_observation.id)})

    try:
        normalized, inference = await pipeline.run(raw_observation)

        if await request.is_disconnected():
            return

        yield _sse({
            "event": "result",
            "normalized": {
                "id": str(normalized.id),
                "source_platform": normalized.source_platform.value,
                "normalized_text": (normalized.normalized_text or "")[:500],
                "original_language": normalized.original_language,
            },
            "inference": {
                "id": str(inference.id),
                "abstained": inference.abstained,
                "abstention_reason": (
                    inference.abstention_reason.value
                    if inference.abstention_reason else None
                ),
                "top_signal_type": (
                    inference.top_prediction.signal_type.value
                    if inference.top_prediction else None
                ),
                "top_probability": (
                    inference.top_prediction.probability
                    if inference.top_prediction else None
                ),
                "rationale": inference.rationale,
                "calibration": (
                    inference.calibration_metrics.model_dump()
                    if inference.calibration_metrics else None
                ),
            },
        })

    except Exception as exc:
        logger.error(f"SSE pipeline error for observation {raw_observation.id}: {exc}", exc_info=True)
        if not await request.is_disconnected():
            yield _sse({"event": "error", "detail": str(exc)})

    finally:
        if not await request.is_disconnected():
            yield _sse({"event": "done"})


@router.post("/stream", summary="Stream signal inference via SSE")
async def stream_signal_inference(
    raw_observation: RawObservation,
    request: Request,
    current_user: User = Depends(get_current_user),  # authentication required
) -> StreamingResponse:
    """Stream signal inference results for a single raw observation via SSE.

    Accepts a :class:`~app.domain.raw_models.RawObservation` payload, runs it
    through the full :class:`~app.intelligence.inference_pipeline.InferencePipeline`,
    and streams back structured events as ``text/event-stream``.

    Requires a valid Bearer token (same as all other signal endpoints).
    Unauthenticated requests are rejected with HTTP 401 before any LLM call
    is made, preventing unauthorised LLM spend.

    Client must handle three event types: ``start``, ``result``, and ``error``,
    followed by a terminal ``done`` event.

    Args:
        raw_observation: The raw social-media observation to classify.
        request: Injected FastAPI request (used for disconnect detection).
        current_user: Authenticated user from JWT (injected by dependency).

    Returns:
        ``StreamingResponse`` with ``Content-Type: text/event-stream``.
    """
    # Build a lightweight pipeline; all components use their default constructors.
    # Translation and entity extraction are disabled for low-latency streaming.
    pipeline = InferencePipeline(
        normalization_engine=None,  # uses default (embeddings on, translation off)
        candidate_retriever=None,
        llm_adjudicator=None,
        calibrator=None,
        abstention_decider=None,
    )

    return StreamingResponse(
        _sse_inference_generator(raw_observation, pipeline, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering for SSE
        },
    )


# ---------------------------------------------------------------------------
# Team Collaboration endpoints (competitive_analysis.md §5.5)
# ---------------------------------------------------------------------------


@router.post("/{signal_id}/assign", response_model=ActionableSignal)
async def assign_signal(
    signal_id: UUID,
    request: AssignRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Assign a signal to a team member.

    Requires the requesting user to hold at least the ``MANAGER`` role in
    ``request.requester_role``.  Returns HTTP 403 if the requester's role is
    ``VIEWER`` or ``ANALYST``.

    Args:
        signal_id: Signal to assign.
        request: Assignment payload (target user, team_id, requester_role).
        current_user: Authenticated user making the request.
        db: Database session.

    Returns:
        Updated :class:`~app.core.signal_models.ActionableSignal`.

    Raises:
        HTTPException 403: If ``request.requester_role`` is below ``MANAGER``.
        HTTPException 404: If the signal does not exist or is not owned by
            the current user.
    """
    # Role gate — MANAGER or higher required
    if not TeamRole.has_role_at_least(request.requester_role, TeamRole.MANAGER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Insufficient privileges: 'assign' requires MANAGER role or higher. "
                f"Current role: {request.requester_role.value}"
            ),
        )

    try:
        result = await db.execute(
            select(ActionableSignalDB).where(
                and_(
                    ActionableSignalDB.id == signal_id,
                    ActionableSignalDB.user_id == current_user.id,
                )
            )
        )
        signal_db = result.scalar_one_or_none()

        if not signal_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Signal not found",
            )

        signal_db.assigned_to = request.user_id
        signal_db.team_id = request.team_id
        signal_db.assigned_role = request.requester_role.value
        if signal_db.status == SignalStatus.NEW:
            signal_db.status = SignalStatus.QUEUED

        await db.commit()
        await db.refresh(signal_db)

        logger.info(
            "Signal %s assigned to user %s by %s (role=%s team=%s)",
            signal_id,
            request.user_id,
            current_user.id,
            request.requester_role.value,
            request.team_id,
        )

        return ActionableSignal(
            id=signal_db.id,
            user_id=signal_db.user_id,
            signal_type=signal_db.signal_type,
            source_item_ids=signal_db.source_item_ids,
            source_platform=signal_db.source_platform,
            source_url=signal_db.source_url,
            source_author=signal_db.source_author,
            title=signal_db.title,
            description=signal_db.description,
            context=signal_db.context,
            urgency_score=signal_db.urgency_score,
            impact_score=signal_db.impact_score,
            confidence_score=signal_db.confidence_score,
            action_score=signal_db.action_score,
            recommended_action=signal_db.recommended_action,
            suggested_channel=signal_db.suggested_channel,
            suggested_tone=signal_db.suggested_tone,
            draft_response=signal_db.draft_response,
            draft_post=signal_db.draft_post,
            draft_dm=signal_db.draft_dm,
            positioning_angle=signal_db.positioning_angle,
            status=signal_db.status,
            assigned_to=signal_db.assigned_to,
            created_at=signal_db.created_at,
            expires_at=signal_db.expires_at,
            acted_at=signal_db.acted_at,
            outcome_feedback=signal_db.outcome_feedback,
            metadata=signal_db.metadata_,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to assign signal %s: %s", signal_id, exc)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign signal",
        )


_TEAM_DIGEST_PAGE_SIZE = 500  # Hard cap to prevent full-table-scan OOM


@router.get("/team", response_model=TeamDigest)
async def get_team_digest(
    team_id: UUID = Query(..., description="Team UUID to generate digest for"),
    requester_role: TeamRole = Query(
        ..., description="Role of the requesting user (VIEWER, ANALYST, MANAGER)"
    ),
    days: int = Query(default=7, ge=1, le=90, description="Digest window in days"),
    offset: int = Query(default=0, ge=0, description="Pagination offset (signals skipped)"),
    response: Response = None,  # type: ignore[assignment]  # injected by FastAPI
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return a team signal digest with counts by status and type.

    The query is capped at **500 signals per page** to prevent full-table-scan
    OOM on large teams.  When the result set is truncated, a ``Link`` response
    header with ``rel="next"`` is included so callers can fetch the next page.

    VIEWERs receive a read-only subset of fields (``total_signals``,
    ``by_status``, ``by_type``).  ``unassigned_count`` and
    ``high_urgency_count`` are only populated for ANALYST and above.

    Args:
        team_id: UUID of the team to summarise.
        requester_role: Role claimed by the caller.  Determines which fields
            are populated in the response.
        days: Number of days back from now to include in the digest window.
        offset: Number of signals to skip (for pagination).
        response: FastAPI response object — used to set ``Link`` header.
        current_user: Authenticated user.
        db: Database session.

    Returns:
        :class:`~app.core.signal_models.TeamDigest` for the requested page.
    """
    from datetime import timedelta

    now = datetime.utcnow()
    period_start = now - timedelta(days=days)

    # Fetch at most _TEAM_DIGEST_PAGE_SIZE + 1 rows so we can detect truncation
    # without a separate COUNT query.
    query = (
        select(ActionableSignalDB)
        .where(
            and_(
                ActionableSignalDB.team_id == team_id,
                ActionableSignalDB.created_at >= period_start,
            )
        )
        .order_by(ActionableSignalDB.created_at.asc())
        .limit(_TEAM_DIGEST_PAGE_SIZE + 1)
        .offset(offset)
    )
    result = await db.execute(query)
    rows = result.scalars().all()

    # Detect whether there is a next page.
    truncated = len(rows) > _TEAM_DIGEST_PAGE_SIZE
    signals = rows[:_TEAM_DIGEST_PAGE_SIZE]

    if truncated and response is not None:
        next_offset = offset + _TEAM_DIGEST_PAGE_SIZE
        response.headers["Link"] = (
            f'</api/v1/signals/team?team_id={team_id}'
            f'&days={days}&offset={next_offset}>; rel="next"'
        )

    by_status: dict = {}
    by_type: dict = {}
    unassigned = 0
    high_urgency = 0

    for sig in signals:
        # Count by status
        s_key = sig.status.value if sig.status else "unknown"
        by_status[s_key] = by_status.get(s_key, 0) + 1

        # Count by type
        t_key = sig.signal_type.value if sig.signal_type else "unknown"
        by_type[t_key] = by_type.get(t_key, 0) + 1

        # Richer fields for ANALYST+
        if TeamRole.has_role_at_least(requester_role, TeamRole.ANALYST):
            if sig.assigned_to is None:
                unassigned += 1
            urgency = getattr(sig, "urgency_score", None) or 0.0
            if urgency >= 0.8:
                high_urgency += 1

    return TeamDigest(
        team_id=team_id,
        period_start=period_start,
        period_end=now,
        total_signals=len(signals),
        by_status=by_status,
        by_type=by_type,
        unassigned_count=unassigned,
        high_urgency_count=high_urgency,
    )


@router.post(
    "/{signal_id}/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a signal-classification correction",
    description=(
        "Record a human correction for a signal's predicted type and trigger "
        "an online calibration update.  Requires TeamRole.ANALYST or higher."
    ),
)
async def submit_feedback(
    signal_id: UUID,
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """Submit a signal-classification correction.

    The endpoint validates that the target signal exists and belongs to the
    current user, enforces ``TeamRole.ANALYST`` access, persists the feedback
    record via ``FeedbackStore``, and triggers a one-step calibration update.

    Args:
        signal_id: UUID path parameter of the signal being corrected.
        request: ``FeedbackRequest`` with ``true_signal_type`` and
            ``requester_role``.
        current_user: Authenticated user from JWT.
        db: Async database session.

    Returns:
        ``FeedbackResponse`` containing the new feedback record's UUID and
        the signal's predicted / true type pair.

    Raises:
        HTTPException 403: If the caller's role is below ``ANALYST``.
        HTTPException 404: If the signal is not found for this user.
        HTTPException 400: If ``true_signal_type`` is not a valid value.
    """
    # Enforce ANALYST-or-higher access gate.
    if not TeamRole.has_role_at_least(request.requester_role, TeamRole.ANALYST):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="TeamRole.ANALYST or higher required to submit feedback.",
        )

    # Resolve the target signal and confirm ownership.
    stmt = select(ActionableSignalDB).where(
        ActionableSignalDB.id == signal_id,
        ActionableSignalDB.user_id == current_user.id,
    )
    result = await db.execute(stmt)
    signal_db = result.scalar_one_or_none()
    if signal_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Signal {signal_id} not found.",
        )

    # Validate the submitted true_signal_type value.
    from app.domain.inference_models import SignalType as InferenceSignalType
    try:
        InferenceSignalType(request.true_signal_type)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{request.true_signal_type}' is not a valid SignalType value.",
        )

    predicted_type: str = (
        signal_db.signal_type.value
        if hasattr(signal_db.signal_type, "value")
        else str(signal_db.signal_type)
    )
    predicted_confidence: float = float(signal_db.action_score or 0.5)
    rec_id = uuid4()
    rec_created_at = datetime.now(timezone.utc)

    # Persist feedback directly to the DB using the already-open session.
    from app.core.db_models import SignalFeedbackDB
    db_row = SignalFeedbackDB(
        id=rec_id,
        signal_id=signal_id,
        predicted_type=predicted_type,
        true_type=request.true_signal_type,
        predicted_confidence=predicted_confidence,
        user_id=current_user.id,
        created_at=rec_created_at,
    )
    db.add(db_row)
    await db.commit()

    # Update ConfidenceCalibrator in-process so the scalar change is visible
    # immediately to subsequent requests in the same worker process.
    from app.intelligence.calibration import ConfidenceCalibrator
    try:
        calibrator = ConfidenceCalibrator(state_path=_CALIBRATION_STATE_PATH)
        calibrator.update(
            InferenceSignalType(predicted_type),
            predicted_confidence,
            predicted_type == request.true_signal_type,
        )
    except Exception as exc:
        # Calibration update failure must never abort the feedback response.
        logger.warning("Calibration update failed after feedback submission: %s", exc)

    logger.info(
        "Feedback submitted: signal=%s predicted=%s true=%s user=%s id=%s",
        signal_id,
        predicted_type,
        request.true_signal_type,
        current_user.id,
        rec_id,
    )

    return FeedbackResponse(
        feedback_id=rec_id,
        signal_id=signal_id,
        predicted_type=predicted_type,
        true_type=request.true_signal_type,
        predicted_confidence=predicted_confidence,
    )



# ---------------------------------------------------------------------------
# Route ordering correction
# ---------------------------------------------------------------------------
# FastAPI/Starlette matches routes in registration order. The parameterised
# route GET /{signal_id} was registered before the literal routes GET /stats
# and GET /team. This means "stats" and "team" would match /{signal_id}
# (UUID validation then raises HTTP 422) instead of their dedicated handlers.
#
# Fix: reorder the router's route list so that every fully-literal GET path
# is moved to the front, ahead of any parameterised path.
_literal_get_paths = frozenset({"/queue", "/stats", "/team", "/stream"})
_literal_first = [r for r in router.routes if getattr(r, "path", None) in _literal_get_paths]
_rest = [r for r in router.routes if r not in _literal_first]
router.routes[:] = _literal_first + _rest

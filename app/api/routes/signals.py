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

import redis.asyncio as aioredis

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.routes.auth import get_current_user
from app.core.config import settings
from app.core.db import get_db
from app.core.db_models import ActionableSignalDB, User
from app.core.models import TeamRole
from app.core.monitoring import MetricsCollector
from app.core.signal_models import (
    ActionableSignal,
    SignalFilter,
    SignalStatus,
    SignalSummary,
    SignalType,
    TeamDigest,
)
from app.domain.raw_models import RawObservation
from app.intelligence.feedback_processor import FeedbackProcessor
from app.intelligence.inference_pipeline import InferencePipeline
from app.intelligence.orchestrator import (
    ConversationTurn,
    DeepResearchReport,
    DraftResponse,
    MultiAgentOrchestrator,
    SignalInteractionAgent,
    VectorSearchTool,
)
from app.domain.inference_models import UserContext

_feedback_processor = FeedbackProcessor()

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

        # Closed-loop calibration — fire-and-forget; never blocks the response
        try:
            fb = await _feedback_processor.process_act(
                signal_id=str(signal_id),
                signal_type_value=signal_db.signal_type.value,
                confidence_score=signal_db.confidence_score,
                current_action_score=signal_db.action_score,
                notes=request.notes,
            )
            signal_db.action_score = fb.new_action_score
        except Exception as _fb_exc:
            logger.warning("FeedbackProcessor.process_act error (non-fatal): %s", _fb_exc)

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

        # Closed-loop calibration
        try:
            fb = await _feedback_processor.process_dismiss(
                signal_id=str(signal_id),
                signal_type_value=signal_db.signal_type.value,
                confidence_score=signal_db.confidence_score,
                current_action_score=signal_db.action_score,
                reason=request.reason,
            )
            signal_db.action_score = fb.new_action_score
        except Exception as _fb_exc:
            logger.warning("FeedbackProcessor.process_dismiss error (non-fatal): %s", _fb_exc)

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
# WebSocket real-time signal stream
# ---------------------------------------------------------------------------

class WebSocketConnectionManager:
    """Manage active WebSocket connections per user, backed by Redis pub/sub.

    Each user gets their own Redis channel ``signals:{user_id}``.  When the
    inference pipeline or a Celery task generates a new actionable signal for
    a user, it publishes a JSON payload to that channel.  All connected
    WebSocket clients for that user receive the broadcast instantly.

    Connection lifecycle
    --------------------
    1. Client opens ``GET /api/v1/signals/ws?token=<JWT>``.
    2. Manager authenticates the token, subscribes to ``signals:{user_id}``.
    3. Incoming Redis messages are forwarded to the WebSocket.
    4. On disconnect, the subscription is torn down and the gauge updated.
    """

    async def connect(
        self,
        ws: WebSocket,
        user_id: str,
        redis_url: str,
    ) -> None:
        """Accept the WebSocket, subscribe to the user's Redis channel, and
        handle both inbound ``chat_message`` events and outbound signal pushes.

        The connection is fully bidirectional:

        * **Outbound** — Redis pub/sub messages on ``signals:{user_id}`` are
          forwarded verbatim as ``{"type": "signal", ...}`` frames.
        * **Inbound** — the client may send ``{"type": "chat_message",
          "signal_id": "<uuid>", "message": "<text>"}`` frames.  These are
          currently acknowledged with a ``{"type": "chat_ack"}`` frame;
          full response generation is handled by the ``/chat`` HTTP endpoint
          which uses ``SignalInteractionAgent`` for RAG-backed replies.

        Args:
            ws: FastAPI WebSocket connection.
            user_id: Authenticated user's UUID string.
            redis_url: Redis connection URL from settings.
        """
        await ws.accept()
        MetricsCollector.record_websocket_connection(+1)
        channel = f"signals:{user_id}"
        client: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=True)
        pubsub = client.pubsub()
        await pubsub.subscribe(channel)
        logger.info("WebSocket subscribed to %s", channel)

        async def _redis_listener() -> None:
            """Forward Redis pub/sub messages to the WebSocket."""
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await ws.send_text(message["data"])

        async def _ws_receiver() -> None:
            """Process inbound WebSocket frames from the client."""
            while True:
                try:
                    raw = await ws.receive_text()
                except WebSocketDisconnect:
                    break
                except Exception:
                    break
                try:
                    frame = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                event_type = frame.get("type")
                if event_type == "chat_message":
                    # Acknowledge receipt; actual response via /chat endpoint
                    ack = json.dumps({
                        "type": "chat_ack",
                        "signal_id": frame.get("signal_id"),
                        "status": "received",
                    })
                    await ws.send_text(ack)
                elif event_type == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

        try:
            # Run both tasks concurrently; stop as soon as either finishes
            await asyncio.gather(
                _redis_listener(),
                _ws_receiver(),
                return_exceptions=True,
            )
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected for user %s", user_id)
        except Exception as exc:
            logger.warning("WebSocket error for user %s: %s", user_id, exc)
        finally:
            await pubsub.unsubscribe(channel)
            await client.aclose()
            MetricsCollector.record_websocket_connection(-1)


_ws_manager = WebSocketConnectionManager()


@router.websocket("/ws")
async def signal_stream_ws(
    ws: WebSocket,
    token: str = Query(..., description="JWT access token (query param — WS cannot send headers)"),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Stream new signals to the client in real time over WebSocket.

    Authentication is performed via a JWT passed as the ``token`` query
    parameter (WebSocket connections cannot carry Authorization headers).

    Each message is a JSON object with a ``type`` field:

    * ``{"type": "signal", "data": {...}}`` — a new actionable signal.
    * ``{"type": "ping"}`` — keepalive sent every 30 seconds.

    The client must reconnect on disconnect; the server does not buffer
    messages sent while the client was offline.

    Args:
        ws: WebSocket connection provided by FastAPI.
        token: Bearer JWT token passed as a query parameter.
        db: Database session for user lookup.

    Raises:
        WebSocketDisconnect: Raised internally when the client closes.
    """
    from app.api.routes.auth import get_current_user_from_token
    try:
        current_user: User = await get_current_user_from_token(token, db)
    except Exception:
        await ws.close(code=1008)  # Policy violation — auth failure
        return

    await _ws_manager.connect(
        ws=ws,
        user_id=str(current_user.id),
        redis_url=settings.redis_url,
    )


# ---------------------------------------------------------------------------
# Deep Research endpoint
# ---------------------------------------------------------------------------

class DeepResearchRequest(BaseModel):
    """Request body for the Deep Research endpoint."""

    question: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Opening research question to drive the recursive analysis.",
    )
    content_history_ids: Optional[List[UUID]] = Field(
        None,
        description="UUIDs of ContentItems to cross-reference during research.",
    )
    max_depth: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum recursion depth (1–5).",
    )


@router.post(
    "/{signal_id}/deep-research",
    response_model=DeepResearchReport,
    summary="Run Deep Research on an actionable signal",
    description=(
        "Trigger a recursive multi-step LLM analysis on a specific signal. "
        "The orchestrator resolves knowledge gaps iteratively up to ``max_depth`` rounds "
        "and returns a structured research report with a final synthesis paragraph."
    ),
)
async def deep_research(
    signal_id: UUID,
    request: DeepResearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DeepResearchReport:
    """Execute Deep Research mode on an actionable signal.

    Args:
        signal_id: UUID of the signal to research.
        request: Research parameters (question, depth, content history).
        current_user: Authenticated user.
        db: Database session.

    Returns:
        :class:`~app.intelligence.orchestrator.DeepResearchReport` with all
        recursive steps and a final synthesis paragraph.

    Raises:
        HTTPException 404: Signal not found or not owned by current user.
        HTTPException 500: Orchestration failure.
    """
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Signal not found")

    signal_context = (
        f"Type: {signal_db.signal_type.value}\n"
        f"Title: {signal_db.title}\n"
        f"Description: {signal_db.description}\n"
        f"Context: {signal_db.context}"
    )
    orchestrator = MultiAgentOrchestrator()
    # Build VectorSearchTool for this request (Req 2)
    try:
        from app.core.db import AsyncSessionLocal
        vec_tool: Optional[VectorSearchTool] = VectorSearchTool(
            db_session_factory=AsyncSessionLocal,
        )
    except Exception:
        vec_tool = None

    # Build temporal signals from recent acted signals of the same type (Req 2)
    temporal_signals: List[dict] = []
    try:
        ts_result = await db.execute(
            select(ActionableSignalDB)
            .where(
                and_(
                    ActionableSignalDB.user_id == current_user.id,
                    ActionableSignalDB.signal_type == signal_db.signal_type,
                    ActionableSignalDB.acted_at.isnot(None),
                )
            )
            .order_by(ActionableSignalDB.acted_at.desc())
            .limit(5)
        )
        for ts in ts_result.scalars().all():
            temporal_signals.append({
                "type": ts.signal_type.value,
                "confidence": float(ts.confidence_score or 0.0),
                "title": ts.title or "",
                "acted_at": ts.acted_at.isoformat() if ts.acted_at else None,
            })
    except Exception as ts_exc:
        logger.debug("Temporal signals query failed: %s", ts_exc)

    try:
        report = await orchestrator.deep_research(
            signal_id=str(signal_id),
            signal_type=signal_db.signal_type.value,
            signal_context=signal_context,
            initial_question=request.question,
            max_depth=request.max_depth,
            vector_search_tool=vec_tool,
            user_id=current_user.id,
            temporal_signals=temporal_signals or None,
        )
    except Exception as exc:
        logger.error("Deep Research failed for signal %s: %s", signal_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Deep Research orchestration failed",
        )
    return report


# ---------------------------------------------------------------------------
# Req 3 — Conversational chat + one-click draft-response endpoints
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """Request body for the signal chat endpoint."""

    message: str = Field(..., min_length=1, max_length=1000)
    history: Optional[List[dict]] = Field(
        default=None,
        description=(
            "Prior conversation turns as ``[{'role': 'user'|'assistant', 'content': '...'}]``. "
            "Last 6 turns are used."
        ),
    )
    report: Optional[dict] = Field(
        None,
        description="Serialised DeepResearchReport from a prior research session.",
    )


class ChatResponse(BaseModel):
    """Response body for the signal chat endpoint."""

    answer: str
    signal_id: str


class OneClickActionRequest(BaseModel):
    """Request body for the one-click-action endpoint."""

    channel: str = Field(
        default="internal_note",
        description="Target channel: 'dm', 'public_reply', 'email', or 'internal_note'.",
    )
    report: Optional[dict] = Field(
        None,
        description="Serialised DeepResearchReport; when absent a lightweight synthesis is used.",
    )


@router.post(
    "/{signal_id}/chat",
    response_model=ChatResponse,
    summary="Ask a follow-up question about a signal (RAG-backed)",
)
async def signal_chat(
    signal_id: UUID,
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ChatResponse:
    """Answer a follow-up question about an actionable signal using the
    ``DeepResearchReport`` as a RAG source.

    The endpoint is the HTTP counterpart to the ``chat_message`` WebSocket event.
    It is stateless: the caller must pass *history* and *report* on every request.

    Args:
        signal_id: UUID of the signal being discussed.
        request: Chat payload (message, optional history, optional report).
        current_user: Authenticated user.
        db: Database session.

    Returns:
        :class:`ChatResponse` with the assistant's answer.

    Raises:
        HTTPException 404: Signal not owned by the authenticated user.
    """
    res = await db.execute(
        select(ActionableSignalDB).where(
            and_(
                ActionableSignalDB.id == signal_id,
                ActionableSignalDB.user_id == current_user.id,
            )
        )
    )
    signal_db = res.scalar_one_or_none()
    if not signal_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Signal not found")

    signal_context = (
        f"Type: {signal_db.signal_type.value}\n"
        f"Title: {signal_db.title}\n"
        f"Description: {signal_db.description}\n"
    )

    # Reconstruct report from request payload if provided
    report_obj: Optional[DeepResearchReport] = None
    if request.report:
        try:
            report_obj = DeepResearchReport(**request.report)
        except Exception:
            report_obj = None

    # Fallback minimal report when none supplied
    if report_obj is None:
        from app.intelligence.orchestrator import DeepResearchStep
        report_obj = DeepResearchReport(
            signal_id=str(signal_id),
            signal_type=signal_db.signal_type.value,
            initial_question="",
            steps=[],
            final_synthesis=signal_db.description or "",
            total_tokens_used=0,
            max_depth_reached=0,
            knowledge_gaps_remaining=[],
            started_at=datetime.now(timezone.utc),
        )

    # Reconstruct conversation history
    history_turns: List[ConversationTurn] = []
    for turn in (request.history or [])[-6:]:
        history_turns.append(
            ConversationTurn(role=turn.get("role", "user"), content=turn.get("content", ""))
        )

    agent = SignalInteractionAgent()
    answer = await agent.chat(
        signal_context=signal_context,
        report=report_obj,
        history=history_turns,
        user_message=request.message,
    )
    return ChatResponse(answer=answer, signal_id=str(signal_id))


@router.post(
    "/{signal_id}/one-click-action",
    summary="Generate a one-click draft response grounded in Deep Research",
)
async def one_click_action(
    signal_id: UUID,
    request: OneClickActionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Generate a tailored draft response for an actionable signal.

    Uses ``SignalInteractionAgent.generate_draft_response()`` to produce copy
    for the requested *channel* grounded in the signal's ``DeepResearchReport``.
    The user's ``StrategicPriorities.tone`` is applied when available.

    Args:
        signal_id: UUID of the signal to respond to.
        request: Channel selection and optional report payload.
        current_user: Authenticated user.
        db: Database session.

    Returns:
        Serialised :class:`~app.intelligence.orchestrator.DraftResponse`.

    Raises:
        HTTPException 404: Signal not found or not owned by user.
        HTTPException 500: Draft generation failed.
    """
    res = await db.execute(
        select(ActionableSignalDB).where(
            and_(
                ActionableSignalDB.id == signal_id,
                ActionableSignalDB.user_id == current_user.id,
            )
        )
    )
    signal_db = res.scalar_one_or_none()
    if not signal_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Signal not found")

    # Determine tone from user's strategic_priorities (Req 1 integration)
    user_ctx = UserContext.from_user(current_user)
    tone = user_ctx.strategic_priorities.tone

    signal_context = (
        f"Type: {signal_db.signal_type.value}\n"
        f"Title: {signal_db.title}\n"
        f"Description: {signal_db.description}\n"
    )

    # Reconstruct report from payload or build minimal fallback
    report_obj: Optional[DeepResearchReport] = None
    if request.report:
        try:
            report_obj = DeepResearchReport(**request.report)
        except Exception:
            report_obj = None
    if report_obj is None:
        report_obj = DeepResearchReport(
            signal_id=str(signal_id),
            signal_type=signal_db.signal_type.value,
            initial_question="",
            steps=[],
            final_synthesis=signal_db.description or "",
            total_tokens_used=0,
            max_depth_reached=0,
            knowledge_gaps_remaining=[],
            started_at=datetime.now(timezone.utc),
        )

    agent = SignalInteractionAgent()
    try:
        draft = await agent.generate_draft_response(
            signal_context=signal_context,
            report=report_obj,
            channel=request.channel,
            tone=tone,
        )
    except Exception as exc:
        logger.error("one-click-action failed for signal %s: %s", signal_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Draft generation failed",
        )
    return {
        "signal_id": draft.signal_id,
        "channel": draft.channel,
        "tone": draft.tone,
        "body": draft.body,
        "suggested_subject": draft.suggested_subject,
        "generated_at": draft.generated_at.isoformat(),
        "source_report_steps": draft.source_report_steps,
    }


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
    from datetime import timedelta, timezone as _tz

    now = datetime.now(_tz.utc)
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

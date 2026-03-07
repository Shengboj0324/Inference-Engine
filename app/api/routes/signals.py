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

import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.dependencies import get_current_user, get_db
from app.core.db_models import ActionableSignalDB, User
from app.core.signal_models import (
    ActionableSignal,
    SignalFilter,
    SignalStatus,
    SignalSummary,
    SignalType,
)

logger = logging.getLogger(__name__)

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
    """Request to assign signal to team member."""

    user_id: UUID = Field(..., description="User ID to assign to")


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
            query = query.where(ActionableSignalDB.signal_type.in_(signal_types))

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

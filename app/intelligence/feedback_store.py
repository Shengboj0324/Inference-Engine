"""Durable feedback store for signal-classification corrections.

``FeedbackStore`` persists per-signal human corrections to the
``signal_feedback`` PostgreSQL table (see ``app/core/db_models.SignalFeedbackDB``).
When no ``session_factory`` is injected it falls back to an in-memory list so
that unit tests remain dependency-free.

The ``record()`` method also triggers a one-step gradient update on the
``ConfidenceCalibrator`` so that newly submitted corrections immediately
influence future probability estimates.

Schema (additive-only migration)::

    CREATE TABLE signal_feedback (
        id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        signal_id            UUID NOT NULL REFERENCES actionable_signals(id) ON DELETE CASCADE,
        predicted_type       VARCHAR(50) NOT NULL,
        true_type            VARCHAR(50) NOT NULL,
        predicted_confidence FLOAT NOT NULL,
        user_id              UUID NOT NULL,
        created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
    );
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, List, Optional
from uuid import UUID, uuid4

from app.domain.inference_models import SignalType

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Immutable record of a single signal-classification correction.

    Attributes:
        id: Unique record identifier.
        signal_id: UUID of the parent ``ActionableSignalDB`` row.
        predicted_type: The signal type predicted by the model.
        true_type: The correct signal type supplied by the human reviewer.
        predicted_confidence: Model confidence at prediction time (0–1).
        user_id: UUID of the reviewer who submitted the correction.
        created_at: UTC timestamp when the feedback was recorded.
    """

    signal_id: UUID
    predicted_type: str
    true_type: str
    predicted_confidence: float
    user_id: UUID
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackStore:
    """Async store for signal-classification feedback records.

    Supports both a live PostgreSQL backend (via an injected
    ``async_session_factory``) and an in-memory backend for testing.

    Args:
        session_factory: Zero-argument async callable that returns an
            ``AsyncSession``.  When ``None`` the store operates in memory.
        confidence_calibrator: Optional ``ConfidenceCalibrator`` instance.
            When provided, ``record()`` calls ``calibrator.update()`` after
            persisting each record so scalars stay fresh.
    """

    def __init__(
        self,
        session_factory: Optional[Callable] = None,
        confidence_calibrator: Optional[object] = None,
    ) -> None:
        """Initialise the store.

        Args:
            session_factory: Async callable returning an ``AsyncSession``.
            confidence_calibrator: Optional ``ConfidenceCalibrator`` for
                online scalar updates.
        """
        self._session_factory: Optional[Callable] = session_factory
        self._calibrator = confidence_calibrator
        self._memory: List[FeedbackRecord] = []
        logger.info(
            "FeedbackStore initialised (backend=%s)",
            "db" if session_factory else "memory",
        )

    async def record(
        self,
        signal_id: UUID,
        predicted_type: str,
        true_type: str,
        predicted_confidence: float,
        user_id: UUID,
    ) -> FeedbackRecord:
        """Persist a feedback record and update the calibrator.

        Args:
            signal_id: UUID of the signal being corrected.
            predicted_type: Model's predicted ``SignalType`` value string.
            true_type: Human-supplied correct ``SignalType`` value string.
            predicted_confidence: Model confidence at prediction time.
            user_id: UUID of the reviewer.

        Returns:
            The newly created ``FeedbackRecord``.
        """
        rec = FeedbackRecord(
            signal_id=signal_id,
            predicted_type=predicted_type,
            true_type=true_type,
            predicted_confidence=max(0.0, min(1.0, predicted_confidence)),
            user_id=user_id,
        )

        if self._session_factory is not None:
            await self._persist_to_db(rec)
        else:
            self._memory.append(rec)

        logger.info(
            "FeedbackStore.record: signal=%s predicted=%s true=%s",
            signal_id,
            predicted_type,
            true_type,
        )

        # Trigger online calibrator update when available.
        if self._calibrator is not None:
            try:
                signal_type = SignalType(predicted_type)
                true_label: bool = predicted_type == true_type
                self._calibrator.update(signal_type, predicted_confidence, true_label)
            except (ValueError, Exception) as exc:
                logger.warning("Calibrator update skipped: %s", exc)

        return rec

    async def get_recent(self, limit: int = 500) -> List[FeedbackRecord]:
        """Return the most recent feedback records ordered by ``created_at`` DESC.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of ``FeedbackRecord`` instances, newest first.
        """
        if self._session_factory is not None:
            return await self._fetch_from_db(limit)
        sorted_records = sorted(self._memory, key=lambda r: r.created_at, reverse=True)
        return sorted_records[:limit]

    # ------------------------------------------------------------------
    # DB helpers (only called when session_factory is set)
    # ------------------------------------------------------------------

    async def _persist_to_db(self, rec: FeedbackRecord) -> None:
        """Insert ``rec`` into the ``signal_feedback`` table.

        Args:
            rec: The ``FeedbackRecord`` to persist.
        """
        from app.core.db_models import SignalFeedbackDB  # deferred to avoid circular imports

        async with self._session_factory() as session:
            db_row = SignalFeedbackDB(
                id=rec.id,
                signal_id=rec.signal_id,
                predicted_type=rec.predicted_type,
                true_type=rec.true_type,
                predicted_confidence=rec.predicted_confidence,
                user_id=rec.user_id,
                created_at=rec.created_at,
            )
            session.add(db_row)
            await session.commit()

    async def _fetch_from_db(self, limit: int) -> List[FeedbackRecord]:
        """Fetch the most recent rows from the ``signal_feedback`` table.

        Args:
            limit: Maximum number of rows to fetch.

        Returns:
            List of ``FeedbackRecord`` instances.
        """
        from sqlalchemy import select, desc
        from app.core.db_models import SignalFeedbackDB

        async with self._session_factory() as session:
            stmt = (
                select(SignalFeedbackDB)
                .order_by(desc(SignalFeedbackDB.created_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                FeedbackRecord(
                    id=row.id,
                    signal_id=row.signal_id,
                    predicted_type=row.predicted_type,
                    true_type=row.true_type,
                    predicted_confidence=row.predicted_confidence,
                    user_id=row.user_id,
                    created_at=row.created_at,
                )
                for row in rows
            ]


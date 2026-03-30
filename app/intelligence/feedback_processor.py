"""Closed-loop accuracy calibration via human signal feedback.

``FeedbackProcessor`` operationalises the ``signal_feedback`` table by
translating act / dismiss events into online ``ConfidenceCalibrator`` updates.

Design
------
* **Act** (``true_label=True``) — the human agreed the signal was actionable;
  the predicted signal type was correct.  The calibrator reinforces confidence
  for this ``SignalType``.
* **Dismiss** (``true_label=False``) — the human rejected the signal.  The
  calibrator down-weights overconfident predictions for this ``SignalType``.
* **action_score EMA** — an exponential moving average (α=0.3) is applied to
  update the per-signal ``action_score`` stored in the DB, creating a
  lightweight online learning signal even without a full retrain.

Both ``process_act`` and ``process_dismiss`` are async so they can be awaited
inside FastAPI route handlers without blocking the event loop.
"""

import asyncio
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.core.monitoring import MetricsCollector
from app.domain.inference_models import SignalType
from app.intelligence.calibration import ConfidenceCalibrator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Smoothing factor for the action_score EMA update.
_EMA_ALPHA: float = 0.3

#: Reward applied to action_score when a signal is acted upon.
_ACT_REWARD: float = 1.0

#: Penalty applied to action_score when a signal is dismissed.
_DISMISS_PENALTY: float = 0.0

#: Default path for the calibration state file.
_DEFAULT_STATE_PATH: Path = Path("training/calibration_state.json")

#: Minimum temperature scalar shift that triggers a global queue re-rank.
#: At lr=0.01 each gradient step shifts T by ~0.001–0.005; 0.05 means roughly
#: 10–50 feedback events have converged before a re-rank fires.
_RERANK_THRESHOLD: float = 0.05


async def _rerank_signals_background(signal_type_value: str, new_temperature: float) -> None:
    """Background coroutine: recompute ``action_score`` for all queued signals.

    Queries every ``ActionableSignalDB`` record whose ``status`` is ``NEW`` or
    ``QUEUED`` and whose ``signal_type`` matches the updated calibration scalar,
    then recalculates ``action_score`` using the new temperature:

    .. code-block::

        calibrated_confidence = sigmoid(logit(raw_confidence) / new_temperature)
        new_action_score = urgency_score * impact_score * calibrated_confidence

    Writes all updates in a single commit per database session.  Runs in a
    separate asyncio task — failures are logged but never propagate.

    Args:
        signal_type_value: ``SignalType.value`` string whose T scalar changed.
        new_temperature: Updated temperature scalar from ``ConfidenceCalibrator``.
    """
    try:
        from sqlalchemy import select, and_
        from app.core.db import AsyncSessionLocal
        from app.core.db_models import ActionableSignalDB
        from app.core.signal_models import SignalStatus, SignalType as DBSignalType

        # Map the string back to the DB enum
        try:
            db_signal_type = DBSignalType(signal_type_value)
        except ValueError:
            logger.warning(
                "_rerank_signals_background: unknown signal_type=%r; skipping",
                signal_type_value,
            )
            return

        active_statuses = [SignalStatus.NEW, SignalStatus.QUEUED]

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ActionableSignalDB).where(
                    and_(
                        ActionableSignalDB.signal_type == db_signal_type,
                        ActionableSignalDB.status.in_(active_statuses),
                    )
                )
            )
            signals = result.scalars().all()
            if not signals:
                logger.debug(
                    "_rerank_signals_background: no active %s signals to re-rank",
                    signal_type_value,
                )
                return

            updated = 0
            for sig in signals:
                raw_conf = max(1e-7, min(1.0 - 1e-7, float(sig.confidence_score)))
                raw_logit = math.log(raw_conf / (1.0 - raw_conf))
                t = max(0.1, min(100.0, new_temperature))
                calibrated = 1.0 / (1.0 + math.exp(-raw_logit / t))
                new_score = float(sig.urgency_score) * float(sig.impact_score) * calibrated
                sig.action_score = max(0.0, min(1.0, new_score))
                updated += 1

            await session.commit()
            logger.info(
                "_rerank_signals_background: re-ranked %d %s signals "
                "using T=%.4f",
                updated, signal_type_value, new_temperature,
            )

    except Exception as exc:
        logger.error(
            "_rerank_signals_background: failed for signal_type=%r: %s",
            signal_type_value, exc, exc_info=True,
        )


@dataclass
class FeedbackResult:
    """Summary of a single feedback processing event.

    Attributes:
        signal_id: UUID string of the processed signal.
        signal_type: The signal's predicted type.
        action: 'act' or 'dismiss'.
        old_action_score: Score before the EMA update.
        new_action_score: Score after the EMA update.
        calibrator_updated: Whether ``ConfidenceCalibrator.update()`` was called.
        processed_at: UTC timestamp of processing.
    """
    signal_id: str
    signal_type: str
    action: str
    old_action_score: float
    new_action_score: float
    calibrator_updated: bool
    processed_at: datetime


class FeedbackProcessor:
    """Translate act / dismiss events into online calibration updates.

    Args:
        calibrator: ``ConfidenceCalibrator`` instance.  When ``None``, a new
            one is created using the default state path.
        state_path: Path to the calibration state JSON file.  Ignored when
            ``calibrator`` is provided.
    """

    def __init__(
        self,
        calibrator: Optional[ConfidenceCalibrator] = None,
        state_path: Optional[Path] = None,
    ) -> None:
        self._calibrator: ConfidenceCalibrator = calibrator or ConfidenceCalibrator(
            state_path=state_path or _DEFAULT_STATE_PATH
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_act(
        self,
        signal_id: str,
        signal_type_value: str,
        confidence_score: float,
        current_action_score: float,
        notes: Optional[str] = None,
    ) -> FeedbackResult:
        """Record that a user acted on a signal (positive label).

        Calls ``ConfidenceCalibrator.update(true_label=True)`` to reinforce the
        temperature scalar for this ``SignalType``, and computes an EMA-adjusted
        ``action_score`` biased toward 1.0.

        Args:
            signal_id: UUID string of the signal.
            signal_type_value: ``SignalType.value`` string.
            confidence_score: The signal's predicted confidence (0–1).
            current_action_score: The signal's current composite priority score.
            notes: Optional free-text notes from the user (logged, not used).

        Returns:
            :class:`FeedbackResult` with old/new action scores.
        """
        return await self._process(
            signal_id=signal_id,
            signal_type_value=signal_type_value,
            confidence_score=confidence_score,
            current_action_score=current_action_score,
            true_label=True,
            action="act",
            context=notes,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _process(
        self,
        signal_id: str,
        signal_type_value: str,
        confidence_score: float,
        current_action_score: float,
        true_label: bool,
        action: str,
        context: Optional[str],
    ) -> FeedbackResult:
        """Core processing logic shared by act and dismiss.

        Args:
            signal_id: UUID string of the signal.
            signal_type_value: Signal type value string.
            confidence_score: Predicted confidence (0–1).
            current_action_score: Current composite priority score.
            true_label: True for act, False for dismiss.
            action: Human-readable action label for logging.
            context: Optional free-text context (notes / reason).

        Returns:
            :class:`FeedbackResult` with updated scores.
        """
        # 1. Resolve SignalType enum — fall back gracefully
        calibrator_updated = False
        try:
            signal_type = SignalType(signal_type_value)

            # Snapshot old scalar before the gradient step so we can measure drift
            old_t: float = self._calibrator._scalars.get(signal_type_value, 1.0)

            self._calibrator.update(
                signal_type=signal_type,
                predicted_prob=confidence_score,
                true_label=true_label,
            )
            calibrator_updated = True

            # Measure scalar shift and trigger re-rank when it exceeds threshold
            new_t: float = self._calibrator._scalars.get(signal_type_value, 1.0)
            scalar_delta = abs(new_t - old_t)
            if scalar_delta >= _RERANK_THRESHOLD:
                logger.info(
                    "FeedbackProcessor: scalar shift %.4f ≥ threshold %.4f for %s — "
                    "scheduling global queue re-rank (T: %.4f → %.4f)",
                    scalar_delta, _RERANK_THRESHOLD, signal_type_value, old_t, new_t,
                )
                try:
                    asyncio.get_event_loop().create_task(
                        _rerank_signals_background(signal_type_value, new_t)
                    )
                except RuntimeError:
                    # No running event loop (e.g. unit-test sync context): skip
                    logger.debug(
                        "FeedbackProcessor: no event loop — skipping background re-rank"
                    )

        except (ValueError, Exception) as exc:
            logger.warning(
                "FeedbackProcessor: calibrator update skipped for type=%r: %s",
                signal_type_value, exc,
            )

        # 2. EMA update for action_score
        target = _ACT_REWARD if true_label else _DISMISS_PENALTY
        new_score = _EMA_ALPHA * target + (1.0 - _EMA_ALPHA) * current_action_score
        new_score = max(0.0, min(1.0, new_score))

        # 3. Metrics
        MetricsCollector.record_token_signal_ratio(
            signal_type=signal_type_value, tokens=0
        )  # placeholder — real token count comes from the LLM router

        logger.info(
            "FeedbackProcessor: signal=%s type=%s action=%s "
            "true_label=%s confidence=%.3f score: %.3f→%.3f context=%r",
            signal_id, signal_type_value, action,
            true_label, confidence_score, current_action_score, new_score,
            (context or "")[:80],
        )

        return FeedbackResult(
            signal_id=signal_id,
            signal_type=signal_type_value,
            action=action,
            old_action_score=current_action_score,
            new_action_score=new_score,
            calibrator_updated=calibrator_updated,
            processed_at=datetime.now(timezone.utc),
        )

    async def process_dismiss(
        self,
        signal_id: str,
        signal_type_value: str,
        confidence_score: float,
        current_action_score: float,
        reason: Optional[str] = None,
    ) -> FeedbackResult:
        """Record that a user dismissed a signal (negative label).

        Calls ``ConfidenceCalibrator.update(true_label=False)`` to increase the
        temperature scalar (widen confidence) for this ``SignalType``, and
        computes an EMA-adjusted ``action_score`` biased toward 0.0.

        Args:
            signal_id: UUID string of the signal.
            signal_type_value: ``SignalType.value`` string.
            confidence_score: The signal's predicted confidence (0–1).
            current_action_score: The signal's current composite priority score.
            reason: Optional dismissal reason from the user (logged, not used).

        Returns:
            :class:`FeedbackResult` with old/new action scores.
        """
        return await self._process(
            signal_id=signal_id,
            signal_type_value=signal_type_value,
            confidence_score=confidence_score,
            current_action_score=current_action_score,
            true_label=False,
            action="dismiss",
            context=reason,
        )


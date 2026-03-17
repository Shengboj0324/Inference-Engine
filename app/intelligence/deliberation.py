"""Structured pre-inference deliberation engine.

``DeliberationEngine`` runs before ``LLMAdjudicator.adjudicate()`` and produces
a ``DeliberationReport`` that guides the downstream reasoning path selection.

Four steps are executed in order:

A. **Signal landscape scan** — query ``ContextMemoryStore`` for the 5 most
   similar past observations and compute a ``signal_type`` frequency
   distribution from their stored labels.

B. **Candidate pruning** — remove any candidate whose ``SignalType`` has **zero**
   historical occurrences for this user **AND** whose retrieval score is below
   ``min_retrieval_score`` (default 0.4).  At least one candidate is always
   preserved to prevent a degenerate empty list.

C. **Risk escalation check** — if any surviving candidate is a high-stakes
   signal type (``CHURN_RISK``, ``LEGAL_RISK``, ``SECURITY_CONCERN``,
   ``REPUTATION_RISK``) and has a score above 0.5, set
   ``DeliberationReport.escalate = True`` and emit a structured audit log
   entry via the ``DataResidencyGuard`` audit logger.

D. **Reasoning mode selection** — choose among ``"single_call"``,
   ``"chain_of_thought"``, and ``"multi_agent"`` based on candidate count,
   text length, and top-two confidence spread.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal

from app.domain.inference_models import SignalType
from app.domain.normalized_models import NormalizedObservation
from app.intelligence.candidate_retrieval import SignalCandidate
from app.intelligence.context_memory import ContextMemoryStore
from app.llm.router import _FRONTIER_SIGNAL_TYPES  # canonical definition, single source of truth

logger = logging.getLogger(__name__)

# Audit logger mirrors the one used by DataResidencyGuard so that risk events
# appear in the same structured audit stream.
_AUDIT_LOGGER = logging.getLogger("radar.data_residency.audit")

# Threshold for Step C risk escalation.
_ESCALATION_SCORE_THRESHOLD: float = 0.5

# Step D thresholds
_TEXT_LENGTH_MULTI_AGENT: int = 1500
_CANDIDATE_COUNT_MULTI_AGENT: int = 6
_COT_CONFIDENCE_SPREAD: float = 0.1
_COT_CONFIDENCE_REQUIRED: float = 0.85


@dataclass
class DeliberationReport:
    """Output of ``DeliberationEngine.deliberate()``.

    Attributes:
        pruned_candidates: Candidate list after Step B pruning.
        reasoning_mode: Chosen reasoning path for the downstream adjudicator.
        escalate: ``True`` when a high-stakes signal exceeded the escalation
            threshold (Step C).
        historical_distribution: Frequency map of ``signal_type`` values from
            the 5 most similar past observations (Step A).
    """

    pruned_candidates: List[SignalCandidate]
    reasoning_mode: Literal["single_call", "chain_of_thought", "multi_agent"]
    escalate: bool
    historical_distribution: Dict[str, int] = field(default_factory=dict)


class DeliberationEngine:
    """Run structured pre-inference deliberation before ``adjudicate()``.

    Args:
        context_memory: ``ContextMemoryStore`` used for Steps A and B.
        min_retrieval_score: Minimum candidate score required to keep a
            candidate that has no historical occurrences (Step B).  Defaults
            to ``0.4``.
    """

    def __init__(
        self,
        context_memory: ContextMemoryStore,
        min_retrieval_score: float = 0.4,
    ) -> None:
        """Initialise the deliberation engine.

        Args:
            context_memory: Store for past observation embeddings.
            min_retrieval_score: Step B pruning threshold.
        """
        self._context_memory: ContextMemoryStore = context_memory
        self._min_retrieval_score: float = min_retrieval_score

    async def deliberate(
        self,
        observation: NormalizedObservation,
        candidates: List[SignalCandidate],
    ) -> DeliberationReport:
        """Execute all four deliberation steps and return a ``DeliberationReport``.

        Args:
            observation: Normalised observation about to be adjudicated.
            candidates: Candidate signal types from the retrieval stage.

        Returns:
            ``DeliberationReport`` with pruned candidates and selected
            reasoning mode.
        """
        # ── Step A: Signal landscape scan ───────────────────────────────────
        historical_distribution = await self._step_a_landscape_scan(observation)

        # ── Step B: Candidate pruning ────────────────────────────────────────
        pruned = self._step_b_prune(candidates, historical_distribution)

        # ── Step C: Risk escalation check ───────────────────────────────────
        escalate = self._step_c_escalation_check(pruned, observation)

        # ── Step D: Reasoning mode selection ────────────────────────────────
        reasoning_mode = self._step_d_reasoning_mode(observation, pruned)

        logger.debug(
            "DeliberationEngine: observation=%s mode=%s escalate=%s pruned=%d/%d",
            observation.id,
            reasoning_mode,
            escalate,
            len(pruned),
            len(candidates),
        )

        return DeliberationReport(
            pruned_candidates=pruned,
            reasoning_mode=reasoning_mode,
            escalate=escalate,
            historical_distribution=historical_distribution,
        )

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    async def _step_a_landscape_scan(
        self, observation: NormalizedObservation
    ) -> Dict[str, int]:
        """Query the 5 most similar past observations and tally signal types.

        Args:
            observation: Current observation; its text is used as the query.

        Returns:
            Frequency map ``{signal_type_value: count}``.
        """
        query_text = observation.normalized_text or observation.title or ""
        try:
            memories = await self._context_memory.retrieve(
                user_id=observation.user_id,
                query_text=query_text,
                top_k=5,
            )
        except Exception as exc:
            logger.warning("Step A landscape scan failed: %s", exc)
            return {}

        distribution: Dict[str, int] = {}
        for mem in memories:
            key = mem.signal_type.value
            distribution[key] = distribution.get(key, 0) + 1
        return distribution

    def _step_b_prune(
        self,
        candidates: List[SignalCandidate],
        historical_distribution: Dict[str, int],
    ) -> List[SignalCandidate]:
        """Remove zero-history, low-score candidates.

        A candidate is removed when ALL of the following are true:
        * Its ``SignalType`` has zero occurrences in ``historical_distribution``.
        * Its retrieval ``score`` is below ``self._min_retrieval_score``.

        At least one candidate is always preserved.

        Args:
            candidates: Full candidate list from retrieval.
            historical_distribution: From Step A.

        Returns:
            Pruned candidate list (never empty if ``candidates`` is non-empty).
        """
        pruned = [
            c for c in candidates
            if historical_distribution.get(c.signal_type.value, 0) > 0
            or c.score >= self._min_retrieval_score
        ]
        # Safety net: never return an empty list.
        return pruned if pruned else candidates

    def _step_c_escalation_check(
        self,
        candidates: List[SignalCandidate],
        observation: NormalizedObservation,
    ) -> bool:
        """Log a structured audit entry if a high-stakes signal exceeds 0.5.

        Args:
            candidates: Pruned candidates from Step B.
            observation: Current observation for audit context.

        Returns:
            ``True`` when escalation was triggered.
        """
        escalate = False
        for candidate in candidates:
            if (
                candidate.signal_type in _FRONTIER_SIGNAL_TYPES
                and candidate.score > _ESCALATION_SCORE_THRESHOLD
            ):
                escalate = True
                _AUDIT_LOGGER.warning(
                    "RISK_ESCALATION signal_type=%s score=%.3f "
                    "observation_id=%s user_id=%s",
                    candidate.signal_type.value,
                    candidate.score,
                    observation.id,
                    observation.user_id,
                )
        return escalate

    @staticmethod
    def _step_d_reasoning_mode(
        observation: NormalizedObservation,
        candidates: List[SignalCandidate],
    ) -> Literal["single_call", "chain_of_thought", "multi_agent"]:
        """Select the downstream reasoning path.

        Rules (evaluated in priority order):

        1. ``multi_agent`` when text length exceeds 1 500 chars OR there
           are more than 6 candidates.
        2. ``chain_of_thought`` when the top two candidate scores differ by
           less than 0.1, OR when ``observation.confidence_required > 0.85``.
        3. ``single_call`` otherwise.

        Args:
            observation: Current observation.
            candidates: Pruned candidates from Step B.

        Returns:
            One of ``"single_call"``, ``"chain_of_thought"``,
            ``"multi_agent"``.
        """
        text_len: int = len(observation.normalized_text or "")
        if text_len > _TEXT_LENGTH_MULTI_AGENT or len(candidates) > _CANDIDATE_COUNT_MULTI_AGENT:
            return "multi_agent"

        scores = sorted([c.score for c in candidates], reverse=True)
        confidence_required: float = getattr(observation, "confidence_required", 0.0) or 0.0

        if confidence_required > _COT_CONFIDENCE_REQUIRED:
            return "chain_of_thought"

        if len(scores) >= 2 and (scores[0] - scores[1]) < _COT_CONFIDENCE_SPREAD:
            return "chain_of_thought"

        return "single_call"


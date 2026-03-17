"""Multi-agent decomposition pipeline for complex signal observations.

``MultiAgentOrchestrator`` decomposes observations with many candidate signal
types or long text into parallel ``SubTaskAgent`` calls that each assess a
single candidate against the observation text.  An ``AggregatorAgent`` then
merges the results using a confidence-weighted vote.

All sub-tasks receive only PII-scrubbed text via
``DataResidencyGuard._scrub_text()``, enforcing the zero-egress contract.

Activation criteria (checked by ``LLMAdjudicator.adjudicate()``):
  * ``len(observation.normalized_text) > 1500``, OR
  * ``len(candidates) > 6``
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.data_residency import DataResidencyGuard
from app.domain.inference_models import SignalType
from app.domain.normalized_models import NormalizedObservation
from app.intelligence.candidate_retrieval import SignalCandidate
from app.intelligence.llm_adjudicator import LLMAdjudicationOutput
from app.llm.models import LLMMessage
from app.llm.router import LLMRouter, get_router

logger = logging.getLogger(__name__)


@dataclass
class SubTaskResult:
    """Result produced by a single ``SubTaskAgent`` invocation.

    Attributes:
        signal_type: The candidate ``SignalType`` assessed by this sub-task.
        confidence: Model confidence that this type applies (0–1).
        evidence: Verbatim text evidence extracted from the observation.
        rationale: One-sentence reasoning from the sub-task LLM call.
    """

    signal_type: SignalType
    confidence: float
    evidence: str
    rationale: str


class SubTaskAgent:
    """Classify a single candidate ``SignalType`` against a text excerpt.

    Args:
        router: ``LLMRouter`` instance to use for generation.
        temperature: Sampling temperature.
        max_tokens: Token budget for the sub-task call.
    """

    def __init__(
        self,
        router: LLMRouter,
        temperature: float = 0.3,
        max_tokens: int = 300,
    ) -> None:
        """Initialise the sub-task agent.

        Args:
            router: LLM router shared across all sub-tasks.
            temperature: Sampling temperature.
            max_tokens: Per-call token budget.
        """
        self._router: LLMRouter = router
        self._temperature: float = temperature
        self._max_tokens: int = max_tokens

    async def classify(
        self,
        candidate: SignalCandidate,
        safe_text: str,
        signal_type: Optional[SignalType] = None,
    ) -> SubTaskResult:
        """Assess whether ``candidate`` applies to ``safe_text``.

        Args:
            candidate: The signal type candidate to evaluate.
            safe_text: PII-scrubbed observation text.
            signal_type: LLMRouter tier override (defaults to ``candidate.signal_type``).

        Returns:
            ``SubTaskResult`` with ``confidence`` in ``[0.0, 1.0]``.  Falls back
            to ``confidence=0.0`` if the LLM call or JSON parse fails.
        """
        routing_type: Optional[SignalType] = signal_type or candidate.signal_type
        prompt = (
            f"Does the following text express a '{candidate.signal_type.value}' signal?\n\n"
            f"Text: {safe_text[:600]}\n\n"
            "Reply with a JSON object: "
            '{"applies": <bool>, "confidence": <float 0-1>, '
            '"evidence": "<verbatim excerpt ≤15 words>", "rationale": "<one sentence>"}'
        )
        try:
            raw = await self._router.generate_for_signal(
                signal_type=routing_type,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            content = raw.strip()
            start, end = content.index("{"), content.rindex("}") + 1
            data: Dict[str, Any] = json.loads(content[start:end])
            confidence: float = float(data.get("confidence", 0.0)) if data.get("applies") else 0.0
            return SubTaskResult(
                signal_type=candidate.signal_type,
                confidence=min(1.0, max(0.0, confidence)),
                evidence=str(data.get("evidence", "")),
                rationale=str(data.get("rationale", "")),
            )
        except Exception as exc:
            logger.warning(
                "SubTaskAgent failed for %s: %s", candidate.signal_type.value, exc
            )
            return SubTaskResult(
                signal_type=candidate.signal_type,
                confidence=0.0,
                evidence="",
                rationale="sub-task failed",
            )


class AggregatorAgent:
    """Merge ``SubTaskResult`` objects via confidence-weighted vote.

    The winning signal type is the one with the highest total confidence across
    all sub-tasks.  When all sub-tasks return ``confidence=0.0`` the aggregator
    produces an abstention output.
    """

    def aggregate(self, results: List[SubTaskResult]) -> LLMAdjudicationOutput:
        """Produce a single ``LLMAdjudicationOutput`` from sub-task results.

        Args:
            results: List of ``SubTaskResult`` objects, one per candidate.

        Returns:
            ``LLMAdjudicationOutput`` where ``primary_signal_type`` is the
            candidate with the highest aggregated confidence score.
        """
        if not results:
            return LLMAdjudicationOutput(
                candidate_signal_types=["unclear"],
                primary_signal_type="unclear",
                confidence=0.0,
                evidence_spans=[],
                rationale="No sub-task results to aggregate.",
                requires_more_context=True,
                abstain=True,
                abstention_reason="insufficient_context",
                risk_labels=[],
                suggested_actions=[],
            )

        # Sum confidence per signal type (weighted vote)
        score_map: Dict[str, float] = {}
        evidence_map: Dict[str, str] = {}
        rationale_parts: List[str] = []

        for result in results:
            key = result.signal_type.value
            score_map[key] = score_map.get(key, 0.0) + result.confidence
            if result.evidence:
                evidence_map[key] = result.evidence
            if result.confidence > 0.0:
                rationale_parts.append(
                    f"{key} ({result.confidence:.2f}): {result.rationale}"
                )

        best_type = max(score_map, key=lambda k: score_map[k])
        best_confidence = min(1.0, score_map[best_type] / len(results))
        should_abstain = best_confidence < 0.3
        candidate_types = sorted(score_map, key=lambda k: score_map[k], reverse=True)

        evidence_spans = [
            {"text": evidence_map[best_type], "reason": "highest confidence sub-task"}
        ] if best_type in evidence_map else []

        return LLMAdjudicationOutput(
            candidate_signal_types=candidate_types[:5],
            primary_signal_type=best_type if not should_abstain else "unclear",
            confidence=best_confidence,
            evidence_spans=evidence_spans,
            rationale=" | ".join(rationale_parts) or "Aggregated from sub-task agents.",
            requires_more_context=should_abstain,
            abstain=should_abstain,
            abstention_reason="low_confidence" if should_abstain else None,
            risk_labels=[],
            suggested_actions=[],
        )


class MultiAgentOrchestrator:
    """Decompose complex observations into parallel ``SubTaskAgent`` calls.

    Triggered when:
    * ``len(observation.normalized_text) > 1500``, **or**
    * ``len(candidates) > 6``

    Each candidate gets its own ``SubTaskAgent`` invocation.  All tasks run
    concurrently via ``asyncio.gather()``.  Results are merged by
    ``AggregatorAgent``.

    The ``DataResidencyGuard`` is applied to ``observation.normalized_text``
    before any text is passed to sub-tasks.

    Args:
        router: ``LLMRouter`` instance; defaults to the global singleton.
        sub_task_temperature: Sampling temperature for ``SubTaskAgent`` calls.
        sub_task_max_tokens: Token budget per sub-task call.
    """

    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        sub_task_temperature: float = 0.3,
        sub_task_max_tokens: int = 300,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            router: LLM router; defaults to the global singleton.
            sub_task_temperature: Sampling temperature for sub-tasks.
            sub_task_max_tokens: Token budget per sub-task call.
        """
        self._router: LLMRouter = router or get_router()
        self._sub_agent: SubTaskAgent = SubTaskAgent(
            router=self._router,
            temperature=sub_task_temperature,
            max_tokens=sub_task_max_tokens,
        )
        self._aggregator: AggregatorAgent = AggregatorAgent()

    async def orchestrate(
        self,
        observation: NormalizedObservation,
        candidates: List[SignalCandidate],
        signal_type: Optional[SignalType] = None,
    ) -> LLMAdjudicationOutput:
        """Run sub-tasks concurrently and aggregate the results.

        The observation text is PII-scrubbed via ``DataResidencyGuard._scrub_text``
        before being forwarded to any ``SubTaskAgent``.

        Args:
            observation: Normalised observation to classify.
            candidates: Candidate signal types from the retrieval stage.
            signal_type: LLMRouter tier override for all sub-tasks.

        Returns:
            ``LLMAdjudicationOutput`` produced by ``AggregatorAgent.aggregate()``.
        """
        safe_text, _ = DataResidencyGuard._scrub_text(observation.normalized_text or "")

        tasks = [
            self._sub_agent.classify(candidate, safe_text, signal_type)
            for candidate in candidates
        ]
        results: List[SubTaskResult] = await asyncio.gather(*tasks)
        logger.info(
            "MultiAgentOrchestrator: %d sub-tasks completed for observation %s",
            len(results),
            observation.id,
        )
        return self._aggregator.aggregate(list(results))


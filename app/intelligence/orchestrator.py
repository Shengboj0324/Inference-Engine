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
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from app.core.data_residency import DataResidencyGuard
from app.core.monitoring import MetricsCollector
from app.domain.inference_models import SignalType
from app.domain.normalized_models import NormalizedObservation
from app.intelligence.candidate_retrieval import SignalCandidate
from app.intelligence.llm_adjudicator import LLMAdjudicationOutput
from app.llm.models import LLMMessage
from app.llm.router import LLMRouter, get_router

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deep Research data structures
# ---------------------------------------------------------------------------

@dataclass
class DeepResearchStep:
    """A single recursive LLM analysis step in a Deep Research session.

    Attributes:
        depth: Zero-based recursion depth of this step.
        question: The knowledge gap this step was designed to resolve.
        answer: LLM-generated answer / analysis for the question.
        knowledge_gaps: Further questions identified by the LLM after answering.
        sources_referenced: ``ContentItem`` IDs consulted at this step.
        tokens_used: Approximate token count for this step.
        completed_at: UTC timestamp when the step finished.
    """
    depth: int
    question: str
    answer: str
    knowledge_gaps: List[str]
    sources_referenced: List[str]
    tokens_used: int
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DeepResearchReport:
    """Aggregated output of a Deep Research session on an actionable signal.

    Attributes:
        signal_id: UUID of the signal that was researched.
        signal_type: The signal's classification type.
        initial_question: The user's or system's initial research question.
        steps: All recursive steps executed (in order).
        final_synthesis: LLM-generated synthesis across all steps.
        total_tokens_used: Sum of tokens across all steps.
        max_depth_reached: Deepest recursion level hit.
        knowledge_gaps_remaining: Unanswered gaps after max_depth.
        started_at: UTC start time.
        completed_at: UTC end time.
    """
    signal_id: str
    signal_type: str
    initial_question: str
    steps: List[DeepResearchStep]
    final_synthesis: str
    total_tokens_used: int
    max_depth_reached: int
    knowledge_gaps_remaining: List[str]
    started_at: datetime
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


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


@dataclass
class VectorSearchResult:
    """A single content item returned by :class:`VectorSearchTool`.

    Attributes:
        content_id: UUID string of the ``ContentItemDB`` row.
        snippet: Truncated ``raw_text`` excerpt for prompt injection.
        similarity: Cosine similarity score in [0, 1].
        platform: Source platform label (e.g. ``"reddit"``).
        published_at: UTC publish timestamp of the original content.
    """

    content_id: str
    snippet: str
    similarity: float
    platform: str
    published_at: Optional[datetime] = None


class VectorSearchTool:
    """Query the pgvector / HNSW store for ``ContentItem`` embeddings.

    Used by :meth:`MultiAgentOrchestrator.deep_research` to retrieve content
    snippets that semantically match each critic-approved knowledge gap.  The
    retrieved snippets are injected into :meth:`DeepResearchAgent.research_step`
    so the researcher grounds its answers in actual ingested content rather than
    relying solely on parametric LLM knowledge.

    Args:
        db_session_factory: Async SQLAlchemy session factory
            (``AsyncSessionLocal`` from ``app.core.db``).  If ``None``, the
            tool silently returns empty results so the rest of the pipeline
            continues without vector augmentation.
        embed_fn: Async callable ``(text) → List[float]``.  Defaults to a
            lightweight synchronous fallback (no API call) when ``None``.
        top_k: Number of nearest-neighbour results to return per query.
    """

    def __init__(
        self,
        db_session_factory: Optional[Callable] = None,
        embed_fn: Optional[Callable] = None,
        top_k: int = 5,
    ) -> None:
        self._db_factory = db_session_factory
        self._embed_fn = embed_fn
        self._top_k = top_k

    async def search(
        self,
        query_text: str,
        user_id: UUID,
    ) -> List[VectorSearchResult]:
        """Return the top-K content items most similar to *query_text*.

        Args:
            query_text: Free-text research question from the researcher agent.
            user_id: Filters results to rows owned by this user.

        Returns:
            Up to ``top_k`` :class:`VectorSearchResult` objects sorted by
            descending cosine similarity.  Returns ``[]`` on any error so the
            caller always receives a valid (possibly empty) list.
        """
        if self._db_factory is None:
            return []
        try:
            # Generate query embedding
            if self._embed_fn is not None:
                query_vec: List[float] = await self._embed_fn(query_text)
            else:
                # Fallback: bag-of-words approximate embedding via character hashes
                import hashlib
                h = hashlib.md5(query_text.encode()).digest()
                query_vec = [float(b) / 255.0 for b in h] * (1536 // len(h) + 1)
                query_vec = query_vec[:1536]

            from sqlalchemy import select
            from app.core.db_models import ContentItemDB

            async with self._db_factory() as session:
                # Use pgvector cosine distance operator
                stmt = (
                    select(ContentItemDB)
                    .where(ContentItemDB.user_id == user_id)
                    .where(ContentItemDB.embedding.isnot(None))
                    .order_by(ContentItemDB.embedding.cosine_distance(query_vec))
                    .limit(self._top_k)
                )
                rows = (await session.execute(stmt)).scalars().all()

            import numpy as np
            q_arr = np.array(query_vec, dtype=np.float32)
            q_norm = np.linalg.norm(q_arr)

            results: List[VectorSearchResult] = []
            for row in rows:
                similarity = 0.0  # default when embedding unavailable
                try:
                    r_arr = np.array(row.embedding, dtype=np.float32)
                    r_norm = np.linalg.norm(r_arr)
                    denom = q_norm * r_norm
                    if denom > 0:
                        similarity = float(np.dot(q_arr, r_arr) / denom)
                except Exception:
                    pass
                results.append(VectorSearchResult(
                    content_id=str(row.id),
                    snippet=(row.raw_text or "")[:300],
                    similarity=round(similarity, 4),
                    platform=str(row.source_platform),
                    published_at=row.published_at,
                ))
            # Enhancement 1 — re-sort by numpy cosine similarity (descending).
            # The pgvector ORDER BY is an approximation; re-sorting here ensures
            # the final list is always in strict similarity order regardless of
            # any DB-side quantisation or HNSW approximation artefacts.
            results.sort(key=lambda r: r.similarity, reverse=True)
            return results
        except Exception as exc:
            logger.warning("VectorSearchTool.search failed: %s", exc)
            return []


@dataclass
class CritiqueResult:
    """Output produced by :class:`ResearchCriticAgent` for one research step.

    Attributes:
        relevant_gaps: Knowledge gaps that are on-topic and worth pursuing.
        filtered_gaps: Gaps removed for being off-topic, speculative, or
            potentially hallucinated.
        quality_score: Critic's assessment of the step answer quality (0–1).
            Scores below 0.4 indicate the answer is likely fabricated or too
            vague to build on.
        critic_reasoning: One-sentence explanation of the filtering decision.
        step_depth: Depth of the ``DeepResearchStep`` that was critiqued.
        correction_strategies: Hallucination self-correction (Req 5).  For
            each filtered gap the critic produces an alternative research
            direction so the loop never dead-ends.  Empty when no gaps were
            filtered.
    """

    relevant_gaps: List[str]
    filtered_gaps: List[str]
    quality_score: float
    critic_reasoning: str
    step_depth: int
    correction_strategies: List[str] = field(default_factory=list)


class ResearchCriticAgent:
    """Validate ``DeepResearchStep`` outputs before the next recursion level.

    The critic is a lightweight LLM call that receives:

    1. The original signal context (so it knows the scope of the research).
    2. The step's question and answer.
    3. The candidate knowledge gaps proposed by the researcher.

    It returns a JSON payload that:
    * Scores the answer quality on 0–1 (flags hallucinated or vague answers).
    * Classifies each proposed gap as ``relevant`` or ``filtered``.
    * Provides a single-sentence explanation.

    Only ``relevant`` gaps are queued for the next recursion round.  When the
    quality score falls below ``quality_threshold``, the step is marked
    unreliable and **no** gaps are forwarded, stopping runaway recursion.

    Args:
        router: Shared ``LLMRouter`` instance.
        temperature: Sampling temperature (default 0.2 — critic should be
            conservative and deterministic).
        max_tokens: Token budget for the critic call.
        quality_threshold: Minimum step quality score; steps below this
            threshold have all their gaps suppressed.
    """

    def __init__(
        self,
        router: LLMRouter,
        temperature: float = 0.2,
        max_tokens: int = 500,
        quality_threshold: float = 0.4,
    ) -> None:
        self._router = router
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._quality_threshold = quality_threshold

    async def critique(
        self,
        step: "DeepResearchStep",
        signal_context: str,
        signal_type: Optional[SignalType] = None,
    ) -> CritiqueResult:
        """Validate a research step and filter its proposed knowledge gaps.

        Args:
            step: The completed ``DeepResearchStep`` to review.
            signal_context: PII-scrubbed original signal description.
            signal_type: Optional LLMRouter tier override.

        Returns:
            :class:`CritiqueResult` with relevant/filtered gap lists and quality
            score.  On LLM or parse failure, returns a conservative fallback
            that passes all gaps through with quality_score=0.5.
        """
        safe_context, _ = DataResidencyGuard._scrub_text(signal_context)
        gaps_json = json.dumps(step.knowledge_gaps)
        prompt = (
            "You are a rigorous research critic reviewing one step of a recursive intelligence analysis.\n\n"
            f"SIGNAL SCOPE: {safe_context[:400]}\n\n"
            f"STEP {step.depth} — Question: {step.question}\n"
            f"Answer: {step.answer[:400]}\n\n"
            f"Proposed follow-up questions: {gaps_json}\n\n"
            "Evaluate:\n"
            "1. Is the answer factual, grounded in the signal scope, and specific enough to act on?\n"
            "2. For each follow-up question, is it RELEVANT (directly advances understanding of the signal)\n"
            "   or FILTERED (off-topic, speculative, too broad, or potentially hallucinated)?\n"
            "3. For each FILTERED question, provide ONE short 'Correction Strategy' — a rephrased or\n"
            "   broadened alternative question that IS on-topic so the research loop never dead-ends.\n\n"
            "Reply with a JSON object ONLY:\n"
            "{\n"
            '  "quality_score": <float 0-1>,\n'
            '  "relevant_gaps": ["<question>", ...],\n'
            '  "filtered_gaps": ["<question>", ...],\n'
            '  "correction_strategies": ["<one on-topic replacement per filtered gap>", ...],\n'
            '  "reasoning": "<one sentence>"\n'
            "}"
        )
        try:
            raw = await self._router.generate_for_signal(
                signal_type=signal_type,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=self._max_tokens + 100,  # +100 for correction strategies
            )
            content = raw.strip()
            start, end = content.index("{"), content.rindex("}") + 1
            data: Dict[str, Any] = json.loads(content[start:end])
            quality = float(data.get("quality_score", 0.5))
            relevant = [str(g) for g in data.get("relevant_gaps", [])]
            filtered = [str(g) for g in data.get("filtered_gaps", [])]
            reasoning = str(data.get("reasoning", ""))
            # Req 5: extract correction strategies for filtered gaps
            corrections = [str(s) for s in data.get("correction_strategies", [])]

            # Suppress all gaps when quality is below the trust threshold
            if quality < self._quality_threshold:
                logger.warning(
                    "ResearchCriticAgent: step %d quality=%.2f below threshold=%.2f — "
                    "suppressing all %d gaps",
                    step.depth, quality, self._quality_threshold, len(relevant),
                )
                filtered = relevant + filtered
                relevant = []
                # When quality is below threshold, inject a broad correction strategy
                if not corrections:
                    corrections = [
                        "Broaden the research scope to industry-wide trends rather than "
                        "signal-specific details."
                    ]

            # Enhancement 3 — Correction strategies completeness guarantee.
            # The LLM may return fewer correction_strategies than filtered_gaps
            # (e.g., it generates 1 for 3 filtered gaps, or none at all).
            # Pad to ensure len(corrections) >= len(filtered) so downstream
            # callers can always zip filtered_gaps and correction_strategies 1-to-1
            # without index errors or silent silences.
            _FALLBACK_CORRECTION = (
                "Broaden the research scope to related market trends or industry benchmarks."
            )
            while len(corrections) < len(filtered):
                corrections.append(_FALLBACK_CORRECTION)

            logger.info(
                "ResearchCriticAgent: step=%d quality=%.2f relevant=%d filtered=%d corrections=%d",
                step.depth, quality, len(relevant), len(filtered), len(corrections),
            )
            return CritiqueResult(
                relevant_gaps=relevant,
                filtered_gaps=filtered,
                quality_score=quality,
                critic_reasoning=reasoning,
                step_depth=step.depth,
                correction_strategies=corrections,
            )
        except Exception as exc:
            logger.warning("ResearchCriticAgent failed for step %d: %s", step.depth, exc)
            # Conservative fallback: pass all gaps through unchanged
            return CritiqueResult(
                relevant_gaps=step.knowledge_gaps,
                filtered_gaps=[],
                quality_score=0.5,
                critic_reasoning="Critic call failed; gaps passed through unchanged.",
                step_depth=step.depth,
                correction_strategies=[],
            )


class DeepResearchAgent:
    """Perform recursive LLM analysis on a signal to resolve knowledge gaps.

    Each call to ``research_step()`` produces one ``DeepResearchStep`` that
    answers a given question and identifies further sub-questions for the next
    recursion level.  All text passed to the LLM is PII-scrubbed via
    ``DataResidencyGuard._scrub_text()`` before leaving the system.

    Args:
        router: ``LLMRouter`` instance for LLM calls.
        temperature: Sampling temperature (lower = more focused analysis).
        max_tokens: Token budget per recursive step.
    """

    def __init__(
        self,
        router: LLMRouter,
        temperature: float = 0.4,
        max_tokens: int = 800,
    ) -> None:
        self._router = router
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def research_step(
        self,
        question: str,
        signal_context: str,
        history: List[DeepResearchStep],
        signal_type: Optional[SignalType] = None,
        retrieved_snippets: Optional[List[VectorSearchResult]] = None,
        correction_context: Optional[str] = None,
    ) -> DeepResearchStep:
        """Execute one recursive research step.

        Args:
            question: The knowledge gap to resolve in this step.
            signal_context: PII-scrubbed signal description / evidence text.
            history: Previous steps (used to build the reasoning chain).
            signal_type: Optional LLMRouter tier override.
            retrieved_snippets: Up to 5 ``VectorSearchResult`` objects from the
                HNSW / pgvector store that semantically match *question*.  When
                present they are injected into the prompt so the agent can
                ground its answer in actual ingested content (Req 2).
            correction_context: Optional string from ``CritiqueResult.
                correction_strategies`` for the *previous* step.  Injected
                when the prior step had filtered gaps to prevent dead-ends (Req 5).

        Returns:
            A completed ``DeepResearchStep`` with the LLM's answer and any
            further knowledge gaps it identified.
        """
        safe_context, _ = DataResidencyGuard._scrub_text(signal_context)
        history_summary = "\n".join(
            f"Step {s.depth}: Q={s.question[:80]} A={s.answer[:80]}"
            for s in history[-3:]  # keep last 3 steps to stay within context window
        )

        # Build vector-augmented evidence block
        evidence_block = ""
        if retrieved_snippets:
            lines = ["### RETRIEVED EVIDENCE (from ingested content — use to ground your answer)"]
            for i, snip in enumerate(retrieved_snippets[:5], 1):
                lines.append(
                    f"{i}. [sim={snip.similarity:.3f} | {snip.platform} | id={snip.content_id}] "
                    f"{snip.snippet[:200]}"
                )
            evidence_block = "\n".join(lines) + "\n\n"

        # Build correction-strategy hint block
        correction_block = ""
        if correction_context:
            correction_block = (
                f"### RESEARCH DIRECTION HINT (from critic self-correction)\n"
                f"{correction_context}\n\n"
            )

        prompt = (
            "You are a senior intelligence analyst performing deep research on a social-media signal.\n\n"
            f"Signal context:\n{safe_context[:800]}\n\n"
            f"Research history (last {len(history[-3:])} steps):\n{history_summary or 'None yet.'}\n\n"
            f"{evidence_block}"
            f"{correction_block}"
            f"Current question to answer:\n{question}\n\n"
            "Reply with a JSON object:\n"
            "{\n"
            '  "answer": "<detailed analysis of the question, 2–4 sentences>",\n'
            '  "knowledge_gaps": ["<follow-up question 1>", "<follow-up question 2>"],\n'
            '  "sources_referenced": ["<content_item_id or URL if known>"],\n'
            '  "tokens_estimate": <int>\n'
            "}"
        )
        routing_type = signal_type
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
            return DeepResearchStep(
                depth=len(history),
                question=question,
                answer=str(data.get("answer", "")),
                knowledge_gaps=[str(g) for g in data.get("knowledge_gaps", [])[:3]],
                sources_referenced=[str(s) for s in data.get("sources_referenced", [])],
                tokens_used=int(data.get("tokens_estimate", self._max_tokens)),
            )
        except Exception as exc:
            logger.warning("DeepResearchAgent step failed: %s", exc)
            return DeepResearchStep(
                depth=len(history),
                question=question,
                answer="Step failed; insufficient data for this question.",
                knowledge_gaps=[],
                sources_referenced=[],
                tokens_used=0,
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
        deep_research_temperature: float = 0.4,
        deep_research_max_tokens: int = 800,
    ) -> None:
        """Initialise the orchestrator.

        Args:
            router: LLM router; defaults to the global singleton.
            sub_task_temperature: Sampling temperature for sub-tasks.
            sub_task_max_tokens: Token budget per sub-task call.
            deep_research_temperature: Temperature for Deep Research steps.
            deep_research_max_tokens: Token budget per Deep Research step.
        """
        self._router: LLMRouter = router or get_router()
        self._sub_agent: SubTaskAgent = SubTaskAgent(
            router=self._router,
            temperature=sub_task_temperature,
            max_tokens=sub_task_max_tokens,
        )
        self._aggregator: AggregatorAgent = AggregatorAgent()
        self._deep_research_agent: DeepResearchAgent = DeepResearchAgent(
            router=self._router,
            temperature=deep_research_temperature,
            max_tokens=deep_research_max_tokens,
        )
        self._critic_agent: ResearchCriticAgent = ResearchCriticAgent(
            router=self._router,
        )

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

    async def deep_research(
        self,
        signal_id: str,
        signal_type: str,
        signal_context: str,
        initial_question: str,
        content_history: Optional[List[Dict[str, Any]]] = None,
        max_depth: int = 3,
        vector_search_tool: Optional[VectorSearchTool] = None,
        user_id: Optional[UUID] = None,
        temporal_signals: Optional[List[Dict[str, Any]]] = None,
        time_budget_seconds: Optional[float] = None,
    ) -> DeepResearchReport:
        """Run recursive LLM analysis on an actionable signal.

        Performs up to ``max_depth`` rounds of LLM-driven research, each round
        answering the knowledge gaps produced by the previous one.  All text is
        PII-scrubbed before being forwarded to the LLM.  Metrics are emitted
        after every step.

        Vector augmentation (Req 2): when *vector_search_tool* and *user_id*
        are provided, after each critic review the tool is queried for content
        snippets matching the *next* pending question.  Snippets are injected
        into the subsequent ``research_step`` call.

        Temporal context (Req 2): when *temporal_signals* are provided, the top
        5 most semantically similar historical signals are injected into the
        final synthesis so ``_synthesise_research`` has long-horizon context.

        Hallucination self-correction (Req 5): when the critic produces
        ``correction_strategies`` for filtered gaps, they are concatenated into
        a ``correction_context`` string and injected into the *next* step.

        Args:
            signal_id: UUID string of the actionable signal being researched.
            signal_type: Signal type value string (used for metrics labelling).
            signal_context: Description / evidence text from the signal.
            initial_question: The user's or system's opening research question.
            content_history: Optional list of related ``ContentItem`` dicts for
                cross-referencing (title and URL are injected into the context).
            max_depth: Maximum recursion depth (default 3, max 5 enforced).
            vector_search_tool: Optional :class:`VectorSearchTool` for HNSW
                snippet retrieval (Req 2).
            user_id: User UUID required when *vector_search_tool* is provided.
            temporal_signals: Top-K most similar historical signals to the
                current one; injected into the synthesis (Req 2).

        Returns:
            A :class:`DeepResearchReport` containing all steps and a final
            synthesis paragraph produced by the LLM.
        """
        max_depth = min(max_depth, 5)
        started_at = datetime.now(timezone.utc)

        # Enrich context with content history excerpts (up to 5 items)
        enriched_context = signal_context
        if content_history:
            excerpts = "\n".join(
                f"- [{item.get('platform', '?')}] {item.get('title', '')[:80]} "
                f"({item.get('url', '')})"
                for item in content_history[:5]
            )
            enriched_context = f"{signal_context}\n\nRelated content:\n{excerpts}"

        routing_type: Optional[SignalType] = None
        try:
            routing_type = SignalType(signal_type)
        except ValueError:
            pass

        steps: List[DeepResearchStep] = []
        pending_questions: List[str] = [initial_question]
        # Track gaps the critic suppressed for the final report
        all_filtered_gaps: List[str] = []
        # Carry correction strategies into the next research step (Req 5)
        next_correction_context: Optional[str] = None
        # Carry vector snippets into the next research step (Req 2)
        next_retrieved_snippets: Optional[List[VectorSearchResult]] = None

        for depth in range(max_depth):
            if not pending_questions:
                break
            # Enhancement 2 — Circuit Breaker: exit gracefully when the wall-clock
            # budget is exceeded rather than hanging or returning a timeout error.
            # The partial report assembled so far is still returned so callers
            # always receive a usable (albeit incomplete) research output.
            if time_budget_seconds is not None:
                elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
                if elapsed > time_budget_seconds:
                    logger.warning(
                        "DeepResearch: time budget %.1fs exceeded at depth=%d "
                        "(elapsed=%.2fs); returning partial report with %d steps",
                        time_budget_seconds, depth, elapsed, len(steps),
                    )
                    pending_questions.insert(0, pending_questions[0] if pending_questions else "")
                    break
            question = pending_questions.pop(0)

            # --- Vector search for this question (Req 2) ---
            retrieved = next_retrieved_snippets  # pre-fetched from previous iteration
            if vector_search_tool is not None and user_id is not None:
                try:
                    retrieved = await vector_search_tool.search(
                        query_text=question, user_id=user_id
                    )
                    logger.debug(
                        "VectorSearchTool: %d snippets retrieved for depth=%d",
                        len(retrieved), depth,
                    )
                except Exception as vs_exc:
                    logger.warning("VectorSearchTool error at depth=%d: %s", depth, vs_exc)
                    retrieved = []

            # --- Research step ---
            step = await self._deep_research_agent.research_step(
                question=question,
                signal_context=enriched_context,
                history=steps,
                signal_type=routing_type,
                retrieved_snippets=retrieved or [],
                correction_context=next_correction_context,
            )
            steps.append(step)
            MetricsCollector.record_deep_research_step(signal_type=signal_type)

            # --- Critic validation (gate before next recursion) ---
            critique = await self._critic_agent.critique(
                step=step,
                signal_context=enriched_context,
                signal_type=routing_type,
            )
            all_filtered_gaps.extend(critique.filtered_gaps)

            # Overwrite the step's knowledge_gaps in-place so the final
            # report accurately reflects what was actually approved.
            step.knowledge_gaps = critique.relevant_gaps

            # Only queue critic-approved gaps (at most 2) for next depth
            pending_questions.extend(critique.relevant_gaps[:2])

            # Carry correction strategies into next iteration (Req 5)
            if critique.correction_strategies:
                next_correction_context = " | ".join(critique.correction_strategies[:3])
            else:
                next_correction_context = None

            # Pre-fetch vector snippets for the next pending question (Req 2)
            next_retrieved_snippets = None
            if (
                vector_search_tool is not None
                and user_id is not None
                and pending_questions
            ):
                try:
                    next_retrieved_snippets = await vector_search_tool.search(
                        query_text=pending_questions[0], user_id=user_id
                    )
                except Exception:
                    next_retrieved_snippets = []

            logger.info(
                "DeepResearch depth=%d question=%r "
                "quality=%.2f approved_gaps=%d suppressed_gaps=%d corrections=%d",
                depth,
                question[:60],
                critique.quality_score,
                len(critique.relevant_gaps),
                len(critique.filtered_gaps),
                len(critique.correction_strategies),
            )

        # Final synthesis — inject temporal signals for long-horizon context (Req 2)
        synthesis = await self._synthesise_research(
            steps, signal_type, routing_type, temporal_signals=temporal_signals
        )

        return DeepResearchReport(
            signal_id=signal_id,
            signal_type=signal_type,
            initial_question=initial_question,
            steps=steps,
            final_synthesis=synthesis,
            total_tokens_used=sum(s.tokens_used for s in steps),
            max_depth_reached=len(steps) - 1 if steps else 0,
            knowledge_gaps_remaining=pending_questions,
            started_at=started_at,
        )

    async def _synthesise_research(
        self,
        steps: List[DeepResearchStep],
        signal_type: str,
        routing_type: Optional[SignalType],
        temporal_signals: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Produce a final synthesis paragraph from all research steps.

        When *temporal_signals* are provided (Req 2 — Temporal Context
        Awareness), the top 5 most semantically similar historical signals are
        injected into the prompt so the synthesiser can compare the current
        findings against the team's longitudinal signal history.

        Args:
            steps: Completed research steps.
            signal_type: Signal type label for the prompt.
            routing_type: LLMRouter tier override.
            temporal_signals: Optional list of dicts with keys ``type``,
                ``confidence``, ``title``, and ``acted_at`` representing
                recent similar signals.  Injected as a ``### HISTORICAL
                CONTEXT`` block when non-empty.

        Returns:
            Single actionable synthesis string.
        """
        if not steps:
            return "No research steps completed."
        step_summaries = "\n".join(
            f"Step {s.depth}: {s.answer[:120]}" for s in steps
        )
        # Build temporal context block (Req 2)
        temporal_block = ""
        if temporal_signals:
            lines = ["### HISTORICAL CONTEXT — top 5 similar signals (temporal awareness)"]
            for sig in temporal_signals[:5]:
                lines.append(
                    f"- [{sig.get('type', '?')} | conf={sig.get('confidence', 0):.2f}] "
                    f"{sig.get('title', '')[:80]} (acted: {sig.get('acted_at', 'unknown')})"
                )
            temporal_block = "\n".join(lines) + "\n\n"
        prompt = (
            f"You researched a '{signal_type}' signal across {len(steps)} recursive steps.\n\n"
            f"Findings:\n{step_summaries}\n\n"
            f"{temporal_block}"
            "Synthesise these findings into a single, actionable paragraph (3–5 sentences) "
            "that a GTM or product team can act on immediately. Be specific."
        )
        try:
            result = await self._router.generate_for_signal(
                signal_type=routing_type,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.5,
                max_tokens=400,
            )
            return result.strip()
        except Exception as exc:
            logger.warning("DeepResearch synthesis failed: %s", exc)
            return step_summaries


# ---------------------------------------------------------------------------
# Req 3 — Conversational Interaction Layer
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """One turn in a :class:`SignalInteractionAgent` conversation.

    Attributes:
        role: ``"user"`` or ``"assistant"``.
        content: Message content (PII-scrubbed before storage).
        timestamp: UTC time the turn was recorded.
    """

    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DraftResponse:
    """Automatically generated response draft produced by ``one-click-action``.

    Attributes:
        signal_id: UUID of the originating actionable signal.
        channel: Target channel — ``"dm"``, ``"public_reply"``, ``"email"``,
            or ``"internal_note"``.
        tone: Prose tone applied — matches ``StrategicPriorities.tone``.
        body: The draft message body.
        suggested_subject: Email subject line when *channel* is ``"email"``.
        generated_at: UTC generation timestamp.
        source_report_steps: Number of ``DeepResearchReport`` steps used.
    """

    signal_id: str
    channel: str
    tone: str
    body: str
    suggested_subject: Optional[str]
    generated_at: datetime
    source_report_steps: int


class SignalInteractionAgent:
    """RAG-backed conversational agent for follow-up questions about a signal.

    Uses the ``DeepResearchReport`` as a retrieval source so answers are
    grounded in the specific evidence gathered during research rather than
    generic LLM parametric knowledge.

    Args:
        router: ``LLMRouter`` instance.
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens: Max response tokens per conversational turn.
    """

    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        temperature: float = 0.4,
        max_tokens: int = 600,
    ) -> None:
        self._router = router or get_router()
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def chat(
        self,
        signal_context: str,
        report: DeepResearchReport,
        history: List[ConversationTurn],
        user_message: str,
        signal_type: Optional[SignalType] = None,
    ) -> str:
        """Answer a follow-up question using the ``DeepResearchReport`` as RAG source.

        The full research history is serialised into a ``### RESEARCH CONTEXT``
        block.  The conversation history (last 6 turns) is serialised as a
        ``### CONVERSATION HISTORY`` block.  PII is scrubbed from both before
        injection.

        Args:
            signal_context: Raw signal context string (will be PII-scrubbed).
            report: The ``DeepResearchReport`` from the prior research session.
                Used as the RAG source — steps, synthesis, and gaps are
                included in the prompt.
            history: Prior conversation turns; last 6 are used.
            user_message: The analyst's follow-up question (will be scrubbed).
            signal_type: Optional LLMRouter routing hint.

        Returns:
            Assistant response string.
        """
        safe_context, _ = DataResidencyGuard._scrub_text(signal_context)
        safe_message, _ = DataResidencyGuard._scrub_text(user_message)

        # Serialise report into RAG source
        rag_lines = [
            f"Signal: {report.signal_type}",
            f"Initial question: {report.initial_question}",
            f"Final synthesis: {report.final_synthesis[:400]}",
        ]
        for step in report.steps:
            rag_lines.append(
                f"Research depth {step.depth}: Q={step.question[:80]} → {step.answer[:120]}"
            )
        rag_block = "\n".join(rag_lines)

        # Serialise conversation history (last 6 turns)
        history_block = "\n".join(
            f"{t.role.upper()}: {t.content[:200]}" for t in history[-6:]
        ) or "None yet."

        prompt = (
            "You are an intelligence assistant helping a GTM analyst understand a market signal.\n\n"
            f"### SIGNAL CONTEXT\n{safe_context[:500]}\n\n"
            f"### RESEARCH CONTEXT (your knowledge base for this conversation)\n{rag_block}\n\n"
            f"### CONVERSATION HISTORY\n{history_block}\n\n"
            f"### ANALYST QUESTION\n{safe_message}\n\n"
            "Answer the question using ONLY information present in the research context above. "
            "If the answer is not there, say so honestly. 2–4 sentences maximum."
        )
        try:
            raw = await self._router.generate_for_signal(
                signal_type=signal_type,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            return raw.strip()
        except Exception as exc:
            logger.warning("SignalInteractionAgent.chat failed: %s", exc)
            return "I was unable to retrieve an answer right now. Please try again."

    async def generate_draft_response(
        self,
        signal_context: str,
        report: DeepResearchReport,
        channel: str = "internal_note",
        tone: str = "neutral",
        signal_type: Optional[SignalType] = None,
    ) -> DraftResponse:
        """Generate a one-click draft response grounded in the research report.

        Args:
            signal_context: Raw signal context (PII-scrubbed before use).
            report: Research report providing the content basis.
            channel: Target channel — ``"dm"``, ``"public_reply"``,
                ``"email"``, or ``"internal_note"``.
            tone: Desired prose tone from ``StrategicPriorities.tone``.
            signal_type: Optional LLMRouter routing hint.

        Returns:
            A populated :class:`DraftResponse`.
        """
        safe_context, _ = DataResidencyGuard._scrub_text(signal_context)
        synthesis_excerpt = (report.final_synthesis or "")[:500]
        channel_instruction = {
            "dm": "Write a short, friendly direct message (2–3 sentences) to a prospect.",
            "public_reply": "Write a public reply that is helpful, professional, and concise.",
            "email": "Write an email with a subject line and 2–3 paragraph body.",
            "internal_note": "Write a crisp internal Slack note for the GTM team.",
        }.get(channel, "Write a concise note.")
        prompt = (
            f"You are a {tone} GTM writer. {channel_instruction}\n\n"
            f"Signal context:\n{safe_context[:400]}\n\n"
            f"Research synthesis:\n{synthesis_excerpt}\n\n"
            "Reply with a JSON object ONLY:\n"
            "{\n"
            '  "subject": "<email subject or null for other channels>",\n'
            '  "body": "<the draft message>"\n'
            "}"
        )
        subject: Optional[str] = None
        body = synthesis_excerpt or "No synthesis available."
        try:
            raw = await self._router.generate_for_signal(
                signal_type=signal_type,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.6,
                max_tokens=700,
            )
            content = raw.strip()
            start, end = content.index("{"), content.rindex("}") + 1
            data: Dict[str, Any] = json.loads(content[start:end])
            body = str(data.get("body", body))
            subject = data.get("subject") or None
        except Exception as exc:
            logger.warning("SignalInteractionAgent.generate_draft_response failed: %s", exc)

        return DraftResponse(
            signal_id=report.signal_id,
            channel=channel,
            tone=tone,
            body=body,
            suggested_subject=subject,
            generated_at=datetime.now(timezone.utc),
            source_report_steps=len(report.steps),
        )


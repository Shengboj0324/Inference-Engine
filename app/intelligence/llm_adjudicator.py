"""LLM adjudication engine for signal classification.

This module implements Stage C of the inference pipeline:
- Structured LLM outputs with JSON schema validation
- Few-shot prompting with canonical exemplars
- Evidence extraction and rationale generation
- Abstention support for uncertain cases
- Retry logic with structured repair prompts
- Optional Chain-of-Thought, multi-agent, and deliberation reasoning paths
- Per-SignalType confidence calibration via ``ConfidenceCalibrator``
- Per-user context-memory RAG retrieval via ``ContextMemoryStore``

All optional components default to ``None`` so existing call-sites that do not
inject them continue to work without any code changes.

Follows the strict contract defined in app/domain/inference_models.py
"""

import asyncio
import logging
import json
import math
import re
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from pydantic import BaseModel, ValidationError

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import (
    AbstentionReason,
    EvidenceSpan,
    SignalInference,
    SignalPrediction,
    SignalType,
)
from app.intelligence.candidate_retrieval import SignalCandidate
from app.llm.models import LLMMessage
from app.llm.router import get_router

logger = logging.getLogger(__name__)


class LLMAdjudicationOutput(BaseModel):
    """Structured output schema for LLM adjudication."""
    
    candidate_signal_types: List[str]
    primary_signal_type: str
    confidence: float
    evidence_spans: List[Dict[str, str]]
    rationale: str
    requires_more_context: bool
    abstain: bool
    abstention_reason: Optional[str] = None
    risk_labels: List[str]
    suggested_actions: List[str]


class LLMAdjudicator:
    """LLM-based signal classification with structured outputs.

    Stage C of the inference pipeline.  Supports three reasoning paths:
    1. **single_call** — default; one prompt, one LLM call (existing behaviour).
    2. **chain_of_thought** — four-turn CoT conversation via ``ChainOfThoughtReasoner``.
    3. **multi_agent** — parallel per-candidate sub-tasks via ``MultiAgentOrchestrator``.

    The active path is selected by ``DeliberationEngine`` (Enhancement 6) when
    injected.  When none of the optional components are provided the adjudicator
    behaves identically to the original implementation.

    Args:
        model_name: Model name passed to ``SignalInference.model_name``.
        temperature: Sampling temperature for LLM calls.
        max_retries: Maximum JSON-repair retries.
        confidence_calibrator: Optional ``ConfidenceCalibrator``; calibrates
            ``SignalPrediction.probability`` in ``_convert_to_inference``.
        cot_reasoner: Optional ``ChainOfThoughtReasoner``; used when the
            deliberation mode is ``"chain_of_thought"`` or CoT conditions met.
        orchestrator: Optional ``MultiAgentOrchestrator``; used when the
            deliberation mode is ``"multi_agent"`` or complexity thresholds hit.
        context_memory: Optional ``ContextMemoryStore``; prepends retrieved
            past observations as few-shot context in ``_build_prompt`` and
            stores each successful inference after adjudication.
        deliberation_engine: Optional ``DeliberationEngine``; runs before the
            main LLM call to prune candidates and select the reasoning path.
    """

    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.3,
        max_retries: int = 3,
        confidence_calibrator: Optional[object] = None,
        cot_reasoner: Optional[object] = None,
        orchestrator: Optional[object] = None,
        context_memory: Optional[object] = None,
        deliberation_engine: Optional[object] = None,
    ) -> None:
        """Initialise the adjudicator.

        Args:
            model_name: LLM model identifier (stored in ``SignalInference``).
            temperature: Sampling temperature.
            max_retries: JSON-repair retry budget.
            confidence_calibrator: Optional ``ConfidenceCalibrator`` instance.
            cot_reasoner: Optional ``ChainOfThoughtReasoner`` instance.
            orchestrator: Optional ``MultiAgentOrchestrator`` instance.
            context_memory: Optional ``ContextMemoryStore`` instance.
            deliberation_engine: Optional ``DeliberationEngine`` instance.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self._confidence_calibrator = confidence_calibrator
        self._cot_reasoner = cot_reasoner
        self._orchestrator = orchestrator
        self._context_memory = context_memory
        self._deliberation_engine = deliberation_engine

        # Initialise LLM router.
        self.llm_router = get_router()

        logger.info(
            "LLMAdjudicator initialized: model=%s temp=%s calibrator=%s cot=%s "
            "orchestrator=%s memory=%s deliberation=%s",
            model_name,
            temperature,
            "yes" if confidence_calibrator else "no",
            "yes" if cot_reasoner else "no",
            "yes" if orchestrator else "no",
            "yes" if context_memory else "no",
            "yes" if deliberation_engine else "no",
        )
    
    async def adjudicate(
        self,
        observation: NormalizedObservation,
        candidates: List[SignalCandidate],
    ) -> SignalInference:
        """Adjudicate signal type for an observation using the active reasoning path.

        Execution flow:
        1. ``DeliberationEngine`` (E6) prunes candidates and selects the mode.
        2. ``ContextMemoryStore`` (E5) retrieves past observations for the user.
        3. The selected reasoning path (single_call / CoT / multi_agent) runs.
        4. ``ConfidenceCalibrator`` (E2) adjusts per-prediction probabilities.
        5. ``ContextMemoryStore`` (E5) stores the result for future retrievals.

        Every optional stage is ``None``-guarded with a ``try/except`` fallback
        so any injection failure degrades gracefully to the single-call path.

        Args:
            observation: Normalized observation to classify.
            candidates: Candidate signal types from the retrieval stage.

        Returns:
            Calibrated ``SignalInference`` produced by the active reasoning path.
        """
        # ── Step 1: Deliberation (E6) ─────────────────────────────────────
        reasoning_mode: str = "single_call"
        if self._deliberation_engine is not None:
            try:
                report = await self._deliberation_engine.deliberate(observation, candidates)
                candidates = report.pruned_candidates
                reasoning_mode = report.reasoning_mode
            except Exception as exc:
                logger.warning("Deliberation failed; using full candidate list: %s", exc)

        # ── Step 2: Context-memory retrieval (E5) ─────────────────────────
        memory_context: str = ""
        if self._context_memory is not None:
            try:
                memories = await self._context_memory.retrieve(
                    user_id=observation.user_id,
                    query_text=observation.normalized_text or "",
                    top_k=3,
                )
                memory_context = self._format_memory_context(memories)
            except Exception as exc:
                logger.warning("Context-memory retrieval failed: %s", exc)

        top_candidate_type: Optional[SignalType] = (
            candidates[0].signal_type if candidates else None
        )

        # ── Step 3: CoT trigger check (E1) ───────────────────────────────
        cot_triggered: bool = (
            (observation.confidence_required > 0.85)
            or (
                len(candidates) >= 2
                and abs(candidates[0].score - candidates[1].score) < 0.1
            )
        )

        # ── Step 4: Routing to the selected reasoning path ────────────────
        adjudication: Optional[LLMAdjudicationOutput] = None

        if (reasoning_mode == "chain_of_thought" or cot_triggered) and self._cot_reasoner is not None:
            try:
                adjudication = await self._cot_reasoner.reason(
                    observation, candidates, top_candidate_type
                )
            except Exception as exc:
                logger.warning("CoT reasoning failed; falling back to single_call: %s", exc)

        if adjudication is None and reasoning_mode == "multi_agent" and self._orchestrator is not None:
            try:
                adjudication = await self._orchestrator.orchestrate(
                    observation, candidates, top_candidate_type
                )
            except Exception as exc:
                logger.warning("Multi-agent orchestration failed; falling back to single_call: %s", exc)

        if adjudication is None:
            # Default: single-call path (original behaviour)
            prompt = self._build_prompt(observation, candidates, memory_context=memory_context)
            llm_output = await self._call_llm_with_retries(prompt, signal_type=top_candidate_type)
            try:
                adjudication = LLMAdjudicationOutput(**llm_output)
            except ValidationError as exc:
                logger.error("Failed to validate LLM output: %s", exc)
                return self._create_abstention_inference(
                    observation,
                    AbstentionReason.MALFORMED_OUTPUT,
                    "LLM output failed validation",
                )

        # ── Step 5: Convert + calibrate (E2) ─────────────────────────────
        inference = self._convert_to_inference(observation, adjudication)

        # ── Step 6: Store in context memory (E5) ─────────────────────────
        if self._context_memory is not None and not inference.abstained:
            try:
                await self._context_memory.store(observation.user_id, observation, inference)
            except Exception as exc:
                logger.warning("Context-memory storage failed: %s", exc)

        return inference

    @staticmethod
    def _format_memory_context(memories: List[Any]) -> str:
        """Format retrieved memory records into a few-shot context block.

        Args:
            memories: ``MemoryRecord`` instances returned by ``ContextMemoryStore.retrieve()``.

        Returns:
            Formatted multi-line string for injection into ``_build_prompt``.
        """
        if not memories:
            return ""
        lines = ["### RELEVANT PAST OBSERVATIONS"]
        for i, mem in enumerate(memories, 1):
            lines.append(
                f"{i}. [{mem.signal_type.value} | conf={mem.confidence:.2f}] "
                f"{mem.normalized_text[:120]}..."
            )
        return "\n".join(lines)
    
    def _build_prompt(
        self,
        observation: NormalizedObservation,
        candidates: List[SignalCandidate],
        memory_context: str = "",
    ) -> str:
        """Build few-shot prompt for LLM adjudication.

        When ``memory_context`` is non-empty (populated by ``ContextMemoryStore``
        retrieval), a ``### RELEVANT PAST OBSERVATIONS`` section is prepended to
        the user message so the LLM has in-context few-shot history.

        Args:
            observation: Normalized observation to classify.
            candidates: Candidate signal types from the retrieval stage.
            memory_context: Pre-formatted past-observation context from
                ``ContextMemoryStore.retrieve()``; injected before the current
                content when non-empty.

        Returns:
            Fully assembled prompt string.
        """
        # Build signal-type glossary from dispatch table for injection into system prompt.
        signal_glossary = "\n".join(
            f"  {st.value}: {desc}"
            for st, desc in self._SIGNAL_DESCRIPTIONS.items()
        )

        # System message — CRITICAL: JSON-only output instruction is explicit.
        system = f"""You are an expert signal classifier for business intelligence.
Your task is to analyse social media content and classify it into exactly one primary actionable signal type.

CRITICAL RULES:
1. OUTPUT ONLY VALID JSON — no markdown fences, no prose before or after.
2. Your entire response MUST be a single JSON object matching the schema below.
3. If you are uncertain (confidence < 0.6), you MUST set "abstain": true and supply "abstention_reason".
4. Never invent signal_type values outside the approved list below.

REQUIRED JSON SCHEMA (every key is mandatory):
{{
  "candidate_signal_types": ["<type>", ...],
  "primary_signal_type": "<type>",
  "confidence": <float 0-1>,
  "evidence_spans": [{{"text": "<verbatim excerpt>", "reason": "<why it matters>"}}],
  "rationale": "<one paragraph explanation>",
  "requires_more_context": <bool>,
  "abstain": <bool>,
  "abstention_reason": "<reason_enum or null>",
  "risk_labels": ["<label>"],
  "suggested_actions": ["<action>"]
}}

APPROVED SIGNAL TYPES (use exact string values):
{signal_glossary}

APPROVED ABSTENTION REASONS:
  low_confidence, ambiguous_multi_label, insufficient_context,
  out_of_distribution, unsafe_to_classify, language_barrier, spam_or_noise

ABSTENTION RULE: Set abstain=true and top type to "unclear" or "not_actionable" when:
- Content is spam, gibberish, or emoji-only
- Your confidence is below 0.6
- Multiple signal types are equally plausible and you cannot distinguish them
- The content is in a language you cannot reliably translate"""

        # Few-shot examples
        examples = self._get_few_shot_examples()

        # Memory context prefix (E5 — prepended when available)
        memory_prefix = f"{memory_context}\n\n" if memory_context else ""

        # User message
        user_message = (
            f"{memory_prefix}"
            f"Analyze this content:\n\n"
            f"Title: {observation.title or 'N/A'}\n"
            f"Text: {observation.normalized_text[:1000] if observation.normalized_text else 'N/A'}\n"
            f"Platform: {observation.source_platform.value}\n"
            f"Language: {observation.original_language or 'unknown'}\n\n"
            f"Candidate signals (from retrieval):\n"
            f"{self._format_candidates(candidates)}\n\n"
            f"Classify this content and output valid JSON."
        )

        # Combine
        full_prompt = f"{system}\n\n{examples}\n\n{user_message}"
        return full_prompt

    # Dispatch table: each SignalType maps to a one-line description used to
    # build examples and system instructions.  O(1) lookup; add new types here.
    _SIGNAL_DESCRIPTIONS: dict = {
        SignalType.SUPPORT_REQUEST: "User needs help with the product (how-to, setup, usage).",
        SignalType.FEATURE_REQUEST: "User wants a new feature or capability added.",
        SignalType.BUG_REPORT: "User reports broken or incorrect behaviour.",
        SignalType.COMPLAINT: "User expresses dissatisfaction without a specific request.",
        SignalType.PRAISE: "User praises or thanks the product/team.",
        SignalType.COMPETITOR_MENTION: "User explicitly names or compares a competitor product.",
        SignalType.ALTERNATIVE_SEEKING: "User is actively looking for an alternative solution.",
        SignalType.PRICE_SENSITIVITY: "User objects to pricing or asks for discounts.",
        SignalType.INTEGRATION_REQUEST: "User wants the product to integrate with another tool.",
        SignalType.CHURN_RISK: "User signals intent to cancel or switch providers.",
        SignalType.SECURITY_CONCERN: "User raises a security, privacy, or compliance issue.",
        SignalType.LEGAL_RISK: "User mentions legal threats, defamation, or regulation risk.",
        SignalType.REPUTATION_RISK: "Content that could damage brand reputation if not addressed.",
        SignalType.EXPANSION_OPPORTUNITY: "Organisation shows signs of growing and needing more.",
        SignalType.UPSELL_OPPORTUNITY: "Current customer could benefit from a higher-tier plan.",
        SignalType.PARTNERSHIP_OPPORTUNITY: "Organisation wants to partner, integrate, or co-market.",
        SignalType.UNCLEAR: "Intent cannot be determined from the content alone.",
        SignalType.NOT_ACTIONABLE: "Content is noise, spam, or requires no business response.",
    }

    def _get_few_shot_examples(self) -> str:
        """Get comprehensive few-shot examples covering all 18 SignalType values.

        Each example demonstrates the expected JSON output format and shows the
        LLM how to handle the full taxonomy including abstention cases.

        Returns:
            Formatted few-shot examples string for insertion into the prompt.
        """
        return """
### FEW-SHOT EXAMPLES (STUDY CAREFULLY)

--- EXAMPLE: alternative_seeking + price_sensitivity ---
Input: "Looking for a better alternative to Slack. Need something with better pricing."
Output:
{"candidate_signal_types":["alternative_seeking","competitor_mention","price_sensitivity"],"primary_signal_type":"alternative_seeking","confidence":0.92,"evidence_spans":[{"text":"better alternative to Slack","reason":"explicit alternative seeking"},{"text":"better pricing","reason":"price sensitivity"}],"rationale":"User explicitly seeks a Slack alternative citing price concerns.","requires_more_context":false,"abstain":false,"abstention_reason":null,"risk_labels":["public_reply_safe"],"suggested_actions":["reply_public","prepare_dm_followup"]}

--- EXAMPLE: praise + upsell_opportunity ---
Input: "Your product is amazing! Just upgraded to Pro plan."
Output:
{"candidate_signal_types":["praise","upsell_opportunity"],"primary_signal_type":"praise","confidence":0.95,"evidence_spans":[{"text":"product is amazing","reason":"positive sentiment"},{"text":"upgraded to Pro plan","reason":"conversion event"}],"rationale":"Clear praise with successful upsell already completed.","requires_more_context":false,"abstain":false,"abstention_reason":null,"risk_labels":["public_reply_safe"],"suggested_actions":["thank_user","request_testimonial"]}

--- EXAMPLE: churn_risk ---
Input: "Been thinking of switching after my renewal comes up. Support has gone downhill."
Output:
{"candidate_signal_types":["churn_risk","complaint"],"primary_signal_type":"churn_risk","confidence":0.85,"evidence_spans":[{"text":"thinking of switching after my renewal","reason":"explicit churn intent"},{"text":"support has gone downhill","reason":"dissatisfaction driver"}],"rationale":"User signals intent to switch on renewal with cited dissatisfaction.","requires_more_context":false,"abstain":false,"abstention_reason":null,"risk_labels":["dm_preferred"],"suggested_actions":["send_dm","offer_discount","escalate_support"]}

--- EXAMPLE: complaint (sarcasm) ---
Input: "Oh amazing, another outage. Truly world-class reliability 🙄"
Output:
{"candidate_signal_types":["complaint","reputation_risk"],"primary_signal_type":"complaint","confidence":0.88,"evidence_spans":[{"text":"another outage","reason":"product failure event"},{"text":"world-class reliability","reason":"sarcasm indicating dissatisfaction"}],"rationale":"Sarcastic praise is a complaint about reliability.","requires_more_context":false,"abstain":false,"abstention_reason":null,"risk_labels":["public_reply_needed"],"suggested_actions":["acknowledge_publicly","escalate_infra"]}

--- EXAMPLE: feature_request ---
Input: "It would be great if you added dark mode. I use this all day and my eyes are killing me."
Output:
{"candidate_signal_types":["feature_request"],"primary_signal_type":"feature_request","confidence":0.93,"evidence_spans":[{"text":"added dark mode","reason":"explicit feature request"}],"rationale":"User directly requests a specific UI feature.","requires_more_context":false,"abstain":false,"abstention_reason":null,"risk_labels":["public_reply_safe"],"suggested_actions":["acknowledge_request","log_to_backlog"]}

--- EXAMPLE: bug_report ---
Input: "The CSV export is broken — it always times out on files over 1MB."
Output:
{"candidate_signal_types":["bug_report","support_request"],"primary_signal_type":"bug_report","confidence":0.91,"evidence_spans":[{"text":"CSV export is broken","reason":"explicit bug statement"},{"text":"times out on files over 1MB","reason":"specific failure condition"}],"rationale":"User reports a reproducible bug with specific threshold.","requires_more_context":false,"abstain":false,"abstention_reason":null,"risk_labels":["internal_ticket_required"],"suggested_actions":["create_bug_ticket","reply_public"]}

--- EXAMPLE: security_concern ---
Input: "I noticed that user emails are visible in URL parameters. Is that a security issue?"
Output:
{"candidate_signal_types":["security_concern","support_request"],"primary_signal_type":"security_concern","confidence":0.89,"evidence_spans":[{"text":"user emails are visible in URL parameters","reason":"potential PII exposure"}],"rationale":"User identifies a possible PII leak via URL parameters.","requires_more_context":false,"abstain":false,"abstention_reason":null,"risk_labels":["internal_security_review","dm_preferred"],"suggested_actions":["send_dm","escalate_security_team"]}

--- EXAMPLE: integration_request ---
Input: "Any plans for a Zapier integration? We use it for all our automations."
Output:
{"candidate_signal_types":["integration_request","feature_request"],"primary_signal_type":"integration_request","confidence":0.87,"evidence_spans":[{"text":"Zapier integration","reason":"named integration target"}],"rationale":"User specifically requests integration with a named automation platform.","requires_more_context":false,"abstain":false,"abstention_reason":null,"risk_labels":["public_reply_safe"],"suggested_actions":["reply_public","log_to_backlog"]}

--- EXAMPLE: partnership_opportunity ---
Input: "We're building a complementary HR tool and would love to explore a joint go-to-market. Who do I contact?"
Output:
{"candidate_signal_types":["partnership_opportunity","expansion_opportunity"],"primary_signal_type":"partnership_opportunity","confidence":0.90,"evidence_spans":[{"text":"joint go-to-market","reason":"explicit partnership intent"},{"text":"complementary HR tool","reason":"product fit signal"}],"rationale":"Organisation seeking a commercial partnership and integration.","requires_more_context":false,"abstain":false,"abstention_reason":null,"risk_labels":["email_preferred"],"suggested_actions":["email_partnerships","schedule_call"]}

--- EXAMPLE: not_actionable (spam) ---
Input: "🔥 BEST DEALS CLICK NOW FREE GIFT 🔥 limited offer!!!"
Output:
{"candidate_signal_types":["not_actionable"],"primary_signal_type":"not_actionable","confidence":0.97,"evidence_spans":[],"rationale":"Content is promotional spam with no actionable business signal.","requires_more_context":false,"abstain":true,"abstention_reason":"spam_or_noise","risk_labels":[],"suggested_actions":[]}

--- EXAMPLE: abstention (low confidence / ambiguous) ---
Input: "meh"
Output:
{"candidate_signal_types":["unclear"],"primary_signal_type":"unclear","confidence":0.12,"evidence_spans":[],"rationale":"Single-word post with no discernible intent or context.","requires_more_context":true,"abstain":true,"abstention_reason":"insufficient_context","risk_labels":[],"suggested_actions":[]}
"""

    def _format_candidates(self, candidates: List[SignalCandidate]) -> str:
        """Format candidates for prompt.

        Args:
            candidates: Candidate signals

        Returns:
            Formatted string
        """
        if not candidates:
            return "No candidates from retrieval"

        lines = []
        for i, candidate in enumerate(candidates, 1):
            lines.append(
                f"{i}. {candidate.signal_type.value} (score={candidate.score:.2f}): {candidate.reasoning}"
            )
        return "\n".join(lines)

    async def _call_llm_with_retries(
        self,
        prompt: str,
        signal_type: Optional["SignalType"] = None,
    ) -> Dict[str, Any]:
        """Call LLM via tiered routing with retry logic for malformed outputs.

        Uses ``LLMRouter.generate_for_signal()`` to route high-stakes signal
        types to the frontier model and other types to the configured
        fine-tuned model (falling back to frontier when not configured).

        Each retry applies exponential backoff (0.5 s × 2^attempt) so that
        transient provider errors are not hammered immediately.  The message
        list is reset to the original single-user-turn on each attempt; the
        old "repair by appending" strategy is removed because appending a
        malformed assistant turn often confuses the model further.

        JSON is extracted with a brace-counting algorithm (``_extract_json``)
        that correctly handles nested objects, unlike the previous
        ``content[first_brace:last_brace]`` approach which could produce
        invalid JSON when the response contained multiple ``{`` occurrences.
        The extracted dict is validated against ``LLMAdjudicationOutput``
        before being returned so callers always receive a schema-conformant
        payload.

        Args:
            prompt: Prompt string built by ``_build_prompt``.
            signal_type: Top-candidate signal type from retrieval; used to
                select the appropriate model tier.  ``None`` defaults to the
                frontier model.

        Returns:
            Parsed JSON dict matching the ``LLMAdjudicationOutput`` schema.

        Raises:
            ValueError: If a valid, schema-conformant JSON response cannot be
                obtained after ``self.max_retries`` attempts.
        """
        messages = [LLMMessage(role="user", content=prompt)]

        for attempt in range(self.max_retries):
            # Exponential backoff before every retry (not before the first attempt)
            if attempt > 0:
                backoff_seconds = 0.5 * (2 ** (attempt - 1))  # 0.5 s, 1.0 s, 2.0 s …
                logger.debug(
                    "LLM adjudicator backoff: %.1f s before attempt %d/%d",
                    backoff_seconds, attempt + 1, self.max_retries,
                )
                await asyncio.sleep(backoff_seconds)

            try:
                # Route to appropriate model tier based on signal type
                content = await self.llm_router.generate_for_signal(
                    signal_type=signal_type,
                    messages=messages,
                    max_tokens=1000,
                    temperature=self.temperature,
                )
                content = content.strip()

                # Brace-counting JSON extraction — correctly handles nested objects
                parsed = self._extract_json(content)
                if parsed is None:
                    raise ValueError("No valid JSON object found in LLM response")

                # Schema validation: raises ValidationError if the LLM output
                # does not conform to LLMAdjudicationOutput before we return it.
                LLMAdjudicationOutput(**parsed)
                return parsed

            except (json.JSONDecodeError, ValueError, ValidationError) as exc:
                logger.warning(
                    "LLM adjudication attempt %d/%d failed: %s",
                    attempt + 1, self.max_retries, exc,
                )
                if attempt == self.max_retries - 1:
                    raise ValueError(
                        f"Failed to get valid JSON response after {self.max_retries} attempts: {exc}"
                    ) from exc

        raise ValueError("Unexpected exit from retry loop")  # pragma: no cover

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract the outermost JSON object from *text* using brace counting.

        This is strictly more correct than ``text[text.index('{'):text.rindex('}')+1]``
        because that approach produces invalid JSON whenever the response
        contains multiple top-level ``{`` tokens (e.g. markdown code fences
        with a JSON block inside).

        The algorithm tracks string-literal boundaries so that ``{`` / ``}``
        inside quoted values are not counted as object delimiters.

        Args:
            text: Raw LLM output, possibly containing prose before or after
                the JSON object.

        Returns:
            Parsed ``dict`` for the first complete JSON object found, or
            ``None`` if no valid object is present.
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False

        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\":
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _convert_to_inference(
        self,
        observation: NormalizedObservation,
        adjudication: LLMAdjudicationOutput,
    ) -> SignalInference:
        """Convert LLM adjudication to a calibrated ``SignalInference``.

        When ``self._confidence_calibrator`` is set, each ``SignalPrediction``
        probability is converted to a log-odds score and passed through
        ``ConfidenceCalibrator.calibrate()`` before the ``SignalInference`` is
        constructed.

        Args:
            observation: Normalized observation.
            adjudication: Structured LLM adjudication output.

        Returns:
            Calibrated ``SignalInference``.
        """
        # Parse predictions, applying per-type calibration (E2) when available.
        predictions: List[SignalPrediction] = []
        for signal_type_str in adjudication.candidate_signal_types:
            try:
                signal_type = SignalType(signal_type_str)
                raw_prob: float = (
                    adjudication.confidence
                    if signal_type_str == adjudication.primary_signal_type
                    else adjudication.confidence * 0.5
                )
                # E2: apply per-SignalType temperature-scaling calibration.
                if self._confidence_calibrator is not None:
                    epsilon = 1e-7
                    p = max(epsilon, min(1.0 - epsilon, raw_prob))
                    raw_logit: float = math.log(p / (1.0 - p))
                    calibrated_prob: float = self._confidence_calibrator.calibrate(
                        raw_logit, signal_type
                    )
                else:
                    calibrated_prob = raw_prob
                predictions.append(
                    SignalPrediction(
                        signal_type=signal_type,
                        probability=calibrated_prob,
                    )
                )
            except ValueError:
                logger.warning("Invalid signal type: %s", signal_type_str)

        # Get top prediction
        top_prediction = predictions[0] if predictions else None

        # Parse evidence spans
        evidence_spans = []
        for span_dict in adjudication.evidence_spans:
            span_text = span_dict.get("text", "")
            evidence_spans.append(
                EvidenceSpan(
                    text=span_text,
                    start_char=0,  # LLM doesn't provide exact character positions
                    end_char=len(span_text),
                    relevance_score=0.8,  # Default relevance; LLM-selected spans are considered high-relevance
                )
            )

        # Parse abstention reason
        abstention_reason = None
        if adjudication.abstain and adjudication.abstention_reason:
            try:
                abstention_reason = AbstentionReason(adjudication.abstention_reason)
            except ValueError:
                abstention_reason = AbstentionReason.LOW_CONFIDENCE

        # Create inference
        inference = SignalInference(
            normalized_observation_id=observation.id,
            user_id=observation.user_id,
            predictions=predictions,
            top_prediction=top_prediction,
            abstained=adjudication.abstain,
            abstention_reason=abstention_reason,
            evidence_spans=evidence_spans,
            rationale=adjudication.rationale,
            model_name=self.model_name,
            model_version=datetime.now(timezone.utc).strftime("%Y-%m"),
            inference_method="llm_few_shot",
        )

        return inference

    def _create_abstention_inference(
        self,
        observation: NormalizedObservation,
        reason: AbstentionReason,
        rationale: str,
    ) -> SignalInference:
        """Create an abstention inference.

        Args:
            observation: Normalized observation
            reason: Abstention reason
            rationale: Rationale for abstention

        Returns:
            Signal inference with abstention
        """
        return SignalInference(
            normalized_observation_id=observation.id,
            user_id=observation.user_id,
            predictions=[],
            top_prediction=None,
            abstained=True,
            abstention_reason=reason,
            rationale=rationale,
            model_name=self.model_name,
            model_version=datetime.now(timezone.utc).strftime("%Y-%m"),
            inference_method="llm_few_shot",
        )


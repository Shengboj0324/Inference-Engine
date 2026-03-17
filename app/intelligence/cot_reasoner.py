"""Structured Chain-of-Thought (CoT) reasoning layer for signal adjudication.

``ChainOfThoughtReasoner`` wraps ``LLMRouter.generate_for_signal()`` with an
explicit four-step multi-turn conversation:

1. **Decompose** — break the input into sub-problems (entities, platforms, intent).
2. **Retrieve** — identify the most relevant ``SignalType`` values and evidence spans.
3. **Synthesise** — combine sub-answers into a primary classification.
4. **Verify** — self-check the conclusion against the ``SignalType`` taxonomy and
   abstention rules, then emit the final ``LLMAdjudicationOutput`` JSON.

Each step is a separate ``LLMMessage`` turn so the model can accumulate context
across the conversation.  The final step always requests valid JSON conforming
to ``LLMAdjudicationOutput``.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from app.domain.inference_models import SignalType
from app.domain.normalized_models import NormalizedObservation
from app.intelligence.candidate_retrieval import SignalCandidate
from app.intelligence.llm_adjudicator import LLMAdjudicationOutput
from app.llm.models import LLMMessage
from app.llm.router import LLMRouter, get_router

logger = logging.getLogger(__name__)


class ChainOfThoughtReasoner:
    """Four-step multi-turn CoT reasoning scaffold for signal classification.

    Each reasoning step is issued as a separate conversation turn so the LLM
    can build up analytical context incrementally.  The final (Verify) step
    instructs the model to emit the canonical ``LLMAdjudicationOutput`` JSON.

    Args:
        router: ``LLMRouter`` instance.  When ``None`` the global singleton
            returned by ``get_router()`` is used.
        temperature: Sampling temperature for all CoT turns.
        max_tokens: Token budget per turn.
    """

    def __init__(
        self,
        router: Optional[LLMRouter] = None,
        temperature: float = 0.3,
        max_tokens: int = 600,
    ) -> None:
        """Initialise the reasoner.

        Args:
            router: LLM router; defaults to the global singleton.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens per turn.
        """
        self._router: LLMRouter = router or get_router()
        self._temperature: float = temperature
        self._max_tokens: int = max_tokens

    async def reason(
        self,
        observation: NormalizedObservation,
        candidates: List[SignalCandidate],
        signal_type: Optional[SignalType] = None,
    ) -> LLMAdjudicationOutput:
        """Execute the four-step CoT conversation and return a validated output.

        Args:
            observation: Normalised observation to classify.
            candidates: Candidate signal types from the retrieval stage.
            signal_type: Top-candidate type used for LLMRouter tier selection.
                ``None`` routes to the frontier model.

        Returns:
            Validated ``LLMAdjudicationOutput`` produced by the Verify step.

        Raises:
            ValueError: If the Verify step does not produce valid JSON or the
                JSON fails ``LLMAdjudicationOutput`` validation.
        """
        text_snippet: str = (observation.normalized_text or "")[:1200]
        candidate_names: str = ", ".join(c.signal_type.value for c in candidates)
        messages: List[LLMMessage] = []

        # ── Step 1: Decompose ────────────────────────────────────────────────
        messages.append(LLMMessage(
            role="user",
            content=(
                "You are a rigorous business-signal analyst.\n\n"
                f"CONTENT:\nTitle: {observation.title or 'N/A'}\n"
                f"Text: {text_snippet}\n"
                f"Platform: {observation.source_platform.value}\n\n"
                "STEP 1 — DECOMPOSE: Break this content into its core sub-problems. "
                "Identify: (a) the author's primary intent, (b) any named entities "
                "(competitors, products, services), (c) emotional tone and urgency. "
                "Be concise — 3–5 bullet points."
            ),
        ))
        decompose_result = await self._router.generate_for_signal(
            signal_type=signal_type,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        messages.append(LLMMessage(role="assistant", content=decompose_result))

        # ── Step 2: Retrieve ─────────────────────────────────────────────────
        messages.append(LLMMessage(
            role="user",
            content=(
                f"STEP 2 — RETRIEVE: From the candidate signal types "
                f"[{candidate_names}], identify which are most relevant "
                "to the decomposed sub-problems above.  For each relevant type, "
                "quote a verbatim text span (≤ 15 words) that serves as evidence. "
                "List at most 4 (type, evidence-span) pairs."
            ),
        ))
        retrieve_result = await self._router.generate_for_signal(
            signal_type=signal_type,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        messages.append(LLMMessage(role="assistant", content=retrieve_result))

        # ── Step 3: Synthesise ───────────────────────────────────────────────
        messages.append(LLMMessage(
            role="user",
            content=(
                "STEP 3 — SYNTHESISE: Combine your findings into a single primary "
                "classification.  State: (a) the primary signal type, (b) your "
                "confidence (0–1), and (c) whether you should abstain (confidence < 0.6 "
                "or content is ambiguous/spam).  One paragraph."
            ),
        ))
        synth_result = await self._router.generate_for_signal(
            signal_type=signal_type,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        messages.append(LLMMessage(role="assistant", content=synth_result))

        # ── Step 4: Verify + JSON output ─────────────────────────────────────
        messages.append(LLMMessage(
            role="user",
            content=(
                "STEP 4 — VERIFY: Review your synthesis against the approved signal "
                "taxonomy and abstention rules.  Correct any errors, then output "
                "ONLY a single valid JSON object — no prose, no markdown — matching "
                "this schema exactly:\n"
                '{"candidate_signal_types":["<type>",...],'
                '"primary_signal_type":"<type>",'
                '"confidence":<float 0-1>,'
                '"evidence_spans":[{"text":"<excerpt>","reason":"<why>"}],'
                '"rationale":"<one paragraph>",'
                '"requires_more_context":<bool>,'
                '"abstain":<bool>,'
                '"abstention_reason":"<enum or null>",'
                '"risk_labels":["<label>"],'
                '"suggested_actions":["<action>"]}'
            ),
        ))
        verify_result = await self._router.generate_for_signal(
            signal_type=signal_type,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return self._parse_verify_result(verify_result)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_verify_result(raw: str) -> LLMAdjudicationOutput:
        """Extract and validate JSON from the Verify step response.

        Args:
            raw: Raw string returned by the LLM for the Verify step.

        Returns:
            Validated ``LLMAdjudicationOutput``.

        Raises:
            ValueError: If no JSON object is found or schema validation fails.
        """
        content = raw.strip()
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            data: Dict[str, Any] = json.loads(content[start:end])
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"CoT Verify step did not produce valid JSON: {exc}"
            ) from exc
        return LLMAdjudicationOutput(**data)


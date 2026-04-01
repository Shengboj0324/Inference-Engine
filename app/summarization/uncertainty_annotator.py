"""Uncertainty annotator.

Scans text for language patterns that signal epistemic uncertainty —
hedging, speculation, anonymous sourcing, and forward-looking statements —
and returns a list of ``UncertaintyAnnotation`` objects with per-span
severity labels.

Annotation strategy (heuristic fast path)
------------------------------------------
Four ordered signal tiers are checked against sentence tokens:

Tier / Severity  Signal examples
───────────────  ────────────────────────────────────────────────────────
CRITICAL         "unverified", "anonymous source", "unconfirmed reports",
                 "we cannot confirm", "rumoured to be"
HIGH             "allegedly", "reportedly", "sources say",
                 "according to sources", "claimed by"
MEDIUM           "might", "may", "could", "possibly", "potentially",
                 "expected to", "planned to", "believed to"
LOW              "will", "would", "should", "likely", "probably",
                 "appears to", "seems to", "suggests"

Sentence-level scanning
------------------------
The input text is split into sentences on ``[.!?]``.  Each sentence is
scanned once; the first (highest-severity) pattern match wins for that
sentence, preventing duplicate annotations for the same span.

Overall uncertainty score
--------------------------
``overall_uncertainty(annotations)`` returns a weighted mean of per-annotation
severity scores mapped as:
    CRITICAL → 1.00, HIGH → 0.75, MEDIUM → 0.50, LOW → 0.25
divided by the fraction of input sentences that carry annotations (clamped
to [0, 1]).

Optional LLM path
-----------------
When ``llm_router`` is provided and heuristic annotation count is zero but
the text length exceeds ``llm_min_chars``, the router is asked to identify
uncertain spans.  Response format expected:
``[{"text_span": str, "severity": str, "reason": str}]``.

Thread safety
-------------
No mutable state is held; all methods are re-entrant.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.summarization.models import UncertaintyAnnotation, UncertaintySeverity

logger = logging.getLogger(__name__)

_DEFAULT_LLM_MIN_CHARS: int = 200

# ---------------------------------------------------------------------------
# Lexicon: (pattern, severity, reason_prefix)
# ---------------------------------------------------------------------------
_LEXICON: List[Tuple[re.Pattern, UncertaintySeverity, str]] = [
    # CRITICAL — unverified / anonymous
    (re.compile(r"\b(unverified|unconfirmed|anonymous source[sd]?|"
                r"we cannot confirm|cannot be confirmed|rumou?red to be|"
                r"not yet official)\b", re.IGNORECASE),
     UncertaintySeverity.CRITICAL, "unverified_source"),

    # HIGH — reportorial hedges
    (re.compile(r"\b(allegedly|reportedly|sources say|according to sources|"
                r"claimed by|said to be|believed by sources|"
                r"insiders say|industry sources)\b", re.IGNORECASE),
     UncertaintySeverity.HIGH, "reportorial_hedge"),

    # MEDIUM — epistemic modals + forward-looking
    (re.compile(r"\b(might|may|could|possibly|potentially|"
                r"expected to|planning to|plans? to|"
                r"believed to|thought to|rumou?red)\b", re.IGNORECASE),
     UncertaintySeverity.MEDIUM, "epistemic_modal"),

    # LOW — weak future / probability markers
    (re.compile(r"\b(will|would|should|likely|probably|"
                r"appears? to|seems? to|suggest[s]?|indicate[s]?)\b", re.IGNORECASE),
     UncertaintySeverity.LOW, "probability_marker"),
]

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_SEVERITY_SCORE: Dict[UncertaintySeverity, float] = {
    UncertaintySeverity.CRITICAL: 1.00,
    UncertaintySeverity.HIGH:     0.75,
    UncertaintySeverity.MEDIUM:   0.50,
    UncertaintySeverity.LOW:      0.25,
}


class UncertaintyAnnotator:
    """Detects hedging and speculative language in text.

    Args:
        llm_router:    Optional LLM router for zero-heuristic-hit fallback.
        llm_min_chars: Minimum text length before LLM fallback is attempted.
        min_severity:  Minimum severity level to include in output.
                       Annotations below this level are filtered out.
    """

    def __init__(
        self,
        llm_router: Optional[Any] = None,
        llm_min_chars: int = _DEFAULT_LLM_MIN_CHARS,
        min_severity: UncertaintySeverity = UncertaintySeverity.LOW,
    ) -> None:
        if llm_min_chars <= 0:
            raise ValueError(f"'llm_min_chars' must be positive, got {llm_min_chars!r}")
        if not isinstance(min_severity, UncertaintySeverity):
            raise TypeError(f"'min_severity' must be UncertaintySeverity, got {type(min_severity)!r}")

        self._router = llm_router
        self._llm_min_chars = llm_min_chars
        self._min_severity = min_severity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def annotate(self, text: str) -> List[UncertaintyAnnotation]:
        """Scan *text* and return uncertainty annotations.

        Args:
            text: Input text (may contain multiple sentences).

        Returns:
            List of ``UncertaintyAnnotation`` ordered by character position.

        Raises:
            TypeError:  *text* is not a string.
            ValueError: *text* is empty.
        """
        if not isinstance(text, str):
            raise TypeError(f"'text' must be str, got {type(text)!r}")
        if not text.strip():
            raise ValueError("'text' must be non-empty")

        annotations = self._heuristic_annotate(text)
        min_score = _SEVERITY_SCORE[self._min_severity]
        annotations = [a for a in annotations if _SEVERITY_SCORE[a.severity] >= min_score]

        if not annotations and self._router is not None and len(text) >= self._llm_min_chars:
            try:
                annotations = self._llm_annotate(text)
                annotations = [a for a in annotations if _SEVERITY_SCORE[a.severity] >= min_score]
                logger.debug("UncertaintyAnnotator: LLM path returned %d annotations", len(annotations))
            except Exception as exc:
                logger.warning("UncertaintyAnnotator: LLM path failed (%s)", exc)

        logger.debug("UncertaintyAnnotator: %d annotations for text len=%d", len(annotations), len(text))
        return sorted(annotations, key=lambda a: a.position)

    def overall_uncertainty(self, annotations: List[UncertaintyAnnotation]) -> float:
        """Compute aggregate uncertainty score from *annotations*.

        Returns:
            Float in [0, 1].  0.0 if *annotations* is empty.

        Raises:
            TypeError: *annotations* is not a list.
        """
        if not isinstance(annotations, list):
            raise TypeError(f"'annotations' must be a list, got {type(annotations)!r}")
        if not annotations:
            return 0.0
        scores = [_SEVERITY_SCORE[a.severity] for a in annotations]
        return round(min(1.0, sum(scores) / len(scores)), 5)

    def classify_severity(self, text_span: str) -> UncertaintySeverity:
        """Return the highest severity triggered by *text_span*.

        Args:
            text_span: A short text excerpt to classify.

        Raises:
            TypeError:  *text_span* is not a string.
            ValueError: *text_span* is empty.
        """
        if not isinstance(text_span, str):
            raise TypeError(f"'text_span' must be str, got {type(text_span)!r}")
        if not text_span.strip():
            raise ValueError("'text_span' must be non-empty")
        for pattern, severity, _ in _LEXICON:
            if pattern.search(text_span):
                return severity
        return UncertaintySeverity.LOW

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heuristic_annotate(self, text: str) -> List[UncertaintyAnnotation]:
        """Sentence-level heuristic scanning."""
        sentences = _SENTENCE_SPLIT_RE.split(text)
        annotations: List[UncertaintyAnnotation] = []
        pos = 0
        for sentence in sentences:
            for pattern, severity, reason_prefix in _LEXICON:
                m = pattern.search(sentence)
                if m:
                    span = sentence.strip()
                    reason = f"{reason_prefix}:{m.group(0).lower()}"
                    annotations.append(
                        UncertaintyAnnotation(
                            text_span=span[:200],
                            severity=severity,
                            reason=reason,
                            position=pos + (sentence.index(m.group(0)) if m.group(0) in sentence else 0),
                        )
                    )
                    break  # highest-severity match wins for this sentence
            pos += len(sentence) + 1
        return annotations

    def _llm_annotate(self, text: str) -> List[UncertaintyAnnotation]:
        """LLM-based annotation fallback."""
        import json
        prompt = (
            "Identify uncertain or speculative spans in the text below.  "
            "Reply ONLY with a JSON array:\n"
            "[{\"text_span\": str, \"severity\": \"low|medium|high|critical\", \"reason\": str}]\n\n"
            f"Text: {text[:1000]}"
        )
        try:
            from app.llm.models import LLMMessage
            import asyncio, inspect
            resp = self._router.generate_for_signal(
                signal_type=None,
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=400,
            )
            if inspect.isawaitable(resp):
                resp = asyncio.get_event_loop().run_until_complete(resp)
            items = json.loads(resp)
            result = []
            for item in items:
                try:
                    sev = UncertaintySeverity(item.get("severity", "medium"))
                except ValueError:
                    sev = UncertaintySeverity.MEDIUM
                result.append(UncertaintyAnnotation(
                    text_span=str(item.get("text_span", ""))[:200],
                    severity=sev,
                    reason=str(item.get("reason", "")),
                ))
            return result
        except Exception as exc:
            logger.warning("UncertaintyAnnotator._llm_annotate failed: %s", exc)
            return []


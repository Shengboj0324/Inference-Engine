"""PublicationGate — seven-step output review chain before user delivery.

Every ``GroundedSummary`` produced by ``GroundedSummaryBuilder`` must pass
this gate before it reaches any user-facing endpoint or digest stream.  The
gate is an explicit linearisation of the recommendation in §2/Priority 2:
refuse low-novelty, low-grounding, or low-confidence outputs automatically.

Steps
-----
1. **Draft completeness** — ``what_happened`` and ``why_it_matters`` are both
   non-empty and meet minimum character thresholds.
2. **Citation verification** — every ``EvidenceSource.source_id`` in
   ``source_attributions`` resolves to a known ``ChunkStore`` record or a
   valid multimodal citation (``mm-img-*`` / ``mm-vid-*``).
3. **Contradiction audit** — flag summaries with unresolved contradictions
   that exceed ``max_contradictions`` without an explicit uncertainty annotation.
4. **Uncertainty annotation** — summaries with ``overall_uncertainty_score``
   above ``max_uncertainty`` are blocked unless ``allow_uncertain`` is True.
5. **Personalization relevance** — optionally checks that at least one
   watchlist entity appears in the summary text or attributions.
6. **Quality gate** — ``confidence_score`` must meet ``min_confidence``.
7. **Policy gate** — summary must not contain tokens from the operator-supplied
   blocklist (prompt-injection / policy compliance backstop).

Usage::

    gate = PublicationGate(min_confidence=0.65, min_chars=80,
                           max_contradictions=2, max_uncertainty=0.75)
    result = gate.evaluate(summary, chunk_store=store, topic="AI safety")
    # result.approved  → bool
    # result.steps     → List[StepResult]  (pass/fail per step with detail)
    # result.blocking_step → Optional[str] (name of first failing step)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

logger = logging.getLogger(__name__)

_MULTIMODAL_PREFIX_RE = re.compile(r"^mm-(img|vid)-", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    step: str
    passed: bool
    detail: str = ""


@dataclass
class PublicationResult:
    """Outcome of a full 7-step publication review."""
    summary_id: str
    approved: bool
    steps: List[StepResult] = field(default_factory=list)
    blocking_step: Optional[str] = None

    @property
    def rejection_reasons(self) -> List[str]:
        return [s.detail for s in self.steps if not s.passed]


# ---------------------------------------------------------------------------
# PublicationGate
# ---------------------------------------------------------------------------

class PublicationGate:
    """Seven-step output review gate.

    Args:
        min_confidence:    Minimum ``GroundedSummary.confidence_score`` (step 6).
        min_chars:         Minimum character length for each narrative field (step 1).
        max_contradictions: Maximum unresolved contradictions before blocking (step 3).
        max_uncertainty:   Maximum ``overall_uncertainty_score`` before blocking (step 4).
        allow_uncertain:   If True, step 4 downgrades to a warning instead of blocking.
        policy_blocklist:  Set of lowercase token patterns that must not appear (step 7).
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        min_chars: int = 80,
        max_contradictions: int = 3,
        max_uncertainty: float = 0.80,
        allow_uncertain: bool = False,
        policy_blocklist: Optional[Set[str]] = None,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        self.min_confidence = min_confidence
        self.min_chars = min_chars
        self.max_contradictions = max_contradictions
        self.max_uncertainty = max_uncertainty
        self.allow_uncertain = allow_uncertain
        self.policy_blocklist: Set[str] = policy_blocklist or set()

    def evaluate(
        self,
        summary: Any,
        chunk_store: Optional[Any] = None,
        topic: str = "",
        watchlist_entities: Optional[Set[str]] = None,
    ) -> PublicationResult:
        """Run all seven steps against ``summary``.

        Args:
            summary:           A ``GroundedSummary`` (duck-typed; accessed by attribute).
            chunk_store:       ``ChunkStore`` used for citation resolution (step 2).
            topic:             Research topic string (used in step 5 if watchlist absent).
            watchlist_entities: Set of entity names that should appear (step 5 optional).

        Returns:
            ``PublicationResult`` with ``approved=True`` iff all steps pass.
        """
        summary_id = str(getattr(summary, "id", id(summary)))
        steps: List[StepResult] = []

        steps.append(self._step1_draft_completeness(summary))
        steps.append(self._step2_citation_verification(summary, chunk_store))
        steps.append(self._step3_contradiction_audit(summary))
        steps.append(self._step4_uncertainty_annotation(summary))
        steps.append(self._step5_personalization(summary, watchlist_entities, topic))
        steps.append(self._step6_quality_gate(summary))
        steps.append(self._step7_policy_gate(summary))

        blocking = next((s.step for s in steps if not s.passed), None)
        approved = blocking is None

        if not approved:
            logger.warning(
                "PublicationGate: summary %s REJECTED at step '%s': %s",
                summary_id, blocking,
                next(s.detail for s in steps if s.step == blocking),
            )
        else:
            logger.debug("PublicationGate: summary %s approved (%d steps)", summary_id, len(steps))

        return PublicationResult(
            summary_id=summary_id,
            approved=approved,
            steps=steps,
            blocking_step=blocking,
        )

    # ── Step implementations ──────────────────────────────────────────────

    def _step1_draft_completeness(self, summary: Any) -> StepResult:
        what = str(getattr(summary, "what_happened", "") or "").strip()
        why = str(getattr(summary, "why_it_matters", "") or "").strip()
        if len(what) < self.min_chars:
            return StepResult(
                "draft_completeness", False,
                f"what_happened is {len(what)} chars (min {self.min_chars})",
            )
        if len(why) < max(10, self.min_chars // 4):
            return StepResult(
                "draft_completeness", False,
                f"why_it_matters is {len(why)} chars (min {max(10, self.min_chars // 4)})",
            )
        return StepResult("draft_completeness", True)

    def _step2_citation_verification(self, summary: Any, chunk_store: Optional[Any]) -> StepResult:
        attributions = getattr(summary, "source_attributions", []) or []
        if not attributions:
            return StepResult("citation_verification", False, "no source_attributions")
        if chunk_store is None:
            return StepResult("citation_verification", True, "chunk_store not provided — skipped")
        known_ids: Set[str] = set()
        try:
            known_ids = set(chunk_store.observation_ids())
        except Exception as exc:
            logger.debug("PublicationGate step2: chunk_store.observation_ids() error: %s", exc)
            return StepResult("citation_verification", True, f"store unavailable ({exc}) — skipped")
        unresolved = []
        for src in attributions:
            sid = str(getattr(src, "source_id", "") or "")
            if _MULTIMODAL_PREFIX_RE.match(sid):
                continue
            if sid not in known_ids:
                unresolved.append(sid)
        if unresolved:
            return StepResult(
                "citation_verification", False,
                f"{len(unresolved)} unresolved citation(s): {unresolved[:3]}",
            )
        return StepResult("citation_verification", True)

    def _step3_contradiction_audit(self, summary: Any) -> StepResult:
        contradictions = getattr(summary, "contradictions", []) or []
        n = len(contradictions)
        uncertainty = getattr(summary, "uncertainty_annotations", []) or []
        # Allow contradictions if explicitly annotated under uncertainty
        unmitigated = n - len(uncertainty)
        if unmitigated > self.max_contradictions:
            return StepResult(
                "contradiction_audit", False,
                f"{unmitigated} unmitigated contradiction(s) (max {self.max_contradictions})",
            )
        return StepResult("contradiction_audit", True)

    def _step4_uncertainty_annotation(self, summary: Any) -> StepResult:
        score = float(getattr(summary, "overall_uncertainty_score", 0.0) or 0.0)
        if score > self.max_uncertainty:
            msg = f"overall_uncertainty_score={score:.3f} > {self.max_uncertainty}"
            if self.allow_uncertain:
                logger.warning("PublicationGate step4: %s (allowed)", msg)
                return StepResult("uncertainty_annotation", True, f"WARNING: {msg}")
            return StepResult("uncertainty_annotation", False, msg)
        return StepResult("uncertainty_annotation", True)

    def _step5_personalization(
        self, summary: Any, watchlist_entities: Optional[Set[str]], topic: str
    ) -> StepResult:
        if not watchlist_entities:
            return StepResult("personalization_relevance", True, "no watchlist — skipped")
        text = " ".join([
            str(getattr(summary, "what_happened", "") or ""),
            str(getattr(summary, "why_it_matters", "") or ""),
        ]).lower()
        matched = [e for e in watchlist_entities if e.lower() in text]
        if not matched:
            return StepResult(
                "personalization_relevance", False,
                f"none of {len(watchlist_entities)} watchlist entities found in summary",
            )
        return StepResult("personalization_relevance", True, f"matched: {matched[:3]}")

    def _step6_quality_gate(self, summary: Any) -> StepResult:
        score = float(getattr(summary, "confidence_score", 0.0) or 0.0)
        if score < self.min_confidence:
            return StepResult(
                "quality_gate", False,
                f"confidence_score={score:.3f} < {self.min_confidence}",
            )
        return StepResult("quality_gate", True)

    def _step7_policy_gate(self, summary: Any) -> StepResult:
        if not self.policy_blocklist:
            return StepResult("policy_gate", True, "no blocklist configured")
        text = " ".join([
            str(getattr(summary, "what_happened", "") or ""),
            str(getattr(summary, "why_it_matters", "") or ""),
        ]).lower()
        hits = [tok for tok in self.policy_blocklist if tok in text]
        if hits:
            return StepResult(
                "policy_gate", False,
                f"policy blocklist hit: {hits[:3]}",
            )
        return StepResult("policy_gate", True)


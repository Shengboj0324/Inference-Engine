"""Response generation evaluation metrics.

Blueprint §8 — Response generation metrics:
- approval_rate          (human-approved fraction when labels present)
- policy_violation_rate  (% of drafts with ≥1 policy violation)
- unsafe_draft_rate      (% of drafts with blocking policy violation)
- length_appropriateness (% of drafts within channel-appropriate length)
- critique_score         (auto-scored 0-1 based on heuristics)

Does NOT require an LLM at evaluation time — all metrics are deterministic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

# Channel-specific length targets (characters)
_CHANNEL_LENGTH_TARGETS = {
    "direct_reply": (50, 500),
    "direct_message": (100, 1200),
    "email": (200, 2000),
    "slack_notification": (50, 300),
    "no_response": (0, 0),
}
_DEFAULT_LENGTH_TARGET = (50, 800)

# Heuristic spam / overclaim patterns
_SPAM_PATTERNS = [re.compile(p, re.I) for p in [
    r"click here", r"limited time", r"act now", r"100%\s*(guaranteed|free)",
    r"best in (the )?world", r"#1 solution", r"miracle",
]]


@dataclass
class DraftEvalResult:
    draft_id: str
    has_policy_violation: bool
    is_blocking_violation: bool
    length_ok: bool
    critique_score: float        # 0.0 (bad) to 1.0 (excellent)
    is_approved: Optional[bool]  # None when no human label available
    char_count: int


@dataclass
class ResponseReport:
    approval_rate: Optional[float]     # None when no labels provided
    policy_violation_rate: float
    unsafe_draft_rate: float
    length_appropriateness_rate: float
    mean_critique_score: float
    per_draft: List[DraftEvalResult]
    total_drafts: int


class ResponseEvaluator:
    """Evaluate a batch of response drafts without calling an LLM."""

    def evaluate(
        self,
        draft_contents: Sequence[str],
        draft_ids: Optional[Sequence[str]] = None,
        policy_violation_flags: Optional[Sequence[bool]] = None,
        blocking_violation_flags: Optional[Sequence[bool]] = None,
        channels: Optional[Sequence[str]] = None,
        human_approvals: Optional[Sequence[Optional[bool]]] = None,
    ) -> ResponseReport:
        """Evaluate drafts.

        Parameters
        ----------
        draft_contents           Raw text of each draft.
        draft_ids                Optional IDs for tracing.
        policy_violation_flags   Pre-computed flag per draft (from PolicyChecker).
        blocking_violation_flags Pre-computed blocking flag per draft.
        channels                 Channel name per draft (for length targets).
        human_approvals          True/False/None per draft (None = not labelled).
        """
        n = len(draft_contents)
        if n == 0:
            raise ValueError("Empty evaluation batch")

        ids     = list(draft_ids)            if draft_ids            else [str(i) for i in range(n)]
        pv      = list(policy_violation_flags)   if policy_violation_flags   else [False] * n
        bv      = list(blocking_violation_flags) if blocking_violation_flags else [False] * n
        chans   = list(channels)             if channels             else ["direct_reply"] * n
        approvals = list(human_approvals)    if human_approvals      else [None] * n

        per_draft: List[DraftEvalResult] = []
        for i, content in enumerate(draft_contents):
            lo, hi = _CHANNEL_LENGTH_TARGETS.get(chans[i], _DEFAULT_LENGTH_TARGET)
            char_count = len(content)
            length_ok = lo <= char_count <= hi if hi > 0 else char_count == 0

            critique = self._critique_score(content, chans[i])

            per_draft.append(DraftEvalResult(
                draft_id=ids[i],
                has_policy_violation=pv[i],
                is_blocking_violation=bv[i],
                length_ok=length_ok,
                critique_score=critique,
                is_approved=approvals[i],
                char_count=char_count,
            ))

        total = len(per_draft)
        pv_rate  = sum(d.has_policy_violation   for d in per_draft) / total
        bv_rate  = sum(d.is_blocking_violation  for d in per_draft) / total
        len_rate = sum(d.length_ok              for d in per_draft) / total
        mean_crit = sum(d.critique_score        for d in per_draft) / total

        labeled = [d for d in per_draft if d.is_approved is not None]
        approval_rate = (
            sum(1 for d in labeled if d.is_approved) / len(labeled)
            if labeled else None
        )

        report = ResponseReport(
            approval_rate=approval_rate,
            policy_violation_rate=pv_rate,
            unsafe_draft_rate=bv_rate,
            length_appropriateness_rate=len_rate,
            mean_critique_score=mean_crit,
            per_draft=per_draft,
            total_drafts=total,
        )

        logger.info(
            "ResponseEvaluator: n=%d pv_rate=%.3f unsafe=%.3f length_ok=%.3f critique=%.3f",
            total, pv_rate, bv_rate, len_rate, mean_crit,
        )
        return report

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _critique_score(content: str, channel: str) -> float:
        """Deterministic heuristic critique score ∈ [0, 1]."""
        if not content.strip():
            return 0.0

        score = 1.0
        # Penalise spam patterns
        spam_hits = sum(1 for p in _SPAM_PATTERNS if p.search(content))
        score -= 0.15 * spam_hits

        # Penalise very short or very long relative to target
        lo, hi = _CHANNEL_LENGTH_TARGETS.get(channel, _DEFAULT_LENGTH_TARGET)
        length = len(content)
        if hi > 0:
            if length < lo:
                score -= 0.2
            elif length > hi * 1.5:
                score -= 0.15

        # Penalise ALL CAPS passages (shouting)
        upper_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        if upper_ratio > 0.4:
            score -= 0.15

        # Reward presence of a question or acknowledgement
        if re.search(r"\?|Thank|understand|appreciate|help", content, re.I):
            score += 0.05

        return max(0.0, min(1.0, score))


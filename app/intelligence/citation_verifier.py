"""Post-generation grounding check for ``SituationReport`` outputs.

The fine-tuned model is instructed to emit citation spans that index into
the input ``observations``. This module is the structural enforcement of
that contract: every citation must reference a real ``observation_id`` and
the ``[char_start, char_end)`` range must fall inside that observation's
``text``. The cited substring must contain at least one non-whitespace
character so the model cannot satisfy the contract by pointing at an empty
slice.

This is a pure structural verifier. It performs no semantic comparison, no
keyword matching, no template lookup; it only confirms that the spans the
model claims as evidence physically exist in the observations it was given.
Per-claim semantic plausibility is evaluated separately by the LLM-as-judge
in ``app/evals/scenario_eval.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from app.evals.scenario_loader import Observation
from app.intelligence.situation_report import Citation, SituationReport


class CitationGroundingError(ValueError):
    """Raised when a ``SituationReport`` fails structural grounding.

    Attributes
    ----------
    failures:
        Ordered list of :class:`CitationFailure` records describing every
        problem detected in the report. Always non-empty when raised.
    """

    def __init__(self, failures: "List[CitationFailure]") -> None:
        self.failures = failures
        joined = "; ".join(str(f) for f in failures)
        super().__init__(
            f"{len(failures)} citation grounding failure(s): {joined}"
        )


@dataclass(frozen=True)
class CitationFailure:
    """Single grounding problem detected on one citation."""

    citation_index: int
    post_id: str
    reason: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return (
            f"citations[{self.citation_index}] post_id={self.post_id!r}: "
            f"{self.reason}"
        )


class CitationVerifier:
    """Verify that every citation in a report points at real observation text.

    The verifier is stateless and side-effect free; callers should construct
    a fresh instance per call (or reuse one across threads, since it carries
    no mutable state).
    """

    def verify_report(
        self,
        report: SituationReport,
        observations: Sequence[Observation],
    ) -> None:
        """Validate a freshly-generated report against its observations.

        Args:
            report: Parsed ``SituationReport`` returned by the model.
            observations: The exact observation list that was fed to the
                model in the user turn.

        Raises:
            CitationGroundingError: One or more citations do not resolve
                to a real, non-empty span in the observations.
        """
        if report.abstain:
            return
        index = self._index_observations(observations)
        failures: List[CitationFailure] = []
        for i, citation in enumerate(report.citations):
            failure = self._verify_one(i, citation, index)
            if failure is not None:
                failures.append(failure)
        if failures:
            raise CitationGroundingError(failures)

    @staticmethod
    def _index_observations(
        observations: Sequence[Observation],
    ) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for o in observations:
            index[o.observation_id] = o.text
        return index

    @staticmethod
    def _verify_one(
        position: int,
        citation: Citation,
        index: Dict[str, str],
    ) -> "CitationFailure | None":
        text = index.get(citation.post_id)
        if text is None:
            return CitationFailure(
                citation_index=position,
                post_id=citation.post_id,
                reason="post_id does not match any observation_id in the input",
            )
        n = len(text)
        if citation.char_start >= n:
            return CitationFailure(
                citation_index=position,
                post_id=citation.post_id,
                reason=(
                    f"char_start {citation.char_start} is past end of text "
                    f"(length {n})"
                ),
            )
        if citation.char_end > n:
            return CitationFailure(
                citation_index=position,
                post_id=citation.post_id,
                reason=(
                    f"char_end {citation.char_end} exceeds text length {n}"
                ),
            )
        span = text[citation.char_start : citation.char_end]
        if not span.strip():
            return CitationFailure(
                citation_index=position,
                post_id=citation.post_id,
                reason="cited span is empty or whitespace-only",
            )
        return None

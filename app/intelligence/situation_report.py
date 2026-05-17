"""Structured schema for grounded situation reports.

A ``SituationReport`` is the single output contract for the post-migration
intelligence engine.  Every report rendered to a user must validate against
this schema, and every claim it contains must point back to an
``Observation`` span via a ``Citation``.

The schema enforces, by construction:

- citation-grounded claims (no claim without ``citation_ids``)
- explicit abstention semantics (``abstain`` requires a reason and forbids
  positive findings)
- calibrated confidence in ``[0.0, 1.0]``
- severity drawn from a closed five-level scale

This module contains **no rules, keyword lists, or report templates**; it is a
format contract used by both the engine output path and the labelling
pipeline.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Severity(str, Enum):
    """Five-level severity scale used across signal types.

    Concrete thresholds per ``SignalType`` are defined in
    ``docs/labelling/guidelines.md`` and are owned by the labelling team, not
    by code.
    """

    SEV_1 = "SEV-1"
    SEV_2 = "SEV-2"
    SEV_3 = "SEV-3"
    SEV_4 = "SEV-4"
    SEV_5 = "SEV-5"


class Citation(BaseModel):
    """Span-level pointer into a source observation.

    ``post_id`` matches an ``observation_id`` from the scenario's
    ``observations.jsonl``; ``char_start``/``char_end`` index into that
    observation's ``text`` field.
    """

    post_id: str = Field(..., min_length=1, max_length=128)
    char_start: int = Field(..., ge=0)
    char_end: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _validate_span(self) -> "Citation":
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be greater than "
                f"char_start ({self.char_start})"
            )
        return self


class Claim(BaseModel):
    """A single factual assertion in the situation report.

    A claim is invalid if it has no citations: every assertion must be
    traceable to evidence in the input observations.
    """

    text: str = Field(..., min_length=1, max_length=2_000)
    citation_ids: List[int] = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)


class SuggestedAction(BaseModel):
    """An action proposed to the operator.

    Actions are advisory; the engine never executes them.  ``rationale`` must
    cite at least one claim by index so operators can audit why the action
    was proposed.
    """

    text: str = Field(..., min_length=1, max_length=1_000)
    rationale_claim_ids: List[int] = Field(..., min_length=1)
    priority: int = Field(..., ge=1, le=5)


class SituationReport(BaseModel):
    """Grounded situation report emitted by the intelligence engine.

    Either ``abstain`` is ``True`` (and ``abstention_reason`` is set, with no
    positive findings), or the report carries at least one claim with at
    least one citation.

    The schema is deliberately small and scenario-agnostic: there are no
    per-scenario sections, no template slots, no rubric encodings.  All
    domain reasoning lives in the LLM, grounded in ``citations``.
    """

    signal_type: str = Field(..., min_length=1, max_length=64)
    severity: Severity
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0)

    summary: str = Field(..., min_length=1, max_length=4_000)
    claims: List[Claim] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    suggested_actions: List[SuggestedAction] = Field(default_factory=list)

    abstain: bool = False
    abstention_reason: Optional[str] = Field(None, max_length=1_000)

    schema_version: str = Field("1.0.0", pattern=r"^\d+\.\d+\.\d+$")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_grounding(self) -> "SituationReport":
        if self.abstain:
            if not self.abstention_reason:
                raise ValueError("abstain=True requires abstention_reason")
            if self.claims or self.suggested_actions:
                raise ValueError(
                    "abstain=True forbids claims and suggested_actions"
                )
            return self

        if not self.claims:
            raise ValueError(
                "non-abstaining report must contain at least one claim"
            )
        if not self.citations:
            raise ValueError(
                "non-abstaining report must contain at least one citation"
            )

        n_citations = len(self.citations)
        for i, claim in enumerate(self.claims):
            for cid in claim.citation_ids:
                if not 0 <= cid < n_citations:
                    raise ValueError(
                        f"claims[{i}].citation_ids contains out-of-range "
                        f"index {cid} (have {n_citations} citations)"
                    )

        n_claims = len(self.claims)
        for i, action in enumerate(self.suggested_actions):
            for cid in action.rationale_claim_ids:
                if not 0 <= cid < n_claims:
                    raise ValueError(
                        f"suggested_actions[{i}].rationale_claim_ids "
                        f"contains out-of-range index {cid} "
                        f"(have {n_claims} claims)"
                    )
        return self

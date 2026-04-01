"""Shared Pydantic models for the grounded-synthesis / summarization package.

Design conventions (same as Phase 2 & 3):
- ``frozen=True`` on value-object models (EvidenceSource, AttributedClaim,
  ContradictionPair, UncertaintyAnnotation, GroundedSummary).
- Mutable on input/config models (SynthesisRequest).
- Every numeric field carries explicit ``ge``/``le`` range validators.
- Every string identifier field carries ``min_length=1``.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    """Semantic category of an extracted claim."""

    FACTUAL = "factual"           # Declarative past-tense statement
    ANNOUNCEMENT = "announcement" # Official / first-party announcement
    BENCHMARK = "benchmark"       # Quantitative performance comparison
    SPECULATION = "speculation"   # Forward-looking or hedged assertion
    OPINION = "opinion"           # Evaluative / subjective statement
    COMPARATIVE = "comparative"   # Explicit comparison between entities


class ContradictionSeverity(str, Enum):
    """How materially two claims contradict each other."""

    MINOR = "minor"       # Framing or emphasis difference
    MODERATE = "moderate" # Factual tension without direct falsification
    MAJOR = "major"       # Directly conflicting factual assertions
    CRITICAL = "critical" # One claim explicitly falsifies the other


class UncertaintySeverity(str, Enum):
    """Confidence impact of detected uncertain language."""

    LOW = "low"           # Soft hedge (e.g. "expected to")
    MEDIUM = "medium"     # Notable uncertainty (e.g. "reportedly")
    HIGH = "high"         # Strong speculation (e.g. "rumoured", "allegedly")
    CRITICAL = "critical" # Unverified / anonymous sourcing


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

class EvidenceSource(BaseModel, frozen=True):
    """A single traceable source that supports one or more claims.

    Attributes:
        source_id:       Unique identifier for this source.
        title:           Display title of the source.
        url:             Source URL (may be empty for offline sources).
        platform:        Publishing platform / connector name.
        trust_score:     Reliability score in [0, 1].
        published_at:    UTC publication datetime.
        content_snippet: Truncated source text used for attribution matching.
        author:          Author or publisher name.
    """

    source_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    title: str = ""
    url: str = ""
    platform: str = ""
    trust_score: float = Field(default=0.5, ge=0.0, le=1.0)
    published_at: Optional[datetime] = None
    content_snippet: str = ""
    author: str = ""


class AttributedClaim(BaseModel, frozen=True):
    """A claim extracted from content, with confidence and source attribution.

    Attributes:
        claim_id:           Unique identifier.
        text:               Claim text (must be non-empty).
        claim_type:         Semantic category.
        confidence:         How well-supported this claim is [0, 1].
        source_ids:         IDs of sources that support this claim.
        negation_detected:  Whether the claim contains a negation.
        extracted_at:       UTC timestamp of extraction.
    """

    claim_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    text: str = Field(..., min_length=1)
    claim_type: ClaimType = ClaimType.FACTUAL
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_ids: List[str] = Field(default_factory=list)
    negation_detected: bool = False
    extracted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContradictionPair(BaseModel, frozen=True):
    """Two claims that contradict each other across sources.

    Attributes:
        claim_a:          First claim.
        claim_b:          Second claim contradicting the first.
        explanation:      Human-readable explanation of the conflict.
        severity:         How serious the contradiction is.
        detected_pattern: Heuristic pattern that triggered detection
                          (e.g. ``"negation_conflict"``, ``"number_conflict"``).
    """

    claim_a: AttributedClaim
    claim_b: AttributedClaim
    explanation: str = ""
    severity: ContradictionSeverity = ContradictionSeverity.MODERATE
    detected_pattern: str = ""

    @field_validator("claim_b")
    @classmethod
    def _claims_differ(cls, v: AttributedClaim, info) -> AttributedClaim:
        claim_a = info.data.get("claim_a")
        if claim_a is not None and v.claim_id == claim_a.claim_id:
            raise ValueError("'claim_a' and 'claim_b' must be different claims")
        return v


class UncertaintyAnnotation(BaseModel, frozen=True):
    """A span of text flagged as uncertain or speculative.

    Attributes:
        text_span: The hedged / uncertain text (must be non-empty).
        severity:  Confidence impact of this uncertainty.
        reason:    Machine-readable reason code (e.g. ``"hedge_word:might"``).
        position:  Character offset of the span in the source text (−1 if unknown).
    """

    text_span: str = Field(..., min_length=1)
    severity: UncertaintySeverity = UncertaintySeverity.MEDIUM
    reason: str = ""
    position: int = Field(default=-1, ge=-1)




class GroundedSummary(BaseModel, frozen=True):
    """The authoritative output of the grounded synthesis pipeline.

    Every factual statement links back to at least one ``EvidenceSource``.
    Contradictions and uncertainty annotations are surfaced explicitly so
    downstream consumers can calibrate how much to trust the output.

    Attributes:
        summary_id:               Unique identifier for this summary run.
        what_happened:            Factual, attribution-backed description.
        why_it_matters:           Significance analysis.
        confidence_score:         Overall confidence in [0, 1].
        source_attributions:      Supporting ``EvidenceSource`` objects.
        key_claims:               Top ``AttributedClaim`` objects.
        contradictions:           Detected ``ContradictionPair`` objects.
        uncertainty_annotations:  Detected uncertainty spans.
        who_it_affects:           List of stakeholder types.
        overall_uncertainty_score: Aggregate hedge / speculation level [0, 1].
        generated_at:             UTC timestamp.
        source_count:             Number of distinct sources used.
    """

    summary_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    what_happened: str = ""
    why_it_matters: str = ""
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source_attributions: List[EvidenceSource] = Field(default_factory=list)
    key_claims: List[AttributedClaim] = Field(default_factory=list)
    contradictions: List[ContradictionPair] = Field(default_factory=list)
    uncertainty_annotations: List[UncertaintyAnnotation] = Field(default_factory=list)
    who_it_affects: List[str] = Field(default_factory=list)
    overall_uncertainty_score: float = Field(default=0.0, ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_count: int = Field(default=0, ge=0)


class SynthesisRequest(BaseModel):
    """Input container for a grounded-synthesis run.

    Attributes:
        topic:            Short topic label (non-empty).
        sources:          List of ``EvidenceSource`` objects; min 1.
        context:          Optional free-text context (e.g. user intent).
        max_claims:       Maximum claims to extract per source.
        min_source_trust: Filter sources below this trust threshold.
        who_it_affects:   Stakeholder hints to guide synthesis.
    """

    topic: str = Field(..., min_length=1)
    sources: List[EvidenceSource] = Field(..., min_length=1)
    context: str = ""
    max_claims: int = Field(default=10, ge=1, le=100)
    min_source_trust: float = Field(default=0.0, ge=0.0, le=1.0)
    who_it_affects: List[str] = Field(default_factory=list)

    @field_validator("sources")
    @classmethod
    def _sources_non_empty(cls, v: List[EvidenceSource]) -> List[EvidenceSource]:
        if not v:
            raise ValueError("'sources' must contain at least one EvidenceSource")
        return v

"""Pydantic schemas for the structured crisis intelligence report.

Every section of the rubric is encoded as a typed slot.  The report
builder fills these slots deterministically; the LLM only contributes
the short executive-summary prose (and the prose must round-trip back
through the model's ``executive_summary`` field).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class RiskEntry(BaseModel):
    title: str = Field(..., min_length=1)
    severity: str = Field(..., pattern=r"^(Low|Medium|High)$")
    evidence_strength: str = Field(..., min_length=1)        # e.g. "High (0.78)"
    core_issue: str = Field(..., min_length=1)
    recommendation: str = Field(..., min_length=1)
    evidence_rationale: str = ""


class ClaimVerdict(BaseModel):
    verdict: str = Field(..., pattern=r"^(REAL|FAKE|UNSUPPORTED|MIXED)$")
    claim: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)
    evidence_strength: str = ""
    references_debunked_claim: Optional[str] = None     # claim_id


class ActorEntry(BaseModel):
    handle: str = Field(..., min_length=1)
    role: str = ""
    credibility_tier: str = Field(..., pattern=r"^(high|medium|low)$")
    influence_tier: str = Field(..., pattern=r"^(high|medium|low)$")
    why_they_matter: str = Field(..., min_length=1)
    handling_note: str = ""


class NarrativeStage(BaseModel):
    stage: str = Field(..., pattern=r"^(morning|midday|afternoon|evening)$")
    summary: str = Field(..., min_length=1)


class ActionItem(BaseModel):
    priority: int = Field(..., ge=1)
    category: str = Field(...,
                          pattern=r"^(copy|onboarding|engineering|comms|misinformation|community)$")
    title: str = Field(..., min_length=1)
    detail: str = Field(..., min_length=1)
    owner: str = ""


class ReplyQueueEntry(BaseModel):
    priority: int = Field(..., ge=1)
    handle: str = Field(..., min_length=1)
    why: str = Field(..., min_length=1)
    draft_reply: str = Field(..., min_length=1)


class HardModeFinding(BaseModel):
    name: str = Field(..., min_length=1)
    status: str = Field(..., pattern=r"^(detected|not_detected|n_a)$")
    detail: str = ""


class CrisisIntelligenceReport(BaseModel):
    """Top-level report.  Every rubric section is a required slot."""

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    subject: str = Field(..., min_length=1)

    # Section 1
    executive_summary: str = Field(..., min_length=1)

    # Section 2
    top_risks: List[RiskEntry] = Field(default_factory=list)

    # Section 3
    real_vs_fake: List[ClaimVerdict] = Field(default_factory=list)

    # Section 4
    important_actors: List[ActorEntry] = Field(default_factory=list)
    low_reliability_actors: List[ActorEntry] = Field(default_factory=list)

    # Section 5
    narrative: List[NarrativeStage] = Field(default_factory=list)

    # Section 6
    recommended_actions: List[ActionItem] = Field(default_factory=list)
    reply_queue: List[ReplyQueueEntry] = Field(default_factory=list)

    # Required explicit yes/no answers from the rubric
    coordinated_amplification_detected: bool = False
    coordinated_amplification_detail: str = ""
    sarcasm_detected: bool = False
    sarcasm_examples: List[str] = Field(default_factory=list)
    ecowatchdog_misquotation_should_be_corrected: bool = False
    ecowatchdog_misquotation_detail: str = ""
    current_onboarding_screenshot: str = ""

    # Hard-mode add-ons (informational)
    hard_mode: List[HardModeFinding] = Field(default_factory=list)

    @field_validator("top_risks")
    @classmethod
    def _at_least_one_risk(cls, v: List[RiskEntry]) -> List[RiskEntry]:
        if not v:
            raise ValueError("top_risks must not be empty")
        return v

    @field_validator("recommended_actions")
    @classmethod
    def _at_least_one_action(cls, v: List[ActionItem]) -> List[ActionItem]:
        if not v:
            raise ValueError("recommended_actions must not be empty")
        return v

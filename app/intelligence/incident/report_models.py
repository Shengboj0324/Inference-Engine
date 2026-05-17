"""Pydantic schema for the NeuroBridge incident-intelligence report.

Mirrors the rubric's required 12-section structure so the renderer can
walk it deterministically and every section can be unit-tested in
isolation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SeverityRow(BaseModel):
    scope: str = Field(..., min_length=1)
    severity: str = Field(..., min_length=1)
    note: str = ""


class ConfirmedFact(BaseModel):
    title: str = Field(..., min_length=1)
    detail: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)        # e.g. "internal investigation"


class UnsupportedClaim(BaseModel):
    claim: str = Field(..., min_length=1)
    classification: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)


class EvidenceMatrixRow(BaseModel):
    finding: str = Field(..., min_length=1)
    strength: str = Field(..., min_length=1)
    note: str = ""


class TimelineEntry(BaseModel):
    stage_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    bullets: List[str] = Field(default_factory=list)


class IncidentActorEntry(BaseModel):
    handle: str = Field(..., min_length=1)
    role: str = Field(..., min_length=1)
    credibility_tier: str = Field(..., min_length=1)
    influence_tier: str = Field(..., min_length=1)
    why_they_matter: str = Field(..., min_length=1)
    handling_note: str = ""


class StakeholderImpact(BaseModel):
    stakeholder: str = Field(..., min_length=1)
    impact: str = Field(..., min_length=1)


class CoordinationFinding(BaseModel):
    near_identical_post_count: int = 0
    new_account_count: int = 0
    new_account_window_days: int = 14
    same_screenshot_reposts: int = 0
    altered_screenshot_count: int = 0
    altered_school_name_from: str = ""
    altered_school_name_to: str = ""
    legitimate_question_count: int = 0
    repeated_phrases: List[str] = Field(default_factory=list)
    rationale: str = ""


class IncidentActionItem(BaseModel):
    team: str = Field(..., min_length=1)   # "security" | "legal" | "comms"
                                           # | "product" | "support"
    priority: int = Field(..., ge=1)
    title: str = Field(..., min_length=1)
    detail: str = Field(..., min_length=1)


class PublicMessagingPosture(BaseModel):
    bullets: List[str] = Field(default_factory=list)

    @field_validator("bullets")
    @classmethod
    def _non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("public messaging posture must not be empty")
        return v


class IncidentIntelligenceReport(BaseModel):
    """Top-level report (rubric sections 1..11 + escalation note)."""
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    subject: str = Field(..., min_length=1)
    incident_label: str = Field(..., min_length=1)

    # Section 1
    executive_summary: List[str] = Field(default_factory=list)

    # Section 2
    current_severity_rows: List[SeverityRow] = Field(default_factory=list)
    current_severity_label: str = Field(..., min_length=1)

    # Section 3
    confirmed_facts: List[ConfirmedFact] = Field(default_factory=list)

    # Section 4
    unsupported_claims: List[UnsupportedClaim] = Field(default_factory=list)

    # Section 5
    evidence_matrix: List[EvidenceMatrixRow] = Field(default_factory=list)

    # Section 6
    timeline: List[TimelineEntry] = Field(default_factory=list)

    # Section 7
    key_actors: List[IncidentActorEntry] = Field(default_factory=list)

    # Section 8
    stakeholder_impact: List[StakeholderImpact] = Field(default_factory=list)

    # Section 9
    coordination: CoordinationFinding = Field(default_factory=CoordinationFinding)

    # Section 10
    messaging_posture: PublicMessagingPosture

    # Section 11
    tomorrow_plan_security: List[IncidentActionItem] = Field(default_factory=list)
    tomorrow_plan_legal: List[IncidentActionItem] = Field(default_factory=list)
    tomorrow_plan_comms: List[IncidentActionItem] = Field(default_factory=list)
    tomorrow_plan_product: List[IncidentActionItem] = Field(default_factory=list)
    tomorrow_plan_support: List[IncidentActionItem] = Field(default_factory=list)

    # Section 12
    internal_escalation_recommendation: str = Field(..., min_length=1)

    @field_validator("executive_summary")
    @classmethod
    def _exec_summary_nonempty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("executive_summary must not be empty")
        return v

    @field_validator("confirmed_facts")
    @classmethod
    def _confirmed_facts_nonempty(cls, v: List[ConfirmedFact]) -> List[ConfirmedFact]:
        if not v:
            raise ValueError("confirmed_facts must not be empty")
        return v

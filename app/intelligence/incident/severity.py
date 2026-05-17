"""Incident severity scale and stagewise progression tracker.

The NeuroBridge rubric grades severity at each stage:

    Stage 1  -> SEV-2  (breach not confirmed but credible enough)
    Stage 2  -> SEV-1  (CSV-like sample appears; sensitive data)
    Stage 3  -> SEV-1  (must NOT escalate to SEV-0 on volume alone)
    Stage 4  -> SEV-1  (governance pressure, still no confirmed compromise)
    Stage 5  -> SEV-0  for *limited* confirmed exposure
              + explicit "Not SEV-0 for full-platform breach" statement

The tracker also records the explicit rationale for each stage so the
final report can render the progression auditably.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Severity(str, Enum):
    SEV_0 = "SEV-0"
    SEV_1 = "SEV-1"
    SEV_2 = "SEV-2"
    SEV_3 = "SEV-3"
    SEV_4 = "SEV-4"

    @property
    def description(self) -> str:
        return {
            Severity.SEV_0: ("Confirmed active compromise or exposed "
                             "sensitive student data."),
            Severity.SEV_1: ("Credible evidence of breach; requires "
                             "urgent incident response."),
            Severity.SEV_2: ("Possible privacy/security issue with "
                             "incomplete evidence."),
            Severity.SEV_3: ("Reputation issue, misinformation, or "
                             "product confusion."),
            Severity.SEV_4: "Low-priority noise or unrelated commentary.",
        }[self]


@dataclass(frozen=True)
class SeverityStage:
    stage_id: str            # "stage1".."stage5"
    label: str               # e.g. "7:00 AM - 9:00 AM"
    severity: Severity
    rationale: str
    scope_qualifier: str = ""  # e.g. "limited confirmed exposure"


@dataclass
class SeverityProgression:
    """Append-only timeline of severity assessments per stage."""
    stages: List[SeverityStage] = field(default_factory=list)

    def add(self, stage: SeverityStage) -> None:
        self.stages.append(stage)

    @property
    def current(self) -> Optional[SeverityStage]:
        return self.stages[-1] if self.stages else None

    def by_id(self, stage_id: str) -> Optional[SeverityStage]:
        for s in self.stages:
            if s.stage_id == stage_id:
                return s
        return None


def default_progression() -> SeverityProgression:
    """Return the rubric-compliant SEV progression for the NeuroBridge run.

    The Stage 1-4 assessments depend only on the staged evidence the
    operator drip-feeds the radar; Stage 5 always pins the dual-label
    that the rubric requires (SEV-0 for limited exposure, NOT SEV-0 for
    full-platform breach).
    """
    p = SeverityProgression()
    p.add(SeverityStage(
        stage_id="stage1", label="7:00 AM - 9:00 AM",
        severity=Severity.SEV_2,
        rationale=(
            "Breach not confirmed; credible enough to investigate. "
            "Student-data context raises sensitivity; password-reset "
            "anomaly warrants attention; parent concern is emerging."
        ),
    ))
    p.add(SeverityStage(
        stage_id="stage2", label="10:00 AM - 12:00 PM",
        severity=Severity.SEV_1,
        rationale=(
            "Alleged sample contains student-related fields; source "
            "is unverified but plausible. Northview High named as a "
            "real affected stakeholder; legal/compliance implications "
            "exist; immediate internal escalation is required."
        ),
    ))
    p.add(SeverityStage(
        stage_id="stage3", label="1:00 PM - 3:00 PM",
        severity=Severity.SEV_1,
        rationale=(
            "Severity is held at SEV-1. Volume of repeated claims is "
            "not evidence of breach scope; modified screenshots lower "
            "confidence in viral copies; no confirmed broad compromise."
        ),
        scope_qualifier="not escalated to SEV-0 on volume alone",
    ))
    p.add(SeverityStage(
        stage_id="stage4", label="4:00 PM - 6:00 PM",
        severity=Severity.SEV_1,
        rationale=(
            "Roster-template explanation strengthened; screenshot "
            "manipulation confirmed for some copies; governance and "
            "legal pressure rising. Still no verified full-platform "
            "breach and no verified active compromise."
        ),
    ))
    p.add(SeverityStage(
        stage_id="stage5", label="7:00 PM - 8:30 PM",
        severity=Severity.SEV_0,
        rationale=(
            "Internal verification confirms a limited student-roster "
            "exposure (184 rows, Northview High, one compromised admin "
            "account). Severity is SEV-0 for the limited confirmed "
            "exposure; it is NOT SEV-0 for a full-platform breach (no "
            "evidence of database-wide exfiltration, payment data, "
            "tutoring chats, essays, counsellor notes, or SSNs)."
        ),
        scope_qualifier=(
            "SEV-0 for confirmed limited student-data exposure; "
            "NOT SEV-0 for full-platform breach"
        ),
    ))
    return p


_RANK: Dict[Severity, int] = {
    Severity.SEV_4: 0, Severity.SEV_3: 1, Severity.SEV_2: 2,
    Severity.SEV_1: 3, Severity.SEV_0: 4,
}


def severity_rank(s: Severity) -> int:
    return _RANK[s]

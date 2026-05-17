"""Render an :class:`IncidentIntelligenceReport` as the rubric-mandated
12-section Markdown report.

Pure formatting; no behavioural logic.  The renderer walks the report
fields in section order so the structure matches the rubric's "Required
Final Report Structure" exactly.
"""
from __future__ import annotations

from typing import List

from app.intelligence.incident.report_models import (
    IncidentActionItem,
    IncidentIntelligenceReport,
)


def render_incident_report_markdown(r: IncidentIntelligenceReport) -> str:
    lines: List[str] = []
    lines.append(f"# Incident Intelligence Report - {r.subject}")
    lines.append("")
    lines.append(f"_Incident label_: {r.incident_label}")
    lines.append("")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary")
    for b in r.executive_summary:
        lines.append(f"- {b}")
    lines.append("")

    # 2. Current Severity
    lines.append("## 2. Current Severity")
    lines.append(f"**{r.current_severity_label}**")
    lines.append("")
    lines.append("| Scope | Severity | Note |")
    lines.append("|---|---|---|")
    for s in r.current_severity_rows:
        lines.append(f"| {s.scope} | {s.severity} | {s.note} |")
    lines.append("")

    # 3. Confirmed Facts
    lines.append("## 3. Confirmed Facts")
    for i, f in enumerate(r.confirmed_facts, 1):
        lines.append(f"{i}. **{f.title}** {f.detail} _(source: {f.source})_")
    lines.append("")

    # 4. Unsupported / False / Misleading Claims
    lines.append("## 4. Unsupported, False, or Misleading Claims")
    lines.append("| Claim | Classification | Rationale |")
    lines.append("|---|---|---|")
    for u in r.unsupported_claims:
        lines.append(f"| {u.claim} | {u.classification} | {u.rationale} |")
    lines.append("")

    # 5. Evidence Strength Matrix
    lines.append("## 5. Evidence Strength Matrix")
    lines.append("| Finding | Evidence Strength | Note |")
    lines.append("|---|---|---|")
    for e in r.evidence_matrix:
        lines.append(f"| {e.finding} | {e.strength} | {e.note} |")
    lines.append("")

    # 6. Timeline
    lines.append("## 6. Timeline of Narrative Evolution")
    for t in r.timeline:
        lines.append(f"### {t.label}")
        for b in t.bullets:
            lines.append(f"- {b}")
        lines.append("")

    # 7. Key Actors
    lines.append("## 7. Key Actors")
    for a in r.key_actors:
        lines.append(
            f"### {a.handle}"
        )
        lines.append(
            f"- **Role:** {a.role}  "
            f"_(credibility={a.credibility_tier}, "
            f"influence={a.influence_tier})_"
        )
        lines.append(f"- **Why they matter:** {a.why_they_matter}")
        if a.handling_note:
            lines.append(f"- **Handling:** {a.handling_note}")
        lines.append("")

    # 8. Stakeholder Impact Map
    lines.append("## 8. Stakeholder Impact Map")
    lines.append("| Stakeholder | Impact |")
    lines.append("|---|---|")
    for s in r.stakeholder_impact:
        lines.append(f"| {s.stakeholder} | {s.impact} |")
    lines.append("")

    # 9. Coordinated Amplification Analysis
    c = r.coordination
    lines.append("## 9. Coordinated Amplification Analysis")
    lines.append(
        f"- **{c.near_identical_post_count:,} posts** repeated "
        "near-identical claims."
    )
    lines.append(
        f"- **{c.new_account_count:,} accounts** were created within the "
        f"last **{c.new_account_window_days} days**."
    )
    lines.append(
        f"- **{c.same_screenshot_reposts:,} posts** reused the same "
        "screenshot."
    )
    lines.append(
        f"- **{c.altered_screenshot_count:,} modified screenshots** "
        f"changed the school name from "
        f"'{c.altered_school_name_from}' to '{c.altered_school_name_to}'."
    )
    lines.append(
        f"- **{c.legitimate_question_count:,} posts** were legitimate "
        "parent/student questions (not bot noise)."
    )
    if c.repeated_phrases:
        lines.append("- Repeated phrases driving the amplification:")
        for p in c.repeated_phrases:
            lines.append(f"  - \"{p}\"")
    lines.append(f"- **Interpretation:** {c.rationale}")
    lines.append("")

    # 10. Public Messaging Posture
    lines.append("## 10. Required Public Messaging Position")
    for b in r.messaging_posture.bullets:
        lines.append(f"- {b}")
    lines.append("")

    # 11. Tomorrow Morning Operating Plan
    lines.append("## 11. Tomorrow Morning Operating Plan")
    _render_team("Security", r.tomorrow_plan_security, lines)
    _render_team("Legal / Compliance", r.tomorrow_plan_legal, lines)
    _render_team("Communications", r.tomorrow_plan_comms, lines)
    _render_team("Product / Engineering", r.tomorrow_plan_product, lines)
    _render_team("Support", r.tomorrow_plan_support, lines)

    # 12. Escalation
    lines.append("## 12. Internal Escalation Recommendation")
    lines.append(r.internal_escalation_recommendation)
    lines.append("")
    lines.append(f"_generated_at_: {r.generated_at.isoformat()}")
    return "\n".join(lines)


def _render_team(title: str, items: List[IncidentActionItem],
                 lines: List[str]) -> None:
    lines.append(f"### {title}")
    for it in sorted(items, key=lambda x: x.priority):
        lines.append(f"{it.priority}. **{it.title}** - {it.detail}")
    lines.append("")

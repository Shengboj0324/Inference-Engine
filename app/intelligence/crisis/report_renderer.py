"""Render a :class:`CrisisIntelligenceReport` to operator-readable Markdown."""

from __future__ import annotations

from app.intelligence.crisis.report_models import CrisisIntelligenceReport


def _yn(b: bool) -> str:
    return "YES" if b else "NO"


def render_report_markdown(report: CrisisIntelligenceReport) -> str:
    lines = []
    lines.append(f"# Crisis Intelligence Report - {report.subject}")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append(report.executive_summary)
    lines.append("")
    lines.append("## 2. Top Risks")
    for i, r in enumerate(report.top_risks, 1):
        lines.append(f"### {i}. {r.title}")
        lines.append(f"- **Severity:** {r.severity}")
        lines.append(f"- **Evidence strength:** {r.evidence_strength}"
                     + (f" ({r.evidence_rationale})" if r.evidence_rationale else ""))
        lines.append(f"- **Core issue:** {r.core_issue}")
        lines.append(f"- **Recommendation:** {r.recommendation}")
        lines.append("")
    lines.append("## 3. Real vs. Fake / Unsupported")
    for v in report.real_vs_fake:
        ref = f" [debunked-claim: {v.references_debunked_claim}]" \
            if v.references_debunked_claim else ""
        lines.append(f"- **[{v.verdict}]**{ref} {v.claim}")
        lines.append(f"  - rationale: {v.rationale}")
        if v.evidence_strength:
            lines.append(f"  - evidence: {v.evidence_strength}")
    lines.append("")
    lines.append("## 4. Important Actors")
    lines.append("### Higher-credibility (lead with these)")
    for a in report.important_actors:
        lines.append(
            f"- **{a.handle}** [{a.role}; "
            f"credibility={a.credibility_tier}, influence={a.influence_tier}]"
        )
        lines.append(f"  - why: {a.why_they_matter}")
        if a.handling_note:
            lines.append(f"  - handling: {a.handling_note}")
    if report.low_reliability_actors:
        lines.append("### Lower-credibility (handle carefully)")
        for a in report.low_reliability_actors:
            lines.append(
                f"- **{a.handle}** [{a.role}; "
                f"credibility={a.credibility_tier}, influence={a.influence_tier}]"
            )
            lines.append(f"  - why: {a.why_they_matter}")
            if a.handling_note:
                lines.append(f"  - handling: {a.handling_note}")
    lines.append("")
    lines.append("## 5. Narrative Evolution")
    for n in report.narrative:
        lines.append(f"- **{n.stage.capitalize()}:** {n.summary}")
    lines.append("")
    lines.append("## 6. Recommended Actions (priority-ordered)")
    for a in sorted(report.recommended_actions, key=lambda x: x.priority):
        lines.append(f"- **P{a.priority} [{a.category}] {a.title}**"
                     + (f"  _(owner: {a.owner})_" if a.owner else ""))
        lines.append(f"  - {a.detail}")
    lines.append("")
    if report.reply_queue:
        lines.append("### 6b. Stakeholder Reply Queue (priority-ordered)")
        for r in sorted(report.reply_queue, key=lambda x: x.priority):
            lines.append(f"- **P{r.priority} -> {r.handle}** ({r.why})")
            lines.append(f"  - draft: \"{r.draft_reply}\"")
        lines.append("")
    lines.append("## 7. Required Explicit Answers")
    lines.append(f"- **(a) Coordinated amplification detected?** "
                 f"{_yn(report.coordinated_amplification_detected)}"
                 + (f" - {report.coordinated_amplification_detail}"
                    if report.coordinated_amplification_detail else ""))
    lines.append(f"- **(b) Sarcasm detected?** {_yn(report.sarcasm_detected)}"
                 + (f" - examples: " + " | ".join(
                     f"'{ex}'" for ex in report.sarcasm_examples
                 ) if report.sarcasm_examples else ""))
    lines.append(f"- **(c) @EcoWatchdog misquotation should be corrected?** "
                 f"{_yn(report.ecowatchdog_misquotation_should_be_corrected)}"
                 + (f" - {report.ecowatchdog_misquotation_detail}"
                    if report.ecowatchdog_misquotation_detail else ""))
    lines.append(f"- **(d) Current onboarding screenshot:** "
                 f"{report.current_onboarding_screenshot}")
    lines.append("")
    lines.append("## 8. Hard-Mode Findings")
    for h in report.hard_mode:
        flag = "[OK]" if h.status == "detected" else "[--]"
        lines.append(f"- {flag} **{h.name}** - {h.status}"
                     + (f": {h.detail}" if h.detail else ""))
    lines.append("")
    lines.append(f"_generated_at_: {report.generated_at.isoformat()}")
    return "\n".join(lines)

"""Drive the NeuroBridge mock-user scenario through the incident pipeline.

Loads ``neurobridge_mock_user_test.md``, isolates the input batches
(Stage 1..4 posts + Stage 5 Internal Security Update) WITHOUT leaking
the rubric / required-output sections to the pipeline, runs the
deterministic incident pipeline, deterministically scores the resulting
report against the rubric in the same file, and prints both the
verbatim 12-section report and the rubric-aligned score breakdown.

Run with::

    DEPLOYMENT_MODE=desktop python scripts/run_neurobridge_stress_test.py

No network access; no LLM call (the incident pipeline is fully
deterministic for this scenario).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("DEPLOYMENT_MODE", "desktop")

from app.intelligence.incident import (  # noqa: E402
    BulkPattern,
    IncidentReportBuilder,
    analyze_incident_signals,
    build_default_registries,
    parse_incident_stream,
    parse_internal_update,
    render_incident_report_markdown,
)
from app.intelligence.incident.severity import default_progression  # noqa: E402
from scripts._neurobridge_scorer import score_report  # noqa: E402


_SCENARIO_PATH = _REPO_ROOT / "neurobridge_mock_user_test.md"


# Hidden Stage-3 metadata the scenario explicitly hands to the system.
# Values copy directly from "## Hidden Pattern in Batch C".
_BULK = BulkPattern(
    near_identical_post_count=1100,
    near_identical_phrases=(
        "Full database leaked.",
        "Every student's SSN leaked.",
        "NeuroBridge exposed college essays.",
        "Credit cards and student files leaked.",
        "Delete NeuroBridge now.",
    ),
    new_account_count=620,
    new_account_window_days=14,
    same_screenshot_reposts=480,
    altered_screenshot_count=90,
    altered_school_name_from="Northview High",
    altered_school_name_to="North Valley High",
    legitimate_question_count=160,
    delayed_reset_mention_count=75,
    duplicate_email_mention_count=40,
    roster_mishandling_mention_count=25,
    technical_analysis_count=12,
    credit_card_claim_count=7,
)


_STAGE_HDR_RX = re.compile(r"^#\s*Stage\s+(\d+)\s*:", re.IGNORECASE)
_STAGE_END_RX = re.compile(
    r"^##\s*(Required\s+Model\s+Output|Hidden\s+Pattern)",
    re.IGNORECASE,
)
_INTERNAL_HDR_RX = re.compile(r"^##\s*Internal\s+Security\s+Update",
                              re.IGNORECASE)


def _extract_input_stream(md: str) -> tuple:
    """Return (stream_text, internal_update_text)."""
    lines = md.splitlines()
    out: list = []
    internal: list = []
    keep = False
    capture_internal = False
    in_internal_block = False
    for line in lines:
        m_stage = _STAGE_HDR_RX.match(line)
        if m_stage:
            keep = True
            in_internal_block = False
            out.append(line)
            continue
        if keep and _INTERNAL_HDR_RX.match(line):
            in_internal_block = True
            capture_internal = True
            internal.append(line)
            continue
        if keep and _STAGE_END_RX.match(line):
            keep = False
            in_internal_block = False
            continue
        if in_internal_block:
            internal.append(line)
        if keep:
            out.append(line)
    return "\n".join(out), "\n".join(internal)


def main() -> int:
    md = _SCENARIO_PATH.read_text()
    stream_text, internal_text = _extract_input_stream(md)

    print("-" * 78)
    print("NEUROBRIDGE INCIDENT INTELLIGENCE PIPELINE")
    print("-" * 78)

    claims, actors, issues = build_default_registries()
    print(f"  registries: {len(claims.all())} debunked claim(s), "
          f"{len(actors.all())} known actor(s), "
          f"{len(issues.all())} known issue(s)")

    items = parse_incident_stream(stream_text)
    print(f"  parsed {len(items)} stream item(s) across "
          f"{len({i.stage_id for i in items})} stage(s); "
          f"{len({i.handle for i in items if i.handle})} unique handle(s)")

    signals = analyze_incident_signals(
        items, debunked=claims, issues=issues, bulk=_BULK,
    )
    print(
        "  signals: "
        f"csv_samples={len(signals.csv_samples)}, "
        f"screenshot_manipulations={len(signals.screenshot_manipulations)}, "
        f"debunked_matches={len(signals.debunked_matches)}, "
        f"known_issues={len(signals.known_issue_matches)}, "
        f"false_claims={len(signals.false_claims)}"
    )

    update = parse_internal_update(internal_text)
    print(f"  internal verification: school={update.affected_school}, "
          f"rows={update.affected_row_count}, "
          f"fields={len(update.exposed_fields)}, "
          f"sample_matches={update.sample_matches_production}")

    progression = default_progression()

    builder = IncidentReportBuilder(
        subject="NeuroBridge Campus Breach", actors=actors,
    )
    report = builder.build(items=items, signals=signals,
                           internal_update=update, progression=progression)
    md_out = render_incident_report_markdown(report)

    print("=" * 78)
    print("FINAL INCIDENT INTELLIGENCE REPORT")
    print("=" * 78)
    print(md_out)
    print("=" * 78)

    breakdown = score_report(md_out)
    print("RUBRIC SCORE")
    print("-" * 78)
    for cat, (earned, total, rows) in breakdown.categories.items():
        print(f"  {cat}: {earned}/{total}")
        for label, pts_earned, pts_total in rows:
            mark = "OK" if pts_earned == pts_total else "--"
            print(f"    [{mark}] {label}: {pts_earned}/{pts_total}")
    print("-" * 78)
    print(f"  TOTAL: {breakdown.total_earned} / {breakdown.total_max}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

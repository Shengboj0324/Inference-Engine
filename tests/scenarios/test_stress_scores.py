"""End-to-end stress-score regression for both mock-user scenarios.

Runs the deterministic crisis (Releaf) and incident (NeuroBridge)
pipelines without any LLM call, scores each rendered report against the
rubric-aligned deterministic scorer, and enforces a >= 98 / 100 floor.

A score regression in either pipeline trips here long before it shows
up in the operator stress-test drivers.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("DEPLOYMENT_MODE", "desktop")

from app.intelligence.crisis import (  # noqa: E402
    CrisisReportBuilder,
    analyze_signals,
    build_default_registries as build_crisis_registries,
    parse_stream,
    render_report_markdown,
)
from app.intelligence.incident import (  # noqa: E402
    BulkPattern,
    IncidentReportBuilder,
    analyze_incident_signals,
    build_default_registries as build_incident_registries,
    parse_incident_stream,
    parse_internal_update,
    render_incident_report_markdown,
)
from app.intelligence.incident.severity import (  # noqa: E402
    default_progression,
)
from scripts._neurobridge_scorer import (  # noqa: E402
    score_report as score_incident_report,
)
from scripts._releaf_scorer import (  # noqa: E402
    score_report as score_crisis_report,
)
from scripts.run_mock_user_stress_test import _STREAM as _RELEAF_STREAM  # noqa: E402


_SCORE_FLOOR = 98
_NEUROBRIDGE_PATH = _REPO_ROOT / "neurobridge_mock_user_test.md"


def _build_releaf_report() -> str:
    """Render the Releaf crisis report with NO LLM round-trip."""
    claims, actors, issues = build_crisis_registries()
    items = parse_stream(_RELEAF_STREAM)
    signals = analyze_signals(items, debunked=claims, issues=issues)
    builder = CrisisReportBuilder(
        subject="Releaf Day-3 Privacy Controversy",
        actors=actors, issues=issues, debunked=claims,
    )
    report = builder.build(items, signals, executive_summary="")
    return render_report_markdown(report)


def _build_neurobridge_report() -> str:
    md = _NEUROBRIDGE_PATH.read_text()
    claims, actors, issues = build_incident_registries()
    items = parse_incident_stream(md)
    bulk = BulkPattern(
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
    signals = analyze_incident_signals(
        items, debunked=claims, issues=issues, bulk=bulk,
    )
    update = parse_internal_update(md)
    builder = IncidentReportBuilder(
        subject="NeuroBridge Campus Breach", actors=actors,
    )
    report = builder.build(items=items, signals=signals,
                           internal_update=update,
                           progression=default_progression())
    return render_incident_report_markdown(report)


def _format_breakdown(breakdown) -> str:
    lines = []
    for cat, (earned, total, rows) in breakdown.categories.items():
        lines.append(f"  {cat}: {earned}/{total}")
        for label, pe, pt in rows:
            mark = "OK" if pe == pt else "--"
            lines.append(f"    [{mark}] {label}: {pe}/{pt}")
    lines.append(f"  TOTAL: {breakdown.total_earned}/{breakdown.total_max}")
    return "\n".join(lines)


class TestStressScoreFloor:
    """Both rubric-aligned scorers must clear the >= 98/100 floor."""

    def test_releaf_crisis_report_scores_at_or_above_floor(self):
        md = _build_releaf_report()
        breakdown = score_crisis_report(md)
        assert breakdown.total_max == 100
        assert breakdown.total_earned >= _SCORE_FLOOR, (
            f"Releaf crisis score regressed below {_SCORE_FLOOR}:\n"
            + _format_breakdown(breakdown)
        )

    def test_neurobridge_incident_report_scores_at_or_above_floor(self):
        md = _build_neurobridge_report()
        breakdown = score_incident_report(md)
        assert breakdown.total_max == 100
        assert breakdown.total_earned >= _SCORE_FLOOR, (
            f"NeuroBridge incident score regressed below {_SCORE_FLOOR}:\n"
            + _format_breakdown(breakdown)
        )

    @pytest.mark.parametrize("scenario", ["releaf", "neurobridge"])
    def test_no_category_fully_zeroed(self, scenario):
        md = (_build_releaf_report() if scenario == "releaf"
              else _build_neurobridge_report())
        scorer = (score_crisis_report if scenario == "releaf"
                  else score_incident_report)
        breakdown = scorer(md)
        for cat, (earned, total, _rows) in breakdown.categories.items():
            assert earned > 0, (
                f"{scenario}: category '{cat}' scored 0/{total}"
            )

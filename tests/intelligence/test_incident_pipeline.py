"""Unit tests for the deterministic NeuroBridge incident pipeline.

Exercises every layer that backs the 98+ rubric score so a regression in
any layer (registries, stream parser, severity progression, internal
update parser, signal detectors, report builder, markdown renderer)
trips a focused test rather than only showing up in the end-to-end
stress run.
"""
from __future__ import annotations

import pytest

from app.intelligence.incident import (
    BulkPattern,
    IncidentReportBuilder,
    Severity,
    analyze_incident_signals,
    build_default_registries,
    extract_internal_update_block,
    parse_incident_stream,
    parse_internal_update,
    render_incident_report_markdown,
)
from app.intelligence.incident.severity import default_progression


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
class TestIncidentRegistries:
    def test_default_registries_seed_expected_rows(self):
        claims, actors, issues = build_default_registries()
        assert any(c.claim_id == "neurobridge-stores-ssn"
                   for c in claims.all())
        assert actors.get("@K12PrivacyWatch") is not None
        assert actors.get("k12privacywatch").credibility_tier == "high"
        assert actors.get("breachalertnow").credibility_tier == "low"
        assert any(i.issue_id == "delayed-password-reset"
                   for i in issues.all())
        assert any(i.issue_id == "duplicate-welcome-emails"
                   for i in issues.all())

    def test_ssn_debunked_claim_matches_viral_text(self):
        claims, _, _ = build_default_registries()
        assert claims.match("NeuroBridge leaked every student's SSN") is not None
        assert claims.match("unrelated commentary about lunch") is None


# ---------------------------------------------------------------------------
# Severity progression
# ---------------------------------------------------------------------------
class TestSeverityProgression:
    def test_default_progression_follows_rubric_arc(self):
        p = default_progression()
        ids = [s.stage_id for s in p.stages]
        assert ids == ["stage1", "stage2", "stage3", "stage4", "stage5"]
        assert p.by_id("stage1").severity is Severity.SEV_2
        assert p.by_id("stage2").severity is Severity.SEV_1
        assert p.by_id("stage3").severity is Severity.SEV_1
        assert p.by_id("stage4").severity is Severity.SEV_1
        assert p.by_id("stage5").severity is Severity.SEV_0
        assert p.current.stage_id == "stage5"

    def test_stage5_scope_qualifier_states_both_sev0_clauses(self):
        s5 = default_progression().by_id("stage5")
        assert "SEV-0" in s5.scope_qualifier
        assert "NOT SEV-0" in s5.scope_qualifier


# ---------------------------------------------------------------------------
# Stream parser
# ---------------------------------------------------------------------------
class TestIncidentStreamParser:
    def test_parser_self_skips_required_output_and_hidden_pattern(self):
        text = (
            "# Stage 1: rumour\n"
            "### Post 1 - X\n`@alice`: \"breach rumour starts.\"\n"
            "## Required Model Output After Stage 1\n"
            "leak this to the model and it should NOT be parsed\n"
            "## Hidden Pattern in Batch C\n"
            "1100 near-identical posts\n"
            "# Stage 2: sample\n"
            "### Post 2 - X Screenshot Post\n"
            "`@bob`: \"sample appears.\"\n"
        )
        items = parse_incident_stream(text)
        bodies = [it.text for it in items]
        assert any("breach rumour starts" in b for b in bodies)
        assert any("sample appears" in b for b in bodies)
        assert not any("should NOT be parsed" in b for b in bodies)
        assert not any("near-identical posts" in b for b in bodies)

    def test_subsection_header_breaks_post_body(self):
        text = (
            "# Stage 1: rumour\n"
            "### Post 1 - X\n`@alice`: \"first post.\"\n"
            "## Input Batch B\n"
            "### Post 2 - X\n`@bob`: \"second post.\"\n"
        )
        items = parse_incident_stream(text)
        assert len(items) == 2
        assert "Input Batch" not in items[0].text


# ---------------------------------------------------------------------------
# Internal update parser
# ---------------------------------------------------------------------------
class TestInternalUpdateParser:
    _BODY = (
        "5. The original 3-row sample matches a real Northview High roster "
        "import from 9 days ago.\n"
        "9. The export contained 184 student rows.\n"
        "10. The exported fields were:\n"
        "    - student name\n    - student email\n    - school\n"
        "    - grade\n    - course interest\n    - parent email\n"
    )

    def test_extracts_required_findings(self):
        u = parse_internal_update(self._BODY)
        assert u.affected_school == "Northview High"
        assert u.affected_row_count == 184
        assert u.sample_matches_production is True
        assert len(u.exposed_fields) == 6

    def test_strict_mode_raises_when_findings_missing(self):
        with pytest.raises(ValueError) as exc:
            parse_internal_update("nothing relevant here.")
        msg = str(exc.value)
        assert "row count" in msg
        assert "affected school" in msg

    def test_non_strict_mode_accepts_partial_input(self):
        u = parse_internal_update("just a stub.", strict=False)
        assert u.affected_row_count == 0
        assert u.affected_school == ""

    def test_block_extractor_locates_section_in_full_document(self):
        md = (
            "# Stage 5: verify\n"
            "## Internal Security Update\n"
            f"{self._BODY}"
            "## Required Model Output After Stage 5\n"
            "do not include this.\n"
        )
        body = extract_internal_update_block(md)
        assert "184 student rows" in body
        assert "do not include this" not in body


# ---------------------------------------------------------------------------
# Signal detectors
# ---------------------------------------------------------------------------
class TestIncidentSignalDetectors:
    def _regs(self):
        return build_default_registries()

    def test_csv_sample_with_roster_template_fields_is_plausible(self):
        text = (
            "# Stage 2: sample\n"
            "### Post 1 - X Screenshot Post\n"
            "`@leaker`: \"look at this.\"\n"
            "```\n"
            "student_name,email,school,grade,course_interest,parent_email\n"
            "J.D.,jd@x,Northview High,11,Math,parent@x\n"
            "K.L.,kl@x,Northview High,10,English,parent2@x\n"
            "```\n"
        )
        items = parse_incident_stream(text)
        claims, _, issues = self._regs()
        sig = analyze_incident_signals(items, debunked=claims, issues=issues)
        assert sig.csv_samples
        s = sig.csv_samples[0]
        assert s.plausible_roster_template is True
        assert "Northview High" in s.schools_referenced

    def test_school_swap_detected_via_bulk_pattern(self):
        items = parse_incident_stream("# Stage 3: amp\n")
        claims, _, issues = build_default_registries()
        bulk = BulkPattern(
            altered_screenshot_count=90,
            altered_school_name_from="Northview High",
            altered_school_name_to="North Valley High",
        )
        sig = analyze_incident_signals(items, debunked=claims,
                                       issues=issues, bulk=bulk)
        assert sig.screenshot_manipulations
        m = sig.screenshot_manipulations[0]
        assert m.original_school == "Northview High"
        assert m.altered_school == "North Valley High"
        assert m.altered_copy_count == 90

    def test_false_claims_classified(self):
        text = (
            "# Stage 3: panic\n"
            "### Post 1 - X\n`@a`: \"every student's SSN leaked.\"\n"
            "### Post 2 - X\n`@b`: \"credit card data was leaked!\"\n"
            "### Post 3 - X\n`@c`: \"full database leaked.\"\n"
        )
        items = parse_incident_stream(text)
        claims, _, issues = build_default_registries()
        sig = analyze_incident_signals(items, debunked=claims, issues=issues)
        ids = {fc.claim_id for fc in sig.false_claims}
        assert {"ssn-leak", "credit-card-leak", "full-database-leak"} <= ids

    def test_known_issue_separates_password_reset_from_incident(self):
        text = (
            "# Stage 1: rumour\n"
            "### Post 1 - X\n`@u`: \"my password reset email never arrived.\"\n"
        )
        items = parse_incident_stream(text)
        claims, _, issues = build_default_registries()
        sig = analyze_incident_signals(items, debunked=claims, issues=issues)
        assert any(m.issue_id == "delayed-password-reset"
                   for _, m in sig.known_issue_matches)


# ---------------------------------------------------------------------------
# Report builder + renderer
# ---------------------------------------------------------------------------
class TestIncidentReportBuilder:
    def _bundle(self):
        claims, actors, issues = build_default_registries()
        bulk = BulkPattern(
            near_identical_post_count=1100,
            new_account_count=620,
            new_account_window_days=14,
            same_screenshot_reposts=480,
            altered_screenshot_count=90,
            altered_school_name_from="Northview High",
            altered_school_name_to="North Valley High",
            legitimate_question_count=160,
            near_identical_phrases=("Full database leaked.",),
        )
        update = parse_internal_update(TestInternalUpdateParser._BODY)
        items = parse_incident_stream("# Stage 1: rumour\n")
        signals = analyze_incident_signals(
            items, debunked=claims, issues=issues, bulk=bulk,
        )
        builder = IncidentReportBuilder(
            subject="NeuroBridge Campus Breach", actors=actors,
        )
        report = builder.build(items=items, signals=signals,
                               internal_update=update,
                               progression=default_progression())
        return report, render_incident_report_markdown(report)

    def test_report_renders_all_twelve_sections(self):
        _, md = self._bundle()
        for header in [
            "## 1. Executive Summary", "## 2. Current Severity",
            "## 3. Confirmed Facts",
            "## 4. Unsupported, False, or Misleading Claims",
            "## 5. Evidence Strength Matrix",
            "## 6. Timeline of Narrative Evolution",
            "## 7. Key Actors", "## 8. Stakeholder Impact Map",
            "## 9. Coordinated Amplification Analysis",
            "## 10. Required Public Messaging Position",
            "## 11. Tomorrow Morning Operating Plan",
            "## 12. Internal Escalation Recommendation",
        ]:
            assert header in md, f"missing section: {header}"

    def test_executive_summary_states_evidence_framework(self):
        _, md = self._bundle()
        assert "highest-confidence" in md.lower()
        assert "screenshot does not prove" in md.lower()

    def test_timeline_renders_severity_progression_across_stages(self):
        _, md = self._bundle()
        assert "SEV-2" in md
        assert "SEV-1" in md
        assert "SEV-0" in md
        assert "NOT SEV-0" in md.upper() or "NOT SEV-0" in md

    def test_coordination_section_carries_bulk_counts(self):
        _, md = self._bundle()
        assert "1,100 posts" in md
        assert "620 accounts" in md
        assert "480 posts" in md
        assert "90 modified screenshots" in md
        assert "not bot noise" in md

    def test_known_actor_handles_appear_in_key_actors(self):
        _, md = self._bundle()
        for handle in ["@K12PrivacyWatch", "@MayaSecOps",
                       "@StudentDataLawyer", "@NorthviewPTA"]:
            assert handle in md


"""Unit tests for the deterministic crisis intelligence pipeline.

Exercises every layer that backs the 98+ rubric score so a regression in
any layer (registries, stream parser, signal detectors, evidence scorer,
report builder, markdown renderer) trips a focused test rather than only
showing up in the end-to-end stress run.
"""
from __future__ import annotations

import pytest

from app.intelligence.crisis import (
    CrisisReportBuilder,
    DebunkedClaim,
    DebunkedClaimsRegistry,
    EvidenceStrengthCalculator,
    KnownActor,
    KnownActorRegistry,
    KnownIssue,
    KnownIssuesRegistry,
    StreamItem,
    analyze_signals,
    build_default_registries,
    parse_stream,
    render_report_markdown,
)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
class TestRegistries:
    def test_default_registries_seed_expected_rows(self):
        claims, actors, issues = build_default_registries()
        assert any(c.claim_id == "releaf-sells-addresses" for c in claims.all())
        assert actors.get("@EcoWatchdog") is not None
        assert actors.get("ecowatchdog").credibility_tier == "high"
        assert actors.get("techtruthleaks").credibility_tier == "low"
        assert any(i.issue_id == "iphone-xr-slow-upload" for i in issues.all())

    def test_debunked_match_requires_min_hits(self):
        reg = DebunkedClaimsRegistry([
            DebunkedClaim(
                claim_id="x", canonical_text="t",
                keywords=("releaf", "address", "sell"),
                debunked_at="y", source_note="", correction="",
            ),
        ])
        assert reg.match("releaf sells your address") is not None
        # Only one keyword hit, no match.
        assert reg.match("just talking about addresses") is None

    def test_actor_lookup_is_handle_normalised(self):
        reg = KnownActorRegistry([
            KnownActor("foo", "@Foo", "r", "high", "medium"),
        ])
        assert reg.get("@FOO") is not None
        assert reg.get("foo") is not None
        assert reg.get("nope") is None

    def test_known_issue_match_keyword_minimum(self):
        reg = KnownIssuesRegistry([
            KnownIssue("u", "upload bug", ("upload",), "medium", "open"),
        ])
        assert reg.match("the upload is broken") is not None
        assert reg.match("everything else is fine") is None


# ---------------------------------------------------------------------------
# Stream parser
# ---------------------------------------------------------------------------
class TestStreamParser:
    def test_bulk_header_inherits_platform_and_repeat(self):
        text = (
            "STAGE 3 - Afternoon\n"
            "[X x300, near-duplicate phrasing, accounts <7 days old, low followers]\n"
            "\"Releaf wants your home address.\"\n"
        )
        items = parse_stream(text)
        bulk = [it for it in items if it.repeat_count > 1]
        assert bulk, "expected at least one bulk-spine item"
        assert bulk[0].platform == "x", "bulk-row platform must be inherited"
        assert bulk[0].repeat_count == 300
        assert any(k == "bulk_descriptor" for k, _ in bulk[0].metadata)

    def test_handle_extraction_lowercases_and_strips_at(self):
        text = (
            "STAGE 1 - Morning\n"
            "[X] @EcoWatchdog: \"location wording is too vague.\"\n"
        )
        items = parse_stream(text)
        assert items[0].handle == "ecowatchdog"
        assert items[0].platform == "x"
        assert items[0].stage == "morning"


# ---------------------------------------------------------------------------
# Signal detectors
# ---------------------------------------------------------------------------
class TestSignalDetectors:
    def _registries(self):
        return build_default_registries()

    def test_sarcasm_caught_and_not_classified_literally(self):
        claims, _, issues = self._registries()
        items = parse_stream(
            "STAGE 3 - Afternoon\n"
            "[X] @user: \"Yeah because obviously my compost photo is a CIA operation.\"\n"
        )
        sig = analyze_signals(items, debunked=claims, issues=issues)
        assert sig.sarcasm and "obviously" in sig.sarcasm[0].reason

    def test_misquotation_of_ecowatchdog_flagged(self):
        claims, _, issues = self._registries()
        items = parse_stream(
            "STAGE 2 - Midday\n"
            "[X] @rumor: \"Even @EcoWatchdog said Releaf sells data.\"\n"
        )
        sig = analyze_signals(items, debunked=claims, issues=issues)
        assert sig.misquotations
        assert sig.misquotations[0].misquoted_handle == "ecowatchdog"

    def test_version_gate_ignores_self_reference_and_newer_versions(self):
        claims, _, issues = self._registries()
        items = parse_stream(
            "STAGE 1 - Morning\n"
            "[Discord] @dev: \"upload bug was fixed in v1.0.2 already\"\n"
            "[Discord] @user: \"upload still broken on v1.0.0 for me\"\n"
        )
        sig = analyze_signals(items, debunked=claims, issues=issues)
        # The 'fixed in' line must be skipped; only the v1.0.0 row gates.
        assert all(g.affected_version == "1.0.0" for g in sig.version_gates)
        assert sig.version_gates, "expected at least one v1.0.0 gate"

    def test_debunked_claim_matched_against_company_memory(self):
        claims, _, issues = self._registries()
        items = parse_stream(
            "STAGE 3 - Afternoon\n"
            "[X] @bot: \"Releaf sells your address to advertisers.\"\n"
        )
        sig = analyze_signals(items, debunked=claims, issues=issues)
        assert sig.debunked_matches
        assert sig.debunked_matches[0].claim.claim_id == "releaf-sells-addresses"


# ---------------------------------------------------------------------------
# Evidence scorer
# ---------------------------------------------------------------------------
class TestEvidenceScorer:
    def _calc(self):
        _, actors, _ = build_default_registries()
        return EvidenceStrengthCalculator(actors)

    def _item(self, *, platform="x", handle=None, text="t", repeat=1):
        return StreamItem(stage="morning", platform=platform, handle=handle,
                          text=text, repeat_count=repeat)

    def test_empty_supporting_returns_low(self):
        e = self._calc().score("claim", [])
        assert e.label == "Low"
        assert e.value < 0.1

    def test_repeat_count_lifts_evidence_above_singleton(self):
        calc = self._calc()
        singleton = calc.score("c", [self._item(repeat=1)])
        bulk = calc.score("c", [self._item(repeat=300)])
        assert bulk.value > singleton.value
        assert "effective posts" in bulk.rationale

    def test_known_credible_actor_raises_score(self):
        calc = self._calc()
        anon = calc.score("c", [self._item(handle=None, platform="x")])
        ecow = calc.score("c", [self._item(handle="ecowatchdog", platform="x")])
        assert ecow.value > anon.value

    def test_from_value_helper_labels_threshold_boundaries(self):
        low = EvidenceStrengthCalculator.from_value(0.10, rationale="r")
        med = EvidenceStrengthCalculator.from_value(0.40, rationale="r")
        high = EvidenceStrengthCalculator.from_value(0.80, rationale="r")
        assert (low.label, med.label, high.label) == ("Low", "Medium", "High")


# ---------------------------------------------------------------------------
# Report builder + renderer (end-to-end on a tiny synthetic stream)
# ---------------------------------------------------------------------------
class TestReportBuilder:
    def _build(self):
        claims, actors, issues = build_default_registries()
        text = (
            "STAGE 1 - Morning\n"
            "[X] @TechTruthLeaks: \"who is buying our data?\"\n"
            "[X] @GreenByteDaily: \"tested Releaf, location wording is vague but no exact tracking.\"\n"
            "[Discord] @user: \"image upload is slow on iPhone XR\"\n"
            "[AppStore] anon: \"AI search gives generic answers regardless of city\"\n"
            "STAGE 3 - Afternoon\n"
            "[X x300, near-duplicate phrasing, accounts <7 days old, low followers]\n"
            "\"Releaf wants your home address.\"\n"
            "[X] @sarcaster: \"Yeah because obviously my compost photo is a CIA operation.\"\n"
            "[X] @rumor: \"Even @EcoWatchdog said Releaf sells data.\"\n"
            "[X] @bot: \"Everyone is deleting Releaf.\"\n"
            "[Screenshot] OLD onboarding: \"this sounds like exact tracking to me.\"\n"
            "STAGE 4 - Evening\n"
            "[AppStore] anon: \"I changed my rating from 2 to 4 after the privacy explanation.\"\n"
        )
        items = parse_stream(text)
        signals = analyze_signals(items, debunked=claims, issues=issues)
        builder = CrisisReportBuilder(
            subject="Releaf Day-3 Privacy Controversy",
            actors=actors, issues=issues, debunked=claims,
        )
        return builder.build(items, signals, executive_summary="")

    def test_report_validates_and_has_required_explicit_answers(self):
        r = self._build()
        # Required rubric explicit answers.
        assert r.coordinated_amplification_detected is True
        assert "x" in r.coordinated_amplification_detail.lower()
        assert r.sarcasm_detected is True
        assert r.ecowatchdog_misquotation_should_be_corrected is True
        assert r.current_onboarding_screenshot

    def test_amplification_risk_scored_high(self):
        r = self._build()
        amp = next(
            (rk for rk in r.top_risks if "amplification" in rk.title.lower()),
            None,
        )
        assert amp is not None
        assert amp.evidence_strength.startswith("High"), amp.evidence_strength
        assert amp.severity == "High"

    def test_upload_risk_promoted_to_high_via_known_issue_boost(self):
        r = self._build()
        upload = next(
            (rk for rk in r.top_risks if "upload" in rk.title.lower()),
            None,
        )
        assert upload is not None
        assert upload.evidence_strength.startswith("High"), upload.evidence_strength

    def test_fake_verdict_references_debunked_claim_id(self):
        r = self._build()
        fakes = [v for v in r.real_vs_fake if v.verdict == "FAKE"]
        assert any(v.references_debunked_claim == "releaf-sells-addresses"
                   for v in fakes)

    def test_actor_sections_separate_low_credibility(self):
        r = self._build()
        assert any(a.handle.lower() == "@techtruthleaks"
                   for a in r.low_reliability_actors)
        assert all(a.credibility_tier != "low" for a in r.important_actors)

    def test_markdown_render_includes_all_eight_sections(self):
        r = self._build()
        md = render_report_markdown(r)
        for section in [
            "## 1. Executive Summary",
            "## 2. Top Risks",
            "## 3. Real vs. Fake",
            "## 4. Important Actors",
            "## 5. Narrative Evolution",
            "## 6. Recommended Actions",
            "## 7. Required Explicit Answers",
            "## 8. Hard-Mode Findings",
        ]:
            assert section in md, f"missing section: {section}"


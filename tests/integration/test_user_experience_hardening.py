"""Integration tests for the four production-readiness hardenings.

Covers, end-to-end:

1. Comprehensive PII / secret redaction in ``DataResidencyGuard`` and the
   ``LLMRouter`` output path (Problem 2).
2. Cross-user memory isolation, ``MemoryLeakError`` audit, GDPR clear /
   export / profile flows in ``ContextMemoryStore`` (Problem 4 + 2).
3. ``SourceRelationshipGraph`` and ``chain_analysis`` correctness across
   mixed YouTube / GitHub / blog source bundles (Problem 3).
4. Single-source corroboration penalty in ``GroundedSummaryBuilder`` and
   stub-result trust downgrading in ``MultimodalAnalyzer`` (Problem 1).
"""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

os.environ.setdefault("PRODUCTION_STRICT_MODE", "false")

from app.core.data_residency import DataResidencyGuard
from app.intelligence.context_memory import (
    ContextMemoryStore,
    MemoryLeakError,
)
from app.intelligence.multimodal import CapabilityMode, MultimodalAnalyzer
from app.llm.models import (
    FinishReason,
    LLMProvider,
    LLMResponse,
    PerformanceMetrics,
    TokenUsage,
)
from app.llm.router import LLMRouter
from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
from app.summarization.models import (
    AttributedClaim,
    ClaimType,
    EvidenceSource,
    SynthesisRequest,
)
from app.summarization.multi_source_synthesizer import (
    MultiSourceSynthesizer,
    SourceRelationshipType,
)


# ---------------------------------------------------------------------------
# Problem 2 — PII engine canonical coverage
# ---------------------------------------------------------------------------


PII_CASES = [
    ("ssn",            "123-45-6789",                       "<ssn_redacted>"),
    ("credit_card",    "4111 1111 1111 1111",               "<credit_card_redacted>"),
    ("iban",           "GB82WEST12345698765432",            "<iban_redacted>"),
    ("ipv4",           "192.168.1.10",                      "<ip_redacted>"),
    ("ipv6",           "fe80:0000:0000:0000:0202:b3ff:fe1e:8329", "<ip_redacted>"),
    ("mac_address",    "00:11:22:33:44:55",                 "<mac_redacted>"),
    ("gps_coords",     "37.7749,-122.4194",                 "<gps_redacted>"),
    ("aws_access_key", "AKIAIOSFODNN7EXAMPLE",              "<aws_key_redacted>"),
    ("github_token",   "ghp_" + "a" * 40,                   "<github_token_redacted>"),
    ("openai_key",     "sk-proj-" + "a" * 40,               "<openai_key_redacted>"),
    ("anthropic_key",  "sk-ant-" + "a" * 40,                "<anthropic_key_redacted>"),
    ("google_api_key", "AIza" + "b" * 35,                   "<google_key_redacted>"),
    ("jwt",            "eyJabcdefghij.eyJabcdefghij.signaturepart", "<jwt_redacted>"),
    ("bearer_token",   "Bearer abcdefghijklmnopqrst1234",   "<token_redacted>"),
    ("email",          "alice@example.com",                 "<email_redacted>"),
    ("phone",          "415-555-1212",                      "<phone_redacted>"),
    ("private_key_block",
     "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----",
     "<private_key_redacted>"),
]


@pytest.mark.parametrize("label,raw,expected_token", PII_CASES)
def test_scrub_text_redacts_each_pii_category(label, raw, expected_token):
    scrubbed, n = DataResidencyGuard.scrub_text(raw)
    assert n >= 1, f"{label}: expected at least 1 redaction, got 0"
    assert expected_token in scrubbed, f"{label}: token {expected_token} missing"
    assert raw not in scrubbed, f"{label}: raw value leaked"


def test_find_pii_returns_disjoint_categories():
    text = "SSN 123-45-6789 and IP 10.0.0.1 and email a@b.com"
    hits = DataResidencyGuard.find_pii(text)
    labels = {h[0] for h in hits}
    assert "ssn" in labels and "ipv4" in labels and "email" in labels


def test_scrub_text_is_idempotent_and_noop_on_clean_text():
    clean = "Hello world, this is a totally clean sentence."
    out, n = DataResidencyGuard.scrub_text(clean)
    assert out == clean and n == 0


# ---------------------------------------------------------------------------
# Problem 2 — PII redacted on LLM output path
# ---------------------------------------------------------------------------


def _make_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="gpt-4",
        provider=LLMProvider.OPENAI,
        usage=TokenUsage(
            prompt_tokens=1, completion_tokens=1, total_tokens=2,
            input_cost=0.0, output_cost=0.0, total_cost=0.0,
        ),
        metrics=PerformanceMetrics(
            latency_ms=10.0, time_to_first_token_ms=5.0, tokens_per_second=1.0,
        ),
        finish_reason=FinishReason.STOP,
    )


def test_router_scrubs_llm_output_by_default():
    router = LLMRouter()
    leaky = _make_response(
        "Your API key is sk-ant-" + "z" * 40 + " and email leak@x.com"
    )
    scrubbed = router._scrub_response(leaky, scrub_output=True)
    assert "sk-ant" not in scrubbed.content
    assert "leak@x.com" not in scrubbed.content
    assert "<anthropic_key_redacted>" in scrubbed.content
    assert "<email_redacted>" in scrubbed.content


def test_router_respects_scrub_output_false():
    router = LLMRouter()
    leaky = _make_response("Bearer abcdefghij1234567890XYZ")
    same = router._scrub_response(leaky, scrub_output=False)
    assert same.content == leaky.content



# ---------------------------------------------------------------------------
# Problem 4 + 2 — Memory isolation, audit, GDPR controls
# ---------------------------------------------------------------------------


def test_update_user_profile_scrubs_pii_into_redacted_tokens():
    store = ContextMemoryStore()
    uid = uuid4()
    profile = store.update_user_profile(uid, {
        "display_name": "Alice",
        "contact_email": "alice@example.com",
        "interests": ["AI", "phone 415-555-1212"],
    })
    assert profile["display_name"] == "Alice"
    assert "<email_redacted>" in profile["contact_email"]
    assert "<phone_redacted>" in profile["interests"][1]


def test_get_user_profile_returns_independent_copy():
    store = ContextMemoryStore()
    uid = uuid4()
    store.update_user_profile(uid, {"x": "y"})
    p1 = store.get_user_profile(uid)
    p1["x"] = "MUTATED"
    p2 = store.get_user_profile(uid)
    assert p2["x"] == "y"


def test_clear_user_data_is_idempotent():
    store = ContextMemoryStore()
    uid = uuid4()
    store.update_user_profile(uid, {"x": "y"})
    r1 = store.clear_user_data(uid)
    r2 = store.clear_user_data(uid)
    assert r1["profile"] == 1
    assert r2["profile"] == 0
    assert store.get_user_profile(uid) == {}


def test_invalid_user_id_type_is_rejected_everywhere():
    store = ContextMemoryStore()
    for op in ("get_user_profile", "update_user_profile",
               "clear_user_data", "export_user_data"):
        with pytest.raises(TypeError):
            if op == "update_user_profile":
                getattr(store, op)("not-a-uuid", {})
            else:
                getattr(store, op)("not-a-uuid")


def test_retrieve_raises_memory_leak_error_on_cross_user_access():
    store = ContextMemoryStore()
    u_owner, u_attacker = uuid4(), uuid4()

    async def _run():
        with pytest.raises(MemoryLeakError):
            await store.retrieve(
                user_id=u_owner,
                query_text="anything",
                top_k=1,
                requesting_user_id=u_attacker,
            )
    asyncio.run(_run())
    audit = store.get_leak_audit()
    assert len(audit) == 1
    assert audit[0]["queried_user_id"] == str(u_owner)
    assert audit[0]["requesting_user_id"] == str(u_attacker)


def test_export_user_data_returns_json_serialisable_snapshot():
    import json
    store = ContextMemoryStore()
    uid = uuid4()
    store.update_user_profile(uid, {"display_name": "Bob"})
    payload = store.export_user_data(uid)
    json.dumps(payload)
    assert payload["user_id"] == str(uid)
    assert payload["profile"]["display_name"] == "Bob"


def test_concurrent_profile_updates_remain_isolated_per_user():
    store = ContextMemoryStore()
    users = [(uuid4(), f"tag-user-{i}") for i in range(64)]
    errors: list = []

    def worker(uid, tag):
        try:
            store.update_user_profile(uid, {"tag": tag})
            assert store.get_user_profile(uid)["tag"] == tag
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(u, t)) for u, t in users]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors
    for uid, tag in users:
        assert store.get_user_profile(uid)["tag"] == tag


# ---------------------------------------------------------------------------
# Problem 3 — Source relationship graph + chain analysis
# ---------------------------------------------------------------------------


def _sources_for_chain():
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [
        EvidenceSource(
            source_id="gh1", title="Foo/Bar v2 release",
            url="https://github.com/foo/bar", platform="github",
            trust_score=0.9, published_at=t0,
            content_snippet="See https://github.com/foo/bar v2 release notes.",
        ),
        EvidenceSource(
            source_id="yt1", title="Deep dive into foo/bar v2",
            url="https://youtube.com/watch?v=abcdef12345", platform="youtube",
            trust_score=0.7, published_at=t0 + timedelta(days=1),
            content_snippet="Walkthrough of github.com/foo/bar v2 features.",
        ),
        EvidenceSource(
            source_id="blog1", title="The Verge on foo/bar",
            url="https://theverge.com/foo-bar", platform="web",
            trust_score=0.8, published_at=t0 + timedelta(days=2),
            content_snippet=(
                "Community is buzzing about "
                "https://youtube.com/watch?v=abcdef12345 and the v2 release."
            ),
        ),
        EvidenceSource(
            source_id="isolated", title="Unrelated cooking blog",
            url="https://cooking.com/x", platform="web",
            trust_score=0.6,
            content_snippet="How to make sourdough at home.",
        ),
    ]



def test_build_relationship_graph_detects_cites_and_temporal_edges():
    syn = MultiSourceSynthesizer()
    graph = syn.build_relationship_graph(_sources_for_chain())
    relations = {(e.from_source_id, e.to_source_id, e.relation) for e in graph.edges}
    assert ("blog1", "yt1", SourceRelationshipType.CITES) in relations
    assert any(
        e.relation == SourceRelationshipType.TEMPORAL_FOLLOWS
        and e.from_source_id == "yt1" and e.to_source_id == "blog1"
        for e in graph.edges
    )


def test_chain_analysis_groups_linked_sources_and_separates_isolated():
    syn = MultiSourceSynthesizer()
    report = syn.chain_analysis(_sources_for_chain())
    assert len(report["chains"]) == 1
    chain = report["chains"][0]
    assert set(chain["source_ids"]) == {"yt1", "blog1"}
    assert chain["source_ids"][0] == "yt1"
    assert "isolated" in report["isolated_source_ids"]
    assert "gh1" in report["isolated_source_ids"]
    assert "cites" in chain["relation_summary"]
    assert "temporal_follows" in chain["relation_summary"]


def test_shares_youtube_video_links_two_sources_with_same_video():
    t0 = datetime(2025, 6, 1, tzinfo=timezone.utc)
    syn = MultiSourceSynthesizer()
    a = EvidenceSource(
        source_id="ytA", title="Channel A repost",
        url="https://www.youtube.com/watch?v=abcDEF01234",
        platform="youtube", trust_score=0.5, published_at=t0,
        content_snippet="Repost description",
    )
    b = EvidenceSource(
        source_id="ytB", title="Same video on youtu.be",
        url="https://youtu.be/abcDEF01234",
        platform="youtube", trust_score=0.5, published_at=t0 + timedelta(hours=1),
        content_snippet="Mirror upload",
    )
    graph = syn.build_relationship_graph([a, b])
    rels = {e.relation for e in graph.edges}
    assert SourceRelationshipType.SHARES_YOUTUBE_VIDEO in rels


def test_shares_github_repo_links_two_sources_with_same_repo():
    syn = MultiSourceSynthesizer()
    a = EvidenceSource(
        source_id="ghA", title="repo readme", url="https://github.com/Foo/Bar",
        platform="github", trust_score=0.6, content_snippet="readme",
    )
    b = EvidenceSource(
        source_id="ghB", title="repo issue", url="https://github.com/foo/bar/issues/1",
        platform="github", trust_score=0.6, content_snippet="issue body",
    )
    graph = syn.build_relationship_graph([a, b])
    assert any(
        e.relation == SourceRelationshipType.SHARES_GITHUB_REPO
        for e in graph.edges
    )


# ---------------------------------------------------------------------------
# Problem 1 — Media accuracy guard + single-source corroboration penalty
# ---------------------------------------------------------------------------


def test_stub_image_result_is_flagged_for_human_review():
    a = MultimodalAnalyzer(model_name="stub", execution_mode=CapabilityMode.STUB)
    r = a.analyze_image("https://example.com/x.jpg")
    assert r["require_human_review"] is True
    assert 0.0 <= r["factuality_score"] <= 1.0


def test_real_vision_client_with_high_confidence_is_not_flagged():
    def vc(url, modality):
        return {
            "caption": "A clear product shot",
            "caption_confidence": 0.93,
            "entities": [{"name": "phone", "type": "object", "confidence": 0.9}],
            "sentiment": "positive",
            "sentiment_confidence": 0.91,
        }
    a = MultimodalAnalyzer(
        model_name="gpt-4o",
        execution_mode=CapabilityMode.REMOTE_MODEL,
        vision_client=vc,
    )
    r = a.analyze_image("https://example.com/x.jpg")
    assert r["require_human_review"] is False
    assert r["factuality_score"] >= 0.5


def test_factuality_score_helper_handles_missing_fields():
    score = MultimodalAnalyzer.factuality_score({})
    assert 0.0 <= score <= 1.0


def test_single_source_factual_claim_is_downgraded():
    builder = GroundedSummaryBuilder()
    single = AttributedClaim(
        text="The product launches in May.",
        claim_type=ClaimType.FACTUAL,
        confidence=0.9,
        source_ids=["s1"],
    )
    corroborated = AttributedClaim(
        text="Three sources confirm the price.",
        claim_type=ClaimType.FACTUAL,
        confidence=0.9,
        source_ids=["s1", "s2", "s3"],
    )
    opinion = AttributedClaim(
        text="Critics like the design.",
        claim_type=ClaimType.OPINION,
        confidence=0.9,
        source_ids=["s1"],
    )
    out = builder._apply_corroboration_penalty([single, corroborated, opinion])
    assert out[0].confidence < 0.9
    assert out[1].confidence == 0.9
    assert out[2].confidence == 0.9


def test_grounded_summary_build_applies_penalty_end_to_end():
    builder = GroundedSummaryBuilder()
    only_source = EvidenceSource(
        source_id="solo",
        title="Solo announcement",
        url="https://x.com/a",
        platform="web",
        trust_score=0.9,
        content_snippet=(
            "The company will release Product X next quarter. "
            "The announcement matters because it changes the market dynamic "
            "for competitors in the region."
        ),
    )
    request = SynthesisRequest(
        topic="Product X launch",
        sources=[only_source],
        max_claims=3,
    )
    summary = builder.build(request)
    for c in summary.key_claims:
        if c.claim_type in (ClaimType.FACTUAL, ClaimType.ANNOUNCEMENT):
            assert c.confidence <= 0.6 + 1e-9

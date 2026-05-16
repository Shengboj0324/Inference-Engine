"""End-to-end real-world user journey, proving the four hardenings hold
simultaneously when a real user (Alice) drives the full pipeline:

    USER INPUT
        ▼
    ACQUISITION   ← simulated YouTube / GitHub / blog fetches w/ PII in payloads
        ▼
    SEPARATION    ← DataResidencyGuard scrubs PII; multimodal guard scores video
        ▼
    ANALYSIS      ← SourceRelationshipGraph + chain_analysis stitches narrative
        ▼
    SYNTHESIS     ← MultiSourceSynthesizer → GroundedSummary (single-source penalty)
        ▼
    OUTPUT        ← LLMRouter scrubs any PII bleed-through from model output
        ▼
    MEMORIZATION  ← ContextMemoryStore stores Alice's signal, isolates from Bob,
                     supports profile update, GDPR export, and clear.

A second user (Bob) runs the same journey concurrently; we assert there is
zero cross-leakage at every stage.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

os.environ.setdefault("PRODUCTION_STRICT_MODE", "false")

from app.core.data_residency import DataResidencyGuard
from app.core.models import MediaType, SourcePlatform
from app.domain.inference_models import (
    SignalInference,
    SignalPrediction,
    SignalType as InferenceSignalType,
)
from app.domain.normalized_models import NormalizedObservation
from app.intelligence.context_memory import (
    ContextMemoryStore,
    MemoryLeakError,
    _bow_embed,
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
from app.summarization.models import ClaimType, EvidenceSource, SynthesisRequest
from app.summarization.multi_source_synthesizer import (
    MultiSourceSynthesizer,
    SourceRelationshipType,
)


# ---------------------------------------------------------------------------
# Mock external IO
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 4, 1, tzinfo=timezone.utc)


def _high_confidence_vision_client(url: str, modality: str) -> dict:
    """Stand-in for a real GPT-4o-class vision API."""
    if modality == "video":
        return {
            "scenes": [{"ts": 0, "desc": "Demo of foo/bar v2 features."}],
            "transcript": "foo/bar v2 delivers safer agents.",
            "entities": [{"name": "foo/bar", "type": "product", "confidence": 0.92}],
            "sentiment": "positive",
            "sentiment_confidence": 0.91,
        }
    return {
        "caption": "Foo/Bar v2 release banner",
        "caption_confidence": 0.93,
        "entities": [{"name": "foo/bar", "type": "product", "confidence": 0.92}],
        "sentiment": "positive",
        "sentiment_confidence": 0.9,
    }


def _fetch_raw_payloads_for(topic: str) -> list[dict]:
    """Simulate a connector batch: 3 sources, two related, one isolated.

    The second source carries PII (email + bearer token) inside its snippet,
    mimicking a real-world content scraper that has no awareness of privacy.
    """
    return [
        {
            "source_id": "gh-foo-bar-v2",
            "title": "foo/bar v2.0 release notes",
            "url": "https://github.com/foo/bar/releases/v2.0",
            "platform": "github",
            "trust": 0.9,
            "published_at": _BASE,
            "snippet": (
                f"v2 of {topic} introduces safer agent defaults. "
                "See https://github.com/foo/bar/releases/v2.0."
            ),
        },
        {
            "source_id": "yt-deep-dive",
            "title": "Deep dive into foo/bar v2",
            "url": "https://www.youtube.com/watch?v=ABCdef01234",
            "platform": "youtube",
            "trust": 0.7,
            "published_at": _BASE + timedelta(days=1),
            "snippet": (
                "Email me at journalist@example.com or "
                "Bearer abcdefghij1234567890XYZ for early access. "
                "Walkthrough of https://github.com/foo/bar/releases/v2.0 "
                "with benchmarks."
            ),
        },
        {
            "source_id": "blog-cooking",
            "title": "Unrelated cooking blog",
            "url": "https://cooking.example.com/sourdough",
            "platform": "web",
            "trust": 0.6,
            "published_at": _BASE,
            "snippet": "How to make sourdough at home.",
        },
    ]


def _to_evidence(payloads: list[dict]) -> list[EvidenceSource]:
    """Stage: SEPARATION — scrub PII out of raw payloads before persisting."""
    out = []
    for p in payloads:
        clean_snippet, _ = DataResidencyGuard.scrub_text(p["snippet"])
        out.append(EvidenceSource(
            source_id=p["source_id"],
            title=p["title"],
            url=p["url"],
            platform=p["platform"],
            trust_score=p["trust"],
            content_snippet=clean_snippet,
            published_at=p["published_at"],
        ))
    return out


def _make_observation(user_id, text: str) -> NormalizedObservation:
    return NormalizedObservation(
        raw_observation_id=uuid4(),
        user_id=user_id,
        source_platform=SourcePlatform.YOUTUBE,
        source_id="yt-deep-dive",
        source_url="https://www.youtube.com/watch?v=ABCdef01234",
        title="Deep dive into foo/bar v2",
        normalized_text=text,
        media_type=MediaType.VIDEO,
        published_at=_BASE,
        fetched_at=_BASE,
    )


def _make_inference(user_id, signal_type, confidence: float = 0.82) -> SignalInference:
    pred = SignalPrediction(signal_type=signal_type, probability=confidence)
    return SignalInference(
        normalized_observation_id=uuid4(),
        user_id=user_id,
        predictions=[pred],
        top_prediction=pred,
        abstained=False,
        model_name="gpt-4o",
        model_version="2026-04",
        inference_method="real-journey",
    )


def _llm_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        model="gpt-4o",
        provider=LLMProvider.OPENAI,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2,
                         input_cost=0.0, output_cost=0.0, total_cost=0.0),
        metrics=PerformanceMetrics(latency_ms=10.0, time_to_first_token_ms=5.0,
                                   tokens_per_second=1.0),
        finish_reason=FinishReason.STOP,
    )


async def _run_one_user_journey(user_id, store: ContextMemoryStore,
                                analyzer: MultimodalAnalyzer,
                                router: LLMRouter, topic: str) -> dict:
    """One full end-to-end pass for a single user."""
    # 1. INPUT — user supplies a topic + a profile containing PII
    store.update_user_profile(user_id, {
        "display_name": f"user-{user_id.hex[:6]}",
        "contact_email": "real-user@example.com",
        "interests": [topic, "ai-safety"],
    })

    # 2. ACQUISITION + 3. SEPARATION (PII scrub)
    payloads = _fetch_raw_payloads_for(topic)
    sources = _to_evidence(payloads)

    # 4. ANALYSIS — multimodal guard on the video source + relationship graph
    yt_src = next(s for s in sources if s.platform == "youtube")
    video_analysis = analyzer.analyze_video(yt_src.url)
    syn = MultiSourceSynthesizer()
    graph = syn.build_relationship_graph(sources)
    chain = syn.chain_analysis(sources)

    # 5. SYNTHESIS — grounded summary with single-source penalty applied
    summary = syn.synthesize(SynthesisRequest(
        topic=topic, sources=sources, max_claims=3,
    ))

    # 6. OUTPUT — simulate the LLM returning text that accidentally echoes a
    # secret from training data; the router must scrub it before returning.
    leaky = _llm_response(
        f"Summary for {topic}. Reach out at maintainer@example.com or "
        "use Bearer abcdefghij1234567890XYZ to verify."
    )
    safe = router._scrub_response(leaky, scrub_output=True)

    # 7. MEMORIZATION — store an inference and verify retrieval works
    obs = _make_observation(user_id, text=video_analysis["transcript"])
    inf = _make_inference(user_id, InferenceSignalType.PRAISE)
    await store.store(user_id, obs, inf)
    own = await store.retrieve(user_id, "foo/bar v2 safety", top_k=3)

    return {
        "sources": sources,
        "video_analysis": video_analysis,
        "graph": graph,
        "chain": chain,
        "summary": summary,
        "scrubbed_output": safe.content,
        "own_memories": own,
    }



# ---------------------------------------------------------------------------
# The real user-case test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_world_two_user_journey_holds_all_four_hardenings():
    """Alice and Bob each drive the full pipeline; verify every guarantee."""
    store = ContextMemoryStore(embed_fn=_bow_embed)
    analyzer = MultimodalAnalyzer(
        model_name="gpt-4o",
        execution_mode=CapabilityMode.REMOTE_MODEL,
        vision_client=_high_confidence_vision_client,
    )
    router = LLMRouter()

    alice_id, bob_id = uuid4(), uuid4()
    alice_result, bob_result = await asyncio.gather(
        _run_one_user_journey(alice_id, store, analyzer, router, "foo/bar v2"),
        _run_one_user_journey(bob_id, store, analyzer, router, "qux v1"),
    )

    # ── HARDENING 1: media accuracy guard ──────────────────────────────────
    for r in (alice_result, bob_result):
        v = r["video_analysis"]
        assert v["model"] == "gpt-4o"
        assert v["require_human_review"] is False
        assert v["factuality_score"] >= 0.5

    # ── HARDENING 2: PII never enters the persisted source set or output ──
    for r in (alice_result, bob_result):
        for s in r["sources"]:
            assert "journalist@example.com" not in s.content_snippet
            assert "Bearer abcdefghij" not in s.content_snippet
        assert "maintainer@example.com" not in r["scrubbed_output"]
        assert "Bearer abcdefghij" not in r["scrubbed_output"]
        assert "<email_redacted>" in r["scrubbed_output"]
        assert "<token_redacted>" in r["scrubbed_output"]

    # ── HARDENING 3: chain analysis ties the related sources together ─────
    for r in (alice_result, bob_result):
        rels = {e.relation for e in r["graph"].edges}
        assert SourceRelationshipType.CITES in rels
        assert SourceRelationshipType.TEMPORAL_FOLLOWS in rels
        chains = r["chain"]["chains"]
        assert len(chains) >= 1
        linked = set(chains[0]["source_ids"])
        assert {"yt-deep-dive", "blog-cooking"}.isdisjoint(linked) or \
               "yt-deep-dive" in linked
        assert "blog-cooking" in r["chain"]["isolated_source_ids"]

    # ── Single-source corroboration penalty visible in summary ────────────
    # 1. Within-journey claims: any FACTUAL/ANNOUNCEMENT claim attributed
    #    to a single source must be downgraded; multi-source ones are exempt.
    for r in (alice_result, bob_result):
        for c in r["summary"].key_claims:
            if (
                c.claim_type in (ClaimType.FACTUAL, ClaimType.ANNOUNCEMENT)
                and len(c.source_ids) <= 1
            ):
                assert c.confidence <= 0.6 + 1e-9, (
                    f"Single-source {c.claim_type} not penalised: {c.confidence}"
                )
    # 2. Direct probe: synthesise from the isolated blog source alone and
    #    verify the penalty path fires end-to-end (proves the wiring works).
    syn_probe = MultiSourceSynthesizer()
    blog_only = [s for s in alice_result["sources"] if s.source_id == "blog-cooking"]
    probe_summary = syn_probe.synthesize(SynthesisRequest(
        topic="sourdough", sources=blog_only, max_claims=2,
    ))
    factual_claims = [
        c for c in probe_summary.key_claims
        if c.claim_type in (ClaimType.FACTUAL, ClaimType.ANNOUNCEMENT)
    ]
    assert factual_claims, "probe should produce at least one factual claim"
    for c in factual_claims:
        assert len(c.source_ids) <= 1
        assert c.confidence <= 0.6 + 1e-9, (
            f"isolated single-source claim not penalised: {c.confidence}"
        )

    # ── HARDENING 4: per-user memory isolation, audit, GDPR ───────────────
    # Alice sees only her own memories
    assert all(m.user_id == alice_id for m in alice_result["own_memories"])
    assert all(m.user_id == bob_id for m in bob_result["own_memories"])
    assert len(alice_result["own_memories"]) >= 1
    assert len(bob_result["own_memories"]) >= 1

    # Profiles are kept independent and PII is scrubbed at write-time
    alice_prof = store.get_user_profile(alice_id)
    bob_prof = store.get_user_profile(bob_id)
    assert alice_prof["display_name"] != bob_prof["display_name"]
    assert "<email_redacted>" in alice_prof["contact_email"]
    assert "<email_redacted>" in bob_prof["contact_email"]

    # Cross-user retrieve must raise + be audited
    with pytest.raises(MemoryLeakError):
        await store.retrieve(
            user_id=alice_id,
            query_text="foo/bar v2 safety",
            top_k=3,
            requesting_user_id=bob_id,
        )
    audit = store.get_leak_audit()
    assert any(
        e["queried_user_id"] == str(alice_id)
        and e["requesting_user_id"] == str(bob_id)
        for e in audit
    )

    # GDPR export → clear → re-export round-trip
    export = store.export_user_data(alice_id)
    assert export["user_id"] == str(alice_id)
    assert export["profile"]["display_name"] == alice_prof["display_name"]
    removed = store.clear_user_data(alice_id)
    assert removed["profile"] == 1
    assert store.get_user_profile(alice_id) == {}
    # Bob is untouched
    assert store.get_user_profile(bob_id) == bob_prof
    # Idempotent re-clear
    assert store.clear_user_data(alice_id)["profile"] == 0

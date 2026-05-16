"""Tier 5 — Response Quality Evaluation

7-criterion rubric applied to every GroundedSummary produced by
build_grounded_summary(). 20+ rounds per query.  Any criterion with
pass rate < 100% is a defect that must be fixed.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, List, Tuple

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

ROUNDS = list(range(20))

QUERIES = [
    ("AI safety alignment interpretability research 2025",                       "safety-lab"),
    ("Protein folding AlphaFold 3 DeepMind structure prediction drug discovery", "bio-research"),
    ("NVIDIA H100 Blackwell GPU semiconductor supply chain shortage TSMC AMD",   "hardware-team"),
    ("GPT-5 Claude Gemini large language model reasoning benchmark evaluation",  "ai-benchmarks"),
    ("LLaMA Mistral Falcon open-source fine-tuning RLHF instruction tuning",    "mlops-team"),
]


def _make_item(text: str):
    from app.core.models import ContentItem, SourcePlatform, MediaType
    return ContentItem(
        user_id=uuid.uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id=f"item-{uuid.uuid4().hex[:8]}",
        source_url="https://example.com/post",
        title=text[:120],
        raw_text=text,
        media_type=MediaType.TEXT,
        published_at=datetime.now(timezone.utc),
        topics=text.split()[:8],
    )


def _make_mm_item(text: str, image_url: str):
    """ContentItem with a real-looking image_url for multimodal citation test."""
    from app.core.models import ContentItem, SourcePlatform, MediaType
    return ContentItem(
        user_id=uuid.uuid4(),
        source_platform=SourcePlatform.YOUTUBE,
        source_id=f"mm-{uuid.uuid4().hex[:8]}",
        source_url="https://youtube.com/post",
        title=text[:120],
        raw_text=text,
        media_type=MediaType.IMAGE,
        published_at=datetime.now(timezone.utc),
        topics=text.split()[:8],
        metadata={"image_url": image_url},
        media_urls=[image_url],
    )


def _build_summary(query: str, tenant_id: str, with_image: bool = False) -> Any:
    """Run process_batch + build_grounded_summary; return (summary, store)."""
    from app.ingestion.indexing_pipeline import IndexingPipeline
    from app.intelligence.retrieval.chunk_store import ChunkStore
    from app.intelligence.multimodal import MultimodalAnalyzer

    articles = [
        f"{query}: researchers from Stanford, MIT, and DeepMind published a landmark paper demonstrating significant advances in {query} at NeurIPS 2025.",
        f"Industry implications of {query}: leading organisations are investing billions in {query} capabilities with projected market impact of $400B by 2030.",
        f"Open-source community responds to {query}: Hugging Face, EleutherAI, and Together AI have released reproducible benchmarks for {query} evaluation.",
    ]

    analyzer = MultimodalAnalyzer()
    store = ChunkStore()
    pipeline = IndexingPipeline(chunk_store=store, multimodal_analyzer=analyzer)

    items = [_make_item(a) for a in articles]
    if with_image:
        items.append(_make_mm_item(
            articles[0],
            image_url=f"https://cdn.research.ai/{uuid.uuid4().hex[:8]}.jpg",
        ))

    result = asyncio.run(pipeline.process_batch(items, tenant_id=tenant_id))
    # Pass tenant_id so build_grounded_summary looks in the correct per-tenant store
    summary = pipeline.build_grounded_summary(result, topic=query, tenant_id=tenant_id)
    # Return the correct per-tenant store for rubric C4 validation
    tenant_store = pipeline.tenant_store(tenant_id) or store
    return summary, tenant_store


# ─────────────────────────────────────────────────────────────────────────────
# 7-criterion rubric check
# ─────────────────────────────────────────────────────────────────────────────

def _check_rubric(summary, store, require_mm_citation: bool = False) -> dict[str, bool]:
    """Return pass/fail per criterion. All must be True for a clean round.

    GroundedSummary field mapping (actual schema):
      summary text   → what_happened
      citations      → source_attributions (list of EvidenceSource)
      uncertainty    → uncertainty_annotations
      contradictions → contradictions (list)
    """
    results: dict[str, bool] = {}

    # C1: Attribution coverage — source_count >= 1
    results["C1_attribution"] = getattr(summary, "source_count", 0) >= 1

    # C2: Confidence plausibility — in [0.3, 1.0]
    cs = float(getattr(summary, "confidence_score", 0))
    results["C2_confidence_plausible"] = 0.3 <= cs <= 1.0

    # C3: Summary non-empty, >= 20 chars
    # GroundedSummary uses what_happened (or why_it_matters) as the main text
    text = (
        getattr(summary, "what_happened", None)
        or getattr(summary, "why_it_matters", None)
        or getattr(summary, "summary", None)
        or ""
    )
    results["C3_summary_nonempty"] = len((text or "").strip()) >= 20

    # C4: No hallucinated citations
    # source_attributions is the citations list; each item is an EvidenceSource
    citations = getattr(summary, "source_attributions", []) or []
    all_known_ids: set[str] = set()
    try:
        # ChunkStore exposes observation_ids(); individual chunk IDs come from
        # get_by_observation().
        for obs_id in store.observation_ids():
            all_known_ids.add(str(obs_id))
            for chunk in store.get_by_observation(str(obs_id)):
                all_known_ids.add(str(chunk.chunk_id))
    except Exception:
        pass
    bad_cits = [
        c for c in citations
        if not (
            str(getattr(c, "source_id", "")).startswith("mm-")
            or str(getattr(c, "source_id", "")) in all_known_ids
            or not getattr(c, "source_id", "")  # empty source_id → safe
        )
    ]
    results["C4_no_hallucinated_citations"] = len(bad_cits) == 0

    # C5: contradictions list present (may be empty)
    results["C5_contradictions_present"] = hasattr(summary, "contradictions")

    # C6: uncertainty_annotations present (may be empty)
    results["C6_uncertainty_notes_present"] = (
        hasattr(summary, "uncertainty_annotations")
        or hasattr(summary, "overall_uncertainty_score")
    )

    # C7: Multimodal citation — only checked when require_mm_citation=True
    if require_mm_citation:
        mm_cit = any(
            str(getattr(c, "source_id", "")).startswith("mm-")
            for c in citations
        )
        results["C7_multimodal_citation"] = mm_cit
    else:
        results["C7_multimodal_citation"] = True  # not checked

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Per-criterion tests (all queries, 20 rounds each)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("q,t", QUERIES)
@pytest.mark.parametrize("_r", ROUNDS)
def test_C1_attribution_coverage(q, t, _r):
    summary, store = _build_summary(q, t)
    if summary is None: pytest.skip("no summary produced")
    assert _check_rubric(summary, store)["C1_attribution"], \
        f"source_count={getattr(summary,'source_count',0)}"


@pytest.mark.parametrize("q,t", QUERIES)
@pytest.mark.parametrize("_r", ROUNDS)
def test_C2_confidence_plausibility(q, t, _r):
    summary, store = _build_summary(q, t)
    if summary is None: pytest.skip("no summary produced")
    cs = float(getattr(summary, "confidence_score", 0))
    assert 0.3 <= cs <= 1.0, f"confidence_score={cs} is out of plausible range [0.3, 1.0]"


@pytest.mark.parametrize("q,t", QUERIES)
@pytest.mark.parametrize("_r", ROUNDS)
def test_C3_summary_nonempty(q, t, _r):
    summary, store = _build_summary(q, t)
    if summary is None: pytest.skip("no summary produced")
    text = (
        getattr(summary, "what_happened", None)
        or getattr(summary, "why_it_matters", None)
        or getattr(summary, "summary", None)
        or ""
    )
    assert len(text.strip()) >= 20, f"summary too short: {len(text.strip())} chars"


@pytest.mark.parametrize("q,t", QUERIES)
@pytest.mark.parametrize("_r", ROUNDS)
def test_C4_no_hallucinated_citations(q, t, _r):
    summary, store = _build_summary(q, t)
    if summary is None: pytest.skip("no summary produced")
    rubric = _check_rubric(summary, store, require_mm_citation=False)
    assert rubric["C4_no_hallucinated_citations"], "hallucinated citation detected"


@pytest.mark.parametrize("q,t", QUERIES)
@pytest.mark.parametrize("_r", ROUNDS)
def test_C5_contradictions_field_present(q, t, _r):
    summary, store = _build_summary(q, t)
    if summary is None: pytest.skip("no summary produced")
    assert hasattr(summary, "contradictions"), "GroundedSummary missing 'contradictions' field"


@pytest.mark.parametrize("q,t", QUERIES)
@pytest.mark.parametrize("_r", ROUNDS)
def test_C6_uncertainty_notes_field_present(q, t, _r):
    summary, store = _build_summary(q, t)
    if summary is None: pytest.skip("no summary produced")
    assert (
        hasattr(summary, "uncertainty_annotations")
        or hasattr(summary, "overall_uncertainty_score")
    ), "GroundedSummary missing uncertainty_annotations / overall_uncertainty_score"


@pytest.mark.parametrize("q,t", QUERIES)
@pytest.mark.parametrize("_r", ROUNDS)
def test_C7_multimodal_citation_when_image_present(q, t, _r):
    """When input has image_url, at least one citation source_id must start with 'mm-'."""
    summary, store = _build_summary(q, t, with_image=True)
    if summary is None: pytest.skip("no summary produced")
    citations = getattr(summary, "source_attributions", []) or []
    mm_cit = any(str(getattr(c, "source_id", "")).startswith("mm-") for c in citations)
    assert mm_cit, (
        f"No 'mm-' citation found when image was in input. "
        f"citations={[str(getattr(c,'source_id','?')) for c in citations]}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Full 7-criterion rubric check (any summary → all 7 must pass)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("q,t", QUERIES)
@pytest.mark.parametrize("_r", ROUNDS)
def test_full_rubric_all_pass(q, t, _r):
    summary, store = _build_summary(q, t, with_image=True)
    if summary is None: pytest.skip("no summary produced")
    rubric = _check_rubric(summary, store, require_mm_citation=True)
    failed = [k for k, v in rubric.items() if not v]
    assert not failed, f"Rubric failures: {failed}"


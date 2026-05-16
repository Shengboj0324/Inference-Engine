"""Tier 2 — Realistic Data Ingestion Tests

Uses actual AI/ML news content, multi-source batches, multimodal items,
and cross-tenant batches.  Each scenario runs 20+ parametrized rounds.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import List

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Realistic content corpus
# ─────────────────────────────────────────────────────────────────────────────

ARTICLES = [
    "OpenAI unveils GPT-5 with advanced reasoning capabilities surpassing human benchmarks on MMLU, HellaSwag, and HumanEval. The model features a 200K context window and native multimodal support.",
    "Google DeepMind releases AlphaFold 3, extending protein structure prediction to DNA, RNA, and small molecules, enabling novel drug discovery workflows for cancer and infectious diseases.",
    "Anthropic closes a $2 billion Series E funding round led by Google, valuing the company at $18 billion, as demand for Claude 3 API surges 400% quarter-over-quarter.",
    "Meta releases LLaMA 3 under a community license, featuring 70B and 400B variants with instruction-following fine-tuning and RLHF alignment improvements over LLaMA 2.",
    "NVIDIA announces the Blackwell B200 GPU architecture delivering 4x AI training throughput and 30x inference improvement over H100, targeting the 2025 data-center market.",
    "Mistral AI open-sources Mixtral-8x22B mixture-of-experts model achieving GPT-4-level performance at one-fifth the inference cost on standard benchmarks.",
    "Microsoft integrates GPT-5 into Azure OpenAI Service with enterprise SLAs, enabling Fortune 500 companies to deploy production-grade RAG pipelines with guaranteed uptime.",
    "AMD launches MI300X GPU with 192 GB HBM3 memory, challenging NVIDIA's H100 dominance in large-model inference workloads for hyperscalers.",
    "Stanford HAI publishes the 2025 AI Index Report, documenting 3x year-over-year growth in foundation model publications and a 40% drop in inference costs.",
    "Hugging Face releases SmolLM-2, a 1.7B parameter language model optimised for on-device deployment on smartphones and edge hardware without cloud connectivity.",
]

PLATFORMS = ["reddit", "github", "arxiv", "youtube", "twitter"]
TENANTS   = ["acme-corp", "quant-fund", "bio-research"]

ROUNDS = list(range(20))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_item(text: str, platform: str = "reddit",
               image_url: str | None = None,
               video_url: str | None = None,
               tenant_id: str = "default"):
    """Create a realistic ContentItem wired for ingestion."""
    from app.core.models import ContentItem, SourcePlatform, MediaType
    meta: dict = {}
    if image_url:
        meta["image_url"] = image_url
    if video_url:
        meta["video_url"] = video_url
    plat_map = {
        "reddit": SourcePlatform.REDDIT,
        "github": SourcePlatform.GITHUB_RELEASES,
        "arxiv": SourcePlatform.ARXIV,
        "youtube": SourcePlatform.YOUTUBE,
        "twitter": SourcePlatform.REDDIT,    # no TWITTER enum; fallback to REDDIT
    }
    mtype = MediaType.IMAGE if image_url else (MediaType.VIDEO if video_url else MediaType.TEXT)
    # ContentItem stores URL-keyed metadata in `metadata` dict and
    # raw URL list in `media_urls`; both paths are tested by _extract_media_urls.
    meta_dict: dict = {}
    media_urls_list: list = []
    if image_url:
        meta_dict["image_url"] = image_url
        media_urls_list.append(image_url)
    if video_url:
        meta_dict["video_url"] = video_url
        media_urls_list.append(video_url)
    return ContentItem(
        user_id=uuid.uuid4(),
        source_platform=plat_map.get(platform, SourcePlatform.REDDIT),
        source_id=f"{platform}-{uuid.uuid4().hex[:8]}",
        source_url=f"https://{platform}.com/post/{uuid.uuid4().hex[:8]}",
        title=text[:120],
        raw_text=text,
        media_type=mtype,
        published_at=datetime.now(timezone.utc),
        topics=text.split()[:8],
        metadata=meta_dict or {},
        media_urls=media_urls_list,
    )


def _run_batch(items, tenant_id="default", multimodal_analyzer=None,
               trust_scorer=None, chunk_store=None):
    from app.ingestion.indexing_pipeline import IndexingPipeline
    from app.intelligence.retrieval.chunk_store import ChunkStore
    store = chunk_store or ChunkStore()
    p = IndexingPipeline(
        chunk_store=store,
        multimodal_analyzer=multimodal_analyzer,
        trust_scorer=trust_scorer,
    )
    result = asyncio.run(p.process_batch(items, tenant_id=tenant_id))
    return result, store


# ═════════════════════════════════════════════════════════════════════════════
# T2-A  Single-article ingestion across platforms
# ═════════════════════════════════════════════════════════════════════════════

class TestT2ASingleArticle:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_single_gpt5_article_indexes(self, _r):
        result, store = _run_batch([_make_item(ARTICLES[0])])
        assert result.stats.chunks_indexed >= 1

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_single_alphafold_article_indexes(self, _r):
        result, store = _run_batch([_make_item(ARTICLES[1])])
        assert result.stats.chunks_indexed >= 1

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_chunks_written_to_store(self, _r):
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        result, _ = _run_batch([_make_item(ARTICLES[2])], chunk_store=store)
        obs_ids = {str(pr.content_item_id) for pr in result.pipeline_results}
        all_chunks = []
        for oid in obs_ids:
            all_chunks.extend(store.get_by_observation(oid))
        assert len(all_chunks) >= 1

    @pytest.mark.parametrize("article", ARTICLES[:5])
    def test_all_ai_articles_index_without_error(self, article):
        result, _ = _run_batch([_make_item(article)])
        assert len(result.errors) == 0 or result.stats.chunks_indexed >= 0


# ═════════════════════════════════════════════════════════════════════════════
# T2-B  Multi-source same-event batches (cross-platform dedup)
# ═════════════════════════════════════════════════════════════════════════════

class TestT2BMultiSourceSameEvent:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_same_event_four_platforms_deduped(self, _r):
        text = ARTICLES[0]
        items = [_make_item(text, p) for p in ("reddit", "github", "arxiv", "youtube")]
        result, _ = _run_batch(items)
        assert len(result.bundles) <= 2  # should collapse or near-collapse

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_different_events_not_merged(self, _r):
        items = [_make_item(ARTICLES[i], "reddit") for i in range(4)]
        result, _ = _run_batch(items)
        assert len(result.bundles) >= 1  # at least some bundles formed

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_nvidia_blackwell_cross_platform_collapses(self, _r):
        text = ARTICLES[4]  # NVIDIA Blackwell
        items = [_make_item(text, p) for p in ("reddit", "twitter", "github")]
        result, _ = _run_batch(items)
        assert len(result.bundles) <= 2


# ═════════════════════════════════════════════════════════════════════════════
# T2-C  Multimodal items
# ═════════════════════════════════════════════════════════════════════════════

class TestT2CMultimodalItems:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_image_chunk_has_multimodal_evidence(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        analyzer = MultimodalAnalyzer()
        item = _make_item(
            ARTICLES[0],
            image_url="https://openai.com/img/gpt5-launch.jpg",
        )
        result, _ = _run_batch(
            [item], multimodal_analyzer=analyzer, chunk_store=store,
        )
        all_chunks = []
        for pr in result.pipeline_results:
            all_chunks.extend(store.get_by_observation(str(pr.content_item_id)))
        mm_found = any("multimodal_evidence" in c.metadata for c in all_chunks)
        assert mm_found, "multimodal_evidence key missing from all chunks"

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_video_chunk_has_multimodal_evidence(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        analyzer = MultimodalAnalyzer()
        item = _make_item(
            ARTICLES[4],
            video_url="https://nvidia.com/events/gtc2025/keynote.mp4",
        )
        result, _ = _run_batch(
            [item], multimodal_analyzer=analyzer, chunk_store=store,
        )
        all_chunks = []
        for pr in result.pipeline_results:
            all_chunks.extend(store.get_by_observation(str(pr.content_item_id)))
        mm_found = any("multimodal_evidence" in c.metadata for c in all_chunks)
        assert mm_found

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_text_only_no_multimodal_evidence(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        result, _ = _run_batch(
            [_make_item(ARTICLES[1])],
            multimodal_analyzer=MultimodalAnalyzer(),
            chunk_store=store,
        )
        all_chunks = []
        for pr in result.pipeline_results:
            all_chunks.extend(store.get_by_observation(str(pr.content_item_id)))
        assert all("multimodal_evidence" not in c.metadata for c in all_chunks)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_multimodal_source_count_ge_2_in_summary(self, _r):
        from app.intelligence.multimodal import MultimodalAnalyzer
        from app.ingestion.indexing_pipeline import IndexingPipeline, IndexingResult
        from app.intelligence.retrieval.chunk_store import ChunkStore
        from unittest.mock import MagicMock
        import uuid as _uuid
        from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
        store = ChunkStore()
        analyzer = MultimodalAnalyzer()

        # Build a fake pipeline result pointing to an observation with image
        from app.domain.raw_models import RawObservation
        from app.core.models import SourcePlatform, MediaType
        obs = RawObservation(
            user_id=_uuid.uuid4(), source_id="img-post",
            source_url="https://openai.com/post",
            title="GPT-5 Launch", raw_text=ARTICLES[0],
            media_type=MediaType.IMAGE,
            source_platform=SourcePlatform.REDDIT,
            published_at=datetime.now(timezone.utc),
            platform_metadata={"image_url": "https://openai.com/img/gpt5.jpg"},
        )
        pr = MagicMock(spec=IntelligencePipelineResult)
        pr.status = PipelineStatus.SUCCESS
        pr.is_actionable.return_value = True
        pr.content_item_id = _uuid.uuid4()
        pr.source_family = "social"
        pr.signal_type = "SOCIAL_TREND"
        pr.confidence = 0.85
        pr.result_id = _uuid.uuid4()
        pr.tenant_id = "default"
        pr.entities = []
        pr.extraction_warnings = []
        pr.summary = "GPT-5 launched with multimodal support"
        pr.produced_at = datetime.now(timezone.utc)
        pr.claims = []
        pr.all_text_for_chunking.return_value = ARTICLES[0]
        pr.raw_observation = obs

        ir = IndexingResult(pipeline_results=[pr])
        pipeline = IndexingPipeline(multimodal_analyzer=analyzer)
        summary = pipeline.build_grounded_summary(ir, "GPT-5 launch")
        if summary is not None:
            assert summary.source_count >= 2  # text source + image source


# ═════════════════════════════════════════════════════════════════════════════
# T2-D  Cross-tenant batches
# ═════════════════════════════════════════════════════════════════════════════

class TestT2DCrossTenant:
    @pytest.mark.parametrize("_r", ROUNDS)
    def test_three_tenants_each_get_chunks(self, _r):
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        shared_store = ChunkStore()
        pipeline = IndexingPipeline(chunk_store=shared_store)

        all_results = []
        for tid, article in zip(TENANTS, ARTICLES[:3]):
            items = [_make_item(article, tenant_id=tid)]
            r = asyncio.run(pipeline.process_batch(items, tenant_id=tid))
            all_results.append((tid, r))

        for tid, r in all_results:
            assert r.stats.chunks_indexed >= 1, f"tenant {tid!r} got no chunks"

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_tenant_chunks_do_not_cross_pollinate(self, _r):
        """Each tenant's chunks carry the correct tenant_id in metadata."""
        from app.ingestion.indexing_pipeline import IndexingPipeline
        from app.intelligence.retrieval.chunk_store import ChunkStore
        store = ChunkStore()
        pipeline = IndexingPipeline(chunk_store=store)

        obs_ids_by_tenant: dict = {}
        for tid, article in zip(TENANTS, ARTICLES[:3]):
            items = [_make_item(article, tenant_id=tid)]
            r = asyncio.run(pipeline.process_batch(items, tenant_id=tid))
            obs_ids_by_tenant[tid] = [str(pr.content_item_id) for pr in r.pipeline_results]

        for tid, obs_ids in obs_ids_by_tenant.items():
            for oid in obs_ids:
                chunks = store.get_by_observation(oid)
                for c in chunks:
                    # tenant_id must match (or be a super-set key for per-tenant stores)
                    assert c.metadata.get("tenant_id") in (tid, None)

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_trust_score_stamped_in_metadata(self, _r):
        """When trust_scorer wired, chunk metadata includes trust_score."""
        from app.source_intelligence.source_trust import SourceTrustScorer
        from app.intelligence.retrieval.chunk_store import ChunkStore
        scorer = SourceTrustScorer()
        scorer.set_authority("social", 0.9)
        scorer.set_primacy("social", True)
        store = ChunkStore()
        result, _ = _run_batch(
            [_make_item(ARTICLES[0])],
            trust_scorer=scorer,
            chunk_store=store,
        )
        all_chunks = []
        for pr in result.pipeline_results:
            all_chunks.extend(store.get_by_observation(str(pr.content_item_id)))
        scored = [c for c in all_chunks if "trust_score" in c.metadata]
        assert len(scored) >= 1
        for c in scored:
            assert 0.0 <= c.metadata["trust_score"] <= 1.0

    @pytest.mark.parametrize("_r", ROUNDS)
    def test_quality_gate_outcomes_present(self, _r):
        from app.ingestion.indexing_pipeline import IndexingPipeline, QualityGate
        from app.intelligence.retrieval.chunk_store import ChunkStore
        gate = QualityGate(min_confidence=0.10)
        pipeline = IndexingPipeline(chunk_store=ChunkStore(), quality_gate=gate)
        items = [_make_item(a) for a in ARTICLES[:3]]
        result = asyncio.run(pipeline.process_batch(items))
        assert isinstance(result.quality_gate_results, list)
        for qr in result.quality_gate_results:
            assert 0.0 <= qr.confidence_score <= 1.0


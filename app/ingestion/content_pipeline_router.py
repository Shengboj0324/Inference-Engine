"""Source-family-specific ingestion pipeline router.

``ContentPipelineRouter`` is the single entry-point that takes a raw
``ContentItem`` (fetched by any connector) and routes it through the
source-family-appropriate extraction chain, returning a normalized
``IntelligencePipelineResult``.

Source-family dispatch table
-----------------------------
+---------------------+-----------------------------------------------------+
| SourceFamily        | Extraction chain                                    |
+=====================+=====================================================+
| MEDIA_AUDIO         | TranscriptSegments → PodcastEpisodeUnderstanding   |
|                     | → EpisodeUnderstanding (topics, claims, quotes,     |
|                     | entities, LLM summary)                              |
+---------------------+-----------------------------------------------------+
| RESEARCH            | raw_text → SectionSegmenter → PaperParser           |
|                     | → PaperParsed (sections, claims, benchmarks,        |
|                     | novelty, LLM summary)                               |
+---------------------+-----------------------------------------------------+
| DEVELOPER_RELEASE   | release body → ReleaseParser → ReleaseNote          |
|                     | → BreakingChangeDetector → IntelligencePipelineResult|
+---------------------+-----------------------------------------------------+
| SOCIAL              | raw_text → entity extraction (keyword-based) +      |
|                     | truncated extractive summary (no LLM call)          |
+---------------------+-----------------------------------------------------+
| NEWS                | raw_text → entity extraction + extractive summary   |
+---------------------+-----------------------------------------------------+
| UNKNOWN             | Extractive summary only; marked PARTIAL             |
+---------------------+-----------------------------------------------------+

Thread-safety
-------------
The router is **stateless** — all pipeline objects it instantiates are either
shared immutable helpers or created per-call.  It is safe to call ``route()``
concurrently from multiple async tasks.

Usage::

    from app.ingestion.content_pipeline_router import ContentPipelineRouter
    from app.source_intelligence.source_registry import SourceFamily

    router = ContentPipelineRouter()
    result = await router.route(content_item, source_family=SourceFamily.RESEARCH)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import List, Optional

from app.core.models import ContentItem, MediaType
from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus
from app.source_intelligence.source_registry import SourceFamily

logger = logging.getLogger(__name__)

# Maximum characters we send to any text-based extractor in a single call.
# This prevents accidental OOM from extremely long blog posts / transcripts.
_MAX_TEXT_CHARS = 120_000

# Regex: simple NER heuristic (capitalised runs, excluding sentence starters)
_ENTITY_RE = re.compile(
    r"(?<!\.\s)(?<!\!\s)(?<!\?\s)\b([A-Z][a-z]{1,30}(?:\s+[A-Z][a-z]{1,30}){0,4})\b"
)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ContentPipelineRouter:
    """Routes ``ContentItem`` objects through source-family-specific pipelines.

    Args:
        llm_router: Optional pre-built ``LLMRouter`` instance.  When ``None``
            a new instance is created lazily on first use (only for families
            that require LLM calls — MEDIA_AUDIO and RESEARCH).
        max_claims: Maximum number of claims to surface in the result.
        max_entities: Maximum number of entities to surface in the result.
    """

    def __init__(
        self,
        llm_router=None,
        max_claims: int = 10,
        max_entities: int = 30,
    ) -> None:
        self._llm_router = llm_router
        self._max_claims = max_claims
        self._max_entities = max_entities

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def route(
        self,
        item: ContentItem,
        source_family: Optional[SourceFamily] = None,
        *,
        timeout_s: float = 120.0,
        tenant_id: str = "default",
    ) -> IntelligencePipelineResult:
        """Route *item* through the appropriate extraction chain.

        Args:
            item:          The ``ContentItem`` to process.
            source_family: Override the source family.  When ``None`` the
                           family is inferred from ``item.source_platform``.
            timeout_s:     Maximum seconds allowed for the full pipeline.
            tenant_id:     Tenant identifier propagated to the result for
                           downstream ``ChunkStore`` partitioning.  Defaults
                           to ``"default"`` (single-tenant deployments).

        Returns:
            ``IntelligencePipelineResult`` — never raises; errors are captured
            in ``status=FAILED`` or ``status=PARTIAL``.
        """
        t0 = time.perf_counter()
        family = source_family or self._infer_family(item)

        try:
            result = await asyncio.wait_for(
                self._dispatch(item, family),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "ContentPipelineRouter: timeout after %.1fs for item=%s family=%s",
                timeout_s, item.id, family,
            )
            result = IntelligencePipelineResult(
                content_item_id=item.id,
                source_family=family.value,
                tenant_id=tenant_id,
                status=PipelineStatus.FAILED,
                extraction_warnings=[f"Pipeline timed out after {timeout_s}s"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "ContentPipelineRouter: unhandled error for item=%s family=%s: %s",
                item.id, family, exc,
            )
            result = IntelligencePipelineResult(
                content_item_id=item.id,
                source_family=family.value,
                tenant_id=tenant_id,
                status=PipelineStatus.FAILED,
                extraction_warnings=[f"Unhandled error: {exc}"],
            )

        result = result.model_copy(
            update={
                "pipeline_duration_s": time.perf_counter() - t0,
                "tenant_id": tenant_id,
            }
        )
        logger.info(
            "ContentPipelineRouter: item=%s family=%s status=%s stages=%s duration=%.2fs",
            item.id, family.value, result.status.value,
            result.stages_run, result.pipeline_duration_s,
        )
        return result

    # ------------------------------------------------------------------
    # Family inference
    # ------------------------------------------------------------------

    def _infer_family(self, item: ContentItem) -> SourceFamily:
        """Infer SourceFamily from platform + media type."""
        platform = item.source_platform.value.lower()

        _audio_platforms   = {"youtube", "youtube_transcript", "podcast_rss", "transcript_feeds"}
        _research_platforms = {"arxiv", "openreview", "semantic_scholar"}
        _dev_platforms     = {"github_releases", "github_repo_events", "github_discussions",
                              "changelog", "docs_monitor"}
        _social_platforms  = {"reddit", "tiktok", "facebook", "instagram", "wechat"}
        _news_platforms    = {"rss", "nytimes", "wsj", "abc_news", "google_news", "apple_news"}

        if item.media_type in {MediaType.AUDIO, MediaType.VIDEO}:
            return SourceFamily.MEDIA_AUDIO
        if platform in _audio_platforms:
            return SourceFamily.MEDIA_AUDIO
        if platform in _research_platforms:
            return SourceFamily.RESEARCH
        if platform in _dev_platforms:
            return SourceFamily.DEVELOPER_RELEASE
        if platform in _social_platforms:
            return SourceFamily.SOCIAL
        if platform in _news_platforms:
            return SourceFamily.NEWS
        return SourceFamily.UNKNOWN

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    async def _dispatch(
        self, item: ContentItem, family: SourceFamily
    ) -> IntelligencePipelineResult:
        if family == SourceFamily.MEDIA_AUDIO:
            return await self._route_audio(item)
        if family == SourceFamily.RESEARCH:
            return await self._route_research(item)
        if family == SourceFamily.DEVELOPER_RELEASE:
            return self._route_developer(item)
        if family in {SourceFamily.SOCIAL, SourceFamily.NEWS}:
            return self._route_social_news(item, family)
        return self._route_unknown(item)

    # ------------------------------------------------------------------
    # MEDIA_AUDIO pipeline
    # ------------------------------------------------------------------

    async def _route_audio(self, item: ContentItem) -> IntelligencePipelineResult:
        from app.media.audio_intelligence.models import TranscriptSegment
        from app.media.audio_intelligence.podcast_episode_understanding import (
            PodcastEpisodeUnderstandingPipeline,
        )

        warnings: List[str] = []
        stages: List[str] = []
        text = (item.raw_text or "").strip()[:_MAX_TEXT_CHARS]

        # Build synthetic transcript segments from raw_text when no audio file
        seg = TranscriptSegment(start_s=0.0, end_s=max(1.0, len(text) / 15.0), text=text or "[no transcript]")
        stages.append("synthetic_segment")

        pipeline = PodcastEpisodeUnderstandingPipeline(
            llm_router=self._llm_router,
        )
        try:
            understanding = await pipeline.process(
                transcript_segments=[seg],
                episode_id=str(item.id),
                title=item.title or "Untitled episode",
                duration_s=max(1.0, len(text) / 15.0),
            )
            stages.append("episode_understanding")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Audio pipeline failed: {exc}")
            return IntelligencePipelineResult(
                content_item_id=item.id,
                source_family=SourceFamily.MEDIA_AUDIO.value,
                status=PipelineStatus.PARTIAL,
                summary=text[:500],
                stages_run=stages,
                extraction_warnings=warnings,
                signal_type="PODCAST_EPISODE",
            )

        entities = understanding.entities[:self._max_entities]
        claims   = [c.text for c in understanding.top_claims[:self._max_claims]]
        keywords = list({t.label.value for t in understanding.topics})
        return IntelligencePipelineResult(
            content_item_id=item.id,
            source_family=SourceFamily.MEDIA_AUDIO.value,
            status=PipelineStatus.SUCCESS,
            signal_type="PODCAST_EPISODE",
            confidence=0.85,
            entities=entities,
            claims=claims,
            keywords=keywords,
            summary=understanding.llm_summary or understanding.transcript[:500],
            episode_detail=understanding,
            stages_run=stages,
            extraction_warnings=warnings,
        )

    # ------------------------------------------------------------------
    # RESEARCH pipeline
    # ------------------------------------------------------------------

    async def _route_research(self, item: ContentItem) -> IntelligencePipelineResult:
        from app.document_intelligence.models import DocumentSection, SectionType
        from app.document_intelligence.paper_parser import PaperParser
        from app.document_intelligence.section_segmenter import SectionSegmenter

        warnings: List[str] = []
        stages: List[str] = []
        text = (item.raw_text or "").strip()[:_MAX_TEXT_CHARS]

        segmenter = SectionSegmenter()
        try:
            sections = segmenter.segment(text)
            stages.append("section_segmenter")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Section segmentation failed: {exc}")
            sections = []

        parser = PaperParser(paper_id=item.source_id or str(item.id), title=item.title or "")
        try:
            paper = parser.parse(full_text=text, sections=sections)
            stages.append("paper_parser")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Paper parsing failed: {exc}")
            return IntelligencePipelineResult(
                content_item_id=item.id,
                source_family=SourceFamily.RESEARCH.value,
                status=PipelineStatus.PARTIAL,
                signal_type="RESEARCH_PAPER",
                summary=text[:500],
                stages_run=stages,
                extraction_warnings=warnings,
            )

        entities  = list({e for c in paper.claims for e in c.entities})[:self._max_entities]
        claims    = [c.claim for c in paper.claims[:self._max_claims]]
        keywords  = (paper.keywords or [])[:15]
        summary   = paper.llm_summary or paper.abstract[:500]
        return IntelligencePipelineResult(
            content_item_id=item.id,
            source_family=SourceFamily.RESEARCH.value,
            status=PipelineStatus.SUCCESS,
            signal_type="RESEARCH_PAPER",
            confidence=0.90,
            entities=entities,
            claims=claims,
            keywords=keywords,
            summary=summary,
            paper_detail=paper,
            stages_run=stages,
            extraction_warnings=warnings,
        )

    # ------------------------------------------------------------------
    # DEVELOPER_RELEASE pipeline
    # ------------------------------------------------------------------

    def _route_developer(self, item: ContentItem) -> IntelligencePipelineResult:
        from app.devintel.breaking_change_detector import BreakingChangeDetector
        from app.devintel.release_parser import ReleaseParser

        warnings: List[str] = []
        stages: List[str] = []
        body = (item.raw_text or "").strip()[:_MAX_TEXT_CHARS]
        version = item.metadata.get("version", item.source_id or "unknown")
        repo    = item.metadata.get("repo", item.channel or "")

        parser = ReleaseParser(version=version, repo=repo, url=item.source_url,
                               published_at=item.published_at, title=item.title or "")
        try:
            release = parser.parse(body)
            stages.append("release_parser")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Release parser failed: {exc}")
            return IntelligencePipelineResult(
                content_item_id=item.id,
                source_family=SourceFamily.DEVELOPER_RELEASE.value,
                status=PipelineStatus.PARTIAL,
                signal_type="DEVELOPER_RELEASE",
                summary=body[:500],
                stages_run=stages,
                extraction_warnings=warnings,
            )

        try:
            bcd = BreakingChangeDetector()
            # detect_from_entries() is synchronous (no LLM call required)
            breaking = bcd.detect_from_entries(release.breaking)
            stages.append("breaking_change_detector")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Breaking change detection failed: {exc}")
            breaking = []

        claims   = [e.text for e in release.features[:5]] + [e.text for e in release.breaking[:5]]
        entities = list({repo} | {e.author for e in release.all_entries() if e.author})
        summary  = release.summary or (f"{repo} released {version}: " + body[:300])
        signal   = "BREAKING_CHANGE" if release.has_breaking_changes else "DEVELOPER_RELEASE"
        return IntelligencePipelineResult(
            content_item_id=item.id,
            source_family=SourceFamily.DEVELOPER_RELEASE.value,
            status=PipelineStatus.SUCCESS,
            signal_type=signal,
            confidence=0.92,
            entities=list(entities)[:self._max_entities],
            claims=claims[:self._max_claims],
            keywords=[e.category.value for e in release.all_entries()[:10]],
            summary=summary,
            release_detail=release,
            stages_run=stages,
            extraction_warnings=warnings,
            extra_detail={"breaking_count": len(breaking)},
        )

    # ------------------------------------------------------------------
    # SOCIAL / NEWS pipeline
    # ------------------------------------------------------------------

    def _route_social_news(
        self, item: ContentItem, family: SourceFamily
    ) -> IntelligencePipelineResult:
        text = (item.raw_text or item.title or "").strip()[:_MAX_TEXT_CHARS]
        entities = self._extract_entities(text)
        summary  = text[:500] if len(text) > 500 else text
        signal   = "SOCIAL_POST" if family == SourceFamily.SOCIAL else "NEWS_ARTICLE"
        return IntelligencePipelineResult(
            content_item_id=item.id,
            source_family=family.value,
            status=PipelineStatus.SUCCESS,
            signal_type=signal,
            confidence=0.70,
            entities=entities[:self._max_entities],
            claims=[],
            keywords=(item.topics or [])[:10],
            summary=summary,
            extra_detail={"platform": item.source_platform.value},
            stages_run=["entity_extraction"],
        )

    # ------------------------------------------------------------------
    # UNKNOWN fallback
    # ------------------------------------------------------------------

    def _route_unknown(self, item: ContentItem) -> IntelligencePipelineResult:
        text = (item.raw_text or item.title or "").strip()[:500]
        return IntelligencePipelineResult(
            content_item_id=item.id,
            source_family=SourceFamily.UNKNOWN.value,
            status=PipelineStatus.PARTIAL,
            summary=text,
            stages_run=["passthrough"],
            extraction_warnings=["Unknown source family; minimal extraction performed"],
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        _STOP = frozenset({"The", "A", "An", "In", "Of", "And", "Or", "But", "So",
                           "We", "It", "Is", "To", "For", "On", "At", "With",
                           "This", "That", "My", "Our", "Your", "Their", "He",
                           "She", "They", "You", "I", "By", "From", "As"})
        seen: dict = {}
        for match in _ENTITY_RE.finditer(text):
            ent = match.group(1)
            if ent not in _STOP:
                seen[ent] = seen.get(ent, 0) + 1
        # Return by frequency descending
        return [e for e, _ in sorted(seen.items(), key=lambda x: -x[1])]


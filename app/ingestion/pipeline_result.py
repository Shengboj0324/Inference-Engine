"""Normalized output model produced by the ``ContentPipelineRouter``.

Every source family (SOCIAL, NEWS, MEDIA_AUDIO, RESEARCH, DEVELOPER_RELEASE)
produces its own family-specific rich object (``EpisodeUnderstanding``,
``PaperParsed``, ``ReleaseNote``, …).  ``IntelligencePipelineResult`` wraps
whichever rich object was produced and exposes a flat, cross-family contract
for downstream consumers such as the personalization ranker, the retrieval
store, and the grounded summarizer.

Design goals
------------
- **Single normalized surface** — downstream components need not know which
  source family generated the content.
- **Optional rich payloads** — at most one of the ``*_detail`` fields is
  populated; all others are ``None``.
- **Auditability** — ``stages_run``, ``pipeline_duration_s``, and
  ``extraction_warnings`` expose exactly what the router did for debugging
  and monitoring.

Relationship to existing models
--------------------------------
- Input:  :class:`app.core.models.ContentItem`
- Audio:  :class:`app.media.audio_intelligence.models.EpisodeUnderstanding`
- Research: :class:`app.document_intelligence.models.PaperParsed`
- Developer: :class:`app.devintel.models.ReleaseNote`
- Social/News: plain text summary (no rich detail object)
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PipelineStatus(str, Enum):
    """Outcome of the routing + extraction pass."""

    SUCCESS  = "success"
    PARTIAL  = "partial"    # some stages failed; usable output still available
    FAILED   = "failed"     # unrecoverable; result should not be indexed
    SKIPPED  = "skipped"    # content was filtered out before extraction


class IntelligencePipelineResult(BaseModel):
    """Normalized output of the ``ContentPipelineRouter`` for one ``ContentItem``.

    Attributes
    ----------
    result_id:
        Unique identifier for this pipeline run.
    content_item_id:
        UUID of the originating ``ContentItem``.
    source_family:
        String value of the ``SourceFamily`` enum used for routing
        (e.g. ``"media_audio"``, ``"research"``).
    tenant_id:
        Tenant identifier propagated from ``ContentPipelineRouter.route()``
        and ``IndexingPipeline.process_batch()``.  Used to partition the
        ``ChunkStore`` per tenant.  ``"default"`` for single-tenant deployments.
    status:
        Overall pipeline outcome.
    signal_type:
        Coarse signal classification (e.g. ``"RESEARCH_PAPER"``,
        ``"DEVELOPER_RELEASE"``, ``"PODCAST_EPISODE"``).  ``None`` when
        classification was skipped.
    confidence:
        Signal classification confidence in ``[0, 1]``.  ``0.0`` when unknown.
    entities:
        Deduplicated named entities extracted across all stages.
    claims:
        Top plain-text claims (up to 10).
    keywords:
        Topic keywords merged from all extraction stages.
    summary:
        Short (≤ 3 sentence) extractive or LLM-generated summary.
    episode_detail:
        Populated for ``MEDIA_AUDIO`` items; ``None`` otherwise.
    paper_detail:
        Populated for ``RESEARCH`` items; ``None`` otherwise.
    release_detail:
        Populated for ``DEVELOPER_RELEASE`` items; ``None`` otherwise.
    extra_detail:
        Arbitrary JSON-serialisable payload for source families without a
        dedicated rich model (e.g. ``SOCIAL``, ``NEWS``).
    stages_run:
        Ordered list of pipeline stage names executed.
    extraction_warnings:
        Non-fatal warnings emitted during extraction.
    pipeline_duration_s:
        Wall-clock seconds from router entry to result assembly.
    produced_at:
        UTC timestamp when this result was assembled.
    """

    result_id:           UUID            = Field(default_factory=uuid4)
    content_item_id:     UUID
    source_family:       str
    tenant_id:           str             = "default"
    status:              PipelineStatus  = PipelineStatus.SUCCESS
    signal_type:         Optional[str]   = None
    confidence:          float           = Field(0.0, ge=0.0, le=1.0)

    # Cross-family normalized fields
    entities:            List[str]       = Field(default_factory=list)
    claims:              List[str]       = Field(default_factory=list)
    keywords:            List[str]       = Field(default_factory=list)
    summary:             str             = ""

    # Rich family-specific payloads (at most one is non-None)
    episode_detail:      Optional[Any]   = None   # EpisodeUnderstanding
    paper_detail:        Optional[Any]   = None   # PaperParsed
    release_detail:      Optional[Any]   = None   # ReleaseNote

    # Arbitrary payload for SOCIAL / NEWS
    extra_detail:        Optional[Dict[str, Any]] = None

    # Audit / observability
    stages_run:          List[str]       = Field(default_factory=list)
    extraction_warnings: List[str]       = Field(default_factory=list)
    pipeline_duration_s: float           = Field(0.0, ge=0.0)
    produced_at:         datetime        = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def is_actionable(self) -> bool:
        """Return True when the result carries extractable intelligence."""
        return self.status in {PipelineStatus.SUCCESS, PipelineStatus.PARTIAL}

    def has_rich_detail(self) -> bool:
        """Return True when at least one family-specific detail is present."""
        return (
            self.episode_detail is not None
            or self.paper_detail is not None
            or self.release_detail is not None
        )

    def all_text_for_chunking(self) -> str:
        """Return a flat text blob suitable for chunk-level indexing."""
        parts: List[str] = []
        if self.summary:
            parts.append(self.summary)
        parts.extend(self.claims)
        parts.extend(self.entities)
        return "\n".join(parts)


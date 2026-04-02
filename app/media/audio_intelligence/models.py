"""Shared Pydantic models for the audio intelligence stack.

All models are immutable (``frozen=True``) and self-validating.
They are used as the contract between every module in this package.
"""

from __future__ import annotations

import re
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class TranscriptSegment(BaseModel, frozen=True):
    """One time-aligned chunk of a raw ASR transcript.

    Attributes:
        start_s:  Segment start time in seconds from episode start.
        end_s:    Segment end time in seconds.
        text:     Transcribed text (may contain ASR errors before correction).
        confidence: ASR confidence score [0, 1]; None if backend doesn't provide it.
        language: BCP-47 language code detected by ASR.
    """

    start_s: float = Field(..., ge=0.0)
    end_s: float = Field(..., ge=0.0)
    text: str
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    language: str = "en"

    @model_validator(mode="after")
    def _end_after_start(self) -> "TranscriptSegment":
        if self.end_s < self.start_s:
            raise ValueError(f"end_s ({self.end_s}) must be >= start_s ({self.start_s})")
        return self

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def duration(self) -> timedelta:
        return timedelta(seconds=self.duration_s)


class TranscriptResult(BaseModel, frozen=True):
    """Complete transcription result with backend provenance.

    Wraps the raw segment list produced by ``TranscriptionRouter.transcribe()``
    and adds backend identification and aggregate confidence so downstream
    callers can make trust decisions without re-inspecting individual segments.

    Attributes:
        segments:        Time-aligned ASR segments (may be empty for silent audio).
        backend_used:    ASR backend that produced this result (e.g.
                         ``"faster_whisper"``, ``"whisper"``, ``"openai"``,
                         ``"stub"``).
        mean_confidence: Mean of per-segment confidence scores; ``None`` if the
                         backend does not emit per-segment confidence (e.g.
                         ``"whisper"`` or ``"openai"`` backends in verbose_json
                         mode without ``avg_logprob`` entries).
        duration_s:      Total audio duration in seconds (sum of all segment
                         durations; ``0.0`` for empty results).
        language:        Detected language BCP-47 code (taken from the first
                         segment; ``"und"`` if no segments were produced).
    """

    segments: List[TranscriptSegment] = Field(default_factory=list)
    backend_used: str = Field(..., min_length=1)
    mean_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    duration_s: float = Field(default=0.0, ge=0.0)
    language: str = "und"

    @model_validator(mode="after")
    def _derive_fields(self) -> "TranscriptResult":
        """Validate that mean_confidence is consistent with segment data."""
        confs = [s.confidence for s in self.segments if s.confidence is not None]
        if self.mean_confidence is not None and confs:
            computed = sum(confs) / len(confs)
            if abs(computed - self.mean_confidence) > 0.01:
                raise ValueError(
                    f"mean_confidence {self.mean_confidence:.4f} deviates "
                    f"more than 0.01 from computed mean {computed:.4f}"
                )
        return self

    @property
    def full_text(self) -> str:
        """Concatenated transcript text across all segments."""
        return " ".join(s.text for s in self.segments if s.text)

    @property
    def is_stub(self) -> bool:
        """True when the backend is the no-op stub (testing / CI only)."""
        return self.backend_used == "stub"


class DiarizedSegment(BaseModel, frozen=True):
    """A TranscriptSegment annotated with speaker identity.

    Attributes:
        segment:      The underlying ASR segment.
        speaker_id:   Short speaker label (e.g. ``"SPEAKER_00"``).
        speaker_role: Inferred role (``"host"``, ``"guest"``, ``"unknown"``).
    """

    segment: TranscriptSegment
    speaker_id: str
    speaker_role: str = "unknown"

    @field_validator("speaker_role")
    @classmethod
    def _valid_role(cls, v: str) -> str:
        allowed = {"host", "guest", "unknown"}
        if v not in allowed:
            raise ValueError(f"speaker_role must be one of {allowed}, got {v!r}")
        return v


class TopicLabel(str, Enum):
    """Taxonomy of topics found in AI/ML podcast episodes."""

    MODEL_RELEASE = "model_release"
    BENCHMARK = "benchmark"
    RESEARCH = "research"
    FUNDING = "funding"
    POLICY = "policy"
    OPINION = "opinion"
    SPECULATION = "speculation"
    INTERVIEW = "interview"
    TUTORIAL = "tutorial"
    INDUSTRY_NEWS = "industry_news"
    SAFETY = "safety"
    OTHER = "other"


class TopicSegment(BaseModel, frozen=True):
    """A coherent thematic segment of an episode.

    Attributes:
        start_s:    Segment start in seconds.
        end_s:      Segment end in seconds.
        label:      ``TopicLabel`` taxonomy value.
        title:      Short human-readable title (≤ 80 chars).
        summary:    3–5 sentence summary of the segment content.
        key_entities: Named entities prominent in this segment.
        confidence: Segmentation confidence [0, 1].
    """

    start_s: float = Field(..., ge=0.0)
    end_s: float = Field(..., ge=0.0)
    label: TopicLabel
    title: str = Field(..., max_length=80)
    summary: str
    key_entities: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _end_after_start(self) -> "TopicSegment":
        if self.end_s < self.start_s:
            raise ValueError(f"end_s ({self.end_s}) must be >= start_s ({self.start_s})")
        return self


class ExtractedQuote(BaseModel, frozen=True):
    """A notable verbatim quote from the episode.

    Attributes:
        text:           Exact quote text.
        speaker_id:     Speaker who said it.
        start_s:        Timestamp in seconds.
        importance:     Importance score [0, 1].
        context:        One sentence of surrounding context.
        topic_label:    Topic this quote belongs to.
    """

    text: str = Field(..., min_length=10)
    speaker_id: str = "unknown"
    start_s: float = Field(..., ge=0.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    context: str = ""
    topic_label: TopicLabel = TopicLabel.OTHER


class ClaimType(str, Enum):
    """The epistemic status of an extracted claim."""

    ANNOUNCEMENT = "announcement"      # "We are releasing X today"
    BENCHMARK = "benchmark"            # "Our model achieves 90% on Y"
    PREDICTION = "prediction"          # "I think X will happen"
    OPINION = "opinion"                # "I believe X is better than Y"
    SPECULATION = "speculation"        # "Maybe X could Y"
    FACT = "fact"                      # Verifiable, source-independent


class ExtractedClaim(BaseModel, frozen=True):
    """A factual or epistemic claim extracted from the episode.

    Attributes:
        text:        Claim text (normalized, not necessarily verbatim).
        claim_type:  ``ClaimType`` classification.
        confidence:  Extraction confidence [0, 1].
        evidence:    Verbatim evidence sentence(s) from transcript.
        speaker_id:  Speaker who made the claim.
        start_s:     Timestamp in seconds.
        entities:    Named entities in the claim.
        supported:   True = evidence supports claim; False = unsupported.
    """

    text: str = Field(..., min_length=5)
    claim_type: ClaimType
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: str = ""
    speaker_id: str = "unknown"
    start_s: float = Field(..., ge=0.0)
    entities: List[str] = Field(default_factory=list)
    supported: bool = True


class EpisodeUnderstanding(BaseModel):
    """Full structured understanding of a podcast/video episode.

    Attributes:
        episode_id:     Unique identifier (e.g. source_id from ContentItem).
        title:          Episode title.
        duration_s:     Total duration in seconds.
        transcript:     Full corrected transcript text.
        segments:       Diarized transcript segments.
        topics:         Topic segments.
        top_claims:     Up to 5 highest-confidence claims.
        key_quotes:     Up to 10 most notable quotes.
        entities:       All named entities mentioned (deduplicated).
        follow_up_sources: Suggested sources to verify claims.
        llm_summary:    LLM-generated executive summary paragraph.
        processing_metadata: Timing, backend, and quality metrics.
    """

    episode_id: str
    title: str
    duration_s: float = Field(..., ge=0.0)
    transcript: str
    segments: List[DiarizedSegment] = Field(default_factory=list)
    topics: List[TopicSegment] = Field(default_factory=list)
    top_claims: List[ExtractedClaim] = Field(default_factory=list)
    key_quotes: List[ExtractedQuote] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    follow_up_sources: List[str] = Field(default_factory=list)
    llm_summary: str = ""
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)


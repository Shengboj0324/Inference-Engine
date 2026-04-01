"""Audio Intelligence — Phase 2 spoken-content understanding stack.

Pipeline for a single podcast/video episode:

    1. TranscriptionRouter  → raw transcript text (Whisper / cloud ASR / stub)
    2. Diarizer             → speaker-segmented DiarizedSegment list
    3. TopicSegmenter       → semantically coherent TopicSegment list
    4. QuoteExtractor       → notable ExtractedQuote list
    5. ClaimExtractor       → factual ExtractedClaim list per segment
    6. EpisodeUnderstanding → assembled EpisodeUnderstanding result

All components share the domain AI/ML correction lexicon defined in
``transcription_router.py`` and the Pydantic models defined here.

Public exports
--------------
TranscriptSegment, DiarizedSegment, TopicSegment,
ExtractedQuote, ExtractedClaim, EpisodeUnderstanding,
TranscriptionRouter, Diarizer, TopicSegmenter,
QuoteExtractor, ClaimExtractor, PodcastEpisodeUnderstandingPipeline
"""

from app.media.audio_intelligence.models import (
    ClaimType,
    DiarizedSegment,
    EpisodeUnderstanding,
    ExtractedClaim,
    ExtractedQuote,
    TopicLabel,
    TopicSegment,
    TranscriptSegment,
)
from app.media.audio_intelligence.transcription_router import (
    ASRBackend,
    TranscriptionRouter,
)
from app.media.audio_intelligence.diarization import Diarizer
from app.media.audio_intelligence.topic_segmentation import TopicSegmenter
from app.media.audio_intelligence.quote_extraction import QuoteExtractor
from app.media.audio_intelligence.claim_extraction import ClaimExtractor
from app.media.audio_intelligence.podcast_episode_understanding import (
    PodcastEpisodeUnderstandingPipeline,
)

__all__ = [
    # Models
    "ClaimType",
    "DiarizedSegment",
    "EpisodeUnderstanding",
    "ExtractedClaim",
    "ExtractedQuote",
    "TopicLabel",
    "TopicSegment",
    "TranscriptSegment",
    # Components
    "ASRBackend",
    "ClaimExtractor",
    "Diarizer",
    "PodcastEpisodeUnderstandingPipeline",
    "QuoteExtractor",
    "TopicSegmenter",
    "TranscriptionRouter",
]


"""Multimodal understanding pipeline for image and video content.

Analyses visual media embedded in ``RawObservation.platform_metadata`` and
produces plain-text summaries that can be fed directly into the RAG pipeline
as retrievable text chunks.

All analysis results are structured as JSON-serialisable ``TypedDict`` schemas
(``ImageAnalysisResult``, ``VideoAnalysisResult``) so the extracted data can
be used directly as training samples for downstream fine-tuning or distillation.

Integration point
-----------------
``ConnectorRegistry.apply_acquisition_filter()`` calls
``MultimodalAnalyzer.visual_to_text()`` for every ``RawObservation`` that
contains ``image_url``, ``image_data``, ``thumbnail_url``, ``video_url``, or
``media_url`` keys in ``platform_metadata``.  The resulting paragraph is
appended to ``raw_text`` before the 8-stage noise filter runs, making visual
content fully searchable by keyword and embedding retrieval.

Design for production
---------------------
* Replace the stub ``analyze_image`` / ``analyze_video`` bodies with calls to
  a vision LLM (e.g. GPT-4o, Claude 3.5 Sonnet, Gemini Pro Vision) or a
  dedicated OCR / ASR service.
* Inject a ``vision_client`` callable at construction time — no other changes
  are needed to wire in a real model.
* The ``visual_to_text`` method must remain synchronous and complete within
  500 ms per observation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TypedDict

from app.domain.raw_models import RawObservation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON-serialisable TypedDict schemas (fine-tuning / distillation ready)
# ---------------------------------------------------------------------------

class FieldConfidence(TypedDict):
    """Wrapper pairing an extracted value with its model confidence score."""
    value: Any    # str | list | float — the extracted output
    confidence: float  # [0.0, 1.0]


class ImageAnalysisResult(TypedDict):
    """Structured output of a single image analysis.

    All fields are JSON-serialisable so the dict can be logged, stored, and
    used directly as a training sample for vision fine-tuning or distillation.

    Fields
    ------
    caption : FieldConfidence
        Natural-language description of the image content and visual scene.
    entities : List[Dict[str, Any]]
        Detected entities — people, brands, objects, on-image text (OCR).
        Each entry: ``{"name": str, "type": str, "confidence": float}``.
    sentiment : FieldConfidence
        Visual sentiment signal: ``"positive"``, ``"neutral"``, or
        ``"negative"`` derived from colour palette, facial expressions, etc.
    source_url : str
        URL or identifier from which the image was sourced.
    model : str
        Name of the model / pipeline used (e.g. ``"gpt-4o"``, ``"stub"``).
    """

    caption: FieldConfidence
    entities: List[Dict[str, Any]]
    sentiment: FieldConfidence
    source_url: str
    model: str


class SceneSummary(TypedDict):
    """Single-scene description extracted from a video.

    Fields
    ------
    timestamp_seconds : int
        Approximate start time of the scene in seconds from the video start.
    description : str
        Natural-language summary of what happens in this scene.
    """

    timestamp_seconds: int
    description: str


class VideoAnalysisResult(TypedDict):
    """Structured output of a single video analysis.

    Fields
    ------
    scenes : List[SceneSummary]
        Scene-by-scene summaries in chronological order.
    transcript : str
        Full audio transcript (empty string when audio is unavailable or
        ASR was not run).  Suitable for downstream keyword / RAG retrieval.
    entities : List[Dict[str, Any]]
        Key entities and topics mentioned across the video.
    sentiment : FieldConfidence
        Overall video sentiment with confidence.
    source_url : str
        Video URL analysed.
    model : str
        Model / pipeline used.
    """

    scenes: List[SceneSummary]
    transcript: str
    entities: List[Dict[str, Any]]
    sentiment: FieldConfidence
    source_url: str
    model: str


# ---------------------------------------------------------------------------
# MultimodalAnalyzer
# ---------------------------------------------------------------------------

class MultimodalAnalyzer:
    """Extracts structured intelligence from image and video metadata embedded
    in ``RawObservation.platform_metadata``.

    Implements a ``visual_to_text`` pipeline that converts multimodal signals
    into a single RAG-ready text paragraph, appended to ``raw_text`` by
    ``ConnectorRegistry.apply_acquisition_filter``.

    Args:
        model_name: Logical name of the analysis model recorded in results.
        vision_client: Optional callable for production vision API calls.
            Signature: ``vision_client(url: str, modality: str) -> dict``.
            When ``None``, deterministic stub results are returned (test-safe).
    """

    #: Platform metadata keys that indicate image content.
    IMAGE_METADATA_KEYS = ("image_url", "thumbnail_url", "image_data")
    #: Platform metadata keys that indicate video content.
    VIDEO_METADATA_KEYS = ("video_url", "media_url")

    def __init__(
        self,
        model_name: str = "stub",
        vision_client: Optional[Any] = None,
    ) -> None:
        self.model_name = model_name
        self._vision_client = vision_client

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze_image(self, url: str) -> ImageAnalysisResult:
        """Return structured analysis of the image at *url*.

        Calls ``vision_client(url, 'image')`` when a client is injected;
        otherwise returns a deterministic stub suitable for test environments.
        Never raises — on client failure the stub result is returned.
        """
        if self._vision_client is not None:
            try:
                raw = self._vision_client(url, "image")
                return self._parse_image_result(raw, url)
            except Exception as exc:
                logger.warning("MultimodalAnalyzer: image analysis failed: %s", exc)
        return self._stub_image_result(url)

    def analyze_video(self, url: str) -> VideoAnalysisResult:
        """Return structured analysis of the video at *url*.

        Calls ``vision_client(url, 'video')`` when a client is injected;
        otherwise returns a deterministic stub.  Never raises.
        """
        if self._vision_client is not None:
            try:
                raw = self._vision_client(url, "video")
                return self._parse_video_result(raw, url)
            except Exception as exc:
                logger.warning("MultimodalAnalyzer: video analysis failed: %s", exc)
        return self._stub_video_result(url)

    def visual_to_text(self, observation: RawObservation) -> str:
        """Produce a RAG-ready paragraph from an observation's visual metadata.

        Completes in < 500 ms per observation (stub always satisfies this).

        The returned paragraph is structured as::

            [Image content] <caption>.
            Detected entities: <names>.
            Visual sentiment: <polarity>.
            Source: <platform> — <url>.

        or::

            [Video content] Scenes: @<t>s: <desc>; …
            Transcript excerpt: <first 200 chars>.
            Key entities: <names>.
            Overall sentiment: <polarity>.
            Source: <platform> — <url>.

        Returns an empty string when no visual metadata is found so callers
        can check truthiness before appending.
        """
        meta = observation.platform_metadata or {}
        parts: List[str] = []

        # Image analysis — first matching key wins
        for key in self.IMAGE_METADATA_KEYS:
            val = meta.get(key)
            if val and isinstance(val, str):
                result = self.analyze_image(val)
                parts.append(self._image_to_paragraph(result, observation))
                break

        # Video analysis — first matching key wins
        for key in self.VIDEO_METADATA_KEYS:
            val = meta.get(key)
            if val and isinstance(val, str):
                result = self.analyze_video(val)
                parts.append(self._video_to_paragraph(result, observation))
                break

        return " ".join(p for p in parts if p)

    def has_visual_content(self, observation: RawObservation) -> bool:
        """Return ``True`` when the observation has image or video metadata."""
        meta = observation.platform_metadata or {}
        return any(
            meta.get(k)
            for k in self.IMAGE_METADATA_KEYS + self.VIDEO_METADATA_KEYS
        )

    # ── Private stub factories ────────────────────────────────────────────────

    def _stub_image_result(self, url: str) -> ImageAnalysisResult:
        return ImageAnalysisResult(
            caption=FieldConfidence(
                value="An image depicting content from a social media post.",
                confidence=0.75,
            ),
            entities=[
                {"name": "unidentified subject", "type": "object", "confidence": 0.6}
            ],
            sentiment=FieldConfidence(value="neutral", confidence=0.8),
            source_url=url,
            model=self.model_name,
        )

    def _stub_video_result(self, url: str) -> VideoAnalysisResult:
        return VideoAnalysisResult(
            scenes=[
                SceneSummary(
                    timestamp_seconds=0,
                    description="Opening scene depicting the main subject.",
                ),
                SceneSummary(
                    timestamp_seconds=30,
                    description="Mid-section showing key content detail.",
                ),
            ],
            transcript="[Transcript not available in stub mode]",
            entities=[
                {"name": "unidentified speaker", "type": "person", "confidence": 0.5}
            ],
            sentiment=FieldConfidence(value="neutral", confidence=0.7),
            source_url=url,
            model=self.model_name,
        )

    # ── Private parsers (production path) ────────────────────────────────────

    def _parse_image_result(self, raw: dict, url: str) -> ImageAnalysisResult:
        return ImageAnalysisResult(
            caption=FieldConfidence(
                value=raw.get("caption", ""),
                confidence=float(raw.get("caption_confidence", 0.5)),
            ),
            entities=raw.get("entities", []),
            sentiment=FieldConfidence(
                value=raw.get("sentiment", "neutral"),
                confidence=float(raw.get("sentiment_confidence", 0.5)),
            ),
            source_url=url,
            model=self.model_name,
        )

    def _parse_video_result(self, raw: dict, url: str) -> VideoAnalysisResult:
        scenes = [
            SceneSummary(
                timestamp_seconds=s.get("ts", 0),
                description=s.get("desc", ""),
            )
            for s in raw.get("scenes", [])
        ]
        return VideoAnalysisResult(
            scenes=scenes,
            transcript=raw.get("transcript", ""),
            entities=raw.get("entities", []),
            sentiment=FieldConfidence(
                value=raw.get("sentiment", "neutral"),
                confidence=float(raw.get("sentiment_confidence", 0.5)),
            ),
            source_url=url,
            model=self.model_name,
        )

    # ── Private paragraph formatters ──────────────────────────────────────────

    def _attribution(self, obs: RawObservation) -> str:
        platform = obs.source_platform.value
        return f"{platform} — {obs.source_url}" if obs.source_url else platform

    def _image_to_paragraph(
        self, result: ImageAnalysisResult, obs: RawObservation
    ) -> str:
        caption = result["caption"]["value"]
        sentiment = result["sentiment"]["value"]
        entity_names = ", ".join(
            e.get("name", "") for e in result["entities"] if e.get("name")
        ) or "none detected"
        return (
            f"[Image content] {caption} "
            f"Detected entities: {entity_names}. "
            f"Visual sentiment: {sentiment}. "
            f"Source: {self._attribution(obs)}."
        )

    def _video_to_paragraph(
        self, result: VideoAnalysisResult, obs: RawObservation
    ) -> str:
        scenes_text = "; ".join(
            f"@{s['timestamp_seconds']}s: {s['description']}"
            for s in result["scenes"]
        )
        sentiment = result["sentiment"]["value"]
        entity_names = ", ".join(
            e.get("name", "") for e in result["entities"] if e.get("name")
        ) or "none detected"
        transcript_snippet = result["transcript"][:200]
        return (
            f"[Video content] Scenes: {scenes_text}. "
            f"Transcript excerpt: {transcript_snippet}. "
            f"Key entities: {entity_names}. "
            f"Overall sentiment: {sentiment}. "
            f"Source: {self._attribution(obs)}."
        )


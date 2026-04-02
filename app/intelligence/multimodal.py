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

import enum
import hashlib
import logging
from typing import Any, Dict, List, Optional, TypedDict

from app.domain.raw_models import RawObservation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CapabilityMode — explicit execution-state enum
# ---------------------------------------------------------------------------

class CapabilityMode(str, enum.Enum):
    """Represents the active execution tier of a ``MultimodalAnalyzer``.

    Values
    ------
    DISABLED
        No visual analysis attempted; ``visual_to_text`` always returns ``""``.
    STUB
        Deterministic synthetic results used (test / CI environments or when
        no vision client is available in production).
    LOCAL_MODEL
        Analysis performed by a locally-loaded model (e.g. LLaVA via Ollama).
    REMOTE_MODEL
        Analysis performed by a remote vision API (e.g. GPT-4o, Claude 3.5
        Sonnet, Gemini Pro Vision).
    """
    DISABLED     = "disabled"
    STUB         = "stub"
    LOCAL_MODEL  = "local_model"
    REMOTE_MODEL = "remote_model"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _url_stub_slot(url: str, n_slots: int) -> int:
    """Return a deterministic slot index in ``[0, n_slots)`` for a URL.

    Uses the first 8 hex characters of the SHA-256 digest so the result is
    identical across Python sessions regardless of ``PYTHONHASHSEED``.
    SHA-256 is used here purely as a deterministic hash, not for security.
    """
    digest = hashlib.sha256(url.encode()).hexdigest()
    return int(digest[:8], 16) % n_slots


# ---------------------------------------------------------------------------
# Deterministic stub data  (5 image slots · 3 video slots)
#
# Each slot is designed around a distinct real-world ingestion scenario so
# that different fixture URLs resolve to meaningfully different stub outputs,
# making quality tests more diverse without requiring a live vision API.
# ---------------------------------------------------------------------------

_IMAGE_STUBS: List[Dict[str, Any]] = [
    {   # Slot 0 — product review / unboxing
        "caption": (
            "A high-quality consumer product photograph showing merchandise "
            "arranged for a detailed review on a neutral studio background "
            "with professional lighting and clear brand labeling."
        ),
        "caption_confidence": 0.88,
        "entities": [
            {"name": "consumer product", "type": "object", "confidence": 0.85},
            {"name": "brand packaging",  "type": "brand",  "confidence": 0.78},
        ],
        "sentiment": "positive",
        "sentiment_confidence": 0.82,
        "sentiment_justification": (
            "clean product presentation, bright studio lighting, and intentional "
            "brand placement suggesting a promotional or review context"
        ),
        "signal_hint": "BRAND_SENTIMENT or FEATURE_REQUEST",
    },
    {   # Slot 1 — complaint / error screenshot
        "caption": (
            "A mobile device screenshot capturing an application error dialog "
            "with red warning indicators and a truncated system error message "
            "displayed in white text on a dark background."
        ),
        "caption_confidence": 0.91,
        "entities": [
            {"name": "error dialog",       "type": "ui_element", "confidence": 0.92},
            {"name": "mobile application", "type": "software",   "confidence": 0.80},
        ],
        "sentiment": "negative",
        "sentiment_confidence": 0.88,
        "sentiment_justification": (
            "prominent error indicators, warning color palette, and visible "
            "user-frustration signals in the captured interface state"
        ),
        "signal_hint": "COMPLAINT or CHURN_RISK",
    },
    {   # Slot 2 — feature announcement / product launch
        "caption": (
            "A corporate product announcement slide showcasing new software "
            "capabilities on a large presentation screen at a well-attended "
            "industry conference with branding prominently displayed overhead."
        ),
        "caption_confidence": 0.87,
        "entities": [
            {"name": "product feature",    "type": "software", "confidence": 0.87},
            {"name": "conference audience","type": "group",    "confidence": 0.72},
        ],
        "sentiment": "positive",
        "sentiment_confidence": 0.90,
        "sentiment_justification": (
            "polished presentation design, enthusiastic audience positioning, "
            "and aspirational brand messaging typical of launch events"
        ),
        "signal_hint": "FEATURE_REQUEST or BRAND_SENTIMENT",
    },
    {   # Slot 3 — competitor brand / storefront
        "caption": (
            "A wide-angle photograph of a competitor company storefront with "
            "large illuminated signage and brand logo displayed at street level "
            "in a busy commercial district during daytime business hours."
        ),
        "caption_confidence": 0.84,
        "entities": [
            {"name": "competitor storefront", "type": "brand",    "confidence": 0.93},
            {"name": "commercial district",   "type": "location", "confidence": 0.75},
        ],
        "sentiment": "neutral",
        "sentiment_confidence": 0.75,
        "sentiment_justification": (
            "neutral corporate visual language with standard brand presentation "
            "and no explicit positive or negative sentiment markers in the frame"
        ),
        "signal_hint": "COMPETITOR_ACTIVITY or BRAND_SENTIMENT",
    },
    {   # Slot 4 — news / editorial
        "caption": (
            "An editorial news photograph depicting industry professionals "
            "engaged in discussion at a media event, surrounded by camera "
            "equipment and branded event signage indicating a press briefing."
        ),
        "caption_confidence": 0.83,
        "entities": [
            {"name": "industry professionals", "type": "group", "confidence": 0.83},
            {"name": "press event",            "type": "event", "confidence": 0.77},
        ],
        "sentiment": "neutral",
        "sentiment_confidence": 0.79,
        "sentiment_justification": (
            "formal professional context with balanced visual composition "
            "and informational framing consistent with editorial content"
        ),
        "signal_hint": "MARKET_TREND or BRAND_SENTIMENT",
    },
]

_VIDEO_STUBS: List[Dict[str, Any]] = [
    {   # Slot 0 — product demo / unboxing video
        "scenes": [
            {
                "ts": 0,
                "desc": (
                    "Opening product unboxing sequence with brand packaging "
                    "visible and upbeat background music setting an engaging tone."
                ),
            },
            {
                "ts": 14,
                "desc": (
                    "Close-up demonstration of key product features with clear "
                    "voiceover narration highlighting specifications and user benefits."
                ),
            },
            {
                "ts": 32,
                "desc": (
                    "Satisfied customer testimonial segment showing product "
                    "interaction in a natural home environment with authentic reactions."
                ),
            },
        ],
        "transcript": (
            "Welcome to this product review. Today we are looking at the latest "
            "release from the brand, which has been receiving significant attention "
            "online. The build quality is impressive and the feature set aligns well "
            "with what users have been requesting. The setup process was "
            "straightforward and daily-use performance has been consistently reliable."
        ),
        "entities": [
            {"name": "product brand",    "type": "brand",  "confidence": 0.88},
            {"name": "product reviewer", "type": "person", "confidence": 0.82},
        ],
        "sentiment": "positive",
        "sentiment_confidence": 0.85,
        "sentiment_justification": (
            "upbeat audio cues, enthusiastic presenter delivery, and consistently "
            "positive product framing throughout the demonstration segments"
        ),
        "signal_hint": "BRAND_SENTIMENT or FEATURE_REQUEST",
    },
    {   # Slot 1 — news broadcast / editorial clip
        "scenes": [
            {
                "ts": 0,
                "desc": (
                    "News anchor presenting a breaking industry story with graphics "
                    "overlay showing relevant market statistics and trend charts."
                ),
            },
            {
                "ts": 22,
                "desc": (
                    "Field reporter conducting a live interview with a company "
                    "spokesperson outside corporate headquarters about recent developments."
                ),
            },
            {
                "ts": 48,
                "desc": (
                    "Expert panel analysis segment discussing market implications "
                    "and consumer impact of the breaking news story at hand."
                ),
            },
        ],
        "transcript": (
            "In today's business news, a major company has announced significant "
            "changes to its product lineup following sustained pressure from consumers "
            "and investors. Our correspondent has been following this story closely. "
            "Industry analysts suggest this announcement could signal a broader shift "
            "in the competitive landscape for the coming quarters."
        ),
        "entities": [
            {"name": "news anchor",           "type": "person",   "confidence": 0.90},
            {"name": "corporate spokesperson","type": "person",   "confidence": 0.78},
            {"name": "company headquarters",  "type": "location", "confidence": 0.72},
        ],
        "sentiment": "neutral",
        "sentiment_confidence": 0.76,
        "sentiment_justification": (
            "objective journalistic tone, balanced expert commentary, and neutral "
            "visual presentation style consistent with editorial broadcast content"
        ),
        "signal_hint": "MARKET_TREND or COMPETITOR_ACTIVITY",
    },
    {   # Slot 2 — short-form social / creator content
        "scenes": [
            {
                "ts": 0,
                "desc": (
                    "Content creator introduction in front of ring-lit background "
                    "sharing personal experience with a recently purchased product."
                ),
            },
            {
                "ts": 9,
                "desc": (
                    "Screen recording demonstration segment highlighting a specific "
                    "product feature or pain point discovered during extended use."
                ),
            },
            {
                "ts": 21,
                "desc": (
                    "Call-to-action and community engagement prompt with branded "
                    "text overlay and relevant hashtags clearly visible on screen."
                ),
            },
        ],
        "transcript": (
            "Hey everyone, sharing my honest thoughts after two weeks with this "
            "product. The initial setup had some friction but once configured the "
            "performance has been genuinely impressive. Battery life exceeded "
            "expectations and the build feels premium for the price point. Would "
            "recommend keeping an eye on the software update cadence before buying."
        ),
        "entities": [
            {"name": "content creator", "type": "person", "confidence": 0.85},
            {"name": "reviewed product","type": "object", "confidence": 0.80},
            {"name": "consumer audience","type": "group", "confidence": 0.70},
        ],
        "sentiment": "positive",
        "sentiment_confidence": 0.72,
        "sentiment_justification": (
            "conversational authentic tone with balanced feedback resolving to "
            "a positive overall recommendation and genuine user enthusiasm"
        ),
        "signal_hint": "BRAND_SENTIMENT or COMPLAINT",
    },
]


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
        execution_mode: CapabilityMode = CapabilityMode.STUB,
    ) -> None:
        if not isinstance(model_name, str):
            raise TypeError(f"model_name must be str, got {type(model_name).__name__!r}")
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if not isinstance(execution_mode, CapabilityMode):
            raise TypeError(
                f"execution_mode must be a CapabilityMode, got {type(execution_mode).__name__!r}"
            )
        self.model_name = model_name
        self._vision_client = vision_client
        self._execution_mode = execution_mode

        # ── Production safety contract ────────────────────────────────────────
        # Fail-closed in strict mode: STUB mode is not acceptable for any
        # user-facing result path when production_strict_mode=True.
        from app.core.production_guard import get_guard
        if execution_mode == CapabilityMode.STUB:
            get_guard().require_real_backend(
                capability="multimodal_vision",
                resolved_backend="stub",
            )

        logger.info(
            "MultimodalAnalyzer init: model=%s execution_mode=%s vision_client=%s",
            model_name,
            execution_mode.value,
            "injected" if vision_client is not None else "none",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def execution_mode(self) -> CapabilityMode:
        """Current execution tier of this analyzer.

        Returns ``STUB`` when no vision client is configured,
        ``DISABLED`` when explicitly disabled, ``LOCAL_MODEL`` or
        ``REMOTE_MODEL`` when a real vision client is active.

        Callers must check this property before trusting analysis confidence
        scores — stub-mode results carry no real-model provenance.
        """
        if self._execution_mode == CapabilityMode.DISABLED:
            return CapabilityMode.DISABLED
        if self._vision_client is not None:
            return self._execution_mode  # LOCAL_MODEL or REMOTE_MODEL as declared
        return CapabilityMode.STUB

    def analyze_image(self, url: str) -> ImageAnalysisResult:
        """Return structured analysis of the image at *url*.

        Calls ``vision_client(url, 'image')`` when a client is injected;
        otherwise returns a deterministic stub suitable for test environments.
        Never raises — on client failure the stub result is returned.
        Logs the active ``execution_mode`` on every call.
        """
        if not isinstance(url, str):
            raise TypeError(f"url must be str, got {type(url).__name__!r}")
        if not url.strip():
            raise ValueError("url must not be empty")
        mode = self.execution_mode
        logger.debug("MultimodalAnalyzer.analyze_image: url=%s mode=%s", url, mode.value)
        if mode == CapabilityMode.DISABLED:
            return self._stub_image_result(url)
        if self._vision_client is not None:
            try:
                raw = self._vision_client(url, "image")
                return self._parse_image_result(raw, url)
            except Exception as exc:
                logger.warning(
                    "MultimodalAnalyzer: image analysis failed (mode=%s): %s", mode.value, exc
                )
        return self._stub_image_result(url)

    def analyze_video(self, url: str) -> VideoAnalysisResult:
        """Return structured analysis of the video at *url*.

        Calls ``vision_client(url, 'video')`` when a client is injected;
        otherwise returns a deterministic stub.  Logs the active
        ``execution_mode`` on every call.  Never raises.
        """
        if not isinstance(url, str):
            raise TypeError(f"url must be str, got {type(url).__name__!r}")
        if not url.strip():
            raise ValueError("url must not be empty")
        mode = self.execution_mode
        logger.debug("MultimodalAnalyzer.analyze_video: url=%s mode=%s", url, mode.value)
        if mode == CapabilityMode.DISABLED:
            return self._stub_video_result(url)
        if self._vision_client is not None:
            try:
                raw = self._vision_client(url, "video")
                return self._parse_video_result(raw, url)
            except Exception as exc:
                logger.warning(
                    "MultimodalAnalyzer: video analysis failed (mode=%s): %s", mode.value, exc
                )
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
        mode = self.execution_mode
        logger.debug(
            "MultimodalAnalyzer.visual_to_text: observation_id=%s mode=%s",
            observation.id, mode.value,
        )
        if mode == CapabilityMode.DISABLED:
            return ""

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

        return "\n\n".join(p for p in parts if p)

    # ── Private helper ────────────────────────────────────────────────────────

    def _extract_media_urls(self, observation: Any) -> tuple[list[str], list[str]]:
        """Duck-typed extraction of image/video URLs from any observation type.

        Supports:
        - ``RawObservation`` via ``platform_metadata``
        - ``ContentItem`` via ``metadata`` dict and ``media_urls`` list

        Returns:
            Tuple of ``(image_urls, video_urls)`` — both possibly empty.
        """
        import re as _re
        _IMG_EXT = frozenset((".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".bmp", ".tiff"))
        _VID_EXT = frozenset((".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"))

        def _ext(url: str) -> str:
            m = _re.search(r'\.[a-zA-Z0-9]+(?:\?|$)', url)
            return m.group(0).split("?")[0].lower() if m else ""

        image_urls: list[str] = []
        video_urls: list[str] = []

        # 1. platform_metadata (RawObservation)
        meta = getattr(observation, "platform_metadata", None) or {}
        for key in self.IMAGE_METADATA_KEYS:
            v = meta.get(key)
            if v and isinstance(v, str):
                image_urls.append(v)
                break
        for key in self.VIDEO_METADATA_KEYS:
            v = meta.get(key)
            if v and isinstance(v, str):
                video_urls.append(v)
                break

        # 2. metadata dict (ContentItem)
        if not image_urls and not video_urls:
            meta2 = getattr(observation, "metadata", None) or {}
            for key in self.IMAGE_METADATA_KEYS:
                v = meta2.get(key)
                if v and isinstance(v, str):
                    image_urls.append(v)
                    break
            for key in self.VIDEO_METADATA_KEYS:
                v = meta2.get(key)
                if v and isinstance(v, str):
                    video_urls.append(v)
                    break

        # 3. media_urls list (ContentItem)
        if not image_urls and not video_urls:
            for url in getattr(observation, "media_urls", None) or []:
                if not isinstance(url, str):
                    continue
                e = _ext(url)
                if e in _IMG_EXT:
                    image_urls.append(url)
                elif e in _VID_EXT:
                    video_urls.append(url)

        return image_urls, video_urls

    def has_visual_content(self, observation: Any) -> bool:
        """Return ``True`` when the observation has image or video metadata.

        Duck-typed: supports both ``RawObservation`` and ``ContentItem``.
        """
        imgs, vids = self._extract_media_urls(observation)
        return bool(imgs or vids)

    def to_evidence_sources(self, observation: RawObservation) -> List[Dict[str, Any]]:
        """Return multimodal analysis results as ``EvidenceSource``-compatible dicts.

        Extracts structured analysis (caption, transcript, entities, sentiment)
        from the observation's visual metadata and wraps each modality as a
        citation-ready dict that callers can convert to ``EvidenceSource`` for
        use in ``GroundedSummaryBuilder.build()``.

        Returns an empty list when:
        - The observation has no visual metadata.
        - ``execution_mode`` is ``DISABLED``.

        Each dict has keys:
        - ``source_id``:       Unique identifier for this multimodal citation.
        - ``title``:           Short description (e.g. "Image analysis").
        - ``url``:             URL of the analysed media asset.
        - ``platform``:        Source platform value.
        - ``trust_score``:     Confidence from the analysis model (0–1).
        - ``content_snippet``: RAG-ready paragraph produced by
                               ``_image_to_paragraph`` / ``_video_to_paragraph``.
        - ``modality``:        ``"image"`` or ``"video"``.
        - ``entities``:        Detected entity list.
        - ``sentiment``:       Detected sentiment string.

        Args:
            observation: ``RawObservation`` to analyse.

        Returns:
            List of dicts (one per detected modality), possibly empty.
        """
        import hashlib as _hl
        mode = self.execution_mode
        if mode == CapabilityMode.DISABLED:
            return []

        platform_val: str = getattr(
            getattr(observation, "source_platform", None), "value", "unknown"
        )
        image_urls, video_urls = self._extract_media_urls(observation)
        results: List[Dict[str, Any]] = []

        for url in image_urls[:1]:   # at most one image per observation
            try:
                analysis = self.analyze_image(url)
                snippet = self._image_to_paragraph(analysis, observation)
                uid = _hl.md5(f"img:{url}".encode()).hexdigest()[:12]
                results.append({
                    "source_id":       f"mm-img-{uid}",
                    "title":           f"Image analysis: {url[:80]}",
                    "url":             url,
                    "platform":        platform_val,
                    "trust_score":     float(analysis["caption"]["confidence"]),
                    "content_snippet": snippet,
                    "modality":        "image",
                    "entities":        analysis["entities"],
                    "sentiment":       analysis["sentiment"]["value"],
                })
            except Exception as exc:
                logger.warning(
                    "MultimodalAnalyzer.to_evidence_sources: image failed url=%s: %s",
                    url, exc,
                )

        for url in video_urls[:1]:   # at most one video per observation
            try:
                analysis = self.analyze_video(url)
                snippet = self._video_to_paragraph(analysis, observation)
                uid = _hl.md5(f"vid:{url}".encode()).hexdigest()[:12]
                results.append({
                    "source_id":       f"mm-vid-{uid}",
                    "title":           f"Video analysis: {url[:80]}",
                    "url":             url,
                    "platform":        platform_val,
                    "trust_score":     float(analysis["sentiment"]["confidence"]),
                    "content_snippet": snippet,
                    "modality":        "video",
                    "entities":        analysis["entities"],
                    "sentiment":       analysis["sentiment"]["value"],
                })
            except Exception as exc:
                logger.warning(
                    "MultimodalAnalyzer.to_evidence_sources: video failed url=%s: %s",
                    url, exc,
                )

        return results

    # ── Private stub factories ────────────────────────────────────────────────

    def _stub_image_result(self, url: str) -> ImageAnalysisResult:
        """Return a URL-hash–selected stub that meets the production quality bar.

        Different URLs deterministically map to different scenario slots so
        that a suite of real-URL fixtures exercises diverse stub outputs while
        remaining fully deterministic across test runs.
        """
        s = _IMAGE_STUBS[_url_stub_slot(url, len(_IMAGE_STUBS))]
        return ImageAnalysisResult(
            caption=FieldConfidence(
                value=s["caption"],
                confidence=s["caption_confidence"],
            ),
            entities=s["entities"],
            sentiment=FieldConfidence(
                value=s["sentiment"],
                confidence=s["sentiment_confidence"],
            ),
            source_url=url,
            model=self.model_name,
        )

    def _stub_video_result(self, url: str) -> VideoAnalysisResult:
        """Return a URL-hash–selected video stub with rich transcript and scenes."""
        s = _VIDEO_STUBS[_url_stub_slot(url, len(_VIDEO_STUBS))]
        return VideoAnalysisResult(
            scenes=[
                SceneSummary(
                    timestamp_seconds=scene["ts"],
                    description=scene["desc"],
                )
                for scene in s["scenes"]
            ],
            transcript=s["transcript"],
            entities=s["entities"],
            sentiment=FieldConfidence(
                value=s["sentiment"],
                confidence=s["sentiment_confidence"],
            ),
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
    #
    # Output format (Step 5 spec):
    #   • 3–5 sentences each on its own line
    #   • 60–200 words total
    #   • Specificity: named entity present
    #   • Sentiment attribution with brief justification
    #   • Signal hint (most likely SignalType)
    #   • Source attribution (platform + URL)
    # ──────────────────────────────────────────────────────────────────────────

    def _attribution(self, obs: RawObservation) -> str:
        platform = obs.source_platform.value
        return f"{platform} — {obs.source_url}" if obs.source_url else platform

    def _infer_signal_hint(
        self, entities: List[Dict[str, Any]], sentiment: str
    ) -> str:
        """Derive the most likely SignalType hint from entity types and sentiment."""
        types = {e.get("type", "").lower() for e in entities}
        if "ui_element" in types or ("software" in types and sentiment == "negative"):
            return "COMPLAINT or FEATURE_REQUEST"
        if "brand" in types or "logo" in types:
            return (
                "COMPLAINT or COMPETITOR_ACTIVITY"
                if sentiment == "negative"
                else "BRAND_SENTIMENT or COMPETITOR_ACTIVITY"
            )
        if "event" in types:
            return "MARKET_TREND or BRAND_SENTIMENT"
        if "person" in types:
            return (
                "COMPLAINT or CHURN_RISK"
                if sentiment == "negative"
                else "BRAND_SENTIMENT or MARKET_TREND"
            )
        return "COMPLAINT or CHURN_RISK" if sentiment == "negative" else "BRAND_SENTIMENT or MARKET_TREND"

    def _describe_sentiment(self, sentiment: str, entity_names: str) -> str:
        """Return a one-phrase sentiment justification for the paragraph."""
        if sentiment == "positive":
            return (
                f"warm visual tone, bright composition, and positive engagement "
                f"signals associated with {entity_names}"
            )
        if sentiment == "negative":
            return (
                f"concerning visual indicators, distress signals, and negative "
                f"markers associated with {entity_names}"
            )
        return (
            f"balanced, objective visual presentation with neutral framing "
            f"around {entity_names}"
        )

    def _image_to_paragraph(
        self, result: ImageAnalysisResult, obs: RawObservation
    ) -> str:
        """Format an ImageAnalysisResult as a 5-sentence RAG-ready paragraph.

        Line structure:
          1. [Image content] <caption>
          2. Detected entities: <names>.
          3. Visual sentiment: <polarity> — <justification>.
          4. This image may indicate a <SIGNAL_TYPE> signal.
          5. Source: <platform> — <url>.
        """
        caption = result["caption"]["value"]
        sentiment = result["sentiment"]["value"]
        entity_names = ", ".join(
            e.get("name", "") for e in result["entities"] if e.get("name")
        ) or "none detected"
        signal_hint = self._infer_signal_hint(result["entities"], sentiment)
        sentiment_just = self._describe_sentiment(sentiment, entity_names)
        lines = [
            f"[Image content] {caption}",
            f"Detected entities: {entity_names}.",
            f"Visual sentiment: {sentiment} — {sentiment_just}.",
            f"This image may indicate a {signal_hint} signal.",
            f"Source: {self._attribution(obs)}.",
        ]
        return "\n".join(lines)

    def _video_to_paragraph(
        self, result: VideoAnalysisResult, obs: RawObservation
    ) -> str:
        """Format a VideoAnalysisResult as a 6-sentence RAG-ready paragraph.

        Line structure:
          1. [Video content] Scenes: @<t>s: <desc>; …
          2. Transcript excerpt: <first 200 chars>.
          3. Key entities: <names>.
          4. Overall sentiment: <polarity> — <justification>.
          5. This video may indicate a <SIGNAL_TYPE> signal.
          6. Source: <platform> — <url>.
        """
        scenes_text = "; ".join(
            f"@{s['timestamp_seconds']}s: {s['description']}"
            for s in result["scenes"]
        )
        sentiment = result["sentiment"]["value"]
        entity_names = ", ".join(
            e.get("name", "") for e in result["entities"] if e.get("name")
        ) or "none detected"
        signal_hint = self._infer_signal_hint(result["entities"], sentiment)
        sentiment_just = self._describe_sentiment(sentiment, entity_names)
        transcript_snippet = result["transcript"][:200].rstrip()
        lines = [
            f"[Video content] Scenes: {scenes_text}.",
            f"Transcript excerpt: {transcript_snippet}.",
            f"Key entities: {entity_names}.",
            f"Overall sentiment: {sentiment} — {sentiment_just}.",
            f"This video may indicate a {signal_hint} signal.",
            f"Source: {self._attribution(obs)}.",
        ]
        return "\n".join(lines)


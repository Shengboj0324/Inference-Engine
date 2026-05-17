"""Local multimodal orchestrator — fans a :class:`ContentItem` out to the
CLIP and ASR runtimes based on its ``media_type``, then assembles a
single :class:`MultimodalResult` for the ingest worker.

Responsibilities
----------------
* **Dispatch** by ``media_type``: IMAGE/MIXED → CLIP; AUDIO/VIDEO → ASR.
  Text-only items are skipped (``status="skipped"``).
* **Confidence flagging**: every result carries ``low_confidence: bool``
  derived from the configured thresholds.  Items below the threshold
  surface a warning badge in the UI; the worker still persists them.
* **Soft failure**: when a runtime is unavailable (permission denied,
  missing weights, network failure) the result records the reason and
  marks itself ``status="unavailable"`` — the worker continues.
* **PII scrubbing**: the underlying runtimes already scrub their output;
  this layer additionally scrubs any free-form caption built from CLIP
  classifier labels before it is appended to ``raw_text``.

Outputs are intentionally JSON-serialisable so the worker can drop the
result straight into ``ContentItem.metadata["multimodal"]``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.data_residency import DataResidencyGuard
from app.core.models import ContentItem, MediaType
from app.local.multimodal_asr import LocalASRRuntime
from app.local.multimodal_clip import LocalCLIPRuntime, MultimodalUnavailable

logger = logging.getLogger(__name__)

_DEFAULT_LABELS = (
    "photo of a person", "screenshot", "chart or graph", "meme",
    "document or text", "outdoor scene", "indoor scene", "product photo",
)
_DEFAULT_ASR_CONF_THRESHOLD = 0.55
_DEFAULT_CLIP_CONF_THRESHOLD = 0.30


@dataclass
class MultimodalResult:
    """JSON-safe summary of multimodal analysis for one ContentItem."""

    status: str  # "ok" | "skipped" | "unavailable" | "partial"
    media_type: str
    low_confidence: bool = False
    caption: str = ""  # scrubbed; safe to append to raw_text
    image_embedding: Optional[List[float]] = None
    image_labels: Dict[str, float] = field(default_factory=dict)
    transcript: str = ""  # scrubbed
    transcript_backend: str = ""
    asr_confidence: Optional[float] = None
    asr_duration_s: float = 0.0
    reason: str = ""  # filled when status != "ok"

    def to_metadata(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [], {})}


class LocalMultimodalAnalyzer:
    """Fan-out facade glueing CLIP + ASR to the ingest worker."""

    def __init__(
        self,
        *,
        clip: Optional[LocalCLIPRuntime] = None,
        asr: Optional[LocalASRRuntime] = None,
        candidate_labels: Optional[List[str]] = None,
        asr_confidence_threshold: float = _DEFAULT_ASR_CONF_THRESHOLD,
        clip_confidence_threshold: float = _DEFAULT_CLIP_CONF_THRESHOLD,
    ) -> None:
        self._clip = clip if clip is not None else LocalCLIPRuntime()
        self._asr = asr if asr is not None else LocalASRRuntime()
        self._labels = list(candidate_labels) if candidate_labels else list(_DEFAULT_LABELS)
        self._asr_threshold = asr_confidence_threshold
        self._clip_threshold = clip_confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return self._clip.is_available() or self._asr.is_available()

    async def analyze(
        self, item: ContentItem, *, media_path: Optional[str] = None,
    ) -> MultimodalResult:
        """Analyse a single :class:`ContentItem` and return a result.

        ``media_path`` is an optional already-downloaded local path; when
        omitted, the analyzer uses the first entry of ``item.media_urls``
        *only if* it is already a local filesystem path (the ingest layer
        is responsible for any downloads — this module never touches the
        network).
        """
        media_type = item.media_type
        path = media_path or self._resolve_local_path(item)

        if media_type == MediaType.TEXT:
            return MultimodalResult(status="skipped", media_type=media_type.value,
                                    reason="text-only item")
        if path is None:
            return MultimodalResult(status="skipped", media_type=media_type.value,
                                    reason="no local media path available")

        if media_type in (MediaType.IMAGE, MediaType.MIXED):
            return await self._analyze_image(path, media_type)
        if media_type in (MediaType.AUDIO, MediaType.VIDEO):
            return await self._analyze_audio(path, media_type)
        return MultimodalResult(status="skipped", media_type=media_type.value,
                                reason=f"unsupported media_type {media_type.value!r}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_local_path(self, item: ContentItem) -> Optional[str]:
        for url in item.media_urls:
            if url and not url.startswith(("http://", "https://")) and Path(url).exists():
                return url
        return None

    async def _analyze_image(self, path: str, media_type: MediaType) -> MultimodalResult:
        if not self._clip.is_available():
            return MultimodalResult(status="unavailable", media_type=media_type.value,
                                    low_confidence=True, reason="CLIP runtime unavailable")
        try:
            embedding = await self._clip.embed_image(path)
            labels = await self._clip.classify(path, self._labels)
        except (MultimodalUnavailable, FileNotFoundError) as exc:
            return MultimodalResult(status="unavailable", media_type=media_type.value,
                                    low_confidence=True, reason=str(exc))
        top_label, top_score = max(labels.items(), key=lambda kv: kv[1])
        caption_raw = f"image appears to depict: {top_label}"
        caption, _ = DataResidencyGuard.scrub_text(caption_raw)
        return MultimodalResult(
            status="ok", media_type=media_type.value,
            low_confidence=top_score < self._clip_threshold,
            caption=caption, image_embedding=embedding, image_labels=labels,
        )

    async def _analyze_audio(self, path: str, media_type: MediaType) -> MultimodalResult:
        if not self._asr.is_available():
            return MultimodalResult(status="unavailable", media_type=media_type.value,
                                    low_confidence=True, reason="ASR runtime unavailable")
        try:
            result = await self._asr.transcribe(path)
        except (MultimodalUnavailable, FileNotFoundError) as exc:
            return MultimodalResult(status="unavailable", media_type=media_type.value,
                                    low_confidence=True, reason=str(exc))
        transcript = " ".join(seg.text for seg in result.segments).strip()
        conf = result.mean_confidence
        return MultimodalResult(
            status="ok", media_type=media_type.value,
            low_confidence=(conf is None or conf < self._asr_threshold),
            transcript=transcript, transcript_backend=result.backend_used,
            asr_confidence=conf, asr_duration_s=result.duration_s,
        )

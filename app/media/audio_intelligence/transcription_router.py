"""Transcription Router — ASR backend selection with domain correction.

Selects the best available ASR backend in priority order:
  1. ``faster-whisper`` (CTranslate2, fastest on CPU/GPU)
  2. ``whisper``        (OpenAI's original torch package)
  3. Cloud OpenAI ``whisper-1`` endpoint (requires API key)
  4. Stub              (returns empty transcript — testing/CI only)

After transcription, a domain-specific correction lexicon replaces
common AI/ML misrecognitions (e.g. "G P T" → "GPT", "lama" → "LLaMA").

Configuration (passed as constructor kwargs or environment):
    backend:          Force a specific backend (``"faster_whisper"``,
                      ``"whisper"``, ``"openai"``, ``"stub"``).
    model_size:       Whisper model size (``"base"`` … ``"large-v3"``).
    language:         Force language (BCP-47); ``None`` = auto-detect.
    openai_api_key:   OpenAI API key for cloud transcription.
    device:           ``"cpu"`` | ``"cuda"`` | ``"auto"`` for local backends.
"""

from __future__ import annotations

import logging
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional

from app.media.audio_intelligence.models import TranscriptSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain-aware AI / ML terminology correction lexicon
# ---------------------------------------------------------------------------
# Maps common ASR misrecognitions → correct form.
# Applied via regex word-boundary replacement (case-insensitive match).
_CORRECTION_LEXICON: dict[str, str] = {
    r"\bG\s*P\s*T\b": "GPT",
    r"\bG\s*P\s*T\s*-?\s*4\b": "GPT-4",
    r"\bG\s*P\s*T\s*-?\s*3\s*\.?\s*5\b": "GPT-3.5",
    r"\bchat\s*G\s*P\s*T\b": "ChatGPT",
    r"\blama\b": "LLaMA",
    r"\bll?ama\s*2\b": "LLaMA 2",
    r"\bll?ama\s*3\b": "LLaMA 3",
    r"\bmistral\b": "Mistral",
    r"\bclaude\b": "Claude",
    r"\bgem[iy]ni\b": "Gemini",
    r"\bgrok\b": "Grok",
    r"\bopen\s*eye\b": "OpenAI",
    r"\banthro\s*pic\b": "Anthropic",
    r"\bhugging\s*face\b": "Hugging Face",
    r"\btrans\s*former[s]?\b": "transformer",
    r"\bat\s*ten\s*tion\s+is\s+all\s+you\s+need\b": "Attention Is All You Need",
    r"\br\s*l\s*h\s*f\b": "RLHF",
    r"\br\s*l\s+h\s+f\b": "RLHF",
    r"\brag\b": "RAG",
    r"\bvector\s+data\s+base\b": "vector database",
    r"\bfine\s+tuning\b": "fine-tuning",
    r"\bfine\s+tune\b": "fine-tune",
    r"\bpython\b": "Python",
    r"\bkeras\b": "Keras",
    r"\bpie\s*torch\b": "PyTorch",
    r"\btensor\s*flow\b": "TensorFlow",
    r"\bopen\s*source\b": "open-source",
    r"\bm\s*l\s+ops\b": "MLOps",
    r"\bl\s+l\s+m[s]?\b": "LLM",
    r"\bsam\s+the\s+model\b": "SAM (Segment Anything Model)",
    r"\bstable\s+diff\s*usion\b": "Stable Diffusion",
    r"\bmid\s*journey\b": "Midjourney",
    r"\bopen\s*ai\s+whis\s*per\b": "OpenAI Whisper",
}

# Pre-compiled patterns (compiled once at import)
_COMPILED_CORRECTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pat, re.IGNORECASE), replacement)
    for pat, replacement in _CORRECTION_LEXICON.items()
]


def apply_domain_corrections(text: str) -> str:
    """Apply AI/ML terminology corrections to ASR output.

    Args:
        text: Raw ASR transcript string.

    Returns:
        Corrected string with AI/ML terms normalized.

    Raises:
        TypeError: If *text* is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"'text' must be str, got {type(text)!r}")
    for pattern, replacement in _COMPILED_CORRECTIONS:
        text = pattern.sub(replacement, text)
    return text


class ASRBackend(str, Enum):
    """Available ASR backends."""

    FASTER_WHISPER = "faster_whisper"
    WHISPER = "whisper"
    OPENAI = "openai"
    STUB = "stub"


class TranscriptionRouter:
    """Selects and invokes the best available ASR backend.

    Args:
        backend:       Force a specific backend.  If ``None`` (default),
                       auto-selects the best available.
        model_size:    Whisper model size string.
        language:      BCP-47 language code or ``None`` for auto-detect.
        openai_api_key: API key for OpenAI cloud transcription.
        device:        Compute device for local backends.
        apply_corrections: Apply domain lexicon corrections (default True).
    """

    def __init__(
        self,
        backend: Optional[ASRBackend] = None,
        model_size: str = "base",
        language: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        device: str = "auto",
        apply_corrections: bool = True,
    ) -> None:
        self._forced_backend = backend
        self._model_size = model_size
        self._language = language
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._device = device
        self._apply_corrections = apply_corrections
        self._resolved: Optional[ASRBackend] = None  # set on first use

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        """Transcribe an audio file and return time-aligned segments.

        Args:
            audio_path: Path to audio file (MP3, WAV, M4A, OGG, etc.).

        Returns:
            List of ``TranscriptSegment`` ordered by start time.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
            ValueError: If the path is empty.
        """
        if not audio_path or not isinstance(audio_path, str):
            raise ValueError("'audio_path' must be a non-empty string")
        p = Path(audio_path)
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        backend = self._resolve_backend()
        t0 = time.perf_counter()
        logger.info("TranscriptionRouter: backend=%s model=%s path=%s", backend.value, self._model_size, audio_path)

        if backend == ASRBackend.FASTER_WHISPER:
            segments = await self._transcribe_faster_whisper(audio_path)
        elif backend == ASRBackend.WHISPER:
            segments = await self._transcribe_whisper(audio_path)
        elif backend == ASRBackend.OPENAI:
            segments = await self._transcribe_openai(audio_path)
        else:
            segments = self._transcribe_stub(audio_path)

        if self._apply_corrections:
            segments = [
                TranscriptSegment(
                    start_s=seg.start_s,
                    end_s=seg.end_s,
                    text=apply_domain_corrections(seg.text),
                    confidence=seg.confidence,
                    language=seg.language,
                )
                for seg in segments
            ]
        logger.info(
            "TranscriptionRouter: done backend=%s segments=%d latency_ms=%.1f",
            backend.value, len(segments), (time.perf_counter() - t0) * 1000,
        )
        return segments

    def resolved_backend(self) -> ASRBackend:
        """Return the resolved backend (triggers resolution if not yet done)."""
        return self._resolve_backend()

    # ------------------------------------------------------------------
    # Backend resolution
    # ------------------------------------------------------------------

    def _resolve_backend(self) -> ASRBackend:
        if self._resolved is not None:
            return self._resolved
        if self._forced_backend is not None:
            self._resolved = self._forced_backend
            return self._resolved
        for backend, checker in [
            (ASRBackend.FASTER_WHISPER, self._check_faster_whisper),
            (ASRBackend.WHISPER, self._check_whisper),
            (ASRBackend.OPENAI, self._check_openai),
        ]:
            if checker():
                self._resolved = backend
                logger.info("TranscriptionRouter: auto-selected backend=%s", backend.value)
                return self._resolved
        self._resolved = ASRBackend.STUB
        logger.warning("TranscriptionRouter: no ASR backend available; using stub")
        return self._resolved

    @staticmethod
    def _check_faster_whisper() -> bool:
        try:
            import faster_whisper  # type: ignore[import]
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_whisper() -> bool:
        try:
            import whisper  # type: ignore[import]
            return True
        except ImportError:
            return False

    def _check_openai(self) -> bool:
        return bool(self._openai_api_key)

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    async def _transcribe_faster_whisper(self, audio_path: str) -> List[TranscriptSegment]:
        import asyncio
        import faster_whisper  # type: ignore[import]

        device = "cpu" if self._device == "auto" else self._device
        model = faster_whisper.WhisperModel(self._model_size, device=device, compute_type="int8")

        def _sync_transcribe():
            segments, info = model.transcribe(
                audio_path,
                language=self._language,
                beam_size=5,
                vad_filter=True,
            )
            return [
                TranscriptSegment(
                    start_s=seg.start,
                    end_s=seg.end,
                    text=seg.text.strip(),
                    confidence=None,
                    language=info.language,
                )
                for seg in segments
            ]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_transcribe)

    async def _transcribe_whisper(self, audio_path: str) -> List[TranscriptSegment]:
        import asyncio
        import whisper  # type: ignore[import]

        model = whisper.load_model(self._model_size)

        def _sync_transcribe():
            result = model.transcribe(audio_path, language=self._language, verbose=False)
            return [
                TranscriptSegment(
                    start_s=seg["start"],
                    end_s=seg["end"],
                    text=seg["text"].strip(),
                    confidence=None,
                    language=result.get("language", "en"),
                )
                for seg in result.get("segments", [])
            ]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_transcribe)

    async def _transcribe_openai(self, audio_path: str) -> List[TranscriptSegment]:
        import httpx

        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._openai_api_key}"},
                files={"file": (Path(audio_path).name, audio_bytes, "audio/mpeg")},
                data={"model": "whisper-1", "response_format": "verbose_json"},
            )
        resp.raise_for_status()
        data = resp.json()
        return [
            TranscriptSegment(
                start_s=seg["start"],
                end_s=seg["end"],
                text=seg["text"].strip(),
                confidence=None,
                language=data.get("language", "en"),
            )
            for seg in data.get("segments", [])
        ]

    @staticmethod
    def _transcribe_stub(audio_path: str) -> List[TranscriptSegment]:
        """Return a single stub segment (for testing / CI)."""
        logger.debug("TranscriptionRouter: stub returning empty segment for %s", audio_path)
        return [TranscriptSegment(start_s=0.0, end_s=1.0, text="[STUB TRANSCRIPT]", language="en")]


"""Speaker diarization.

Assigns speaker identities to raw ``TranscriptSegment`` objects.

Two modes:
1. **pyannote.audio** (optional) — high-quality neural diarization via
   ``pyannote.audio.Pipeline``.  Requires ``pyannote.audio`` and a Hugging
   Face auth token.
2. **Heuristic fallback** — assigns speakers based on configurable silence
   gap thresholds and sentence-boundary rules.  Works without any ML
   dependencies, suitable for CI / resource-constrained deployments.

Speaker role inference labels the first speaker in a two-speaker episode
as ``"host"`` and subsequent speakers as ``"guest"``.  For >2 speakers all
are ``"unknown"`` (role labeling requires domain knowledge).
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional

from app.media.audio_intelligence.models import DiarizedSegment, TranscriptSegment

logger = logging.getLogger(__name__)

_HOST_INDICATORS = re.compile(
    r"\b(welcome|today\s+we|joining\s+me|my\s+guest|let\s+me\s+introduce)\b",
    re.IGNORECASE,
)


class Diarizer:
    """Assigns speaker labels to transcript segments.

    Args:
        hf_token:        Hugging Face token for pyannote.audio (optional).
        pyannote_model:  pyannote model repo (default ``"pyannote/speaker-diarization-3.1"``).
        min_speakers:    Minimum speaker count hint.
        max_speakers:    Maximum speaker count hint.
        silence_gap_s:   Seconds of silence between turns (heuristic mode).
        audio_path:      Path to audio file (required for pyannote mode).
        use_pyannote:    Force pyannote (``True``) or heuristic (``False``).
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        pyannote_model: str = "pyannote/speaker-diarization-3.1",
        min_speakers: int = 1,
        max_speakers: int = 6,
        silence_gap_s: float = 1.0,
        audio_path: Optional[str] = None,
        use_pyannote: Optional[bool] = None,
    ) -> None:
        if min_speakers < 1:
            raise ValueError(f"'min_speakers' must be >= 1, got {min_speakers!r}")
        if max_speakers < min_speakers:
            raise ValueError(f"'max_speakers' ({max_speakers}) must be >= 'min_speakers' ({min_speakers})")
        if silence_gap_s < 0:
            raise ValueError(f"'silence_gap_s' must be >= 0, got {silence_gap_s!r}")

        self._hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self._pyannote_model = pyannote_model
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._silence_gap_s = silence_gap_s
        self._audio_path = audio_path
        # Auto-detect pyannote availability
        if use_pyannote is None:
            self._use_pyannote = self._pyannote_available() and bool(self._hf_token) and bool(audio_path)
        else:
            self._use_pyannote = use_pyannote

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def diarize(self, segments: List[TranscriptSegment]) -> List[DiarizedSegment]:
        """Assign speaker IDs to transcript segments.

        Args:
            segments: Raw ASR segments (must be ordered by start time).

        Returns:
            List of ``DiarizedSegment`` in the same order as *segments*.

        Raises:
            TypeError: If *segments* is not a list.
        """
        if not isinstance(segments, list):
            raise TypeError(f"'segments' must be a list, got {type(segments)!r}")
        if not segments:
            return []

        if self._use_pyannote and self._audio_path:
            try:
                return self._diarize_pyannote(segments)
            except Exception as exc:
                logger.warning("Diarizer: pyannote failed (%s), falling back to heuristic", exc)

        return self._diarize_heuristic(segments)

    # ------------------------------------------------------------------
    # Pyannote implementation
    # ------------------------------------------------------------------

    def _diarize_pyannote(self, segments: List[TranscriptSegment]) -> List[DiarizedSegment]:
        """Use pyannote.audio pipeline for diarization."""
        from pyannote.audio import Pipeline  # type: ignore[import]
        import torch  # type: ignore[import]

        pipeline = Pipeline.from_pretrained(self._pyannote_model, use_auth_token=self._hf_token)
        diarization = pipeline(
            self._audio_path,
            min_speakers=self._min_speakers,
            max_speakers=self._max_speakers,
        )
        # Build a time → speaker_id lookup
        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, speaker))

        result: List[DiarizedSegment] = []
        speaker_order: list[str] = []

        for seg in segments:
            mid = (seg.start_s + seg.end_s) / 2.0
            speaker_id = "SPEAKER_00"
            for t_start, t_end, spk in turns:
                if t_start <= mid <= t_end:
                    speaker_id = spk
                    break
            if speaker_id not in speaker_order:
                speaker_order.append(speaker_id)
            role = self._infer_role(speaker_id, speaker_order, seg.text)
            result.append(DiarizedSegment(segment=seg, speaker_id=speaker_id, speaker_role=role))

        return result

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _diarize_heuristic(self, segments: List[TranscriptSegment]) -> List[DiarizedSegment]:
        """Assign speakers using silence gaps and sentence-boundary heuristics."""
        result: List[DiarizedSegment] = []
        current_speaker_idx = 0
        speaker_order: list[str] = []
        prev_end = 0.0

        for i, seg in enumerate(segments):
            gap = seg.start_s - prev_end if i > 0 else 0.0
            # Speaker change heuristic: large gap or explicit introduction cue
            if gap >= self._silence_gap_s and i > 0:
                current_speaker_idx = (current_speaker_idx + 1) % max(self._max_speakers, 2)

            speaker_id = f"SPEAKER_{current_speaker_idx:02d}"
            if speaker_id not in speaker_order:
                speaker_order.append(speaker_id)
            role = self._infer_role(speaker_id, speaker_order, seg.text)
            result.append(DiarizedSegment(segment=seg, speaker_id=speaker_id, speaker_role=role))
            prev_end = seg.end_s

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_role(speaker_id: str, speaker_order: list[str], text: str) -> str:
        """Infer host / guest role from speaker order and text cues."""
        if len(speaker_order) <= 2:
            if speaker_order and speaker_id == speaker_order[0]:
                return "host"
            if _HOST_INDICATORS.search(text):
                return "host"
            return "guest" if len(speaker_order) > 1 else "host"
        return "unknown"

    @staticmethod
    def _pyannote_available() -> bool:
        try:
            import pyannote.audio  # type: ignore[import]
            return True
        except ImportError:
            return False


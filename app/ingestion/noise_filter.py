"""AcquisitionNoiseFilter — pre-normalization content gating.

Runs immediately after content is fetched from a connector and *before*
``NormalizationEngine`` to eliminate low-value content early, saving
embedding budget and pipeline latency.

Filter stages (applied in order):
  1. Platform / content-type gate   — ``StrategicPriorities.platforms_enabled``
  2. Keyword blocklist              — ``StrategicPriorities.keywords_blocklist``
  3. Keyword allowlist              — ``StrategicPriorities.keywords_allowlist``
  4. Engagement threshold           — ``StrategicPriorities.min_engagement_threshold``
  5. Near-duplicate fingerprint     — 24 h sliding window per user
  6. Bot / spam heuristics          — caps-ratio, URL density, post-frequency
  7. Minimum text length            — configurable floor (default 20 chars)
  8. Language filter                — ``StrategicPriorities`` (reserved hook)

Every drop appends a ``noise_filter_decision`` entry to
``RawObservation.platform_metadata`` for full auditability.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from app.domain.raw_models import RawObservation
from app.domain.inference_models import StrategicPriorities

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level deduplication state (in-process 24-h window)
# ---------------------------------------------------------------------------
# Maps user_id -> set of content fingerprints seen in the last 24 h.
# In production this would be backed by Redis (SADD + EXPIRE), but for a
# single-process deployment or unit-testing the in-memory set is sufficient.
_seen_fingerprints: Dict[str, Dict[str, float]] = defaultdict(dict)
_DEDUP_WINDOW_SECONDS: int = 86_400  # 24 h


def _make_fingerprint(obs: RawObservation) -> str:
    """Return a short fingerprint for near-duplicate detection.

    Uses the first 64 characters of raw_text (or title if no text),
    the author, and the platform.  Collisions are extremely rare for
    distinct posts and acceptably frequent for true near-duplicates.
    """
    body = (obs.raw_text or obs.title or "")[:64]
    raw = f"{obs.source_platform.value}|{obs.author or ''}|{body}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _purge_expired(user_key: str) -> None:
    """Remove fingerprints older than the dedup window."""
    now = time.monotonic()
    expired = [fp for fp, ts in _seen_fingerprints[user_key].items()
               if now - ts > _DEDUP_WINDOW_SECONDS]
    for fp in expired:
        del _seen_fingerprints[user_key][fp]


# ---------------------------------------------------------------------------
# Bot/spam heuristics
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+")


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _url_density(text: str) -> float:
    """URLs per 100 chars of text."""
    if not text:
        return 0.0
    urls = len(_URL_RE.findall(text))
    return urls / (len(text) / 100)


def _is_likely_bot(obs: RawObservation, caps_threshold: float = 0.7,
                   url_density_threshold: float = 3.0) -> bool:
    text = obs.raw_text or ""
    if _caps_ratio(text) >= caps_threshold:
        return True
    if _url_density(text) >= url_density_threshold:
        return True
    return False




# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class FilterDecision:
    """Result of a single filter evaluation."""

    __slots__ = ("passed", "reason", "stage")

    def __init__(self, passed: bool, stage: str, reason: str = "") -> None:
        self.passed = passed
        self.stage = stage
        self.reason = reason

    def as_dict(self) -> dict:
        return {"passed": self.passed, "stage": self.stage, "reason": self.reason}


class AcquisitionNoiseFilter:
    """Pre-normalization content gating.

    Parameters
    ----------
    min_text_length:
        Drop observations whose ``raw_text`` (or ``title``) is shorter
        than this many characters.  Default: 20.
    """

    def __init__(self, min_text_length: int = 20) -> None:
        self._min_text_length = min_text_length

    def filter(
        self,
        obs: RawObservation,
        priorities: Optional[StrategicPriorities] = None,
    ) -> Tuple[bool, List[FilterDecision]]:
        """Evaluate all filter stages for *obs*.

        Returns ``(should_ingest, decisions)`` where ``should_ingest`` is
        ``True`` when the observation clears every enabled stage.
        ``decisions`` is appended to ``obs.platform_metadata`` for auditability.
        """
        sp = priorities or StrategicPriorities()
        decisions: List[FilterDecision] = []

        def _drop(stage: str, reason: str) -> Tuple[bool, List[FilterDecision]]:
            decisions.append(FilterDecision(False, stage, reason))
            obs.platform_metadata["noise_filter_decision"] = [d.as_dict() for d in decisions]
            logger.debug("DROP [%s] obs=%s: %s", stage, obs.id, reason)
            return False, decisions

        def _pass(stage: str) -> None:
            decisions.append(FilterDecision(True, stage))

        text = (obs.raw_text or obs.title or "").lower()

        # Stage 1 — Platform gate
        if sp.platforms_enabled:
            if obs.source_platform.value not in sp.platforms_enabled:
                return _drop("platform_gate",
                             f"platform '{obs.source_platform.value}' not in platforms_enabled")
        _pass("platform_gate")

        # Stage 2 — Keyword blocklist
        for kw in sp.keywords_blocklist:
            if kw.lower() in text:
                return _drop("keyword_blocklist", f"blocked keyword: '{kw}'")
        _pass("keyword_blocklist")

        # Stage 3 — Keyword allowlist
        if sp.keywords_allowlist:
            if not any(kw.lower() in text for kw in sp.keywords_allowlist):
                return _drop("keyword_allowlist", "no allowlist keyword found")
        _pass("keyword_allowlist")

        # Stage 4 — Engagement threshold
        if sp.min_engagement_threshold > 0:
            engagement = (
                obs.platform_metadata.get("upvotes")
                or obs.platform_metadata.get("likes")
                or obs.platform_metadata.get("shares")
                or 0
            )
            if engagement < sp.min_engagement_threshold:
                return _drop("engagement_threshold",
                             f"engagement {engagement} < {sp.min_engagement_threshold}")
        _pass("engagement_threshold")

        # Stage 5 — Near-duplicate deduplication
        user_key = str(obs.user_id)
        _purge_expired(user_key)
        fp = _make_fingerprint(obs)
        if fp in _seen_fingerprints[user_key]:
            return _drop("deduplication", "near-duplicate fingerprint seen within 24 h")
        _seen_fingerprints[user_key][fp] = time.monotonic()
        _pass("deduplication")

        # Stage 6 — Bot/spam heuristics
        if _is_likely_bot(obs):
            return _drop("bot_detection", "caps-ratio or URL density exceeds threshold")
        _pass("bot_detection")

        # Stage 7 — Minimum text length
        raw = obs.raw_text or obs.title or ""
        if len(raw) < self._min_text_length:
            return _drop("min_text_length",
                         f"text length {len(raw)} < {self._min_text_length}")
        _pass("min_text_length")

        obs.platform_metadata["noise_filter_decision"] = [d.as_dict() for d in decisions]
        return True, decisions

    def filter_batch(
        self,
        observations: List[RawObservation],
        priorities: Optional[StrategicPriorities] = None,
    ) -> Tuple[List[RawObservation], int]:
        """Filter a batch; returns ``(accepted, noise_filtered_count)``."""
        accepted: List[RawObservation] = []
        dropped = 0
        for obs in observations:
            ok, _ = self.filter(obs, priorities)
            if ok:
                accepted.append(obs)
            else:
                dropped += 1
        return accepted, dropped

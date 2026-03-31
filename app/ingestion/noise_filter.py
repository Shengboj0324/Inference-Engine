"""AcquisitionNoiseFilter — pre-normalization content gating.

Runs immediately after content is fetched from a connector and *before*
``NormalizationEngine`` to eliminate low-value content early, saving
embedding budget and pipeline latency.

Filter stages (applied in order):
  1. Platform / content-type gate   — ``StrategicPriorities.platforms_enabled``
  2. Keyword blocklist              — ``StrategicPriorities.keywords_blocklist``
  3. Keyword allowlist              — ``StrategicPriorities.keywords_allowlist``
  4. Engagement threshold           — ``StrategicPriorities.min_engagement_threshold``
                                      Uses ``normalize_engagement()`` which maps each
                                      platform's native metric to a canonical
                                      ``engagement_score`` field.
  5. Near-duplicate fingerprint     — 24 h sliding window per user.  When a
                                      canonical ``source_url`` is present the
                                      fingerprint is URL-based (UTM params stripped)
                                      enabling *cross-platform* deduplication.
  6. Bot / spam heuristics          — caps-ratio, URL density, post-frequency
  7. Minimum text length            — configurable floor (default 20 chars)
  8. Trending gate                  — ``StrategicPriorities.trending_only``
                                      Drops observations whose
                                      ``platform_metadata["is_trending"]`` is falsy.
                                      Connectors that do not expose trending signals
                                      must emit ``is_trending=False`` by default;
                                      ``ConnectorRegistry.apply_acquisition_filter``
                                      guarantees this invariant.

Every drop appends a ``noise_filter_decision`` entry to
``RawObservation.platform_metadata`` for full auditability.
Every passing observation receives a ``relevance_score`` (0.0–1.0) written
by ``RelevanceScorer`` based on ``StrategicPriorities`` keyword density.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

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

# URL query parameters that carry zero content-identity information.
_TRACKING_PARAMS: frozenset = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "utm_source_platform", "fbclid", "gclid", "msclkid",
    "ref", "source", "s", "via", "_hsenc", "_hsmi", "mc_eid",
})


def _normalize_url(url: str) -> str:
    """Return a canonical form of *url* for cross-platform deduplication.

    Transformations applied (all safe / invertible):
    - Scheme and netloc lowercased.
    - ``www.`` prefix stripped from netloc.
    - Trailing slashes removed from path.
    - Tracking / UTM query parameters removed.
    - URL fragment discarded (fragments are client-only, never server-canonical).

    Falls back to the original URL if parsing fails (e.g. relative URL).
    """
    try:
        p = urlparse(url.strip())
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        qs_pairs = [
            (k, v) for k, v in parse_qsl(p.query)
            if k.lower() not in _TRACKING_PARAMS
        ]
        return urlunparse((
            p.scheme.lower(),
            netloc,
            p.path.rstrip("/"),
            p.params,
            urlencode(qs_pairs),
            "",  # discard fragment
        ))
    except Exception:
        return url.strip()


def _make_fingerprint(obs: RawObservation) -> str:
    """Return a short fingerprint for near-duplicate detection.

    Strategy
    --------
    When *obs* has a non-empty ``source_url`` we derive the fingerprint
    exclusively from its *normalised* form (UTM params stripped, ``www.``
    removed, trailing slash removed).  This means the **same article URL
    ingested from multiple connectors** (e.g. NYTimes RSS + Google News +
    Apple News) produces the **same fingerprint**, enabling cross-platform
    deduplication.

    When no URL is available we fall back to the original platform-scoped
    strategy (platform + author + first 64 chars of body text), which is
    appropriate for social-media posts that lack canonical URLs.
    """
    if obs.source_url:
        raw = _normalize_url(obs.source_url)
    else:
        body = (obs.raw_text or obs.title or "")[:64]
        raw = f"{obs.source_platform.value}|{obs.author or ''}|{body}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Engagement normalisation
# ---------------------------------------------------------------------------

# Maps SourcePlatform.value → ordered list of metadata keys to try.
# The first key found with a non-None value is used as the canonical score.
_ENGAGEMENT_KEY_MAP: Dict[str, List[str]] = {
    # Social media — primary engagement metrics
    "reddit":      ["score", "upvotes"],            # PRAW exposes .score
    "youtube":     ["view_count", "like_count"],
    "tiktok":      ["play_count", "digg_count", "like_count"],
    "facebook":    ["reactions_count", "shares_count", "like_count"],
    "instagram":   ["like_count", "play_count"],
    "wechat":      ["read_count", "like_count"],
    # News / RSS — engagement signals are sparse; views from NYTimes
    # most_popular mode is the only real signal
    "rss":         ["like_count", "share_count", "views"],
    "nytimes":     ["views", "share_count"],
    "wsj":         ["views", "share_count", "like_count"],
    "abc_news":    ["views", "share_count"],
    "abc_news_au": ["views", "share_count"],
    "google_news": ["like_count", "share_count"],
    "apple_news":  ["like_count", "share_count"],
}


def _coerce_to_int(val: Any) -> Optional[int]:
    """Safely coerce *val* to a non-negative ``int``, or return ``None`` on failure.

    Handles the full range of values connectors may produce:

    * Native ``int`` / ``float`` / ``bool`` — cast directly.
    * Numeric strings with locale-style commas (e.g. ``"1,234"``).
    * Numeric strings with surrounding whitespace (e.g. ``" 500 "``).

    Returns ``None`` (instead of raising) for non-numeric strings (``"n/a"``)
    and for any other value that cannot be meaningfully interpreted as an
    integer engagement count.  Negative values are clamped to ``0``.
    """
    if val is None:
        return None
    # bool is a subclass of int in Python — handle explicitly to avoid
    # treating True/False as meaningful engagement signals.
    if isinstance(val, bool):
        return int(val)  # True→1, False→0; still valid (rare but correct)
    if isinstance(val, (int, float)):
        return max(0, int(val))
    # String path — strip whitespace and locale commas before converting.
    try:
        cleaned = str(val).replace(",", "").strip()
        return max(0, int(float(cleaned)))
    except (ValueError, TypeError):
        return None


def normalize_engagement(
    platform_metadata: Dict[str, Any],
    source_platform: "Any",  # SourcePlatform; avoid circular import
) -> int:
    """Return a canonical ``engagement_score`` (int ≥ 0) from *platform_metadata*.

    The function reads platform-specific engagement metric keys (in priority
    order per :data:`_ENGAGEMENT_KEY_MAP`) and returns the first non-``None``
    value found, coerced to a non-negative ``int`` via :func:`_coerce_to_int`.
    Returns ``0`` when no valid engagement data is available.

    **Side effect:** the result is written back under the ``"engagement_score"``
    key so that the Stage 4 engagement-threshold check and downstream analytics
    can rely on a single, platform-agnostic field regardless of connector.

    Parameters
    ----------
    platform_metadata:
        The ``RawObservation.platform_metadata`` dict (mutated in place).
    source_platform:
        A :class:`~app.core.models.SourcePlatform` enum instance.

    Notes
    -----
    Idempotent: if ``"engagement_score"`` is already present and is a valid
    numeric value, it is returned immediately without re-scanning the key map.
    If the cached value is itself non-numeric (a misconfigured connector wrote
    a string like ``"n/a"``), the cache is cleared and the key map is re-scanned
    so Stage 4 always receives a reliable integer.
    """
    # Short-circuit: already normalised (e.g. called twice in the same request).
    # Guard against a cached non-numeric value left by a misconfigured connector.
    if "engagement_score" in platform_metadata:
        cached = _coerce_to_int(platform_metadata["engagement_score"])
        if cached is not None:
            return cached
        # Cached value is non-numeric; evict it and fall through to re-scan.
        del platform_metadata["engagement_score"]

    platform_value = getattr(source_platform, "value", str(source_platform))
    keys = _ENGAGEMENT_KEY_MAP.get(platform_value, [])
    for key in keys:
        val = platform_metadata.get(key)
        if val is None:
            continue
        score = _coerce_to_int(val)
        if score is None:
            # This key exists but holds a non-numeric value; skip to the next
            # key in priority order rather than treating it as 0.
            continue
        platform_metadata["engagement_score"] = score
        return score

    platform_metadata["engagement_score"] = 0
    return 0


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
        # ``normalize_engagement`` maps every platform's native key(s) to the
        # canonical ``engagement_score`` field and returns the value.  This
        # replaces the previous hard-coded ``upvotes / likes / shares`` lookup
        # which never matched Reddit (``score``) or TikTok (``play_count``).
        if sp.min_engagement_threshold > 0:
            engagement = normalize_engagement(
                obs.platform_metadata, obs.source_platform
            )
            if engagement < sp.min_engagement_threshold:
                return _drop(
                    "engagement_threshold",
                    f"engagement {engagement} < {sp.min_engagement_threshold}",
                )
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

        # Stage 8 — Trending gate (enforces StrategicPriorities.trending_only)
        # ``is_trending`` must be injected by the connector or defaulted to False
        # by ``ConnectorRegistry.apply_acquisition_filter`` before this filter
        # runs.  Connectors that do not expose platform trending signals emit
        # ``is_trending=False``; enabling ``trending_only`` for those platforms
        # effectively mutes them (which is the correct, explicit behaviour).
        if sp.trending_only:
            if not obs.platform_metadata.get("is_trending"):
                return _drop(
                    "trending_gate",
                    "trending_only=True but is_trending is absent or False",
                )
        _pass("trending_gate")

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



# ---------------------------------------------------------------------------
# Relevance Scorer
# ---------------------------------------------------------------------------

class RelevanceScorer:
    """Score a *passing* observation by keyword relevance to StrategicPriorities.

    This replaces the binary pass/fail allowlist with a ranked signal.  Every
    observation that clears the 8-stage filter receives a ``relevance_score``
    written to ``obs.platform_metadata["relevance_score"]``.

    Scoring formula
    ---------------
    Each unique keyword match in the observation text contributes a weight
    according to its category:

    * ``keywords_allowlist`` → weight **1.0** per match
    * ``focus_areas``        → weight **1.5** per match (domain-specific)
    * ``competitors``        → weight **2.0** per match (highest GTM value)

    The raw sum is normalised through ``tanh(raw / scale)`` to produce a
    monotonically increasing score in **[0.0, 1.0]**:

    * 0 matching terms → **0.0**
    * 1 allowlist hit  → ≈ **0.48**
    * 1 focus-area hit → ≈ **0.64**
    * 1 competitor hit → ≈ **0.76**
    * 3 mixed hits     → ≈ **0.95**

    The ``scale`` parameter (default **2.1**) controls the knee of the curve.
    It can be tuned per-deployment without changing the interface.

    Parameters
    ----------
    scale:
        Denominator for the tanh argument.  Larger values flatten the curve
        (more hits required to reach 1.0); smaller values steepen it.
    """

    _WEIGHTS: Dict[str, float] = {
        "keywords_allowlist": 1.0,
        "focus_areas": 1.5,
        "competitors": 2.0,
    }

    def __init__(self, scale: float = 2.1) -> None:
        self._scale = scale

    def score(
        self,
        obs: RawObservation,
        priorities: StrategicPriorities,
    ) -> float:
        """Compute a relevance score and stamp it into ``obs.platform_metadata``.

        Parameters
        ----------
        obs:
            The observation to score (mutated in place via ``platform_metadata``).
        priorities:
            The user's ``StrategicPriorities`` supplying the keyword sets.

        Returns
        -------
        float
            The computed relevance score in **[0.0, 1.0]**.
        """
        text = (obs.raw_text or obs.title or "").lower()
        raw_score = 0.0

        for kw in priorities.keywords_allowlist:
            if kw.lower() in text:
                raw_score += self._WEIGHTS["keywords_allowlist"]

        for kw in priorities.focus_areas:
            if kw.lower() in text:
                raw_score += self._WEIGHTS["focus_areas"]

        for kw in priorities.competitors:
            if kw.lower() in text:
                raw_score += self._WEIGHTS["competitors"]

        relevance = round(
            math.tanh(raw_score / self._scale) if raw_score > 0 else 0.0,
            4,
        )
        obs.platform_metadata["relevance_score"] = relevance
        return relevance

    def score_batch(
        self,
        observations: List[RawObservation],
        priorities: StrategicPriorities,
    ) -> List[float]:
        """Score a batch of observations in-place.

        Returns the list of scores in the same order as *observations*.
        """
        return [self.score(obs, priorities) for obs in observations]

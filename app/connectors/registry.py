"""Connector registry for platform-specific connector instantiation.

Connector Capability Matrix
===========================
Documents each of the 13 connectors across four capability dimensions that
the acquisition pipeline depends on.  Update this table whenever a connector
is modified.

Legend
------
 ✓  : Capability is implemented.
 ~  : Partially implemented or dependent on API mode.
 ✗  : Not implemented; downstream stages must not assume this field.
 N/A: Not applicable (public/RSS connector, no OAuth).

+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| Platform          | SP fields used   | is_trending  | engagement_score | source_url          | Rate-limit handling | Query strategy   |
|                   | at query time    | emitted      | (native key)     | populated           |                     |                  |
+===================+==================+==============+==================+=====================+=====================+==================+
| reddit            | ✗ (uses subscr.) | ✗ (default)  | ✓ (score)        | ✓ permalink         | PlatformError       | Broad fetch,     |
|                   |                  |              |                  |                     | raised on exc.      | post-fetch filter|
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| youtube           | ~ (config query) | ✗ (default)  | ✗ (no view_count)| ✓ watch URL         | PlatformError       | Channel search;  |
|                   |                  |              | view_count absent| built in connector  | on HttpError        | relevanceLang ✗  |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| tiktok            | ~ (config kw/ht) | ✗ (default)  | ~ (play_count    | ✓ user/video URL    | ConnectorError      | Keyword & hashtag|
|                   |                  |              | via view_count)  | built in _parse     | raised; partial FR  | query at API     |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| facebook          | ✗ (config pages) | ✗ (default)  | ✓ reactions_count| ✓ post link         | ConnectorError      | Feed/page IDs;   |
|                   |                  |              |                  | or FB permalink     | raised on 4xx       | no keyword query |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| instagram         | ~ (config tags)  | ✗ (default)  | ✓ like_count     | ✓ permalink         | ConnectorError      | Hashtag & account|
|                   |                  |              |                  |                     | raised on non-200   | query at API     |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| wechat            | ✗ (account-wide) | ✗ (default)  | ✗ (no engagement | ✓ article URL       | ConnectorError      | Account material |
|                   |                  |              | data in API)     |                     | raised on errcode   | batch; no query  |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| rss               | ✗ (feed URLs)    | ✗ (default)  | ✗ (no engagement)| ✓ entry link        | PlatformError       | Fixed URLs;      |
|                   |                  |              |                  |                     | raised on HTTPError | no keyword query |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| nytimes           | ~ (config query) | ~ (most_pop) | ~ views (most_pop| ✓ web_url field     | ConnectorError      | Article search   |
|                   |                  | mode only)   | mode only)       |                     | on 429; raised      | w/ config query  |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| wsj               | ✗ (config feeds) | ✗ (default)  | ✗ (no engagement)| ✓ entry link        | Silent warning;     | Named feed URLs; |
|                   |                  |              |                  |                     | returns partial FR  | no keyword query |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| abc_news          | ✗ (config feeds) | ✗ (default)  | ✗ (no engagement)| ✓ entry link        | Silent warning;     | Named feed URLs; |
| abc_news_au       |                  |              |                  |                     | returns partial FR  | no keyword query |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| google_news       | ~ (config kw/top)| ✗ (default)  | ✗ (no engagement)| ✓ entry link        | Silent warning;     | Keyword & topic  |
|                   |                  |              |                  |                     | returns partial FR  | search at API    |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+
| apple_news        | ✗ (scraping)     | ✗ (default)  | ✗ (no engagement)| ✓ scraped URL       | Silent warning;     | Newsroom scrape; |
|                   |                  |              |                  |                     | returns partial FR  | no keyword query |
+-------------------+------------------+--------------+------------------+---------------------+---------------------+------------------+

Key observations from the audit
--------------------------------
1. Reddit uses ``score`` (not ``upvotes``) — the old Stage 4 engagement check
   always read 0 for Reddit.  Fixed via ``normalize_engagement()``.
2. YouTube's ``_fetch_from_channel`` does not include ``view_count`` in
   ``metadata``; only ``channel_id``, ``video_id``, and ``thumbnails`` are
   present.  ``normalize_engagement`` falls back to 0 for YouTube until the
   connector is updated to request statistics.
3. WeChat and all RSS/news connectors do not expose engagement data; they
   safely score 0 for ``min_engagement_threshold`` when the threshold is
   non-zero.  Users should set ``min_engagement_threshold=0`` when monitoring
   news-only sources.
4. No connector populates ``is_trending``; ``apply_acquisition_filter``
   therefore injects ``is_trending=False`` before running the filter.  Users
   enabling ``trending_only=True`` must also configure a connector that
   actively queries platform trending APIs (e.g. TikTok ``trending`` endpoint).
5. The deduplication fingerprint is now URL-based when ``source_url`` is
   present, enabling cross-platform dedup (same article from NYTimes RSS +
   Google News shares one fingerprint).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Type
from uuid import UUID

from app.connectors.abc_news import ABCNewsConnector
from app.connectors.apple_news import AppleNewsConnector
from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult
from app.connectors.facebook import FacebookConnector
from app.connectors.google_news import GoogleNewsConnector
from app.connectors.instagram import InstagramConnector
from app.connectors.nytimes import NYTimesConnector
from app.connectors.reddit import RedditConnector
from app.connectors.rss import RSSConnector
from app.connectors.tiktok import TikTokConnector
from app.connectors.wechat import WeChatConnector
from app.connectors.wsj import WSJConnector
from app.connectors.youtube import YouTubeConnector
from app.core.errors import ConnectorError
from app.core.models import SourcePlatform
from app.domain.inference_models import StrategicPriorities
from app.ingestion.noise_filter import AcquisitionNoiseFilter, normalize_engagement

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OAuth scope documentation per connector (read-only minimum)
# ---------------------------------------------------------------------------
# Platforms that rely on public / API-key access do not appear here —
# they are documented in ConnectorRegistry.get_platform_info().
OAUTH_SCOPES: Dict[SourcePlatform, List[str]] = {
    SourcePlatform.REDDIT:    ["identity", "read", "mysubreddits", "history"],
    SourcePlatform.YOUTUBE:   ["https://www.googleapis.com/auth/youtube.readonly"],
    SourcePlatform.TIKTOK:    ["user.info.basic", "video.list"],
    SourcePlatform.FACEBOOK:  ["public_profile", "user_posts", "pages_read_engagement"],
    SourcePlatform.INSTAGRAM: ["user_profile", "user_media"],
    SourcePlatform.WECHAT:    ["snsapi_userinfo"],
}

# Public / API-key platforms — no OAuth required.
PUBLIC_ACCESS_PLATFORMS: List[SourcePlatform] = [
    SourcePlatform.RSS,
    SourcePlatform.NYTIMES,
    SourcePlatform.WSJ,
    SourcePlatform.ABC_NEWS,
    SourcePlatform.ABC_NEWS_AU,
    SourcePlatform.GOOGLE_NEWS,
    SourcePlatform.APPLE_NEWS,
]


class ConnectorRegistry:
    """Registry for platform-specific connectors."""

    _connectors: Dict[SourcePlatform, Type[BaseConnector]] = {
        # Social Media Platforms
        SourcePlatform.REDDIT: RedditConnector,
        SourcePlatform.YOUTUBE: YouTubeConnector,
        SourcePlatform.TIKTOK: TikTokConnector,
        SourcePlatform.FACEBOOK: FacebookConnector,
        SourcePlatform.INSTAGRAM: InstagramConnector,
        SourcePlatform.WECHAT: WeChatConnector,
        # News Sources
        SourcePlatform.RSS: RSSConnector,
        SourcePlatform.NYTIMES: NYTimesConnector,
        SourcePlatform.WSJ: WSJConnector,
        SourcePlatform.ABC_NEWS: ABCNewsConnector,
        SourcePlatform.ABC_NEWS_AU: ABCNewsConnector,
        SourcePlatform.GOOGLE_NEWS: GoogleNewsConnector,
        SourcePlatform.APPLE_NEWS: AppleNewsConnector,
    }

    @classmethod
    def get_connector(
        cls,
        platform: SourcePlatform,
        config: ConnectorConfig,
        user_id: UUID,
    ) -> BaseConnector:
        """Get connector instance for platform.
        
        Args:
            platform: Source platform
            config: Connector configuration
            user_id: User ID
            
        Returns:
            Connector instance
            
        Raises:
            ConnectorError: If platform not supported
        """
        connector_class = cls._connectors.get(platform)
        
        if not connector_class:
            raise ConnectorError(f"Unsupported platform: {platform}")
        
        return connector_class(config, user_id)

    @classmethod
    def get_supported_platforms(cls) -> list[SourcePlatform]:
        """Get list of supported platforms."""
        return list(cls._connectors.keys())

    @classmethod
    def is_platform_supported(cls, platform: SourcePlatform) -> bool:
        """Check if platform is supported."""
        return platform in cls._connectors

    @classmethod
    def register_connector(
        cls,
        platform: SourcePlatform,
        connector_class: Type[BaseConnector],
    ) -> None:
        """Register a custom connector.
        
        Args:
            platform: Source platform
            connector_class: Connector class
        """
        cls._connectors[platform] = connector_class

    @classmethod
    def get_oauth_scopes(cls, platform: SourcePlatform) -> List[str]:
        """Return the minimum required OAuth scopes for *platform*.

        Returns an empty list for public/API-key platforms that do not
        require user-delegated OAuth.
        """
        return OAUTH_SCOPES.get(platform, [])

    @classmethod
    def apply_acquisition_filter(
        cls,
        fetch_result: "FetchResult",
        priorities: Optional[StrategicPriorities],
        user_id: UUID,
        noise_filter: Optional[AcquisitionNoiseFilter] = None,
        max_downstream_chars: int = 500_000,
    ) -> "FetchResult":
        """Run ``AcquisitionNoiseFilter`` against every item in *fetch_result*.

        Content that fails any filter stage is removed from ``fetch_result.items``
        before the items enter ``NormalizationEngine``.  A log line is emitted
        for each batch summarising how many items were accepted vs. dropped.

        After the 8-stage noise filter the accepted list is further trimmed by
        the downstream character budget so that the cumulative ``raw_text``
        length of all accepted items never exceeds *max_downstream_chars*.
        When ``priorities.max_downstream_chars`` is set it overrides the
        call-site default.  Pass ``max_downstream_chars=0`` to disable budget
        enforcement.

        Args:
            fetch_result: Raw fetch result from a connector.
            priorities: User's ``StrategicPriorities``; ``None`` = no filtering.
            user_id: Used for per-user dedup state.
            noise_filter: Optional pre-built filter instance; defaults to a new
                          ``AcquisitionNoiseFilter()`` with default settings.
            max_downstream_chars: System-level character budget ceiling.

        Returns:
            A mutated ``FetchResult`` with only accepted, budget-fitting items.
        """
        from app.domain.raw_models import RawObservation
        from app.core.models import MediaType

        if not fetch_result.items:
            return fetch_result

        nf = noise_filter or AcquisitionNoiseFilter()

        # ── Hoist per-batch constants outside the per-item loop ──────────────
        # `priorities` never changes between iterations; computing `_sp` and
        # constructing `MultimodalAnalyzer` once avoids 500× object allocation
        # in large batches and makes patching in tests straightforward.
        _sp = priorities or StrategicPriorities()
        _mm_analyzer = None
        if _sp.multimodal_enabled:
            try:
                from app.intelligence.multimodal import MultimodalAnalyzer
                _mm_analyzer = MultimodalAnalyzer()
            except Exception as _mm_init_exc:
                logger.debug(
                    "AcquisitionFilter: MultimodalAnalyzer init failed; "
                    "multimodal enrichment disabled for this batch: %s",
                    _mm_init_exc,
                )
        # ─────────────────────────────────────────────────────────────────────

        # Wrap ContentItems into RawObservation shells for the filter
        accepted_items = []
        dropped = 0
        for item in fetch_result.items:
            # Build a minimal RawObservation proxy so the filter can run
            try:
                raw = RawObservation(
                    user_id=item.user_id,
                    source_platform=item.source_platform,
                    source_id=item.source_id,
                    source_url=item.source_url,
                    author=item.author,
                    title=item.title,
                    raw_text=item.raw_text,
                    media_type=item.media_type,
                    published_at=item.published_at,
                    platform_metadata=dict(item.metadata),
                )
            except Exception as exc:
                logger.debug("AcquisitionFilter: failed to wrap item %s: %s", item.id, exc)
                accepted_items.append(item)
                continue

            # ── Pre-filter enrichment ────────────────────────────────────────
            # 1. Guarantee ``is_trending`` is present (Stage 8 contract).
            #    No current connector populates this field; injecting the
            #    default here means Stage 8 is always safe to run.
            raw.platform_metadata.setdefault("is_trending", False)

            # 2. Normalise engagement to ``engagement_score`` before Stage 4.
            #    This maps platform-specific keys (Reddit ``score``,
            #    TikTok ``play_count``, etc.) to a single canonical field so
            #    the filter never reads a stale or missing key.
            normalize_engagement(raw.platform_metadata, raw.source_platform)

            # 3. Multimodal enrichment — when the observation contains image
            #    or video metadata keys AND multimodal_enabled=True in the
            #    user's StrategicPriorities, append a RAG-ready visual
            #    description to raw_text before the 8-stage noise filter runs.
            #    (_mm_analyzer is pre-built once per batch above the loop.)
            if _mm_analyzer is not None:
                try:
                    if _mm_analyzer.has_visual_content(raw):
                        visual_text = _mm_analyzer.visual_to_text(raw)
                        if visual_text:
                            raw.raw_text = (raw.raw_text or "") + " " + visual_text
                except Exception as _mm_exc:
                    logger.debug(
                        "AcquisitionFilter: multimodal enrichment failed: %s",
                        _mm_exc,
                    )
            # ────────────────────────────────────────────────────────────────

            ok, _ = nf.filter(raw, priorities)
            if ok:
                accepted_items.append(item)
            else:
                dropped += 1

        # ── Downstream character-budget enforcement ───────────────────────────
        # Mirrors the identical logic in AcquisitionNoiseFilter.filter_batch().
        # Evaluated here because apply_acquisition_filter operates on
        # ContentItem objects (the connector's native output) whereas
        # filter_batch works on RawObservation shells.  Both use the same
        # effective_budget resolution: per-user SP value → call-site default.
        sp = priorities or StrategicPriorities()
        effective_budget: int = (
            sp.max_downstream_chars
            if sp.max_downstream_chars is not None
            else max_downstream_chars
        )
        if effective_budget > 0 and accepted_items:
            budget_items = []
            total_chars = 0
            truncated = 0
            for item in accepted_items:
                item_chars = len(item.raw_text or item.title or "")
                if total_chars + item_chars > effective_budget:
                    truncated += 1
                else:
                    total_chars += item_chars
                    budget_items.append(item)
            if truncated:
                logger.warning(
                    "AcquisitionFilter: downstream char budget (%d) reached — "
                    "truncated %d of %d accepted items (chars used: %d).",
                    effective_budget,
                    truncated,
                    len(accepted_items),
                    total_chars,
                )
            accepted_items = budget_items
        # ─────────────────────────────────────────────────────────────────────

        logger.info(
            "AcquisitionFilter: %d/%d items passed (dropped %d)",
            len(accepted_items), len(fetch_result.items), dropped,
        )
        fetch_result.items = accepted_items
        return fetch_result

    @classmethod
    def get_platform_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get information about all supported platforms.
        
        Returns:
            Dictionary mapping platform names to their info
        """
        return {
            # Social Media
            "reddit": {
                "name": "Reddit",
                "platform": SourcePlatform.REDDIT,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://www.reddit.com/dev/api/",
            },
            "youtube": {
                "name": "YouTube",
                "platform": SourcePlatform.YOUTUBE,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://developers.google.com/youtube/v3",
            },
            "tiktok": {
                "name": "TikTok",
                "platform": SourcePlatform.TIKTOK,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://developers.tiktok.com/doc/research-api-overview",
            },
            "facebook": {
                "name": "Facebook",
                "platform": SourcePlatform.FACEBOOK,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://developers.facebook.com/docs/graph-api/",
            },
            "instagram": {
                "name": "Instagram",
                "platform": SourcePlatform.INSTAGRAM,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://developers.facebook.com/docs/instagram-api/",
            },
            "wechat": {
                "name": "WeChat",
                "platform": SourcePlatform.WECHAT,
                "type": "social_media",
                "requires_oauth": True,
                "api_docs": "https://developers.weixin.qq.com/doc/offiaccount/en/",
            },
            # News Sources
            "nytimes": {
                "name": "New York Times",
                "platform": SourcePlatform.NYTIMES,
                "type": "news",
                "requires_oauth": False,
                "api_docs": "https://developer.nytimes.com/",
            },
            "wsj": {
                "name": "Wall Street Journal",
                "platform": SourcePlatform.WSJ,
                "type": "news",
                "requires_oauth": False,
                "api_docs": "https://www.wsj.com/news/rss-news-and-feeds",
            },
            "abc_news": {
                "name": "ABC News (US)",
                "platform": SourcePlatform.ABC_NEWS,
                "type": "news",
                "requires_oauth": False,
                "api_docs": None,
            },
            "abc_news_au": {
                "name": "ABC News (Australia)",
                "platform": SourcePlatform.ABC_NEWS_AU,
                "type": "news",
                "requires_oauth": False,
                "api_docs": "https://www.abc.net.au/news/feeds/",
            },
            "google_news": {
                "name": "Google News",
                "platform": SourcePlatform.GOOGLE_NEWS,
                "type": "news",
                "requires_oauth": False,
                "api_docs": None,
            },
            "apple_news": {
                "name": "Apple News",
                "platform": SourcePlatform.APPLE_NEWS,
                "type": "news",
                "requires_oauth": False,
                "api_docs": None,
            },
            "rss": {
                "name": "RSS Feed",
                "platform": SourcePlatform.RSS,
                "type": "generic",
                "requires_oauth": False,
                "api_docs": None,
            },
        }


# Singleton instance
connector_registry = ConnectorRegistry()


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
from app.connectors.arxiv import ArxivConnector
from app.connectors.base import BaseConnector, ConnectorConfig, FetchResult
from app.connectors.changelog import ChangelogConnector
from app.connectors.docs_monitor import DocsMonitorConnector
from app.connectors.facebook import FacebookConnector
from app.connectors.github_discussions import GitHubDiscussionsConnector
from app.connectors.github_releases import GitHubReleasesConnector
from app.connectors.github_repo_events import GitHubRepoEventsConnector
from app.connectors.google_news import GoogleNewsConnector
from app.connectors.instagram import InstagramConnector
from app.connectors.nytimes import NYTimesConnector
from app.connectors.openreview import OpenReviewConnector
from app.connectors.podcast_rss import PodcastRSSConnector
from app.connectors.reddit import RedditConnector
from app.connectors.rss import RSSConnector
from app.connectors.semantic_scholar import SemanticScholarConnector
from app.connectors.tiktok import TikTokConnector
from app.connectors.transcript_feeds import TranscriptFeedConnector
from app.connectors.wechat import WeChatConnector
from app.connectors.wsj import WSJConnector
from app.connectors.youtube import YouTubeConnector
from app.connectors.youtube_transcript import YouTubeTranscriptConnector
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
    # Phase 1 — Source Intelligence Layer (no OAuth; optional PAT/key)
    SourcePlatform.GITHUB_RELEASES,
    SourcePlatform.GITHUB_REPO_EVENTS,
    # GitHub Discussions uses a PAT (not user-delegated OAuth) → public/key category
    SourcePlatform.GITHUB_DISCUSSIONS,
    SourcePlatform.CHANGELOG,
    SourcePlatform.DOCS_MONITOR,
    SourcePlatform.ARXIV,
    SourcePlatform.OPENREVIEW,
    SourcePlatform.SEMANTIC_SCHOLAR,
    SourcePlatform.PODCAST_RSS,
    SourcePlatform.TRANSCRIPT_FEED,
    SourcePlatform.YOUTUBE_TRANSCRIPT,
]

# Phase 1 platforms that accept an optional PAT/API-key for higher rate limits
# but do NOT require user-delegated OAuth.
OPTIONAL_AUTH_PLATFORMS: List[SourcePlatform] = [
    SourcePlatform.GITHUB_RELEASES,
    SourcePlatform.GITHUB_REPO_EVENTS,
    SourcePlatform.GITHUB_DISCUSSIONS,  # GraphQL requires a PAT
    SourcePlatform.SEMANTIC_SCHOLAR,
    SourcePlatform.YOUTUBE_TRANSCRIPT,
]


class ConnectorRegistry:
    """Registry for platform-specific connectors."""

    _connectors: Dict[SourcePlatform, Type[BaseConnector]] = {
        # ── Social Media ──────────────────────────────────────────────────────
        SourcePlatform.REDDIT: RedditConnector,
        SourcePlatform.YOUTUBE: YouTubeConnector,
        SourcePlatform.TIKTOK: TikTokConnector,
        SourcePlatform.FACEBOOK: FacebookConnector,
        SourcePlatform.INSTAGRAM: InstagramConnector,
        SourcePlatform.WECHAT: WeChatConnector,
        # ── News / Editorial ──────────────────────────────────────────────────
        SourcePlatform.RSS: RSSConnector,
        SourcePlatform.NYTIMES: NYTimesConnector,
        SourcePlatform.WSJ: WSJConnector,
        SourcePlatform.ABC_NEWS: ABCNewsConnector,
        SourcePlatform.ABC_NEWS_AU: ABCNewsConnector,
        SourcePlatform.GOOGLE_NEWS: GoogleNewsConnector,
        SourcePlatform.APPLE_NEWS: AppleNewsConnector,
        # ── Developer / Release (Phase 1) ─────────────────────────────────────
        SourcePlatform.GITHUB_RELEASES: GitHubReleasesConnector,
        SourcePlatform.GITHUB_REPO_EVENTS: GitHubRepoEventsConnector,
        SourcePlatform.GITHUB_DISCUSSIONS: GitHubDiscussionsConnector,
        SourcePlatform.CHANGELOG: ChangelogConnector,
        SourcePlatform.DOCS_MONITOR: DocsMonitorConnector,
        # ── Research (Phase 1) ────────────────────────────────────────────────
        SourcePlatform.ARXIV: ArxivConnector,
        SourcePlatform.OPENREVIEW: OpenReviewConnector,
        SourcePlatform.SEMANTIC_SCHOLAR: SemanticScholarConnector,
        # ── Media / Audio (Phase 1) ───────────────────────────────────────────
        SourcePlatform.PODCAST_RSS: PodcastRSSConnector,
        SourcePlatform.TRANSCRIPT_FEED: TranscriptFeedConnector,
        SourcePlatform.YOUTUBE_TRANSCRIPT: YouTubeTranscriptConnector,
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
            # ── Developer / Release (Phase 1) ─────────────────────────────────
            "github_releases": {
                "name": "GitHub Releases",
                "platform": SourcePlatform.GITHUB_RELEASES,
                "type": "developer",
                "source_family": "developer_release",
                "requires_oauth": False,
                "auth_optional": True,
                "rate_limit_unauth": "60/hour",
                "rate_limit_auth": "5000/hour",
                "content_types": ["release_notes", "changelogs"],
                "supports_since": True,
                "api_docs": "https://docs.github.com/en/rest/releases/releases",
            },
            "github_repo_events": {
                "name": "GitHub Repo Events",
                "platform": SourcePlatform.GITHUB_REPO_EVENTS,
                "type": "developer",
                "source_family": "developer_release",
                "requires_oauth": False,
                "auth_optional": True,
                "content_types": ["push_events", "issue_events", "pr_events"],
                "supports_since": True,
                "api_docs": "https://docs.github.com/en/rest/activity/events",
            },
            "github_discussions": {
                "name": "GitHub Discussions",
                "platform": SourcePlatform.GITHUB_DISCUSSIONS,
                "type": "developer",
                "source_family": "developer_release",
                "requires_oauth": True,
                "auth_type": "github_pat",
                "content_types": ["rfc", "announcements", "feedback"],
                "supports_since": True,
                "api_docs": "https://docs.github.com/en/graphql",
            },
            "changelog": {
                "name": "Changelog Monitor",
                "platform": SourcePlatform.CHANGELOG,
                "type": "developer",
                "source_family": "developer_release",
                "requires_oauth": False,
                "content_types": ["version_sections"],
                "supports_since": True,
                "api_docs": None,
            },
            "docs_monitor": {
                "name": "Documentation Monitor",
                "platform": SourcePlatform.DOCS_MONITOR,
                "type": "developer",
                "source_family": "developer_release",
                "requires_oauth": False,
                "content_types": ["docs_changes"],
                "supports_since": False,
                "change_detection": ["etag", "hash"],
                "api_docs": None,
            },
            # ── Research (Phase 1) ────────────────────────────────────────────
            "arxiv": {
                "name": "arXiv",
                "platform": SourcePlatform.ARXIV,
                "type": "research",
                "source_family": "research",
                "requires_oauth": False,
                "rate_limit": "1 req/3s (courtesy)",
                "content_types": ["preprints", "abstracts"],
                "supports_since": True,
                "api_docs": "https://info.arxiv.org/help/api/user-manual",
            },
            "openreview": {
                "name": "OpenReview",
                "platform": SourcePlatform.OPENREVIEW,
                "type": "research",
                "source_family": "research",
                "requires_oauth": False,
                "content_types": ["papers", "reviews", "decisions"],
                "supports_since": True,
                "api_docs": "https://docs.openreview.net/reference/api-v2",
            },
            "semantic_scholar": {
                "name": "Semantic Scholar",
                "platform": SourcePlatform.SEMANTIC_SCHOLAR,
                "type": "research",
                "source_family": "research",
                "requires_oauth": False,
                "auth_optional": True,
                "content_types": ["papers", "author_papers"],
                "supports_since": True,
                "api_docs": "https://api.semanticscholar.org/api-docs/",
            },
            # ── Media / Audio (Phase 1) ───────────────────────────────────────
            "podcast_rss": {
                "name": "Podcast RSS",
                "platform": SourcePlatform.PODCAST_RSS,
                "type": "media",
                "source_family": "media_audio",
                "requires_oauth": False,
                "content_types": ["episode_metadata", "show_notes"],
                "media_type": "audio",
                "transcript_pending": True,
                "supports_since": True,
                "api_docs": None,
            },
            "transcript_feed": {
                "name": "Transcript Feed",
                "platform": SourcePlatform.TRANSCRIPT_FEED,
                "type": "media",
                "source_family": "media_audio",
                "requires_oauth": False,
                "content_types": ["transcripts"],
                "media_type": "text",
                "transcript_pending": False,
                "supports_since": True,
                "api_docs": None,
            },
            "youtube_transcript": {
                "name": "YouTube Transcript",
                "platform": SourcePlatform.YOUTUBE_TRANSCRIPT,
                "type": "media",
                "source_family": "media_audio",
                "requires_oauth": False,
                "auth_optional": True,
                "content_types": ["transcripts"],
                "media_type": "text",
                "transcript_pending": False,
                "supports_since": True,
                "api_docs": "https://developers.google.com/youtube/v3/docs/captions",
            },
        }

    @classmethod
    def get_source_family(cls, platform: SourcePlatform) -> str:
        """Return the source family string for *platform*.

        Source families group platforms for the Source Intelligence Layer:
        - ``social``           — Reddit, YouTube, TikTok, Facebook, Instagram, WeChat
        - ``news``             — RSS, NYTimes, WSJ, ABC News, Google News, Apple News
        - ``developer_release``— GitHub Releases/Events/Discussions, Changelog, DocsMonitor
        - ``research``         — arXiv, OpenReview, Semantic Scholar
        - ``media_audio``      — Podcast RSS, Transcript Feeds, YouTube Transcript
        - ``unknown``          — anything not yet classified
        """
        _FAMILY_MAP: Dict[SourcePlatform, str] = {
            SourcePlatform.REDDIT: "social",
            SourcePlatform.YOUTUBE: "social",
            SourcePlatform.TIKTOK: "social",
            SourcePlatform.FACEBOOK: "social",
            SourcePlatform.INSTAGRAM: "social",
            SourcePlatform.WECHAT: "social",
            SourcePlatform.RSS: "news",
            SourcePlatform.NEWSAPI: "news",
            SourcePlatform.NYTIMES: "news",
            SourcePlatform.WSJ: "news",
            SourcePlatform.ABC_NEWS: "news",
            SourcePlatform.ABC_NEWS_AU: "news",
            SourcePlatform.GOOGLE_NEWS: "news",
            SourcePlatform.APPLE_NEWS: "news",
            SourcePlatform.GITHUB_RELEASES: "developer_release",
            SourcePlatform.GITHUB_REPO_EVENTS: "developer_release",
            SourcePlatform.GITHUB_DISCUSSIONS: "developer_release",
            SourcePlatform.CHANGELOG: "developer_release",
            SourcePlatform.DOCS_MONITOR: "developer_release",
            SourcePlatform.ARXIV: "research",
            SourcePlatform.OPENREVIEW: "research",
            SourcePlatform.SEMANTIC_SCHOLAR: "research",
            SourcePlatform.PODCAST_RSS: "media_audio",
            SourcePlatform.TRANSCRIPT_FEED: "media_audio",
            SourcePlatform.YOUTUBE_TRANSCRIPT: "media_audio",
        }
        return _FAMILY_MAP.get(platform, "unknown")


# Singleton instance
connector_registry = ConnectorRegistry()


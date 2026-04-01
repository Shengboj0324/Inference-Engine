"""Feed expander.

Given an entity name or homepage URL, auto-discovers related feeds
without making live HTTP calls (Phase 1 implementation is heuristic-only;
Phase 2 will add live URL probing via the ``httpx`` client).

Discovery heuristics:
- GitHub org name → release/event/discussion feeds
- arXiv author ID → author paper feed
- YouTube channel name → known channel ID lookup
- Podcast platform slug → RSS feed URL patterns
- Domain name → ``{domain}/feed``, ``{domain}/rss``, ``{domain}/atom.xml``

Each expansion returns a ``FeedCandidate`` with a suggested ``ConnectorConfig``
ready for use with ``ConnectorRegistry.get_connector()``.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.models import SourcePlatform

logger = logging.getLogger(__name__)

_YOUTUBE_KNOWN_CHANNELS: Dict[str, str] = {
    "andrej karpathy": "UCbmNph6atAoGfqLoCL_duAg",
    "george hotz": "UCwgKmJM4ZJQRJ-U5NjvR2dg",
    "yannic kilcher": "UCZHmQk67mSJgfCCTn7xBfew",
    "two minute papers": "UCbfYPyITQ-7l4upoX8nvctg",
    "sentdex": "UCfzlCWGWYyIQ0aLC5w48gBQ",
}

_GITHUB_FEED_TEMPLATES: Dict[str, str] = {
    "releases": "https://github.com/{repo}/releases.atom",
    "commits": "https://github.com/{repo}/commits/{branch}.atom",
    "tags": "https://github.com/{repo}/tags.atom",
}

_SUBSTACK_PATTERN = re.compile(r"^(?:https?://)?([a-z0-9-]+)\.substack\.com", re.IGNORECASE)
_DOMAIN_PATTERN = re.compile(r"^(?:https?://)?([a-z0-9.-]+\.[a-z]{2,})", re.IGNORECASE)


@dataclass
class FeedCandidate:
    """A candidate feed discovered by ``FeedExpander``.

    Attributes:
        source_id:      Suggested source ID for ``SourceSpec``.
        platform:       Suggested ``SourcePlatform``.
        connector_settings: Settings dict suitable for ``ConnectorConfig.settings``.
        display_name:   Human-readable label.
        confidence:     Heuristic confidence score [0, 1].
        discovery_reason: Why this feed was suggested.
    """

    source_id: str
    platform: SourcePlatform
    connector_settings: Dict[str, Any]
    display_name: str = ""
    confidence: float = 0.5
    discovery_reason: str = ""

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"'confidence' must be in [0, 1], got {self.confidence!r}")


class FeedExpander:
    """Expands entity names / URLs into connector-ready feed candidates.

    All methods are stateless and safe to call concurrently.
    """

    def expand_github_org(self, org: str, repos: Optional[List[str]] = None) -> List[FeedCandidate]:
        """Generate GitHub feed candidates for an org and optional repo list.

        Args:
            org: GitHub organisation or user name.
            repos: Optional explicit list of ``owner/repo`` slugs.  If None,
                   generates candidates for the org profile page only.

        Returns:
            List of ``FeedCandidate`` for releases, events, and discussions.
        """
        if not org or not isinstance(org, str):
            raise ValueError("'org' must be a non-empty string")

        candidates: List[FeedCandidate] = []
        repo_list = repos or [f"{org}/{org}"]  # best-guess default

        for repo in repo_list:
            repo_slug = repo if "/" in repo else f"{org}/{repo}"
            candidates.append(FeedCandidate(
                source_id=repo_slug,
                platform=SourcePlatform.GITHUB_RELEASES,
                connector_settings={"repos": [repo_slug]},
                display_name=f"{repo_slug} — GitHub Releases",
                confidence=0.90,
                discovery_reason=f"GitHub release feed for org '{org}'",
            ))
            candidates.append(FeedCandidate(
                source_id=f"{repo_slug}:events",
                platform=SourcePlatform.GITHUB_REPO_EVENTS,
                connector_settings={"repos": [repo_slug]},
                display_name=f"{repo_slug} — GitHub Events",
                confidence=0.80,
                discovery_reason=f"GitHub event stream for org '{org}'",
            ))
        logger.debug("FeedExpander.expand_github_org: org=%r candidates=%d", org, len(candidates))
        return candidates

    def expand_arxiv_author(self, author_id: str, display_name: str = "") -> List[FeedCandidate]:
        """Generate a Semantic Scholar author feed for a known author ID.

        Args:
            author_id: Semantic Scholar numeric author ID.
            display_name: Optional human-readable name.

        Returns:
            Single-element list with a ``SEMANTIC_SCHOLAR`` feed candidate.
        """
        if not author_id or not isinstance(author_id, str):
            raise ValueError("'author_id' must be a non-empty string")

        label = display_name or f"Author {author_id}"
        return [FeedCandidate(
            source_id=f"ss:author:{author_id}",
            platform=SourcePlatform.SEMANTIC_SCHOLAR,
            connector_settings={"author_ids": [author_id]},
            display_name=f"{label} — Semantic Scholar Papers",
            confidence=0.85,
            discovery_reason=f"Semantic Scholar author feed for '{label}'",
        )]

    def expand_youtube_channel(self, channel_name: str) -> List[FeedCandidate]:
        """Generate a YouTube transcript feed for a known channel.

        Args:
            channel_name: Channel display name (case-insensitive lookup).

        Returns:
            List of ``FeedCandidate`` if channel is in known catalogue.
        """
        if not channel_name or not isinstance(channel_name, str):
            raise ValueError("'channel_name' must be a non-empty string")

        lower = channel_name.lower().strip()
        channel_id = _YOUTUBE_KNOWN_CHANNELS.get(lower)
        if not channel_id:
            logger.debug("FeedExpander.expand_youtube_channel: %r not in known catalogue", channel_name)
            return []
        return [FeedCandidate(
            source_id=f"yt:{channel_id}",
            platform=SourcePlatform.YOUTUBE_TRANSCRIPT,
            connector_settings={"channel_ids": [channel_id]},
            display_name=f"{channel_name} — YouTube Transcripts",
            confidence=0.88,
            discovery_reason=f"Known YouTube channel catalogue match for '{channel_name}'",
        )]

    def expand_domain_feeds(self, domain_or_url: str) -> List[FeedCandidate]:
        """Guess RSS/Atom feed URLs for a given domain.

        Tries canonical patterns: ``/feed``, ``/rss``, ``/atom.xml``.
        Detects Substack domains and returns the Substack RSS URL.

        Args:
            domain_or_url: Domain name or full URL.

        Returns:
            List of ``FeedCandidate`` for the guessed RSS URLs.
        """
        if not domain_or_url or not isinstance(domain_or_url, str):
            raise ValueError("'domain_or_url' must be a non-empty string")

        candidates: List[FeedCandidate] = []

        # Substack special case
        sub_match = _SUBSTACK_PATTERN.match(domain_or_url)
        if sub_match:
            slug = sub_match.group(1)
            rss_url = f"https://{slug}.substack.com/feed"
            candidates.append(FeedCandidate(
                source_id=rss_url,
                platform=SourcePlatform.TRANSCRIPT_FEED,
                connector_settings={"feed_urls": [rss_url]},
                display_name=f"{slug}.substack.com RSS",
                confidence=0.88,
                discovery_reason="Substack domain detected",
            ))
            return candidates

        # Generic domain patterns
        dom_match = _DOMAIN_PATTERN.match(domain_or_url)
        domain = dom_match.group(1) if dom_match else domain_or_url.strip("/")
        for path, conf in [("/feed", 0.65), ("/rss", 0.60), ("/atom.xml", 0.55), ("/rss.xml", 0.55)]:
            url = f"https://{domain}{path}"
            candidates.append(FeedCandidate(
                source_id=url,
                platform=SourcePlatform.RSS,
                connector_settings={"feed_urls": [url]},
                display_name=f"{domain}{path}",
                confidence=conf,
                discovery_reason=f"Canonical RSS path guess for '{domain}'",
            ))
        logger.debug("FeedExpander.expand_domain_feeds: domain=%r candidates=%d", domain, len(candidates))
        return candidates

    def expand_podcast_platform(self, platform_slug: str, podcast_id: str) -> List[FeedCandidate]:
        """Generate a podcast RSS candidate from a platform slug and show ID.

        Supported platforms: ``apple``, ``spotify``, ``google``.

        Args:
            platform_slug: One of ``"apple"``, ``"spotify"``.
            podcast_id: Platform-specific podcast identifier.

        Returns:
            List with one ``PODCAST_RSS`` candidate, or empty if unrecognised.
        """
        if not platform_slug or not isinstance(platform_slug, str):
            raise ValueError("'platform_slug' must be a non-empty string")
        if not podcast_id or not isinstance(podcast_id, str):
            raise ValueError("'podcast_id' must be a non-empty string")

        slug = platform_slug.lower().strip()
        url_map: Dict[str, str] = {
            "apple": f"https://podcasts.apple.com/podcast/id{podcast_id}",
            "spotify": f"https://open.spotify.com/show/{podcast_id}",
        }
        if slug not in url_map:
            logger.debug("FeedExpander.expand_podcast_platform: unknown platform %r", platform_slug)
            return []
        return [FeedCandidate(
            source_id=f"podcast:{slug}:{podcast_id}",
            platform=SourcePlatform.PODCAST_RSS,
            connector_settings={"feed_urls": [url_map[slug]]},
            display_name=f"{platform_slug.title()} Podcast {podcast_id}",
            confidence=0.70,
            discovery_reason=f"Podcast platform slug: {slug}",
        )]


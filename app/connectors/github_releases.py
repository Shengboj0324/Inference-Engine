"""GitHub Releases connector.

Ingests release notes, changelogs, and version metadata from GitHub's
Releases API (``GET /repos/{owner}/{repo}/releases``).

Each release maps to one ``ContentItem`` with:
- ``title``          : ``"{repo} {tag_name}"``
- ``raw_text``       : release body (Markdown)
- ``source_url``     : HTML URL of the release
- ``media_type``     : ``MediaType.TEXT``
- ``published_at``   : release ``published_at`` timestamp

Configuration (``ConnectorConfig.settings``)::

    repos: List[str]          # e.g. ["openai/openai-python", "anthropics/sdk"]
    github_token: str         # Optional PAT for higher rate limits (5000 req/h)
    include_prereleases: bool # default False
    include_drafts: bool      # default False

Rate limits
-----------
Unauthenticated: 60 req/h.  Authenticated (PAT): 5 000 req/h.
The connector raises ``RateLimitError`` on HTTP 403/429 so
``fetch_content_with_retry`` and ``CircuitBreaker`` handle back-off.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx

from app.connectors.base import (
    BaseConnector,
    ConnectorConfig,
    FetchResult,
    PlatformError,
    RateLimitError,
)
from app.core.models import ContentItem, MediaType, SourcePlatform

logger = logging.getLogger(__name__)

_GITHUB_API_BASE = "https://api.github.com"
_DEFAULT_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


class GitHubReleasesConnector(BaseConnector):
    """Fetches GitHub release notes for a configured list of repositories.

    Thread-safe: ``fetch_content`` is stateless between calls; all mutable
    state (``last_successful_fetch``, ``consecutive_failures``, etc.) is
    inherited from ``BaseConnector`` and only written from async context.
    """

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        settings: Dict[str, Any] = config.settings or {}
        self._repos: List[str] = settings.get("repos", [])
        self._token: Optional[str] = settings.get("github_token") or config.credentials.get("github_token")
        self._include_prereleases: bool = bool(settings.get("include_prereleases", False))
        self._include_drafts: bool = bool(settings.get("include_drafts", False))

        if not isinstance(self._repos, list):
            raise TypeError(f"'repos' must be a list of strings, got {type(self._repos)!r}")

    # ------------------------------------------------------------------
    # BaseConnector interface
    # ------------------------------------------------------------------

    async def validate_credentials(self) -> bool:
        """Returns True when the GitHub API responds; False on auth failure."""
        headers = self._build_headers()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{_GITHUB_API_BASE}/rate_limit", headers=headers)
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("GitHubReleasesConnector.validate_credentials failed: %s", exc)
            return False

    async def get_user_feeds(self) -> List[str]:
        """Return the list of monitored repos as feed identifiers."""
        return list(self._repos)

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        """Fetch releases from all configured repositories.

        Args:
            since: Ignore releases published before this UTC timestamp.
            cursor: Unused; pagination is handled per-repo internally.
            max_items: Maximum total items to return across all repos.

        Returns:
            ``FetchResult`` with ``ContentItem`` per release.
        """
        if not self._repos:
            raise ValueError("GitHubReleasesConnector: 'repos' setting is empty")

        t0 = time.perf_counter()
        items: List[ContentItem] = []
        errors: List[str] = []
        headers = self._build_headers()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for repo in self._repos:
                if len(items) >= max_items:
                    break
                try:
                    repo_items = await self._fetch_repo_releases(
                        client, repo, headers, since, max_items - len(items)
                    )
                    items.extend(repo_items)
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"{repo}: {exc}")
                    logger.warning("GitHubReleasesConnector: error fetching %s: %s", repo, exc)

        logger.info(
            "GitHubReleasesConnector.fetch_content: repos=%d items=%d errors=%d latency_ms=%.1f",
            len(self._repos), len(items), len(errors),
            (time.perf_counter() - t0) * 1000,
        )
        return FetchResult(items=items, errors=errors)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        headers = dict(_DEFAULT_HEADERS)
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _fetch_repo_releases(
        self,
        client: httpx.AsyncClient,
        repo: str,
        headers: Dict[str, str],
        since: Optional[datetime],
        remaining: int,
    ) -> List[ContentItem]:
        """Fetch releases for one ``owner/repo`` slug."""
        url = f"{_GITHUB_API_BASE}/repos/{repo}/releases"
        params: Dict[str, Any] = {"per_page": min(remaining, 30)}
        resp = await client.get(url, headers=headers, params=params)

        if resp.status_code in (403, 429):
            raise RateLimitError(f"GitHub rate-limited on {repo}: HTTP {resp.status_code}")
        if resp.status_code == 404:
            raise PlatformError(f"Repo not found: {repo}")
        if resp.status_code != 200:
            raise PlatformError(f"GitHub API error {resp.status_code} for {repo}")

        releases = resp.json()
        items: List[ContentItem] = []
        for rel in releases:
            if rel.get("draft") and not self._include_drafts:
                continue
            if rel.get("prerelease") and not self._include_prereleases:
                continue
            pub_raw = rel.get("published_at") or rel.get("created_at")
            pub_at = datetime.fromisoformat(pub_raw.replace("Z", "+00:00")) if pub_raw else datetime.now(timezone.utc)
            if since and pub_at.replace(tzinfo=timezone.utc) <= since.replace(tzinfo=timezone.utc):
                continue
            items.append(self._create_content_item(
                source_id=str(rel.get("id", "")),
                source_url=rel.get("html_url", ""),
                title=f"{repo} {rel.get('tag_name', 'release')}",
                raw_text=rel.get("body") or "",
                media_type=MediaType.TEXT,
                published_at=pub_at,
                metadata={
                    "repo": repo,
                    "tag_name": rel.get("tag_name"),
                    "name": rel.get("name"),
                    "prerelease": rel.get("prerelease", False),
                    "draft": rel.get("draft", False),
                    "author": (rel.get("author") or {}).get("login"),
                    "assets_count": len(rel.get("assets", [])),
                    "tarball_url": rel.get("tarball_url"),
                    "zipball_url": rel.get("zipball_url"),
                },
            ))
        return items


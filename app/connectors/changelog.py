"""Changelog connector.

Monitors ``CHANGELOG.md`` (or equivalent) files hosted at URLs or on GitHub,
extracting individual version sections as ``ContentItem`` objects.

Supports two fetch modes:

1. **Raw URL mode** — fetches the changelog file directly from a URL
   (e.g. ``https://raw.githubusercontent.com/org/repo/main/CHANGELOG.md``).
2. **GitHub API mode** — uses the GitHub Releases API (falling back to the
   ``CHANGELOG.md`` raw content) when a ``repos`` setting is given.

Each parsed version section produces one ``ContentItem`` with:
- ``title``    : ``"{repo/source} v{version_tag}"``
- ``raw_text`` : changelog section body (Markdown)
- ``media_type``: ``MediaType.TEXT``

Version sections are detected by headings matching the pattern
``## [x.y.z]`` or ``## vX.Y.Z`` (configurable via regex).

Configuration (``ConnectorConfig.settings``)::

    changelog_urls: List[str]    # Raw changelog file URLs
    repos: List[str]             # GitHub repos — fetches CHANGELOG.md from default branch
    github_token: str            # Optional PAT
    version_regex: str           # Regex to split sections; default: ^##\s+[vV]?\d+
    max_versions: int            # Versions per source; default 10
"""

import logging
import re
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
from app.core.models import ContentItem, MediaType

logger = logging.getLogger(__name__)

_GH_RAW_BASE = "https://raw.githubusercontent.com"
_DEFAULT_VERSION_REGEX = r"^##\s+[vV]?\d+"


class ChangelogConnector(BaseConnector):
    """Parses structured CHANGELOG.md files into per-version ContentItems."""

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        s: Dict[str, Any] = config.settings or {}
        self._changelog_urls: List[str] = s.get("changelog_urls", [])
        self._repos: List[str] = s.get("repos", [])
        self._token: Optional[str] = s.get("github_token") or config.credentials.get("github_token")
        self._version_regex: str = s.get("version_regex", _DEFAULT_VERSION_REGEX)
        self._max_versions: int = int(s.get("max_versions", 10))
        try:
            re.compile(self._version_regex)
        except re.error as exc:
            raise ValueError(f"'version_regex' is invalid: {exc}") from exc
        if not self._changelog_urls and not self._repos:
            raise ValueError("ChangelogConnector: provide 'changelog_urls' or 'repos'")

    async def validate_credentials(self) -> bool:
        if self._changelog_urls:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(self._changelog_urls[0])
                return resp.status_code == 200
            except Exception:
                return False
        return True

    async def get_user_feeds(self) -> List[str]:
        return self._changelog_urls + [f"github:{r}" for r in self._repos]

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        t0 = time.perf_counter()
        items: List[ContentItem] = []
        errors: List[str] = []
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}

        async with httpx.AsyncClient(timeout=30.0) as client:
            for url in self._changelog_urls:
                if len(items) >= max_items:
                    break
                try:
                    items.extend(await self._fetch_url(client, url, url, headers, since, max_items - len(items)))
                except Exception as exc:
                    errors.append(f"{url}: {exc}")
            for repo in self._repos:
                if len(items) >= max_items:
                    break
                try:
                    items.extend(await self._fetch_repo(client, repo, headers, since, max_items - len(items)))
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"{repo}: {exc}")

        logger.info(
            "ChangelogConnector.fetch_content: sources=%d items=%d latency_ms=%.1f",
            len(self._changelog_urls) + len(self._repos), len(items), (time.perf_counter() - t0) * 1000,
        )
        return FetchResult(items=items, errors=errors)

    async def _fetch_repo(
        self, client: httpx.AsyncClient, repo: str, headers: Dict[str, str], since: Optional[datetime], remaining: int
    ) -> List[ContentItem]:
        """Fetch CHANGELOG.md from the default branch of a GitHub repo."""
        gh_headers = {**headers, "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        # Get default branch
        meta = await client.get(f"https://api.github.com/repos/{repo}", headers=gh_headers)
        if meta.status_code in (403, 429):
            raise RateLimitError(f"GitHub rate-limited on {repo}")
        if meta.status_code != 200:
            raise PlatformError(f"GitHub API error {meta.status_code} for {repo}")
        default_branch = meta.json().get("default_branch", "main")
        raw_url = f"{_GH_RAW_BASE}/{repo}/{default_branch}/CHANGELOG.md"
        return await self._fetch_url(client, raw_url, repo, {}, since, remaining)

    async def _fetch_url(
        self, client: httpx.AsyncClient, url: str, label: str, headers: Dict[str, str], since: Optional[datetime], remaining: int
    ) -> List[ContentItem]:
        resp = await client.get(url, headers=headers)
        if resp.status_code == 404:
            logger.debug("ChangelogConnector: no CHANGELOG at %s", url)
            return []
        if resp.status_code in (403, 429):
            raise RateLimitError(f"Rate-limited fetching {url}")
        if resp.status_code != 200:
            raise PlatformError(f"HTTP {resp.status_code} fetching {url}")
        return self._parse_changelog(resp.text, label, since, remaining)

    def _parse_changelog(
        self, content: str, label: str, since: Optional[datetime], remaining: int
    ) -> List[ContentItem]:
        pattern = re.compile(self._version_regex, re.MULTILINE)
        matches = list(pattern.finditer(content))
        items: List[ContentItem] = []
        for i, match in enumerate(matches[: self._max_versions]):
            if len(items) >= remaining:
                break
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section = content[start:end].strip()
            heading = section.split("\n")[0].strip()
            body = "\n".join(section.split("\n")[1:]).strip()
            # Extract date from heading e.g. "## v1.2.3 - 2024-01-15"
            date_match = re.search(r"(\d{4}-\d{2}-\d{2})", heading)
            pub_at = datetime.now(timezone.utc)
            if date_match:
                try:
                    pub_at = datetime.strptime(date_match.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            if since and pub_at <= since.replace(tzinfo=timezone.utc):
                continue
            version_match = re.search(r"[vV]?(\d+\.\d+[\.\d]*)", heading)
            version_tag = version_match.group(0) if version_match else heading[:40]
            items.append(self._create_content_item(
                source_id=f"{label}:{version_tag}",
                source_url=label if label.startswith("http") else f"https://github.com/{label}",
                title=f"{label} {version_tag}",
                raw_text=body,
                media_type=MediaType.TEXT,
                published_at=pub_at,
                metadata={"source_label": label, "version_tag": version_tag, "heading": heading},
            ))
        return items


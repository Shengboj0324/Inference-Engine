"""GitHub Repository Events connector.

Ingests repository activity events (stars, forks, pushes, issues, pull requests)
from the GitHub Events API: ``GET /repos/{owner}/{repo}/events``.

Each event produces one ``ContentItem``.  Only these event types are surfaced
by default (all others are silently skipped):

- ``PushEvent``     — commit activity; raw_text = commit messages joined
- ``CreateEvent``   — branch/tag creation; indicates release preparation
- ``ReleaseEvent``  — mirrors GitHubReleasesConnector but from the event stream
- ``IssuesEvent``   — issue opened/closed (useful for ``breaking-change`` labels)
- ``PullRequestEvent`` — merged PRs that may signal meaningful changes

Configuration (``ConnectorConfig.settings``)::

    repos: List[str]         # e.g. ["openai/openai-python"]
    github_token: str        # Optional PAT
    event_types: List[str]   # filter; default = all types listed above
    max_events_per_repo: int # default 30
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
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

_GITHUB_API_BASE = "https://api.github.com"
_DEFAULT_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
_DEFAULT_EVENT_TYPES: Set[str] = {
    "PushEvent", "CreateEvent", "ReleaseEvent",
    "IssuesEvent", "PullRequestEvent",
}


class GitHubRepoEventsConnector(BaseConnector):
    """Streams per-repo GitHub events filtered by event type."""

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        s: Dict[str, Any] = config.settings or {}
        self._repos: List[str] = s.get("repos", [])
        self._token: Optional[str] = s.get("github_token") or config.credentials.get("github_token")
        self._max_per_repo: int = int(s.get("max_events_per_repo", 30))
        raw_types = s.get("event_types", list(_DEFAULT_EVENT_TYPES))
        self._event_types: Set[str] = set(raw_types) if raw_types else _DEFAULT_EVENT_TYPES
        if not isinstance(self._repos, list):
            raise TypeError(f"'repos' must be a list, got {type(self._repos)!r}")

    async def validate_credentials(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{_GITHUB_API_BASE}/rate_limit", headers=self._headers())
            return resp.status_code == 200
        except Exception:
            return False

    async def get_user_feeds(self) -> List[str]:
        return list(self._repos)

    async def fetch_content(
        self,
        since: Optional[datetime] = None,
        cursor: Optional[str] = None,
        max_items: int = 100,
    ) -> FetchResult:
        if not self._repos:
            raise ValueError("GitHubRepoEventsConnector: 'repos' setting is empty")

        t0 = time.perf_counter()
        items: List[ContentItem] = []
        errors: List[str] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for repo in self._repos:
                if len(items) >= max_items:
                    break
                try:
                    repo_items = await self._fetch_repo_events(client, repo, since, max_items - len(items))
                    items.extend(repo_items)
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"{repo}: {exc}")
                    logger.warning("GitHubRepoEventsConnector: error on %s: %s", repo, exc)

        logger.info(
            "GitHubRepoEventsConnector.fetch_content: repos=%d items=%d latency_ms=%.1f",
            len(self._repos), len(items), (time.perf_counter() - t0) * 1000,
        )
        return FetchResult(items=items, errors=errors)

    def _headers(self) -> Dict[str, str]:
        h = dict(_DEFAULT_HEADERS)
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def _fetch_repo_events(
        self,
        client: httpx.AsyncClient,
        repo: str,
        since: Optional[datetime],
        remaining: int,
    ) -> List[ContentItem]:
        url = f"{_GITHUB_API_BASE}/repos/{repo}/events"
        resp = await client.get(url, headers=self._headers(), params={"per_page": min(self._max_per_repo, 100)})
        if resp.status_code in (403, 429):
            raise RateLimitError(f"GitHub rate-limited on {repo}: HTTP {resp.status_code}")
        if resp.status_code == 404:
            raise PlatformError(f"Repo not found: {repo}")
        if resp.status_code != 200:
            raise PlatformError(f"GitHub API error {resp.status_code} for {repo}")

        events = resp.json()
        items: List[ContentItem] = []
        for ev in events:
            if len(items) >= remaining:
                break
            etype = ev.get("type", "")
            if etype not in self._event_types:
                continue
            created = ev.get("created_at", "")
            pub_at = datetime.fromisoformat(created.replace("Z", "+00:00")) if created else datetime.now(timezone.utc)
            if since and pub_at.replace(tzinfo=timezone.utc) <= since.replace(tzinfo=timezone.utc):
                continue
            payload = ev.get("payload", {})
            title = self._make_title(repo, etype, payload)
            raw_text = self._make_text(etype, payload)
            items.append(self._create_content_item(
                source_id=str(ev.get("id", "")),
                source_url=f"https://github.com/{repo}",
                title=title,
                raw_text=raw_text,
                media_type=MediaType.TEXT,
                published_at=pub_at,
                metadata={"repo": repo, "event_type": etype, "actor": (ev.get("actor") or {}).get("login"), "payload": payload},
            ))
        return items

    @staticmethod
    def _make_title(repo: str, etype: str, payload: Dict[str, Any]) -> str:
        if etype == "PushEvent":
            branch = payload.get("ref", "").replace("refs/heads/", "")
            n = payload.get("size", 0)
            return f"{repo}: {n} commit(s) pushed to {branch}"
        if etype == "CreateEvent":
            return f"{repo}: {payload.get('ref_type', 'ref')} {payload.get('ref', '')} created"
        if etype == "ReleaseEvent":
            rel = payload.get("release", {})
            return f"{repo}: release {rel.get('tag_name', '')} {payload.get('action', '')}"
        if etype == "IssuesEvent":
            issue = payload.get("issue", {})
            return f"{repo}: issue #{issue.get('number', '')} {payload.get('action', '')} — {issue.get('title', '')}"
        if etype == "PullRequestEvent":
            pr = payload.get("pull_request", {})
            return f"{repo}: PR #{pr.get('number', '')} {payload.get('action', '')} — {pr.get('title', '')}"
        return f"{repo}: {etype}"

    @staticmethod
    def _make_text(etype: str, payload: Dict[str, Any]) -> str:
        if etype == "PushEvent":
            commits = payload.get("commits", [])
            return "\n".join(c.get("message", "") for c in commits[:20])
        if etype == "ReleaseEvent":
            return (payload.get("release") or {}).get("body") or ""
        if etype == "IssuesEvent":
            return (payload.get("issue") or {}).get("body") or ""
        if etype == "PullRequestEvent":
            return (payload.get("pull_request") or {}).get("body") or ""
        return ""


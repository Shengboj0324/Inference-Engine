"""GitHub Discussions connector.

Fetches GitHub Discussions via the GitHub GraphQL API
(``POST https://api.github.com/graphql``).

Discussions are high-signal long-form content: RFC proposals, feature
debates, and release feedback.  Each discussion becomes one ``ContentItem``
with ``MediaType.TEXT`` containing the top-level body (and, optionally, the
first N replies concatenated).

Configuration (``ConnectorConfig.settings``)::

    repos: List[str]           # e.g. ["openai/openai-python"]
    github_token: str          # Required for GraphQL API (PAT or OAuth app token)
    categories: List[str]      # Optional: filter by category name e.g. ["Announcements"]
    include_replies: bool      # Include first-page comments; default False
    max_discussions_per_repo: int  # default 20
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
from app.core.models import ContentItem, MediaType

logger = logging.getLogger(__name__)

_GRAPHQL_URL = "https://api.github.com/graphql"

_DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $first: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: $first, after: $after, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes {
        id
        number
        title
        body
        createdAt
        url
        author { login }
        category { name }
        comments(first: 5) {
          nodes { body author { login } createdAt }
        }
      }
      pageInfo { endCursor hasNextPage }
    }
  }
}
"""


class GitHubDiscussionsConnector(BaseConnector):
    """Fetches GitHub Discussions via GraphQL for configured repositories."""

    def __init__(self, config: ConnectorConfig, user_id: UUID) -> None:
        super().__init__(config, user_id)
        s: Dict[str, Any] = config.settings or {}
        self._repos: List[str] = s.get("repos", [])
        self._token: Optional[str] = s.get("github_token") or config.credentials.get("github_token")
        self._categories: Optional[List[str]] = s.get("categories")
        self._include_replies: bool = bool(s.get("include_replies", False))
        self._max_per_repo: int = int(s.get("max_discussions_per_repo", 20))
        if not isinstance(self._repos, list):
            raise TypeError(f"'repos' must be a list, got {type(self._repos)!r}")
        if not self._token:
            raise ValueError("GitHubDiscussionsConnector requires 'github_token' (GraphQL API needs auth)")

    async def validate_credentials(self) -> bool:
        """Validate via a minimal GraphQL introspection ping."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    _GRAPHQL_URL,
                    json={"query": "{ viewer { login } }"},
                    headers=self._headers(),
                )
            return resp.status_code == 200 and "data" in resp.json()
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
            raise ValueError("GitHubDiscussionsConnector: 'repos' setting is empty")

        t0 = time.perf_counter()
        items: List[ContentItem] = []
        errors: List[str] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for repo in self._repos:
                if len(items) >= max_items:
                    break
                try:
                    repo_items = await self._fetch_discussions(client, repo, since, max_items - len(items))
                    items.extend(repo_items)
                except RateLimitError:
                    raise
                except Exception as exc:
                    errors.append(f"{repo}: {exc}")
                    logger.warning("GitHubDiscussionsConnector: error on %s: %s", repo, exc)

        logger.info(
            "GitHubDiscussionsConnector.fetch_content: repos=%d items=%d latency_ms=%.1f",
            len(self._repos), len(items), (time.perf_counter() - t0) * 1000,
        )
        return FetchResult(items=items, errors=errors)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _fetch_discussions(
        self,
        client: httpx.AsyncClient,
        repo: str,
        since: Optional[datetime],
        remaining: int,
    ) -> List[ContentItem]:
        parts = repo.split("/", 1)
        if len(parts) != 2:
            raise PlatformError(f"Invalid repo slug: {repo!r} (expected 'owner/name')")
        owner, name = parts
        variables: Dict[str, Any] = {"owner": owner, "name": name, "first": min(self._max_per_repo, remaining)}
        resp = await client.post(_GRAPHQL_URL, json={"query": _DISCUSSIONS_QUERY, "variables": variables}, headers=self._headers())
        if resp.status_code in (403, 429):
            raise RateLimitError(f"GitHub GraphQL rate-limited on {repo}: HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise PlatformError(f"GraphQL API error {resp.status_code} for {repo}")

        data = resp.json()
        if "errors" in data:
            raise PlatformError(f"GraphQL errors for {repo}: {data['errors']}")

        discussions = (data.get("data") or {}).get("repository", {}).get("discussions", {}).get("nodes", [])
        items: List[ContentItem] = []
        for d in discussions:
            created = d.get("createdAt", "")
            pub_at = datetime.fromisoformat(created.replace("Z", "+00:00")) if created else datetime.now(timezone.utc)
            if since and pub_at.replace(tzinfo=timezone.utc) <= since.replace(tzinfo=timezone.utc):
                continue
            cat = (d.get("category") or {}).get("name", "")
            if self._categories and cat not in self._categories:
                continue
            body = d.get("body") or ""
            if self._include_replies:
                comments = (d.get("comments") or {}).get("nodes", [])
                body += "\n\n" + "\n\n".join(c.get("body", "") for c in comments if c.get("body"))
            items.append(self._create_content_item(
                source_id=str(d.get("id", "")),
                source_url=d.get("url", ""),
                title=f"{repo} #{d.get('number', '')}: {d.get('title', '')}",
                raw_text=body,
                media_type=MediaType.TEXT,
                published_at=pub_at,
                metadata={"repo": repo, "category": cat, "author": (d.get("author") or {}).get("login"), "number": d.get("number")},
            ))
        return items


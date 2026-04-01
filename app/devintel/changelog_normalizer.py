"""Changelog normalizer.

Normalizes various changelog formats to a unified list of ``ReleaseNote``
objects:

- **Keep-a-Changelog** (``## [1.2.0] - 2024-01-15``)
- **GitHub Release Markdown** (processed via ``ReleaseParser``)
- **Free-form** (version header + bullet items)

Each call to ``normalize()`` returns releases sorted newest-first.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from app.devintel.models import ReleaseNote
from app.devintel.release_parser import ReleaseParser

logger = logging.getLogger(__name__)

# Keep-a-Changelog version header: ## [1.2.0] - 2024-01-15
_KACL_HEADER = re.compile(
    r"^##\s+\[(?P<version>[^\]]+)\](?:\s*-\s*(?P<date>\d{4}-\d{2}-\d{2}))?\s*$",
    re.MULTILINE,
)
# Free-form version header: ## v1.2.0 or # Version 1.2.0
_FREE_HEADER = re.compile(
    r"^#{1,3}\s+(?:v|version\s+)?(?P<version>\d+\.\d+[\.\d]*(?:[-+]\S*)?)\b",
    re.IGNORECASE | re.MULTILINE,
)
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class ChangelogNormalizer:
    """Normalizes a full changelog text into structured ``ReleaseNote`` objects.

    Args:
        repo:      Optional ``owner/repo`` slug applied to all notes.
        max_notes: Maximum number of notes to return (newest first).
    """

    def __init__(self, repo: str = "", max_notes: int = 50) -> None:
        if max_notes <= 0:
            raise ValueError(f"'max_notes' must be positive, got {max_notes!r}")
        self._repo = repo
        self._max_notes = max_notes

    def normalize(self, changelog_text: str) -> List[ReleaseNote]:
        """Parse *changelog_text* into a list of ``ReleaseNote`` objects.

        Detects the changelog format automatically.

        Args:
            changelog_text: Full changelog string (Markdown or plain text).

        Returns:
            List of ``ReleaseNote`` sorted by version date descending.

        Raises:
            TypeError: If *changelog_text* is not a string.
        """
        if not isinstance(changelog_text, str):
            raise TypeError(f"'changelog_text' must be str, got {type(changelog_text)!r}")
        if not changelog_text.strip():
            return []

        if _KACL_HEADER.search(changelog_text):
            notes = self._parse_kacl(changelog_text)
        elif _FREE_HEADER.search(changelog_text):
            notes = self._parse_free_form(changelog_text)
        else:
            # Treat entire text as a single release body
            notes = self._parse_single_body(changelog_text)

        notes.sort(key=self._sort_key, reverse=True)
        logger.debug("ChangelogNormalizer: %d release notes detected", len(notes))
        return notes[: self._max_notes]

    # ------------------------------------------------------------------
    # Format-specific parsers
    # ------------------------------------------------------------------

    def _parse_kacl(self, text: str) -> List[ReleaseNote]:
        """Parse Keep-a-Changelog format."""
        notes: List[ReleaseNote] = []
        blocks = self._split_by_header(text, _KACL_HEADER)
        for version, date_str, body in blocks:
            if version.lower() in ("unreleased", "upcoming"):
                continue
            published_at = self._parse_date(date_str)
            parser = ReleaseParser(version=version, repo=self._repo, published_at=published_at)
            notes.append(parser.parse(body))
        return notes

    def _parse_free_form(self, text: str) -> List[ReleaseNote]:
        """Parse free-form version-header changelog."""
        notes: List[ReleaseNote] = []
        blocks = self._split_by_header(text, _FREE_HEADER)
        for version, date_str, body in blocks:
            published_at = self._parse_date(date_str) or self._extract_date_from_body(body)
            parser = ReleaseParser(version=version, repo=self._repo, published_at=published_at)
            notes.append(parser.parse(body))
        return notes

    def _parse_single_body(self, text: str) -> List[ReleaseNote]:
        """Treat the entire text as one release body with unknown version."""
        parser = ReleaseParser(version="unknown", repo=self._repo)
        return [parser.parse(text)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _split_by_header(
        text: str,
        pattern: re.Pattern[str],
    ) -> List[Tuple[str, str, str]]:
        """Split *text* at each header match.

        Returns:
            List of (version, date_str, body) triples.
        """
        results: List[Tuple[str, str, str]] = []
        matches = list(pattern.finditer(text))
        for i, m in enumerate(matches):
            version = m.group("version") if "version" in m.groupdict() else m.group(0).strip("#").strip()
            date_str = m.group("date") if "date" in m.groupdict() and m.group("date") else ""
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()
            results.append((version.strip(), date_str, body))
        return results

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _extract_date_from_body(body: str) -> Optional[datetime]:
        m = _ISO_DATE.search(body[:200])
        if m:
            try:
                return datetime.strptime(m.group(0), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return None

    @staticmethod
    def _sort_key(note: ReleaseNote):
        if note.published_at:
            return (1, note.published_at.timestamp())
        # Fallback: parse version numerically
        parts = re.findall(r"\d+", note.version)
        return (0, tuple(int(p) for p in parts[:3]))


"""Release note parser.

Converts GitHub release body Markdown (and similar formats) into a
structured ``ReleaseNote`` object with categorized ``ChangeEntry`` lists.

Detection strategy (in order):
1. **Keep-a-Changelog** headers: ``### Added``, ``### Breaking Changes``, etc.
2. **GitHub Release sections**: ``## What's Changed``, ``## New Features``, etc.
3. **Bullet-prefix heuristics**: ``[BREAKING]``, ``[FIX]``, ``feat:``, ``fix:``
   (Conventional Commits prefix style)
4. **Fallback**: All bullet lines → ``OTHER`` category
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from app.devintel.models import ChangeCategory, ChangeEntry, ReleaseNote

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Section-heading → ChangeCategory mapping
# ---------------------------------------------------------------------------
_SECTION_MAP: List[Tuple[re.Pattern[str], ChangeCategory]] = [
    (re.compile(r"^#+\s*(breaking\s*changes?|incompatible)\s*$", re.I), ChangeCategory.BREAKING),
    (re.compile(r"^#+\s*(new\s*features?|added|what'?s?\s*new)\s*$", re.I), ChangeCategory.FEATURE),
    (re.compile(r"^#+\s*(bug\s*fix(es)?|fixed|fixes)\s*$", re.I), ChangeCategory.FIX),
    (re.compile(r"^#+\s*(deprecated?|deprecations?)\s*$", re.I), ChangeCategory.DEPRECATION),
    (re.compile(r"^#+\s*(security)\s*$", re.I), ChangeCategory.SECURITY),
    (re.compile(r"^#+\s*(performance|perf)\s*$", re.I), ChangeCategory.PERFORMANCE),
    (re.compile(r"^#+\s*(documentation?|docs)\s*$", re.I), ChangeCategory.DOCUMENTATION),
    (re.compile(r"^#+\s*(dependencies?|dependency\s*updates?)\s*$", re.I), ChangeCategory.DEPENDENCY),
    (re.compile(r"^#+\s*(migration)\s*$", re.I), ChangeCategory.MIGRATION),
]

# Conventional Commits prefixes
_CONV_COMMIT_PREFIX: List[Tuple[re.Pattern[str], ChangeCategory]] = [
    (re.compile(r"^feat(!)?[\s:(]", re.I), ChangeCategory.FEATURE),
    (re.compile(r"^fix[\s:(]", re.I), ChangeCategory.FIX),
    (re.compile(r"^break(ing)?[\s:(]|\[BREAKING\]", re.I), ChangeCategory.BREAKING),
    (re.compile(r"^dep(recate)?[\s:(]|\[DEPRECATED?\]", re.I), ChangeCategory.DEPRECATION),
    (re.compile(r"^sec(urity)?[\s:(]", re.I), ChangeCategory.SECURITY),
    (re.compile(r"^perf[\s:(]", re.I), ChangeCategory.PERFORMANCE),
    (re.compile(r"^docs?[\s:(]", re.I), ChangeCategory.DOCUMENTATION),
    (re.compile(r"^refactor[\s:(]", re.I), ChangeCategory.REFACTOR),
    (re.compile(r"^(chore|deps?)[\s:(]", re.I), ChangeCategory.DEPENDENCY),
]

_BULLET = re.compile(r"^\s*[-*•]\s+(.+)$")
_PR_REF = re.compile(r"#(\d+)")
_COMMIT_SHA = re.compile(r"\b([0-9a-f]{7,40})\b")
_AUTHOR_REF = re.compile(r"@([a-zA-Z0-9\-]+)")
_MIGRATION_SECTION = re.compile(r"^#+\s*(migration|upgrade\s*guide|how\s+to\s+upgrade)", re.I)

_BREAKING_INLINE = re.compile(r"\[BREAKING\]|⚠️\s*BREAKING|BREAKING\s*CHANGE", re.I)


class ReleaseParser:
    """Parses GitHub release body text into a structured ``ReleaseNote``.

    Args:
        version:      Release version string.
        repo:         ``owner/repo`` GitHub slug.
        url:          Release HTML URL.
        published_at: Release datetime.
        title:        Release title (default = version).
    """

    def __init__(
        self,
        version: str,
        repo: str = "",
        url: str = "",
        published_at: Optional[datetime] = None,
        title: str = "",
    ) -> None:
        if not version or not isinstance(version, str):
            raise ValueError("'version' must be a non-empty string")
        self._version = version.strip()
        self._repo = repo
        self._url = url
        self._published_at = published_at
        self._title = title or version

    def parse(self, body: str) -> ReleaseNote:
        """Parse a release body string into a ``ReleaseNote``.

        Args:
            body: Markdown release body text.

        Returns:
            ``ReleaseNote`` with categorized change entries.

        Raises:
            TypeError: If *body* is not a string.
        """
        if not isinstance(body, str):
            raise TypeError(f"'body' must be str, got {type(body)!r}")

        features: List[ChangeEntry] = []
        fixes: List[ChangeEntry] = []
        breaking: List[ChangeEntry] = []
        deprecations: List[ChangeEntry] = []
        security: List[ChangeEntry] = []
        migration_lines: List[str] = []
        other: List[ChangeEntry] = []

        current_category = ChangeCategory.OTHER
        in_migration = False
        lines = body.splitlines()

        for line in lines:
            # Section heading detection
            heading_cat = self._classify_heading(line)
            if heading_cat is not None:
                current_category = heading_cat
                in_migration = bool(_MIGRATION_SECTION.match(line))
                continue

            # Bullet extraction
            bullet_m = _BULLET.match(line)
            if bullet_m:
                text = bullet_m.group(1).strip()
                if in_migration:
                    migration_lines.append(text)
                    continue
                entry = self._make_entry(text, current_category)
                self._route_entry(entry, features, fixes, breaking, deprecations, security, other)
                continue

            # Migration section prose
            if in_migration and line.strip():
                migration_lines.append(line.strip())

        note = ReleaseNote(
            version=self._version,
            repo=self._repo,
            published_at=self._published_at,
            url=self._url,
            title=self._title,
            features=features,
            fixes=fixes,
            breaking=breaking,
            deprecations=deprecations,
            security=security,
            migration_notes="\n".join(migration_lines),
            raw_body=body,
        )
        logger.debug(
            "ReleaseParser: v%s feat=%d fix=%d breaking=%d dep=%d sec=%d",
            self._version, len(features), len(fixes), len(breaking), len(deprecations), len(security),
        )
        return note

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_heading(line: str) -> Optional[ChangeCategory]:
        for pattern, category in _SECTION_MAP:
            if pattern.match(line):
                return category
        return None

    def _make_entry(self, text: str, default_category: ChangeCategory) -> ChangeEntry:
        category = default_category
        is_breaking = bool(_BREAKING_INLINE.search(text))
        if is_breaking:
            category = ChangeCategory.BREAKING
        else:
            for pattern, cat in _CONV_COMMIT_PREFIX:
                if pattern.match(text):
                    category = cat
                    if "!" in text[:8]:
                        is_breaking = True
                        category = ChangeCategory.BREAKING
                    break

        pr_m = _PR_REF.search(text)
        sha_m = _COMMIT_SHA.search(text)
        author_m = _AUTHOR_REF.search(text)
        return ChangeEntry(
            text=text,
            category=category,
            is_breaking=is_breaking,
            pr_number=int(pr_m.group(1)) if pr_m else None,
            commit_sha=sha_m.group(1) if sha_m else "",
            author=author_m.group(1) if author_m else "",
        )

    @staticmethod
    def _route_entry(
        entry: ChangeEntry,
        features: List[ChangeEntry],
        fixes: List[ChangeEntry],
        breaking: List[ChangeEntry],
        deprecations: List[ChangeEntry],
        security: List[ChangeEntry],
        other: List[ChangeEntry],
    ) -> None:
        if entry.is_breaking or entry.category == ChangeCategory.BREAKING:
            breaking.append(entry)
        elif entry.category == ChangeCategory.FEATURE:
            features.append(entry)
        elif entry.category == ChangeCategory.FIX:
            fixes.append(entry)
        elif entry.category == ChangeCategory.DEPRECATION:
            deprecations.append(entry)
        elif entry.category == ChangeCategory.SECURITY:
            security.append(entry)
        else:
            other.append(entry)


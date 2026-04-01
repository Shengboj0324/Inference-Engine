"""Version difference analyzer.

Compares two semantic versions and classifies the change type:
- ``patch``   — backwards-compatible bug fix (0.0.X)
- ``minor``   — backwards-compatible new feature (0.X.0)
- ``major``   — potentially breaking (X.0.0)
- ``breaking`` — confirmed breaking based on release notes

Provides ``upgrade_urgency_score`` ∈ [0, 1] combining:
- Change magnitude (patch < minor < major)
- Presence of security fixes (+0.3)
- Days since release (staleness penalty)
- Number of confirmed breaking changes
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import List, Optional

from app.devintel.models import BreakingChange, ImpactLevel, ReleaseNote

logger = logging.getLogger(__name__)

_SEMVER = re.compile(
    r"^v?(?P<major>\d+)\.(?P<minor>\d+)\.?(?P<patch>\d+)?(?:[-+].+)?$"
)


def _parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a semver string into (major, minor, patch) ints.

    Returns:
        Tuple of (major, minor, patch).

    Raises:
        ValueError: If *version* is not parseable as semver.
    """
    if not version or not isinstance(version, str):
        raise ValueError(f"'version' must be a non-empty string, got {version!r}")
    m = _SEMVER.match(version.strip())
    if not m:
        raise ValueError(f"Cannot parse as semver: {version!r}")
    major = int(m.group("major"))
    minor = int(m.group("minor"))
    patch = int(m.group("patch") or 0)
    return major, minor, patch


class VersionDiff:
    """Represents the semantic difference between two versions.

    Attributes:
        from_version: Starting version string.
        to_version:   Target version string.
        change_type:  ``"patch"`` | ``"minor"`` | ``"major"`` | ``"breaking"`` | ``"unknown"``.
        from_parts:   (major, minor, patch) tuple for *from_version*.
        to_parts:     (major, minor, patch) tuple for *to_version*.
        upgrade_urgency_score: Float ∈ [0, 1].
        breaking_changes: Detected breaking changes in *to_version*.
    """

    def __init__(
        self,
        from_version: str,
        to_version: str,
        change_type: str,
        from_parts: tuple[int, int, int],
        to_parts: tuple[int, int, int],
        upgrade_urgency_score: float,
        breaking_changes: Optional[List[BreakingChange]] = None,
    ) -> None:
        self.from_version = from_version
        self.to_version = to_version
        self.change_type = change_type
        self.from_parts = from_parts
        self.to_parts = to_parts
        self.upgrade_urgency_score = upgrade_urgency_score
        self.breaking_changes = breaking_changes or []

    def __repr__(self) -> str:
        return (
            f"VersionDiff({self.from_version!r} → {self.to_version!r}, "
            f"type={self.change_type!r}, urgency={self.upgrade_urgency_score:.2f})"
        )


class VersionDiffAnalyzer:
    """Analyzes semantic version differences and computes upgrade urgency.

    Args:
        security_urgency_bonus: Bonus added to urgency when security fixes exist.
        staleness_half_life_days: Days until staleness penalty reaches 0.5.
    """

    def __init__(
        self,
        security_urgency_bonus: float = 0.3,
        staleness_half_life_days: int = 90,
    ) -> None:
        if not (0.0 <= security_urgency_bonus <= 1.0):
            raise ValueError(f"'security_urgency_bonus' must be in [0, 1]")
        if staleness_half_life_days <= 0:
            raise ValueError(f"'staleness_half_life_days' must be positive")
        self._security_bonus = security_urgency_bonus
        self._staleness_half_life = staleness_half_life_days

    def analyze(
        self,
        from_version: str,
        to_version: str,
        release_note: Optional[ReleaseNote] = None,
        breaking_changes: Optional[List[BreakingChange]] = None,
        released_at: Optional[datetime] = None,
    ) -> VersionDiff:
        """Compare two version strings and compute upgrade urgency.

        Args:
            from_version:     Currently installed version.
            to_version:       Available version to upgrade to.
            release_note:     Structured release note for *to_version*.
            breaking_changes: Pre-detected breaking changes.
            released_at:      When *to_version* was released (for staleness).

        Returns:
            ``VersionDiff`` with change type and urgency score.

        Raises:
            ValueError: If either version string is invalid semver.
        """
        try:
            from_parts = _parse_semver(from_version)
        except ValueError:
            from_parts = (0, 0, 0)

        try:
            to_parts = _parse_semver(to_version)
        except ValueError:
            to_parts = (0, 0, 0)

        change_type = self._classify(from_parts, to_parts, breaking_changes or [])
        urgency = self._compute_urgency(
            change_type,
            release_note,
            breaking_changes or [],
            released_at,
        )
        return VersionDiff(
            from_version=from_version,
            to_version=to_version,
            change_type=change_type,
            from_parts=from_parts,
            to_parts=to_parts,
            upgrade_urgency_score=urgency,
            breaking_changes=breaking_changes or [],
        )

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(
        from_parts: tuple[int, int, int],
        to_parts: tuple[int, int, int],
        breaking: List[BreakingChange],
    ) -> str:
        if breaking:
            return "breaking"
        if to_parts[0] != from_parts[0]:
            return "major"
        if to_parts[1] != from_parts[1]:
            return "minor"
        if to_parts[2] != from_parts[2]:
            return "patch"
        return "unknown"

    def _compute_urgency(
        self,
        change_type: str,
        note: Optional[ReleaseNote],
        breaking: List[BreakingChange],
        released_at: Optional[datetime],
    ) -> float:
        base = {"breaking": 0.80, "major": 0.65, "minor": 0.40, "patch": 0.20, "unknown": 0.30}
        score = base.get(change_type, 0.30)

        # Security bonus
        has_security = note and bool(note.security)
        if has_security:
            score = min(score + self._security_bonus, 1.0)

        # Critical breaking change bonus
        has_critical = any(bc.impact_level == ImpactLevel.CRITICAL for bc in breaking)
        if has_critical:
            score = min(score + 0.15, 1.0)

        # Staleness: older unreplaced releases reduce urgency slightly
        if released_at:
            now = datetime.now(timezone.utc)
            days_old = (now - released_at).days
            staleness = max(0.0, 1.0 - days_old / (self._staleness_half_life * 2))
            score *= (0.8 + 0.2 * staleness)

        return round(min(max(score, 0.0), 1.0), 3)


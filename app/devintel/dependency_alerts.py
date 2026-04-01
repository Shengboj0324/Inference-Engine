"""Dependency alert engine.

Watches a set of (package_name → current_version) pairs.  When the caller
calls ``process_release(release_note)``, the engine checks whether the
released package matches a watched dependency, computes a ``VersionDiff``,
and emits a ``DependencyAlert``.

Thread-safe: all mutations use a ``threading.Lock``.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional

from app.devintel.models import DependencyAlert, ImpactLevel, ReleaseNote
from app.devintel.version_diff_analyzer import VersionDiffAnalyzer

logger = logging.getLogger(__name__)

_AlertCallback = Callable[[DependencyAlert], None]


class DependencyAlertEngine:
    """Tracks watched packages and emits ``DependencyAlert`` on new releases.

    Args:
        version_analyzer:  ``VersionDiffAnalyzer`` instance.
        callbacks:         List of callables invoked with each alert.
        package_repo_map:  Optional dict mapping package names to repo slugs.
    """

    def __init__(
        self,
        version_analyzer: Optional[VersionDiffAnalyzer] = None,
        callbacks: Optional[List[_AlertCallback]] = None,
        package_repo_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self._analyzer = version_analyzer or VersionDiffAnalyzer()
        self._callbacks: List[_AlertCallback] = list(callbacks or [])
        self._package_repo_map: Dict[str, str] = dict(package_repo_map or {})
        # {package_name: current_pinned_version}
        self._watched: Dict[str, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def watch(self, package_name: str, current_version: str, repo: str = "") -> None:
        """Register a package to watch.

        Args:
            package_name:    Package name (case-insensitive).
            current_version: Currently installed version.
            repo:            Optional ``owner/repo`` slug.

        Raises:
            ValueError: If *package_name* or *current_version* is empty.
        """
        if not package_name or not isinstance(package_name, str):
            raise ValueError("'package_name' must be a non-empty string")
        if not current_version or not isinstance(current_version, str):
            raise ValueError("'current_version' must be a non-empty string")
        with self._lock:
            key = package_name.strip().lower()
            self._watched[key] = current_version.strip()
            if repo:
                self._package_repo_map[key] = repo

    def unwatch(self, package_name: str) -> bool:
        """Remove a package from the watch list.

        Returns:
            True if the package was watched, False if it was not.
        """
        if not package_name:
            raise ValueError("'package_name' must be a non-empty string")
        with self._lock:
            key = package_name.strip().lower()
            if key in self._watched:
                del self._watched[key]
                self._package_repo_map.pop(key, None)
                return True
            return False

    def register_callback(self, callback: _AlertCallback) -> None:
        """Register a callable to be invoked with each ``DependencyAlert``.

        Raises:
            TypeError: If *callback* is not callable.
        """
        if not callable(callback):
            raise TypeError(f"'callback' must be callable, got {type(callback)!r}")
        with self._lock:
            self._callbacks.append(callback)

    def watched_packages(self) -> Dict[str, str]:
        """Return a snapshot of {package_name: current_version} dict."""
        with self._lock:
            return dict(self._watched)

    def __len__(self) -> int:
        with self._lock:
            return len(self._watched)

    # ------------------------------------------------------------------
    # Alert processing
    # ------------------------------------------------------------------

    def process_release(self, release_note: ReleaseNote) -> Optional[DependencyAlert]:
        """Check whether *release_note* matches a watched dependency.

        If matched, computes a ``VersionDiff``, creates a ``DependencyAlert``,
        invokes all registered callbacks, and returns the alert.

        Args:
            release_note: Structured release note from ``ReleaseParser``.

        Returns:
            ``DependencyAlert`` if the release matches a watched package,
            ``None`` otherwise.

        Raises:
            TypeError: If *release_note* is not a ``ReleaseNote``.
        """
        if not isinstance(release_note, ReleaseNote):
            raise TypeError(f"Expected ReleaseNote, got {type(release_note)!r}")

        package_name = self._resolve_package_name(release_note)
        if package_name is None:
            return None

        with self._lock:
            current_version = self._watched.get(package_name, "")

        if not current_version:
            return None

        new_version = release_note.version
        if new_version == current_version:
            return None

        try:
            diff = self._analyzer.analyze(
                from_version=current_version,
                to_version=new_version,
                release_note=release_note,
                released_at=release_note.published_at,
            )
        except ValueError:
            diff = None

        impact = ImpactLevel.MEDIUM
        if diff:
            impact = {
                "breaking": ImpactLevel.HIGH,
                "major": ImpactLevel.HIGH,
                "minor": ImpactLevel.MEDIUM,
                "patch": ImpactLevel.LOW,
            }.get(diff.change_type, ImpactLevel.MEDIUM)
            if release_note.security:
                impact = ImpactLevel.CRITICAL

        urgency = diff.upgrade_urgency_score if diff else 0.5
        repo = release_note.repo or self._package_repo_map.get(package_name, "")

        alert = DependencyAlert(
            package_name=package_name,
            old_version=current_version,
            new_version=new_version,
            repo=repo,
            impact_level=impact,
            breaking_changes=diff.breaking_changes if diff else [],
            upgrade_urgency_score=urgency,
        )

        logger.info(
            "DependencyAlertEngine: alert %s %s→%s impact=%s urgency=%.2f",
            package_name, current_version, new_version, impact.value, urgency,
        )
        self._fire_callbacks(alert)
        return alert

    def _resolve_package_name(self, note: ReleaseNote) -> Optional[str]:
        """Match release note repo or title to a watched package name."""
        with self._lock:
            watched = set(self._watched.keys())
        # Try repo slug match: owner/repo-name → repo-name
        if note.repo:
            repo_parts = note.repo.lower().split("/")
            for part in repo_parts:
                if part in watched:
                    return part
        # Try title / version prefix
        for pkg in watched:
            if pkg in note.repo.lower() or pkg in note.title.lower():
                return pkg
        return None

    def _fire_callbacks(self, alert: DependencyAlert) -> None:
        with self._lock:
            callbacks = list(self._callbacks)
        for cb in callbacks:
            try:
                cb(alert)
            except Exception as exc:
                logger.warning("DependencyAlertEngine: callback failed (%s)", exc)


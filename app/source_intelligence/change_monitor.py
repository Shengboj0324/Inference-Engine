"""Change monitor.

Tracks per-source state (last-seen item ID, last-seen content hash, fetch
cursor) so that repeated calls to connectors only retrieve *new* content.

The monitor persists its state atomically to a JSON file and supports
loading from that file on restart.  Between restarts, all state lives in
memory and is protected by a ``threading.Lock``.

``ChangeEvent`` objects are emitted when a source's state changes.  Callers
subscribe via ``register_listener(callback)`` and receive events
synchronously in the thread that calls ``record_fetch_result()``.

Typical usage::

    monitor = ChangeMonitor(state_path=Path("/var/smr/change_monitor.json"))
    monitor.set_last_cursor("openai/openai-python", "2024-01-01T00:00:00Z")
    cursor = monitor.get_last_cursor("openai/openai-python")
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_STATE_SCHEMA_VERSION = "1.0"


@dataclass
class ChangeEvent:
    """Emitted when a source produces new content.

    Attributes:
        source_id:      Source that changed.
        new_item_count: Number of new items in this fetch.
        cursor_before:  Cursor/timestamp before this fetch.
        cursor_after:   Cursor/timestamp after this fetch.
        detected_at:    UTC wall-clock time of detection.
        metadata:       Arbitrary extra data.
    """

    source_id: str
    new_item_count: int
    cursor_before: Optional[str]
    cursor_after: Optional[str]
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChangeMonitor:
    """Thread-safe monitor that tracks per-source fetch state.

    State schema per source::

        {
          "last_cursor":       str | null,
          "last_item_id":      str | null,
          "last_content_hash": str | null,
          "last_fetch_at":     ISO-8601 str | null,
          "total_items_seen":  int,
          "consecutive_errors": int,
        }

    Args:
        state_path: Optional path for atomic JSON persistence.
                    If None, state is in-memory only.
    """

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self._state: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._listeners: List[Callable[[ChangeEvent], None]] = []
        self._state_path = state_path
        if state_path and Path(state_path).exists():
            self._load(Path(state_path))

    # ------------------------------------------------------------------
    # Listener management
    # ------------------------------------------------------------------

    def register_listener(self, callback: Callable[[ChangeEvent], None]) -> None:
        """Register a callback to receive ``ChangeEvent`` objects.

        Raises:
            TypeError: If *callback* is not callable.
        """
        if not callable(callback):
            raise TypeError(f"'callback' must be callable, got {type(callback)!r}")
        with self._lock:
            self._listeners.append(callback)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_last_cursor(self, source_id: str) -> Optional[str]:
        """Return the last-seen pagination cursor for *source_id*."""
        self._require_source_id(source_id)
        with self._lock:
            return self._state.get(source_id, {}).get("last_cursor")

    def set_last_cursor(self, source_id: str, cursor: Optional[str]) -> None:
        """Update the pagination cursor for *source_id*."""
        self._require_source_id(source_id)
        with self._lock:
            self._ensure_entry(source_id)["last_cursor"] = cursor

    def get_last_item_id(self, source_id: str) -> Optional[str]:
        """Return the last-seen item ID for *source_id*."""
        self._require_source_id(source_id)
        with self._lock:
            return self._state.get(source_id, {}).get("last_item_id")

    def set_last_item_id(self, source_id: str, item_id: Optional[str]) -> None:
        """Set the last-seen item ID for *source_id*."""
        self._require_source_id(source_id)
        with self._lock:
            self._ensure_entry(source_id)["last_item_id"] = item_id

    def record_fetch_result(
        self,
        source_id: str,
        new_item_count: int,
        cursor_after: Optional[str] = None,
        content_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ChangeEvent]:
        """Record the outcome of a fetch for *source_id*.

        Emits a ``ChangeEvent`` to registered listeners when ``new_item_count > 0``.

        Args:
            source_id:       Source that was fetched.
            new_item_count:  Number of genuinely new items returned.
            cursor_after:    New cursor/timestamp to persist.
            content_hash:    SHA-256 of concatenated content (for idempotency).
            metadata:        Arbitrary extra data for the event.

        Returns:
            The emitted ``ChangeEvent``, or None if no change detected.
        """
        self._require_source_id(source_id)
        if not isinstance(new_item_count, int) or new_item_count < 0:
            raise ValueError(f"'new_item_count' must be a non-negative int, got {new_item_count!r}")

        t0 = time.perf_counter()
        with self._lock:
            entry = self._ensure_entry(source_id)
            cursor_before = entry.get("last_cursor")
            entry["last_cursor"] = cursor_after or entry.get("last_cursor")
            entry["last_content_hash"] = content_hash or entry.get("last_content_hash")
            entry["last_fetch_at"] = datetime.now(timezone.utc).isoformat()
            entry["total_items_seen"] = entry.get("total_items_seen", 0) + new_item_count
            if new_item_count > 0:
                entry["consecutive_errors"] = 0
            listeners_snapshot = list(self._listeners)

        logger.debug(
            "ChangeMonitor.record_fetch_result: source=%r new_items=%d latency_ms=%.2f",
            source_id, new_item_count, (time.perf_counter() - t0) * 1000,
        )

        if new_item_count == 0:
            return None

        event = ChangeEvent(
            source_id=source_id,
            new_item_count=new_item_count,
            cursor_before=cursor_before,
            cursor_after=cursor_after,
            metadata=metadata or {},
        )
        for listener in listeners_snapshot:
            try:
                listener(event)
            except Exception as exc:
                logger.warning("ChangeMonitor: listener raised %s: %s", type(exc).__name__, exc)

        if self._state_path:
            self._persist(Path(self._state_path))
        return event

    def record_error(self, source_id: str) -> None:
        """Increment the consecutive-error counter for *source_id*."""
        self._require_source_id(source_id)
        with self._lock:
            entry = self._ensure_entry(source_id)
            entry["consecutive_errors"] = entry.get("consecutive_errors", 0) + 1
        logger.debug("ChangeMonitor.record_error: source=%r", source_id)

    def get_consecutive_errors(self, source_id: str) -> int:
        """Return current consecutive error count for *source_id*."""
        self._require_source_id(source_id)
        with self._lock:
            return self._state.get(source_id, {}).get("consecutive_errors", 0)

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Return a deep copy of the full state dict."""
        import copy
        with self._lock:
            return copy.deepcopy(self._state)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, path: Path) -> None:
        t0 = time.perf_counter()
        tmp = path.with_suffix(".json.tmp")
        with self._lock:
            payload = {"version": _STATE_SCHEMA_VERSION, "state": dict(self._state)}
        try:
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("ChangeMonitor: failed to persist state: %s", exc)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        logger.debug("ChangeMonitor._persist: path=%s latency_ms=%.2f", path, (time.perf_counter() - t0) * 1000)

    def _load(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8")
            payload = json.loads(text)
            with self._lock:
                self._state = payload.get("state", {})
            logger.info("ChangeMonitor: loaded state from %s (%d sources)", path, len(self._state))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ChangeMonitor: failed to load state from %s: %s", path, exc)

    def persist(self, path: Optional[Path] = None) -> None:
        """Manually trigger persistence to *path* (or configured state_path)."""
        target = Path(path) if path else (Path(self._state_path) if self._state_path else None)
        if target is None:
            raise ValueError("No state_path configured and no path argument provided")
        self._persist(target)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_entry(self, source_id: str) -> Dict[str, Any]:
        """Return the state dict for *source_id*, creating if absent. Must be called under _lock."""
        if source_id not in self._state:
            self._state[source_id] = {
                "last_cursor": None, "last_item_id": None, "last_content_hash": None,
                "last_fetch_at": None, "total_items_seen": 0, "consecutive_errors": 0,
            }
        return self._state[source_id]

    @staticmethod
    def _require_source_id(source_id: str) -> None:
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError(f"'source_id' must be a non-empty string, got {source_id!r}")


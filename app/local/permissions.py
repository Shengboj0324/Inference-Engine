"""Local permission manager for desktop-agent actions.

The desktop sidecar will eventually expose tool-call surfaces (file
access, outbound HTTP fetches, subprocess execution, use of a stored
LLM key for billable calls).  Each surface is gated by a
:class:`Permission` whose grant state is persisted to a JSON file in
the user data directory so that consent survives restarts.

Three states are tracked per permission:

* ``ASK``    — default; callers must surface a prompt before proceeding.
* ``GRANT``  — caller may proceed without prompting.
* ``DENY``   — caller must refuse without prompting.

Grants may carry an ``expires_at`` Unix timestamp (``None`` = forever).
Expired grants automatically degrade back to ``ASK`` on next read.

This module is intentionally synchronous and lock-protected — the file
is rewritten atomically on every change so even an OS-level crash
cannot leave a half-written permission state.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from app.local.user_data_dir import get_user_data_dir

_FILE_NAME = "permissions.json"


class Permission(str, Enum):
    """Discrete capability the user can grant to the local agent."""

    USE_LLM_KEY = "use_llm_key"
    READ_FILES = "read_files"
    WRITE_FILES = "write_files"
    NETWORK_FETCH = "network_fetch"
    EXECUTE_SHELL = "execute_shell"


class Decision(str, Enum):
    """User decision recorded for a :class:`Permission`."""

    ASK = "ask"
    GRANT = "grant"
    DENY = "deny"


@dataclass(frozen=True)
class Grant:
    """Per-permission state record."""

    permission: Permission
    decision: Decision = Decision.ASK
    granted_at: Optional[float] = None
    expires_at: Optional[float] = None
    note: Optional[str] = None

    def is_active(self, now: Optional[float] = None) -> bool:
        if self.decision is not Decision.GRANT:
            return False
        if self.expires_at is None:
            return True
        return (now or time.time()) < self.expires_at


class PermissionError(Exception):
    """Raised when an operation is invoked without the required grant."""


class PermissionManager:
    """Persistent permission store.

    Thread-safe via an internal lock; file writes use the standard
    write-tmp-then-rename atomic pattern.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or (get_user_data_dir() / _FILE_NAME)
        self._lock = threading.RLock()
        self._cache: Dict[Permission, Grant] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> Dict[Permission, Grant]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        out: Dict[Permission, Grant] = {}
        for entry in raw.get("grants", []):
            try:
                p = Permission(entry["permission"])
                d = Decision(entry.get("decision", "ask"))
                out[p] = Grant(
                    permission=p,
                    decision=d,
                    granted_at=entry.get("granted_at"),
                    expires_at=entry.get("expires_at"),
                    note=entry.get("note"),
                )
            except (KeyError, ValueError):
                continue
        return out

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "grants": [
                {
                    "permission": g.permission.value,
                    "decision": g.decision.value,
                    "granted_at": g.granted_at,
                    "expires_at": g.expires_at,
                    "note": g.note,
                }
                for g in self._cache.values()
            ]
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_grants(self) -> List[Grant]:
        """Return one :class:`Grant` per known :class:`Permission`."""
        with self._lock:
            out: List[Grant] = []
            for p in Permission:
                g = self._cache.get(p, Grant(permission=p))
                if (g.decision is Decision.GRANT and g.expires_at is not None
                        and g.expires_at <= time.time()):
                    # Auto-degrade expired grants to ASK on the way out.
                    g = Grant(permission=p)
                out.append(g)
            return out

    def get(self, permission: Permission) -> Grant:
        with self._lock:
            g = self._cache.get(permission, Grant(permission=permission))
            if (g.decision is Decision.GRANT and g.expires_at is not None
                    and g.expires_at <= time.time()):
                # Persist the degradation so future reads do not re-check.
                degraded = Grant(permission=permission)
                self._cache[permission] = degraded
                self._save()
                return degraded
            return g

    def grant(
        self,
        permission: Permission,
        *,
        ttl_seconds: Optional[float] = None,
        note: Optional[str] = None,
    ) -> Grant:
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive or None")
        now = time.time()
        g = Grant(
            permission=permission,
            decision=Decision.GRANT,
            granted_at=now,
            expires_at=(now + ttl_seconds) if ttl_seconds else None,
            note=note,
        )
        with self._lock:
            self._cache[permission] = g
            self._save()
        return g

    def revoke(self, permission: Permission, *, note: Optional[str] = None) -> Grant:
        g = Grant(permission=permission, decision=Decision.ASK, note=note)
        with self._lock:
            self._cache[permission] = g
            self._save()
        return g

    def deny(self, permission: Permission, *, note: Optional[str] = None) -> Grant:
        g = Grant(permission=permission, decision=Decision.DENY,
                  granted_at=time.time(), note=note)
        with self._lock:
            self._cache[permission] = g
            self._save()
        return g

    def is_allowed(self, permission: Permission) -> bool:
        return self.get(permission).is_active()

    def require(self, permission: Permission) -> None:
        """Raise :class:`PermissionError` unless ``permission`` is granted."""
        if not self.is_allowed(permission):
            raise PermissionError(
                f"permission {permission.value!r} is required but not granted"
            )


_global_manager: Optional[PermissionManager] = None
_global_manager_lock = threading.Lock()


def get_permission_manager() -> PermissionManager:
    """Return the process-wide :class:`PermissionManager` singleton."""
    global _global_manager
    with _global_manager_lock:
        if _global_manager is None:
            _global_manager = PermissionManager()
        return _global_manager


def reset_permission_manager() -> None:
    """Drop the cached singleton (test-only)."""
    global _global_manager
    with _global_manager_lock:
        _global_manager = None

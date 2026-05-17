"""Shell-handshake manifest builder for the desktop sidecar.

The Tauri shell bootstraps by reading two pieces of state written by
:mod:`app.desktop.launcher`:

* ``sidecar.port`` — legacy port-only file (kept for backwards-compat).
* ``sidecar.json`` — atomic manifest with port, host, token, contract
  version, and capability flags.

Once it has the port and token it calls ``GET /api/v1/desktop/manifest``
to fetch the same capability snapshot live (so a stale on-disk file is
never authoritative).  This module is the single source of truth for the
snapshot — both the on-disk writer and the HTTP route share
:func:`build_capability_snapshot` so they cannot drift.

The snapshot is intentionally minimal: it describes *what the sidecar
binary can do* (compiled-in features), not *what the user has currently
permitted* (which is a per-request concern owned by the existing
``/api/v1/permissions`` and ``/api/v1/multimodal/status`` routes).
"""

from __future__ import annotations

from typing import Any, Dict

#: Schema version for the manifest payload itself.  Bump when the *shape*
#: of the dict returned by :func:`build_capability_snapshot` changes in a
#: breaking way (added fields are non-breaking and do not require a bump).
MANIFEST_SCHEMA_VERSION = "1"


def _module_importable(dotted_path: str) -> bool:
    """Return True when ``dotted_path`` imports without raising.

    Used to probe optional desktop modules so the shell can hide UI
    panels that have no backing implementation in this build.
    """
    import importlib

    try:
        importlib.import_module(dotted_path)
        return True
    except Exception:  # noqa: BLE001 - any import failure means "absent"
        return False


def _detect_capabilities() -> Dict[str, bool]:
    """Return the static capability map for this sidecar build.

    The values reflect *compile-time* presence of each subsystem and do
    not check whether the user has granted the corresponding permission.
    """
    return {
        "key_vault": _module_importable("app.local.key_vault"),
        "permissions": _module_importable("app.local.permissions"),
        "chat": _module_importable("app.local.chat_store"),
        "ingestion": _module_importable("app.local.ingest_runtime"),
        "multimodal": _module_importable("app.local.multimodal_analyzer"),
        # Phase 5 — flipped to True once the local RAG layer lands.
        "rag": False,
    }


def build_capability_snapshot() -> Dict[str, Any]:
    """Return the manifest payload (sans port/host/token/started_at).

    Pure: no I/O, no side effects, safe to call from any thread or
    process at any time.
    """
    # Imported lazily so this module stays importable in environments
    # where the FastAPI app construction is too expensive (e.g. tests
    # that only want the schema constants).
    from app.api.main import API_CONTRACT_VERSION
    from app.core.config import settings

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "contract_version": API_CONTRACT_VERSION,
        "app_version": getattr(settings, "version", None) or "0.1.0",
        "deployment_mode": settings.deployment_mode,
        "capabilities": _detect_capabilities(),
    }

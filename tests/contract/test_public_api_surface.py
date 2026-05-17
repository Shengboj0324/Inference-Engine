"""Phase 0 contract freeze — public FastAPI surface snapshot.

Pins every route the desktop UI sidecar will depend on:

* Path
* HTTP method set (sorted, ``HEAD``/``OPTIONS`` stripped — those are
  framework-generated and uninteresting for a UI client).
* Route class (``APIRoute`` vs ``APIWebSocketRoute``) so a route cannot
  silently change from REST to websocket or vice-versa.

If this test fails, the change is breaking for the UI build.  Update the
``FROZEN_ROUTES`` set deliberately and bump ``CONTRACT_VERSION``.
"""

from __future__ import annotations

from typing import FrozenSet, Tuple

import pytest

from app.api.main import app


CONTRACT_VERSION = "phase2.0"


# Methods that are framework-generated for every GET route — excluded from the
# snapshot so adding/removing them never trips the diff.
_NOISE_METHODS = {"HEAD", "OPTIONS"}


def _normalise_methods(methods) -> Tuple[str, ...]:
    return tuple(sorted(m for m in (methods or []) if m not in _NOISE_METHODS))


def _collect_surface() -> FrozenSet[Tuple[str, str, Tuple[str, ...]]]:
    """Return the (route_class, path, methods) triples for the live app."""
    surface = set()
    for route in app.routes:
        rclass = type(route).__name__
        if rclass not in {"APIRoute", "APIWebSocketRoute"}:
            continue
        path = getattr(route, "path", None)
        methods = _normalise_methods(getattr(route, "methods", None))
        surface.add((rclass, path, methods))
    return frozenset(surface)


# ----------------------------------------------------------------------------
# FROZEN SURFACE — phase2.0
# ----------------------------------------------------------------------------
# Each entry is (route_class, path, sorted-methods-tuple).
FROZEN_ROUTES: FrozenSet[Tuple[str, str, Tuple[str, ...]]] = frozenset({
    # Auth
    ("APIRoute", "/api/v1/auth/register", ("POST",)),
    ("APIRoute", "/api/v1/auth/login", ("POST",)),
    ("APIRoute", "/api/v1/auth/logout", ("POST",)),
    # Signals (primary product interface)
    ("APIRoute", "/api/v1/signals/queue", ("GET",)),
    ("APIRoute", "/api/v1/signals/{signal_id}", ("GET",)),
    ("APIRoute", "/api/v1/signals/{signal_id}/act", ("POST",)),
    ("APIRoute", "/api/v1/signals/{signal_id}/dismiss", ("POST",)),
    ("APIRoute", "/api/v1/signals/stats", ("GET",)),
    ("APIRoute", "/api/v1/signals/stream", ("POST",)),
    ("APIWebSocketRoute", "/api/v1/signals/ws", ()),
    ("APIRoute", "/api/v1/signals/{signal_id}/deep-research", ("POST",)),
    ("APIRoute", "/api/v1/signals/{signal_id}/chat", ("POST",)),
    ("APIRoute", "/api/v1/signals/{signal_id}/one-click-action", ("POST",)),
    ("APIRoute", "/api/v1/signals/{signal_id}/assign", ("POST",)),
    ("APIRoute", "/api/v1/signals/team", ("GET",)),
    ("APIRoute", "/api/v1/signals/{signal_id}/feedback", ("POST",)),
    # Sources
    ("APIRoute", "/api/v1/sources/", ("GET",)),
    ("APIRoute", "/api/v1/sources/", ("POST",)),
    ("APIRoute", "/api/v1/sources/{platform}/test", ("GET",)),
    ("APIRoute", "/api/v1/sources/{platform}", ("DELETE",)),
    # Digest
    ("APIRoute", "/api/v1/digest/generate", ("POST",)),
    ("APIRoute", "/api/v1/digest/latest", ("GET",)),
    ("APIRoute", "/api/v1/digest/latest/html", ("GET",)),
    ("APIRoute", "/api/v1/digest/latest/markdown", ("GET",)),
    ("APIRoute", "/api/v1/digest/history", ("GET",)),
    # Search
    ("APIRoute", "/api/v1/search/", ("POST",)),
    ("APIRoute", "/api/v1/search/topics", ("GET",)),
    # LLM
    ("APIRoute", "/api/llm/generate", ("POST",)),
    ("APIRoute", "/api/llm/chat", ("POST",)),
    ("APIRoute", "/api/llm/health", ("GET",)),
    ("APIRoute", "/api/llm/stats", ("GET",)),
    ("APIRoute", "/api/llm/tiers", ("GET",)),
    ("APIRoute", "/api/llm/tier", ("GET",)),
    ("APIRoute", "/api/llm/tier", ("POST",)),
    # Health / root
    ("APIRoute", "/", ("GET",)),
    ("APIRoute", "/health", ("GET",)),
    ("APIRoute", "/health/ready", ("GET",)),
    ("APIRoute", "/health/live", ("GET",)),
    ("APIRoute", "/api/v1/health", ("GET",)),
    # Phase 2 — Local key vault (desktop mode only; 404 in server mode)
    ("APIRoute", "/api/v1/keys", ("GET",)),
    ("APIRoute", "/api/v1/keys/{provider}", ("PUT",)),
    ("APIRoute", "/api/v1/keys/{provider}", ("DELETE",)),
    ("APIRoute", "/api/v1/keys/{provider}/test", ("POST",)),
    # Phase 2 — Chat sessions + SSE streaming (desktop mode only)
    ("APIRoute", "/api/v1/chat/sessions", ("GET",)),
    ("APIRoute", "/api/v1/chat/sessions", ("POST",)),
    ("APIRoute", "/api/v1/chat/sessions/{session_id}", ("GET",)),
    ("APIRoute", "/api/v1/chat/sessions/{session_id}", ("PATCH",)),
    ("APIRoute", "/api/v1/chat/sessions/{session_id}", ("DELETE",)),
    ("APIRoute", "/api/v1/chat/sessions/{session_id}/messages", ("POST",)),
    ("APIRoute", "/api/v1/chat/sessions/{session_id}/stream", ("POST",)),
    # Phase 2 — Local permissions (desktop mode only)
    ("APIRoute", "/api/v1/permissions", ("GET",)),
    ("APIRoute", "/api/v1/permissions/{name}", ("GET",)),
    ("APIRoute", "/api/v1/permissions/{name}/grant", ("POST",)),
    ("APIRoute", "/api/v1/permissions/{name}/revoke", ("POST",)),
    ("APIRoute", "/api/v1/permissions/{name}/deny", ("POST",)),
})


class TestPhase0PublicAPISurface:
    """Pin the routes the desktop UI sidecar will depend on."""

    def test_contract_version_stamp(self) -> None:
        """The contract version stamp must be set; bump it on any breaking edit."""
        assert CONTRACT_VERSION, "CONTRACT_VERSION must be a non-empty string"

    def test_no_routes_removed(self) -> None:
        live = _collect_surface()
        missing = FROZEN_ROUTES - live
        assert not missing, (
            "Frozen routes disappeared from the live surface — this is a "
            f"breaking change for the desktop UI client:\n  {sorted(missing)}"
        )

    def test_no_routes_silently_added(self) -> None:
        """Catches accidental endpoints that bypass the contract review.

        New endpoints are not forbidden — but they must be added to
        ``FROZEN_ROUTES`` *in the same commit* that introduces them so that
        the desktop UI client and any external consumer can be updated in
        lock-step.
        """
        live = _collect_surface()
        extras = live - FROZEN_ROUTES
        assert not extras, (
            "New routes were added without updating the contract snapshot. "
            "Add them to FROZEN_ROUTES (and bump CONTRACT_VERSION if the "
            "shape is breaking):\n  " + "\n  ".join(repr(e) for e in sorted(extras))
        )

    @pytest.mark.parametrize("route_class,path,methods", sorted(FROZEN_ROUTES))
    def test_each_frozen_route_present(
        self, route_class: str, path: str, methods: Tuple[str, ...]
    ) -> None:
        live = _collect_surface()
        assert (route_class, path, methods) in live, (
            f"Contract route missing: {route_class} {path} {methods}"
        )

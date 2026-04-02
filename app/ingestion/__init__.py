"""Ingestion pipeline package.

Exposes the source-family-specific content routing layer that acts as the
primary integration point between raw connector output and downstream
intelligence modules.

Public API
----------
- :class:`ContentPipelineRouter` — async ``route()`` dispatches a
  ``ContentItem`` to the appropriate extraction chain and returns an
  ``IntelligencePipelineResult``.
- :class:`IntelligencePipelineResult` — normalized cross-family output model.
- :class:`PipelineStatus` — ``SUCCESS``, ``PARTIAL``, ``FAILED``, ``SKIPPED``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ContentPipelineRouter",
    "IntelligencePipelineResult",
    "PipelineStatus",
]


def __getattr__(name: str):
    if name == "ContentPipelineRouter":
        from app.ingestion.content_pipeline_router import ContentPipelineRouter
        return ContentPipelineRouter
    if name in {"IntelligencePipelineResult", "PipelineStatus"}:
        import importlib
        mod = importlib.import_module("app.ingestion.pipeline_result")
        return getattr(mod, name)
    raise AttributeError(f"module 'app.ingestion' has no attribute {name!r}")


if TYPE_CHECKING:
    from app.ingestion.content_pipeline_router import ContentPipelineRouter
    from app.ingestion.pipeline_result import IntelligencePipelineResult, PipelineStatus

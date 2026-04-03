"""Startup capability validation — importable without pulling in FastAPI routes.

``validate_capabilities`` is the canonical function that verifies every
enabled capability has a real (non-stub) backend before the application begins
serving traffic.  It is called:

1. From ``app.api.main.lifespan`` via a thin shim so the function lives here,
   not buried inside ``main.py`` where it would be untestable without loading
   all route modules and their module-scope side-effects.
2. Directly in tests and in the production safety CI step, with an injectable
   ``settings`` argument to avoid any live service dependency.

Capabilities checked
--------------------
* ``enable_multimodal_vision`` — needs an LLM API key or ``local_llm_url``.
* ``enable_asr``               — needs ``faster_whisper``, ``whisper``, or
                                 ``openai_api_key``.
* ``enable_pdf_extraction``    — needs ``pdfplumber`` or ``pypdf``.

In strict mode (``settings.is_strict = True``), any capability that cannot
resolve a real backend causes a ``RuntimeError`` which must not be caught.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def validate_capabilities(settings: Optional[Any] = None) -> None:
    """Validate that every enabled capability has a real backend.

    Args:
        settings: A settings object exposing ``enable_*`` flags and
                  ``is_strict``.  When ``None``, imports the global
                  ``app.core.config.settings`` singleton.

    Raises:
        RuntimeError: In strict mode when any enabled capability lacks a
                      real backend.
    """
    if settings is None:
        from app.core.config import settings as _s
        settings = _s

    from app.core.production_guard import ProductionSafetyContract
    guard = ProductionSafetyContract(strict=settings.is_strict)

    # ── Multimodal vision ─────────────────────────────────────────────────
    if settings.enable_multimodal_vision:
        has_key = bool(
            getattr(settings, "openai_api_key", None)
            or getattr(settings, "anthropic_api_key", None)
            or getattr(settings, "local_llm_url", None)
        )
        mode = "remote_model" if has_key else "stub"
        guard.require_real_backend(
            capability="multimodal_vision",
            resolved_backend=mode,
            allowed_stubs=frozenset({"stub", "disabled"}),
        )
        logger.info("Capability check: multimodal_vision → %s", mode)

    # ── ASR ───────────────────────────────────────────────────────────────
    if settings.enable_asr:
        asr_backend = _resolve_asr_backend(settings)
        guard.require_real_backend(
            capability="asr",
            resolved_backend=asr_backend,
        )
        logger.info("Capability check: asr → %s", asr_backend)

    # ── PDF extraction ────────────────────────────────────────────────────
    if settings.enable_pdf_extraction:
        pdf_backend = _resolve_pdf_backend()
        guard.require_real_backend(
            capability="pdf_extraction",
            resolved_backend=pdf_backend,
        )
        logger.info("Capability check: pdf_extraction → %s", pdf_backend)

    logger.info(
        "Startup capability validation complete (strict=%s)", settings.is_strict
    )


def _resolve_asr_backend(settings: Any) -> str:
    """Return the name of the available ASR backend, or 'stub'."""
    try:
        import faster_whisper  # type: ignore[import]  # noqa: F401
        return "faster_whisper"
    except ImportError:
        pass
    try:
        import whisper  # type: ignore[import]  # noqa: F401
        return "whisper"
    except ImportError:
        pass
    if getattr(settings, "openai_api_key", None):
        return "openai"
    return "stub"


def _resolve_pdf_backend() -> str:
    """Return the name of the available PDF backend, or 'stub'."""
    try:
        import pdfplumber  # type: ignore[import]  # noqa: F401
        return "pdfplumber"
    except ImportError:
        pass
    try:
        import pypdf  # type: ignore[import]  # noqa: F401
        return "pypdf"
    except ImportError:
        pass
    return "stub"


"""Desktop-friendly facade around :class:`TranscriptionRouter`.

Resolves an ASR backend in this order:

1. **Local** (``faster_whisper`` → ``whisper``) — requires
   :data:`Permission.USE_LOCAL_MODELS`.
2. **BYOK OpenAI** — requires :data:`Permission.USE_LLM_KEY` *and* an
   ``openai`` key in the Phase 2 :class:`KeyVault`.

If neither path is available the runtime reports ``is_available() ==
False`` and ``transcribe()`` raises :class:`MultimodalUnavailable`.  The
underlying router's stub backend is **never** invoked from the desktop
sidecar — stubbed transcripts would silently pollute the
``ContentItem.raw_text`` and downstream embeddings.

Every segment is scrubbed through :class:`DataResidencyGuard` before the
``TranscriptResult`` leaves the runtime, so user-facing transcripts
respect the same zero-egress contract as text embeddings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from app.core.data_residency import DataResidencyGuard
from app.local.key_vault import KeyVault, get_key_vault
from app.local.multimodal_clip import MultimodalUnavailable
from app.local.permissions import (
    Permission,
    PermissionManager,
    get_permission_manager,
)

logger = logging.getLogger(__name__)


def _local_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


class LocalASRRuntime:
    """Permission-gated ASR facade for the desktop sidecar."""

    def __init__(
        self,
        *,
        permissions: Optional[PermissionManager] = None,
        vault: Optional[KeyVault] = None,
        router_factory: Optional[Any] = None,
        model_size: str = "base",
        language: Optional[str] = None,
    ) -> None:
        self._permissions = permissions or get_permission_manager()
        self._vault = vault if vault is not None else get_key_vault()
        self._router_factory = router_factory
        self._model_size = model_size
        self._language = language
        self._router: Optional[Any] = None
        self._init_failed = False

    # ------------------------------------------------------------------
    # Capability probes
    # ------------------------------------------------------------------

    def _has_byok(self) -> bool:
        if not self._permissions.is_allowed(Permission.USE_LLM_KEY):
            return False
        try:
            return bool(self._vault.get("openai"))
        except Exception:
            return False

    def _has_local(self) -> bool:
        if not self._permissions.is_allowed(Permission.USE_LOCAL_MODELS):
            return False
        return _local_whisper_available()

    def is_available(self) -> bool:
        if self._init_failed:
            return False
        return self._has_local() or self._has_byok()

    def resolved_backend(self) -> str:
        """Return a stable string describing which backend would be used."""
        if self._has_local():
            return "local_whisper"
        if self._has_byok():
            return "openai_byok"
        return "unavailable"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_router(self) -> Any:
        if self._router_factory is not None:
            return self._router_factory()
        from app.media.audio_intelligence.transcription_router import (
            ASRBackend, TranscriptionRouter,
        )
        if self._has_local():
            return TranscriptionRouter(
                model_size=self._model_size, language=self._language, device="cpu",
            )
        # BYOK path: force the OpenAI backend so the router doesn't probe
        # for local deps and trip the strict-mode guard on the stub fallback.
        api_key = self._vault.get("openai") or ""
        return TranscriptionRouter(
            backend=ASRBackend.OPENAI, model_size=self._model_size,
            language=self._language, openai_api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe(self, audio_path: str):
        """Transcribe ``audio_path`` and return a scrubbed ``TranscriptResult``."""
        if not self.is_available():
            raise MultimodalUnavailable(
                "no ASR backend available (need USE_LOCAL_MODELS + whisper, "
                "or USE_LLM_KEY + an 'openai' key in the vault)"
            )
        if not Path(audio_path).exists():
            raise FileNotFoundError(audio_path)
        if self._router is None:
            try:
                self._router = self._build_router()
            except Exception as exc:
                self._init_failed = True
                logger.warning("LocalASRRuntime: router build failed: %s", exc)
                raise MultimodalUnavailable(f"ASR build failed: {exc}") from exc
        result = await self._router.transcribe_with_provenance(audio_path)
        scrubbed = [
            seg.model_copy(update={"text": DataResidencyGuard.scrub_text(seg.text)[0]})
            for seg in result.segments
        ]
        return result.model_copy(update={"segments": scrubbed})

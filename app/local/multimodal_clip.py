"""Desktop-friendly facade around :class:`app.media.multimodal_models.CLIPModel`.

Adds three desktop concerns the upstream class does not provide:

* **Permission gate** — every call requires ``Permission.USE_LOCAL_MODELS``;
  denied calls raise :class:`MultimodalUnavailable` rather than silently
  returning zeros, so the orchestrator can mark items as ``low_confidence``.
* **Lifecycle hygiene** — the heavy model loads lazily on first use; if
  ``transformers``/``torch``/``Pillow`` are not installed (or model download
  fails offline) the runtime degrades to ``available=False`` and the
  orchestrator skips image analysis without crashing the worker.
* **PII scrubbing** — generated captions / classifier labels pass through
  :func:`DataResidencyGuard.scrub_text` before being returned, so nothing
  leaves the box that wasn't already permitted by the residency contract.

This module deliberately exposes a *narrow* surface (``embed_image``,
``classify``) — Phase 4 does not need the full ``align_image_text`` /
``semantic_search`` API.  The underlying :class:`CLIPModel` remains
available for code that does.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.data_residency import DataResidencyGuard
from app.local.permissions import (
    Permission,
    PermissionManager,
    get_permission_manager,
)

logger = logging.getLogger(__name__)


class MultimodalUnavailable(RuntimeError):
    """Raised when the local CLIP runtime cannot service a request.

    The orchestrator catches this to mark the item as ``low_confidence``
    rather than failing the whole ingestion pass.
    """


class LocalCLIPRuntime:
    """Permission-gated, lazy-loaded CLIP facade for the desktop sidecar."""

    def __init__(
        self,
        *,
        permissions: Optional[PermissionManager] = None,
        model_factory: Optional[Any] = None,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
    ) -> None:
        self._permissions = permissions or get_permission_manager()
        self._model_name = model_name
        self._device = device
        self._model_factory = model_factory
        self._model: Optional[Any] = None
        self._init_failed = False

    # ------------------------------------------------------------------
    # Availability probes
    # ------------------------------------------------------------------

    def is_permitted(self) -> bool:
        return self._permissions.is_allowed(Permission.USE_LOCAL_MODELS)

    def is_available(self) -> bool:
        """True iff the runtime *could* execute right now (perm + deps)."""
        if not self.is_permitted():
            return False
        if self._init_failed:
            return False
        if self._model is not None:
            return True
        # Cheap check: are the heavy deps importable?
        try:
            import transformers  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            return False
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._init_failed:
            raise MultimodalUnavailable("CLIP runtime previously failed to load")
        try:
            if self._model_factory is not None:
                self._model = self._model_factory()
            else:
                from app.media.multimodal_models import CLIPConfig, CLIPModel
                self._model = CLIPModel(CLIPConfig(
                    model_name=self._model_name, device=self._device,
                ))
            await self._model.initialize()
        except Exception as exc:
            self._init_failed = True
            logger.warning("LocalCLIPRuntime: model load failed: %s", exc)
            raise MultimodalUnavailable(f"CLIP load failed: {exc}") from exc
        return self._model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed_image(self, image_path: str) -> List[float]:
        """Return the CLIP image embedding for ``image_path``."""
        self._permissions.require(Permission.USE_LOCAL_MODELS)
        if not Path(image_path).exists():
            raise FileNotFoundError(image_path)
        model = await self._ensure_model()
        alignment = await model.align_image_text(image_path, ["a photo"])
        return list(alignment.image_embedding)

    async def classify(
        self,
        image_path: str,
        candidate_labels: List[str],
        *,
        hypothesis_template: str = "a photo of {}",
    ) -> Dict[str, float]:
        """Zero-shot classify ``image_path`` against ``candidate_labels``.

        Returns a dict mapping each (scrubbed) label to its softmax
        probability.  The label set is scrubbed because user-supplied
        prompts may contain PII that would otherwise be persisted in
        the ContentItem metadata blob.
        """
        self._permissions.require(Permission.USE_LOCAL_MODELS)
        if not candidate_labels:
            raise ValueError("candidate_labels must not be empty")
        if not Path(image_path).exists():
            raise FileNotFoundError(image_path)
        model = await self._ensure_model()
        scrubbed = [DataResidencyGuard.scrub_text(l)[0] for l in candidate_labels]
        return await model.zero_shot_classify(
            image_path, scrubbed, hypothesis_template=hypothesis_template,
        )

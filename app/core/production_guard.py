"""ProductionSafetyContract — global fail-closed enforcement.

All pipeline components that can fall back to stub, synthetic, or partially
implemented execution modes must call into this module before producing any
user-facing result.  When ``settings.is_strict`` is True (automatic in
``environment=production``), any stub backend causes an explicit RuntimeError
rather than a silent quality degradation.

Design principles
-----------------
* Central, not distributed — every quality gate lives here, not scattered
  across individual modules.
* Evidence provenance is mandatory — every result object must carry a
  ``BackendProvenance`` describing whether upstream evidence was full,
  partial, synthetic, or missing.
* Fail explicitly — the system must never silently fabricate quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evidence quality tiers
# ---------------------------------------------------------------------------

class EvidenceQuality(str, Enum):
    """Describes how complete upstream evidence was for a given result."""
    FULL       = "full"       # real backend, complete extraction
    PARTIAL    = "partial"    # real backend, incomplete (e.g. timeout, page limit)
    SYNTHETIC  = "synthetic"  # stub or demo-grade substitute
    MISSING    = "missing"    # no upstream evidence available at all


@dataclass(frozen=True)
class BackendProvenance:
    """Provenance record stamped on every pipeline result.

    Attributes:
        capability:      Human-readable capability name (e.g. "multimodal_vision").
        backend:         Resolved backend identifier (e.g. "gpt-4o", "stub", "pdfplumber").
        evidence_quality: Completeness tier of the upstream evidence.
        notes:           Optional freeform detail for partial/synthetic tiers.
    """
    capability: str
    backend: str
    evidence_quality: EvidenceQuality
    notes: Optional[str] = None

    @property
    def is_stub(self) -> bool:
        return self.evidence_quality in (EvidenceQuality.SYNTHETIC, EvidenceQuality.MISSING)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "backend": self.backend,
            "evidence_quality": self.evidence_quality.value,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Central safety contract
# ---------------------------------------------------------------------------

class ProductionSafetyContract:
    """Enforces fail-closed semantics for all stub / partial backend paths.

    Usage::

        guard = ProductionSafetyContract()

        # At call time — raises RuntimeError in strict mode if backend is stub
        guard.require_real_backend(
            capability="multimodal_vision",
            resolved_backend="stub",
            allowed_stubs={"stub"},
        )

        # At publication time — blocks stub-tainted results
        prov = BackendProvenance("pdf", "stub", EvidenceQuality.SYNTHETIC)
        guard.validate_publishable(prov, context="grounded_summary")
    """

    def __init__(self, strict: Optional[bool] = None) -> None:
        if strict is None:
            from app.core.config import settings
            strict = settings.is_strict
        self._strict = strict

    @property
    def strict(self) -> bool:
        return self._strict

    def require_real_backend(
        self,
        capability: str,
        resolved_backend: str,
        allowed_stubs: frozenset[str] = frozenset({"stub", "STUB"}),
    ) -> None:
        """Raise ``RuntimeError`` in strict mode if ``resolved_backend`` is a stub.

        Args:
            capability:       Name of the capability being checked.
            resolved_backend: The backend the module selected (e.g. "stub").
            allowed_stubs:    Set of backend names considered stub/synthetic.

        Raises:
            RuntimeError: If strict mode is active and the backend is a stub.
        """
        if resolved_backend.lower() in {s.lower() for s in allowed_stubs}:
            msg = (
                f"ProductionSafetyContract: capability '{capability}' resolved to "
                f"stub backend '{resolved_backend}' but production_strict_mode=True.  "
                f"Install the required dependency or disable the capability via "
                f"ENABLE_{capability.upper()}=false before deploying."
            )
            if self._strict:
                logger.critical(msg)
                raise RuntimeError(msg)
            else:
                logger.warning(
                    "ProductionSafetyContract: stub backend '%s' for capability '%s' "
                    "(allowed in non-strict mode).",
                    resolved_backend, capability,
                )

    def validate_publishable(
        self,
        provenance: BackendProvenance,
        context: str = "output",
    ) -> None:
        """Raise ``RuntimeError`` in strict mode if provenance is stub-tainted.

        Args:
            provenance: The ``BackendProvenance`` of the upstream result.
            context:    Human-readable label for the publication context.

        Raises:
            RuntimeError: If strict mode is active and evidence is synthetic/missing.
        """
        if provenance.is_stub:
            msg = (
                f"ProductionSafetyContract: refusing to publish '{context}' — "
                f"upstream evidence quality is '{provenance.evidence_quality.value}' "
                f"(backend='{provenance.backend}', capability='{provenance.capability}').  "
                f"Ensure a real backend is available or qualify the output explicitly."
            )
            if self._strict:
                logger.critical(msg)
                raise RuntimeError(msg)
            else:
                logger.warning(
                    "ProductionSafetyContract: publishing stub-tainted '%s' result "
                    "(evidence_quality=%s) — allowed in non-strict mode.",
                    context, provenance.evidence_quality.value,
                )

    def downgrade_or_block(
        self,
        provenance: BackendProvenance,
        context: str = "output",
    ) -> bool:
        """Return True if the output may proceed; False if it must be blocked.

        In strict mode, stub-tainted outputs are always blocked (False).
        In non-strict mode, they proceed with a warning (True).
        """
        if provenance.is_stub and self._strict:
            logger.warning(
                "ProductionSafetyContract: blocking '%s' — stub evidence in strict mode.",
                context,
            )
            return False
        return True


# Module-level singleton — used by components that import directly.
_guard: Optional[ProductionSafetyContract] = None


def get_guard() -> ProductionSafetyContract:
    """Return the module-level ``ProductionSafetyContract`` singleton."""
    global _guard
    if _guard is None:
        _guard = ProductionSafetyContract()
    return _guard


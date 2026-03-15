"""Typed workflow contracts — blueprint §6.

Defines the strict handler contract every workflow step must honour:
- required / optional input artifacts
- declared output artifacts
- failure and retry behaviour
- transition policy (conditions + fallback routing)

These complement workflow_models.py (runtime state) with compile-time
type safety for handler authors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Artifact taxonomy
# ---------------------------------------------------------------------------

class ArtifactType(str, Enum):
    """First-class artifact types that flow between workflow steps."""

    RAW_OBSERVATION = "raw_observation"
    NORMALIZED_OBSERVATION = "normalized_observation"
    SIGNAL_INFERENCE = "signal_inference"
    ACTIONABLE_SIGNAL = "actionable_signal"
    RESPONSE_DRAFT = "response_draft"
    POLICY_REPORT = "policy_report"
    METRIC_RECORD = "metric_record"
    NOTIFICATION_RECEIPT = "notification_receipt"
    EXECUTION_RECEIPT = "execution_receipt"
    ARBITRARY = "arbitrary"  # Escape hatch — prefer explicit types


class FailureBehavior(str, Enum):
    """What the engine does when a step raises an exception."""

    ABORT_WORKFLOW = "abort_workflow"   # Propagate failure upward; halt execution
    SKIP_STEP = "skip_step"             # Log error; continue with next step
    RETRY_THEN_ABORT = "retry_then_abort"  # Honour retry_count, then abort
    RETRY_THEN_SKIP = "retry_then_skip"   # Honour retry_count, then skip


# ---------------------------------------------------------------------------
# Typed execution context (replaces bare Dict[str, Any])
# ---------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    """Strongly typed wrapper around the mutable workflow context dict.

    The underlying ``data`` dict is the same object stored in
    ``WorkflowExecution.context`` so mutations are reflected immediately.
    """

    execution_id: UUID
    signal_id: UUID
    user_id: UUID
    data: Dict[str, Any] = field(default_factory=dict)

    # ---- artifact accessors -------------------------------------------------

    def put(self, key: ArtifactType | str, value: Any) -> None:
        self.data[str(key)] = value

    def get(self, key: ArtifactType | str, default: Any = None) -> Any:
        return self.data.get(str(key), default)

    def require(self, key: ArtifactType | str) -> Any:
        """Return value or raise KeyError with a clear message."""
        k = str(key)
        if k not in self.data:
            raise KeyError(
                f"Required artifact '{k}' missing from execution context "
                f"(execution_id={self.execution_id})"
            )
        return self.data[k]

    def has(self, key: ArtifactType | str) -> bool:
        return str(key) in self.data


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Typed step output — replaces bare Dict[str, Any] step results."""

    success: bool
    outputs: Dict[ArtifactType | str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_recommended: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, **outputs: Any) -> "StepResult":
        return cls(success=True, outputs=outputs)

    @classmethod
    def fail(cls, message: str, *, retry: bool = False) -> "StepResult":
        return cls(success=False, error_message=message, retry_recommended=retry)


# ---------------------------------------------------------------------------
# Transition policy
# ---------------------------------------------------------------------------

@dataclass
class TransitionPolicy:
    """Rules for routing after a step completes or fails.

    Each step handler *declares* its transition policy; the engine
    enforces it so there is no silent fallback behaviour.
    """

    # Step to run on success (None = next step in linear order)
    on_success_step: Optional[str] = None
    # Step to run on failure (None = apply failure_behavior)
    on_failure_step: Optional[str] = None
    # Python expression evaluated against the StepResult to decide branch
    # e.g. "result.outputs.get('abstained') is True"
    condition: Optional[str] = None
    # Step to run when condition is True
    condition_true_step: Optional[str] = None
    # Step to run when condition is False (None = on_success_step)
    condition_false_step: Optional[str] = None


# ---------------------------------------------------------------------------
# Handler contract declaration
# ---------------------------------------------------------------------------

@dataclass
class HandlerContract:
    """Declares the full contract a step handler must satisfy.

    Step authors register this alongside the handler function so the
    engine can validate artifact availability before execution and
    validate outputs after execution — eliminating runtime surprises.
    """

    # Human-readable name used in logs / dashboards
    handler_name: str

    # Artifacts that MUST be present in context before this step runs
    required_inputs: List[ArtifactType] = field(default_factory=list)
    # Artifacts that MAY be present (step degrades gracefully without them)
    optional_inputs: List[ArtifactType] = field(default_factory=list)
    # Artifacts this step WILL write into the context on success
    declared_outputs: List[ArtifactType] = field(default_factory=list)

    failure_behavior: FailureBehavior = FailureBehavior.RETRY_THEN_ABORT
    retry_count: int = 3
    retry_delay_seconds: float = 5.0

    transition: TransitionPolicy = field(default_factory=TransitionPolicy)

    def validate_inputs(self, ctx: ExecutionContext) -> None:
        """Raise ValueError if any required input is absent."""
        missing = [a for a in self.required_inputs if not ctx.has(a)]
        if missing:
            raise ValueError(
                f"[{self.handler_name}] Missing required inputs: "
                + ", ".join(str(a) for a in missing)
            )


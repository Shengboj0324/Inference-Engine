"""Workflow orchestration system for automated action execution.

This package provides a complete workflow engine for executing multi-step
automated actions based on actionable signals.

Main components:
- WorkflowOrchestrator: High-level interface for workflow execution
- WorkflowEngine: Core execution engine with state management
- WorkflowRegistry: Catalog of predefined workflows
- WorkflowStepHandlers: Implementation of step logic

Example usage:
    from app.workflows import WorkflowOrchestrator
    from app.core.signal_models import ActionableSignal

    orchestrator = WorkflowOrchestrator()
    execution = await orchestrator.execute_for_signal(signal)
"""

from app.workflows.orchestrator import WorkflowOrchestrator
from app.workflows.workflow_engine import WorkflowEngine
from app.workflows.workflow_models import (
    StepExecution,
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStatus,
    WorkflowStep,
    WorkflowType,
)
from app.workflows.workflow_registry import WorkflowRegistry

__all__ = [
    "WorkflowOrchestrator",
    "WorkflowEngine",
    "WorkflowRegistry",
    "WorkflowDefinition",
    "WorkflowExecution",
    "WorkflowStep",
    "StepExecution",
    "WorkflowType",
    "StepType",
    "WorkflowStatus",
    "StepStatus",
]



"""Workflow orchestrator - high-level workflow management.

This module provides the main interface for workflow execution,
integrating the engine, registry, and step handlers.
"""

import logging
from typing import Optional

from app.core.signal_models import ActionableSignal
from app.intelligence.response_generator import ResponseGenerator
from app.workflows.step_handlers import WorkflowStepHandlers
from app.workflows.workflow_engine import WorkflowEngine
from app.workflows.workflow_models import (
    StepType,
    WorkflowExecution,
    WorkflowType,
)
from app.workflows.workflow_registry import WorkflowRegistry

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """High-level workflow orchestration service.

    This class provides the main interface for executing workflows
    based on signals. It coordinates the engine, registry, and handlers.

    Usage:
        orchestrator = WorkflowOrchestrator()
        execution = await orchestrator.execute_for_signal(signal)
    """

    def __init__(
        self,
        response_generator: Optional[ResponseGenerator] = None,
        max_concurrent_workflows: int = 10,
    ):
        """Initialize workflow orchestrator.

        Args:
            response_generator: Response generator for content generation
            max_concurrent_workflows: Maximum concurrent workflow executions
        """
        # Initialize components
        self.registry = WorkflowRegistry()
        self.engine = WorkflowEngine(max_concurrent_workflows=max_concurrent_workflows)

        # Initialize step handlers
        if not response_generator:
            response_generator = ResponseGenerator()

        self.handlers = WorkflowStepHandlers(response_generator=response_generator)

        # Register step handlers with engine
        self._register_handlers()

        logger.info("WorkflowOrchestrator initialized successfully")

    def _register_handlers(self) -> None:
        """Register step handlers with the workflow engine."""
        self.engine.register_step_handler(StepType.ANALYZE, self.handlers.handle_analyze)
        self.engine.register_step_handler(StepType.SCORE, self.handlers.handle_score)
        self.engine.register_step_handler(StepType.DECIDE, self.handlers.handle_decide)
        self.engine.register_step_handler(StepType.GENERATE, self.handlers.handle_generate)
        self.engine.register_step_handler(StepType.NOTIFY, self.handlers.handle_notify)
        self.engine.register_step_handler(StepType.EXECUTE, self.handlers.handle_execute)
        self.engine.register_step_handler(StepType.TRACK, self.handlers.handle_track)
        self.engine.register_step_handler(StepType.WAIT, self.handlers.handle_wait)

        logger.info("Registered 8 step handlers with workflow engine")

    async def execute_for_signal(
        self,
        signal: ActionableSignal,
    ) -> Optional[WorkflowExecution]:
        """Execute appropriate workflow for a signal.

        Args:
            signal: Actionable signal to process

        Returns:
            Workflow execution or None if no workflow available
        """
        # Get workflow for signal type
        workflow_def = self.registry.get_workflow_for_signal(signal)

        if not workflow_def:
            logger.info(
                f"No workflow available for signal type {signal.signal_type.value}"
            )
            return None

        logger.info(
            f"Executing workflow {workflow_def.type.value} for signal {signal.id}"
        )

        # Execute workflow
        execution = await self.engine.execute_workflow(
            workflow_def=workflow_def,
            signal=signal,
        )

        return execution

    async def execute_workflow_by_type(
        self,
        workflow_type: WorkflowType,
        signal: ActionableSignal,
    ) -> Optional[WorkflowExecution]:
        """Execute a specific workflow type for a signal.

        Args:
            workflow_type: Type of workflow to execute
            signal: Actionable signal to process

        Returns:
            Workflow execution or None if workflow not found
        """
        # Get workflow definition
        workflow_def = self.registry.get_workflow(workflow_type)

        if not workflow_def:
            logger.error(f"Workflow type {workflow_type.value} not found")
            return None

        logger.info(
            f"Executing workflow {workflow_type.value} for signal {signal.id}"
        )

        # Execute workflow
        execution = await self.engine.execute_workflow(
            workflow_def=workflow_def,
            signal=signal,
        )

        return execution

    async def get_execution_status(
        self,
        execution_id: str,
    ) -> Optional[WorkflowExecution]:
        """Get status of a workflow execution.

        Args:
            execution_id: Workflow execution ID

        Returns:
            Workflow execution or None if not found
        """
        from uuid import UUID

        try:
            exec_uuid = UUID(execution_id)
            return await self.engine.get_execution(exec_uuid)
        except ValueError:
            logger.error(f"Invalid execution ID: {execution_id}")
            return None

    async def cancel_execution(
        self,
        execution_id: str,
    ) -> bool:
        """Cancel a running workflow execution.

        Args:
            execution_id: Workflow execution ID

        Returns:
            True if cancelled, False otherwise
        """
        from uuid import UUID

        try:
            exec_uuid = UUID(execution_id)
            return await self.engine.cancel_workflow(exec_uuid)
        except ValueError:
            logger.error(f"Invalid execution ID: {execution_id}")
            return False

    def list_available_workflows(self) -> dict:
        """List all available workflow types.

        Returns:
            Dictionary of workflow type to definition
        """
        workflows = self.registry.list_workflows()
        return {
            wf_type.value: {
                "name": wf_def.name,
                "description": wf_def.description,
                "steps": len(wf_def.steps),
                "version": wf_def.version,
            }
            for wf_type, wf_def in workflows.items()
        }



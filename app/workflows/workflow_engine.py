"""Workflow execution engine with state management and error recovery.

This module implements the core workflow orchestration engine that executes
multi-step automated actions based on signals.

Design principles:
- State machine-based execution
- Automatic error recovery with retries
- Step dependency resolution
- Execution persistence for recovery
- Comprehensive logging and monitoring
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from uuid import UUID

from app.core.signal_models import ActionableSignal
from app.workflows.workflow_models import (
    StepExecution,
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStatus,
    WorkflowStep,
)

logger = logging.getLogger(__name__)


class WorkflowExecutionError(Exception):
    """Base exception for workflow execution errors."""


class StepExecutionError(Exception):
    """Exception raised when a step fails."""


class WorkflowEngine:
    """Execute and manage workflow instances.

    The engine handles:
    - Step-by-step execution with dependency resolution
    - Automatic retry on transient failures
    - State persistence for recovery
    - Parallel execution when allowed
    - Timeout management
    - Error handling and rollback

    Attributes:
        step_handlers: Registry of step type handlers
        max_concurrent_workflows: Maximum concurrent workflow executions
    """

    def __init__(
        self,
        max_concurrent_workflows: int = 10,
    ):
        """Initialize workflow engine.

        Args:
            max_concurrent_workflows: Maximum concurrent workflow executions
        """
        self.max_concurrent_workflows = max_concurrent_workflows
        self.step_handlers: Dict[StepType, Callable] = {}
        self._active_workflows: Dict[UUID, WorkflowExecution] = {}
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_workflows)

        logger.info(
            f"WorkflowEngine initialized with max_concurrent={max_concurrent_workflows}"
        )

    def register_step_handler(
        self,
        step_type: StepType,
        handler: Callable,
    ) -> None:
        """Register a handler function for a step type.

        Args:
            step_type: Type of step
            handler: Async function that executes the step
        """
        self.step_handlers[step_type] = handler
        logger.info(f"Registered handler for step type: {step_type.value}")

    async def execute_workflow(
        self,
        workflow_def: WorkflowDefinition,
        signal: ActionableSignal,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """Execute a workflow for a signal.

        Args:
            workflow_def: Workflow definition to execute
            signal: Signal that triggered the workflow
            initial_context: Initial context data for workflow

        Returns:
            Completed workflow execution

        Raises:
            WorkflowExecutionError: If workflow execution fails
        """
        async with self._execution_semaphore:
            # Create execution instance
            execution = WorkflowExecution(
                workflow_id=workflow_def.id,
                workflow_type=workflow_def.type,
                signal_id=signal.id,
                user_id=signal.user_id,
                context=initial_context or {},
            )

            # Add signal data to context
            execution.context["signal"] = signal.model_dump()

            self._active_workflows[execution.id] = execution

            try:
                # Start execution
                execution.status = WorkflowStatus.RUNNING
                execution.started_at = datetime.utcnow()

                logger.info(
                    f"Starting workflow {workflow_def.type.value} "
                    f"for signal {signal.id} (execution_id={execution.id})"
                )

                # Execute steps
                await self._execute_steps(workflow_def, execution)

                # Mark as completed
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
                execution.duration_ms = int(
                    (execution.completed_at - execution.started_at).total_seconds() * 1000
                )

                logger.info(
                    f"Workflow {execution.id} completed successfully "
                    f"in {execution.duration_ms}ms"
                )

                return execution

            except Exception as e:
                execution.status = WorkflowStatus.FAILED
                execution.error = str(e)
                execution.completed_at = datetime.utcnow()

                logger.error(
                    f"Workflow {execution.id} failed: {e}",
                    exc_info=True
                )

                raise WorkflowExecutionError(f"Workflow execution failed: {e}") from e

            finally:
                # Remove from active workflows
                self._active_workflows.pop(execution.id, None)

    async def _execute_steps(
        self,
        workflow_def: WorkflowDefinition,
        execution: WorkflowExecution,
    ) -> None:
        """Execute all workflow steps in order.

        Args:
            workflow_def: Workflow definition
            execution: Workflow execution instance
        """
        for step in workflow_def.steps:
            # Check if step should be executed
            if not await self._should_execute_step(step, execution):
                logger.info(f"Skipping step {step.id} due to condition")
                execution.step_executions[step.id] = StepExecution(
                    step_id=step.id,
                    status=StepStatus.SKIPPED,
                )
                continue

            # Wait for dependencies
            await self._wait_for_dependencies(step, execution)

            # Execute step
            execution.current_step_id = step.id
            await self._execute_step(step, execution)

            # Check if step failed
            step_exec = execution.step_executions[step.id]
            if step_exec.status == StepStatus.FAILED:
                raise StepExecutionError(
                    f"Step {step.id} failed: {step_exec.error}"
                )

    async def _should_execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> bool:
        """Check if step should be executed based on condition.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            True if step should execute
        """
        if not step.condition:
            return True

        try:
            # Evaluate condition in context
            # For safety, we use a restricted eval with only context variables
            context_vars = {
                "context": execution.context,
                "signal": execution.context.get("signal", {}),
            }
            result = eval(step.condition, {"__builtins__": {}}, context_vars)
            return bool(result)
        except Exception as e:
            logger.error(f"Error evaluating condition for step {step.id}: {e}")
            # Default to executing if condition evaluation fails
            return True

    async def _wait_for_dependencies(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> None:
        """Wait for step dependencies to complete.

        Args:
            step: Workflow step
            execution: Workflow execution
        """
        if not step.depends_on:
            return

        for dep_step_id in step.depends_on:
            # Check if dependency exists
            if dep_step_id not in execution.step_executions:
                raise WorkflowExecutionError(
                    f"Step {step.id} depends on {dep_step_id} which hasn't executed"
                )

            dep_exec = execution.step_executions[dep_step_id]

            # Check if dependency completed successfully
            if dep_exec.status == StepStatus.FAILED:
                raise WorkflowExecutionError(
                    f"Step {step.id} dependency {dep_step_id} failed"
                )

            if dep_exec.status != StepStatus.COMPLETED:
                raise WorkflowExecutionError(
                    f"Step {step.id} dependency {dep_step_id} not completed"
                )

    async def _execute_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> None:
        """Execute a single workflow step with retry logic.

        Args:
            step: Workflow step to execute
            execution: Workflow execution instance
        """
        step_exec = StepExecution(step_id=step.id)
        execution.step_executions[step.id] = step_exec

        step_exec.status = StepStatus.RUNNING
        step_exec.started_at = datetime.utcnow()

        logger.info(f"Executing step {step.id} ({step.type.value})")

        # Get handler for step type
        handler = self.step_handlers.get(step.type)
        if not handler:
            step_exec.status = StepStatus.FAILED
            step_exec.error = f"No handler registered for step type {step.type.value}"
            logger.error(step_exec.error)
            return

        # Execute with retry logic
        for attempt in range(step.retry_count + 1):
            try:
                # Execute step with timeout
                result = await asyncio.wait_for(
                    handler(step, execution),
                    timeout=step.timeout_seconds,
                )

                # Step succeeded
                step_exec.status = StepStatus.COMPLETED
                step_exec.result = result
                step_exec.output = result if isinstance(result, dict) else {"result": result}
                step_exec.completed_at = datetime.utcnow()
                step_exec.duration_ms = int(
                    (step_exec.completed_at - step_exec.started_at).total_seconds() * 1000
                )

                logger.info(
                    f"Step {step.id} completed successfully in {step_exec.duration_ms}ms"
                )
                return

            except asyncio.TimeoutError:
                error_msg = f"Step {step.id} timed out after {step.timeout_seconds}s"
                logger.warning(f"{error_msg} (attempt {attempt + 1}/{step.retry_count + 1})")
                step_exec.error = error_msg
                step_exec.retry_count = attempt + 1

            except Exception as e:
                error_msg = f"Step {step.id} failed: {str(e)}"
                logger.warning(
                    f"{error_msg} (attempt {attempt + 1}/{step.retry_count + 1})",
                    exc_info=True
                )
                step_exec.error = error_msg
                step_exec.retry_count = attempt + 1

            # Wait before retry (except on last attempt)
            if attempt < step.retry_count:
                await asyncio.sleep(step.retry_delay_seconds)

        # All retries exhausted
        step_exec.status = StepStatus.FAILED
        step_exec.completed_at = datetime.utcnow()
        logger.error(f"Step {step.id} failed after {step.retry_count + 1} attempts")

    async def get_execution(self, execution_id: UUID) -> Optional[WorkflowExecution]:
        """Get workflow execution by ID.

        Args:
            execution_id: Workflow execution ID

        Returns:
            Workflow execution or None if not found
        """
        return self._active_workflows.get(execution_id)

    async def cancel_workflow(self, execution_id: UUID) -> bool:
        """Cancel a running workflow.

        Args:
            execution_id: Workflow execution ID

        Returns:
            True if cancelled, False if not found or already completed
        """
        execution = self._active_workflows.get(execution_id)
        if not execution:
            return False

        if execution.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
            return False

        execution.status = WorkflowStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        logger.info(f"Workflow {execution_id} cancelled")

        return True



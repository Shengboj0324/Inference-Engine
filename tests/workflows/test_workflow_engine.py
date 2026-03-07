"""Tests for workflow engine core functionality."""

import pytest
from datetime import datetime
from uuid import uuid4

from app.core.signal_models import (
    ActionableSignal,
    SignalType,
    ActionType,
    ResponseTone,
    SignalStatus,
)
from app.workflows.workflow_engine import WorkflowEngine
from app.workflows.workflow_models import (
    StepType,
    StepStatus,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowType,
    WorkflowStatus,
)


@pytest.fixture
def sample_signal():
    """Create a sample actionable signal for testing."""
    return ActionableSignal(
        user_id=uuid4(),
        signal_type=SignalType.LEAD_OPPORTUNITY,
        source_platform="reddit",
        source_url="https://reddit.com/r/test/comments/123",
        source_author="test_user",
        title="Looking for alternatives to CompetitorX",
        description="I'm frustrated with CompetitorX's pricing and looking for better options",
        context="User is actively seeking alternatives, high intent",
        urgency_score=0.8,
        impact_score=0.7,
        confidence_score=0.9,
        action_score=0.8,
        recommended_action=ActionType.REPLY_PUBLIC,
        suggested_channel="reddit",
        suggested_tone=ResponseTone.HELPFUL,
    )


@pytest.fixture
def simple_workflow():
    """Create a simple workflow for testing."""
    return WorkflowDefinition(
        type=WorkflowType.ALTERNATIVE_SEEKER,
        name="Test Workflow",
        description="Simple test workflow",
        steps=[
            WorkflowStep(
                id="step1",
                type=StepType.ANALYZE,
                name="Analyze",
                description="Analyze signal",
                config={"extract_fields": ["pain_points"]},
            ),
            WorkflowStep(
                id="step2",
                type=StepType.SCORE,
                name="Score",
                description="Score signal",
                depends_on=["step1"],
                config={"scoring_criteria": ["fit_score"]},
            ),
        ],
    )


@pytest.fixture
def workflow_engine():
    """Create workflow engine instance."""
    return WorkflowEngine(max_concurrent_workflows=5)


class TestWorkflowEngine:
    """Test workflow engine functionality."""

    def test_engine_initialization(self, workflow_engine):
        """Test engine initializes correctly."""
        assert workflow_engine.max_concurrent_workflows == 5
        assert len(workflow_engine.step_handlers) == 0
        assert len(workflow_engine._active_workflows) == 0

    def test_register_step_handler(self, workflow_engine):
        """Test registering step handlers."""
        async def dummy_handler(step, execution):
            return {"result": "success"}

        workflow_engine.register_step_handler(StepType.ANALYZE, dummy_handler)
        
        assert StepType.ANALYZE in workflow_engine.step_handlers
        assert workflow_engine.step_handlers[StepType.ANALYZE] == dummy_handler

    @pytest.mark.asyncio
    async def test_execute_workflow_basic(
        self,
        workflow_engine,
        simple_workflow,
        sample_signal,
    ):
        """Test basic workflow execution."""
        # Register dummy handlers
        async def analyze_handler(step, execution):
            return {"pain_points": ["expensive", "slow"]}

        async def score_handler(step, execution):
            return {"fit_score": 0.8, "overall_score": 0.8, "quality_level": "high_quality"}

        workflow_engine.register_step_handler(StepType.ANALYZE, analyze_handler)
        workflow_engine.register_step_handler(StepType.SCORE, score_handler)

        # Execute workflow
        execution = await workflow_engine.execute_workflow(
            workflow_def=simple_workflow,
            signal=sample_signal,
        )

        # Verify execution
        assert execution.status == WorkflowStatus.COMPLETED
        assert execution.signal_id == sample_signal.id
        assert execution.user_id == sample_signal.user_id
        assert len(execution.step_executions) == 2
        
        # Verify step 1
        step1_exec = execution.step_executions["step1"]
        assert step1_exec.status == StepStatus.COMPLETED
        assert step1_exec.result == {"pain_points": ["expensive", "slow"]}
        
        # Verify step 2
        step2_exec = execution.step_executions["step2"]
        assert step2_exec.status == StepStatus.COMPLETED
        assert step2_exec.result["fit_score"] == 0.8

    @pytest.mark.asyncio
    async def test_step_dependency_resolution(
        self,
        workflow_engine,
        sample_signal,
    ):
        """Test that step dependencies are resolved correctly."""
        workflow = WorkflowDefinition(
            type=WorkflowType.ALTERNATIVE_SEEKER,
            name="Dependency Test",
            description="Test dependency resolution",
            steps=[
                WorkflowStep(
                    id="step1",
                    type=StepType.ANALYZE,
                    name="Step 1",
                    description="First step",
                ),
                WorkflowStep(
                    id="step2",
                    type=StepType.SCORE,
                    name="Step 2",
                    description="Depends on step1",
                    depends_on=["step1"],
                ),
                WorkflowStep(
                    id="step3",
                    type=StepType.DECIDE,
                    name="Step 3",
                    description="Depends on step2",
                    depends_on=["step2"],
                ),
            ],
        )

        # Register handlers
        execution_order = []

        async def handler_factory(step_id):
            async def handler(step, execution):
                execution_order.append(step_id)
                return {"step": step_id}
            return handler

        workflow_engine.register_step_handler(StepType.ANALYZE, await handler_factory("step1"))
        workflow_engine.register_step_handler(StepType.SCORE, await handler_factory("step2"))
        workflow_engine.register_step_handler(StepType.DECIDE, await handler_factory("step3"))

        # Execute
        execution = await workflow_engine.execute_workflow(workflow, sample_signal)

        # Verify execution order
        assert execution_order == ["step1", "step2", "step3"]
        assert execution.status == WorkflowStatus.COMPLETED


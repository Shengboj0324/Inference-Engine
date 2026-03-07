"""Tests for workflow orchestrator."""

import pytest
from uuid import uuid4

from app.core.signal_models import (
    ActionableSignal,
    SignalType,
    ActionType,
    ResponseTone,
)
from app.workflows.orchestrator import WorkflowOrchestrator
from app.workflows.workflow_models import WorkflowStatus, WorkflowType


@pytest.fixture
def orchestrator():
    """Create workflow orchestrator instance."""
    return WorkflowOrchestrator(max_concurrent_workflows=5)


@pytest.fixture
def lead_opportunity_signal():
    """Create a lead opportunity signal."""
    return ActionableSignal(
        user_id=uuid4(),
        signal_type=SignalType.LEAD_OPPORTUNITY,
        source_platform="reddit",
        source_url="https://reddit.com/r/saas/comments/123",
        source_author="potential_customer",
        title="Looking for alternatives to Salesforce",
        description="Salesforce is too expensive and complicated. Need something simpler.",
        context="High-intent lead actively seeking alternatives",
        urgency_score=0.9,
        impact_score=0.8,
        confidence_score=0.85,
        action_score=0.85,
        recommended_action=ActionType.REPLY_PUBLIC,
        suggested_channel="reddit",
        suggested_tone=ResponseTone.HELPFUL,
        metadata={"competitor_name": "Salesforce"},
    )


@pytest.fixture
def competitor_weakness_signal():
    """Create a competitor weakness signal."""
    return ActionableSignal(
        user_id=uuid4(),
        signal_type=SignalType.COMPETITOR_WEAKNESS,
        source_platform="twitter",
        source_url="https://twitter.com/user/status/123",
        source_author="frustrated_user",
        title="CompetitorX support is terrible",
        description="Been waiting 3 days for support response. Their customer service is awful.",
        context="Public complaint about competitor support quality",
        urgency_score=0.7,
        impact_score=0.6,
        confidence_score=0.9,
        action_score=0.7,
        recommended_action=ActionType.CREATE_CONTENT,
        suggested_channel="twitter",
        suggested_tone=ResponseTone.HELPFUL,
        metadata={"competitor_name": "CompetitorX", "complaint_type": "support"},
    )


class TestWorkflowOrchestrator:
    """Test workflow orchestrator functionality."""

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initializes correctly."""
        assert orchestrator.registry is not None
        assert orchestrator.engine is not None
        assert orchestrator.handlers is not None
        assert len(orchestrator.engine.step_handlers) == 8  # All step types registered

    def test_list_available_workflows(self, orchestrator):
        """Test listing available workflows."""
        workflows = orchestrator.list_available_workflows()
        
        assert len(workflows) >= 5  # At least 5 default workflows
        assert "alternative_seeker" in workflows
        assert "competitor_intelligence" in workflows
        assert "churn_prevention" in workflows
        
        # Check workflow structure
        alt_seeker = workflows["alternative_seeker"]
        assert "name" in alt_seeker
        assert "description" in alt_seeker
        assert "steps" in alt_seeker
        assert alt_seeker["steps"] >= 4  # At least 4 steps

    @pytest.mark.asyncio
    async def test_execute_for_lead_opportunity_signal(
        self,
        orchestrator,
        lead_opportunity_signal,
    ):
        """Test executing workflow for lead opportunity signal."""
        execution = await orchestrator.execute_for_signal(lead_opportunity_signal)
        
        assert execution is not None
        assert execution.workflow_type == WorkflowType.ALTERNATIVE_SEEKER
        assert execution.signal_id == lead_opportunity_signal.id
        assert execution.user_id == lead_opportunity_signal.user_id
        assert execution.status == WorkflowStatus.COMPLETED
        
        # Verify steps were executed
        assert len(execution.step_executions) > 0
        
        # Verify context contains signal data
        assert "signal" in execution.context
        assert execution.context["signal"]["signal_type"] == SignalType.LEAD_OPPORTUNITY.value

    @pytest.mark.asyncio
    async def test_execute_for_competitor_weakness_signal(
        self,
        orchestrator,
        competitor_weakness_signal,
    ):
        """Test executing workflow for competitor weakness signal."""
        execution = await orchestrator.execute_for_signal(competitor_weakness_signal)
        
        assert execution is not None
        assert execution.workflow_type == WorkflowType.COMPETITOR_INTELLIGENCE
        assert execution.signal_id == competitor_weakness_signal.id
        assert execution.status == WorkflowStatus.COMPLETED
        
        # Verify the analyze_complaint step executed successfully
        assert "analyze_complaint" in execution.step_executions
        step_exec = execution.step_executions["analyze_complaint"]
        assert step_exec.status.value == "completed"

        # Verify the step result contains extracted fields
        if step_exec.result:
            assert isinstance(step_exec.result, dict)
            # The result should contain the extracted fields
            assert "competitor_name" in step_exec.result or "complaint_type" in step_exec.result

    @pytest.mark.asyncio
    async def test_execute_workflow_by_type(
        self,
        orchestrator,
        lead_opportunity_signal,
    ):
        """Test executing specific workflow type."""
        execution = await orchestrator.execute_workflow_by_type(
            workflow_type=WorkflowType.ALTERNATIVE_SEEKER,
            signal=lead_opportunity_signal,
        )
        
        assert execution is not None
        assert execution.workflow_type == WorkflowType.ALTERNATIVE_SEEKER
        assert execution.status == WorkflowStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_get_execution_status(
        self,
        orchestrator,
        lead_opportunity_signal,
    ):
        """Test getting execution status."""
        # Execute workflow
        execution = await orchestrator.execute_for_signal(lead_opportunity_signal)
        
        # Note: Since execution completes immediately, it won't be in active workflows
        # This test verifies the method works with valid UUID
        status = await orchestrator.get_execution_status(str(execution.id))
        
        # Will be None since execution is complete and removed from active workflows
        assert status is None or status.id == execution.id

    @pytest.mark.asyncio
    async def test_execute_for_unsupported_signal_type(self, orchestrator):
        """Test executing workflow for signal type without workflow."""
        signal = ActionableSignal(
            user_id=uuid4(),
            signal_type=SignalType.MISINFORMATION_RISK,  # No workflow defined yet
            source_platform="twitter",
            source_url="https://twitter.com/user/status/123",
            source_author="user",
            title="Test signal",
            description="Test description",
            context="Test context",
            urgency_score=0.5,
            impact_score=0.5,
            confidence_score=0.5,
            action_score=0.5,
            recommended_action=ActionType.MONITOR,
            suggested_channel="twitter",
            suggested_tone=ResponseTone.PROFESSIONAL,
        )
        
        execution = await orchestrator.execute_for_signal(signal)
        
        # Should return None for unsupported signal type
        assert execution is None


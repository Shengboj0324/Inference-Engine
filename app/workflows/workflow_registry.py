"""Workflow registry and builder for predefined workflows.

This module provides a registry of workflow definitions and builders
for creating workflow instances from signals.
"""

import logging
from typing import Dict, Optional
from uuid import uuid4

from app.core.signal_models import ActionableSignal, SignalType
from app.workflows.workflow_models import (
    StepType,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowType,
)

logger = logging.getLogger(__name__)


class WorkflowRegistry:
    """Registry of workflow definitions.

    This class maintains a catalog of predefined workflows and provides
    methods to retrieve and instantiate them based on signal types.
    """

    def __init__(self):
        """Initialize workflow registry."""
        self._workflows: Dict[WorkflowType, WorkflowDefinition] = {}
        self._signal_to_workflow: Dict[SignalType, WorkflowType] = {}

        # Build default workflows
        self._build_default_workflows()

        logger.info(f"WorkflowRegistry initialized with {len(self._workflows)} workflows")

    def _build_default_workflows(self) -> None:
        """Build and register default workflow definitions."""
        # Alternative Seeker Workflow (Lead Opportunity)
        self.register_workflow(self._build_alternative_seeker_workflow())
        self._signal_to_workflow[SignalType.LEAD_OPPORTUNITY] = WorkflowType.ALTERNATIVE_SEEKER

        # Competitor Intelligence Workflow
        self.register_workflow(self._build_competitor_intelligence_workflow())
        self._signal_to_workflow[SignalType.COMPETITOR_WEAKNESS] = WorkflowType.COMPETITOR_INTELLIGENCE

        # Churn Prevention Workflow
        self.register_workflow(self._build_churn_prevention_workflow())
        self._signal_to_workflow[SignalType.CHURN_RISK] = WorkflowType.CHURN_PREVENTION

        # Support Escalation Workflow
        self.register_workflow(self._build_support_escalation_workflow())
        self._signal_to_workflow[SignalType.SUPPORT_ESCALATION] = WorkflowType.SUPPORT_ESCALATION

        # Feature Request Workflow
        self.register_workflow(self._build_feature_request_workflow())
        self._signal_to_workflow[SignalType.FEATURE_REQUEST_PATTERN] = WorkflowType.FEATURE_REQUEST

    def _build_alternative_seeker_workflow(self) -> WorkflowDefinition:
        """Build Alternative Seeker workflow for lead opportunities."""
        return WorkflowDefinition(
            id=uuid4(),
            type=WorkflowType.ALTERNATIVE_SEEKER,
            name="Alternative Seeker - Lead Qualification",
            description="Qualify leads looking for alternatives and generate personalized outreach",
            version="1.0.0",
            steps=[
                WorkflowStep(
                    id="analyze_intent",
                    type=StepType.ANALYZE,
                    name="Analyze Lead Intent",
                    description="Analyze the lead's specific pain points and requirements",
                    config={
                        "extract_fields": ["pain_points", "current_tool", "requirements", "budget_signals"]
                    },
                ),
                WorkflowStep(
                    id="qualify_lead",
                    type=StepType.SCORE,
                    name="Qualify Lead",
                    description="Score lead quality based on fit and intent",
                    depends_on=["analyze_intent"],
                    config={
                        "scoring_criteria": ["fit_score", "intent_score", "urgency_score"],
                        "min_threshold": 0.6,
                    },
                ),
                WorkflowStep(
                    id="decide_approach",
                    type=StepType.DECIDE,
                    name="Decide Outreach Approach",
                    description="Determine best outreach strategy based on lead quality",
                    depends_on=["qualify_lead"],
                    config={
                        "decision_tree": {
                            "high_quality": "personalized_outreach",
                            "medium_quality": "helpful_response",
                            "low_quality": "monitor_only",
                        }
                    },
                ),
                WorkflowStep(
                    id="generate_response",
                    type=StepType.GENERATE,
                    name="Generate Personalized Response",
                    description="Generate tailored response highlighting relevant features",
                    depends_on=["decide_approach"],
                    condition="context.get('lead_quality') in ['high_quality', 'medium_quality']",
                    config={
                        "num_variants": 3,
                        "include_case_study": True,
                        "include_demo_offer": True,
                    },
                ),
                WorkflowStep(
                    id="schedule_followup",
                    type=StepType.EXECUTE,
                    name="Schedule Follow-up",
                    description="Schedule follow-up reminder for high-quality leads",
                    depends_on=["generate_response"],
                    condition="context.get('lead_quality') == 'high_quality'",
                    config={
                        "followup_delay_hours": 48,
                        "max_followups": 2,
                    },
                ),
                WorkflowStep(
                    id="track_outcome",
                    type=StepType.TRACK,
                    name="Track Outcome",
                    description="Track engagement and conversion metrics",
                    depends_on=["generate_response"],
                    config={
                        "metrics": ["response_posted", "engagement_rate", "conversion"],
                    },
                ),
            ],
            max_execution_time_seconds=600,
        )

    def _build_competitor_intelligence_workflow(self) -> WorkflowDefinition:
        """Build Competitor Intelligence workflow."""
        return WorkflowDefinition(
            id=uuid4(),
            type=WorkflowType.COMPETITOR_INTELLIGENCE,
            name="Competitor Intelligence - Weakness Analysis",
            description="Analyze competitor weaknesses and draft positioning content",
            version="1.0.0",
            steps=[
                WorkflowStep(
                    id="analyze_complaint",
                    type=StepType.ANALYZE,
                    name="Analyze Competitor Complaint",
                    description="Extract specific pain points and competitor weaknesses",
                    config={
                        "extract_fields": ["competitor_name", "complaint_type", "pain_points", "severity"]
                    },
                ),
                WorkflowStep(
                    id="identify_positioning",
                    type=StepType.ANALYZE,
                    name="Identify Positioning Angle",
                    description="Determine how our product addresses these pain points",
                    depends_on=["analyze_complaint"],
                    config={
                        "match_features": True,
                        "find_differentiators": True,
                    },
                ),
                WorkflowStep(
                    id="generate_content",
                    type=StepType.GENERATE,
                    name="Generate Positioning Content",
                    description="Create helpful content that subtly positions our solution",
                    depends_on=["identify_positioning"],
                    config={
                        "content_types": ["public_response", "blog_post_idea", "comparison_chart"],
                        "tone": "helpful_not_salesy",
                    },
                ),
            ],
            max_execution_time_seconds=300,
        )

    def _build_churn_prevention_workflow(self) -> WorkflowDefinition:
        """Build Churn Prevention workflow."""
        return WorkflowDefinition(
            id=uuid4(),
            type=WorkflowType.CHURN_PREVENTION,
            name="Churn Prevention - Retention Outreach",
            description="Assess churn risk and execute retention strategy",
            version="1.0.0",
            steps=[
                WorkflowStep(
                    id="assess_severity",
                    type=StepType.SCORE,
                    name="Assess Churn Risk Severity",
                    description="Score the severity and urgency of churn risk",
                    config={
                        "risk_factors": ["sentiment", "account_value", "tenure", "complaint_history"],
                        "severity_levels": ["low", "medium", "high", "critical"],
                    },
                ),
                WorkflowStep(
                    id="draft_retention_outreach",
                    type=StepType.GENERATE,
                    name="Draft Retention Outreach",
                    description="Generate empathetic retention message",
                    depends_on=["assess_severity"],
                    config={
                        "tone": "supportive",
                        "include_solutions": True,
                        "escalation_offer": True,
                    },
                ),
                WorkflowStep(
                    id="escalate_to_cs",
                    type=StepType.NOTIFY,
                    name="Escalate to Customer Success",
                    description="Alert CS team for high-risk accounts",
                    depends_on=["assess_severity"],
                    condition="context.get('severity') in ['high', 'critical']",
                    config={
                        "notification_channels": ["slack", "email"],
                        "priority": "high",
                    },
                ),
                WorkflowStep(
                    id="track_resolution",
                    type=StepType.TRACK,
                    name="Track Resolution",
                    description="Monitor resolution and retention outcome",
                    depends_on=["draft_retention_outreach"],
                    config={
                        "metrics": ["response_time", "resolution_status", "retention_outcome"],
                        "followup_days": 7,
                    },
                ),
            ],
            max_execution_time_seconds=600,
        )

    def _build_support_escalation_workflow(self) -> WorkflowDefinition:
        """Build Support Escalation workflow."""
        return WorkflowDefinition(
            id=uuid4(),
            type=WorkflowType.SUPPORT_ESCALATION,
            name="Support Escalation - Public Issue Response",
            description="Handle support issues that went public",
            version="1.0.0",
            steps=[
                WorkflowStep(
                    id="analyze_issue",
                    type=StepType.ANALYZE,
                    name="Analyze Support Issue",
                    description="Extract issue details and severity",
                    config={
                        "extract_fields": ["issue_type", "severity", "customer_sentiment", "public_visibility"],
                    },
                ),
                WorkflowStep(
                    id="generate_public_response",
                    type=StepType.GENERATE,
                    name="Generate Public Response",
                    description="Draft empathetic public acknowledgment",
                    depends_on=["analyze_issue"],
                    config={
                        "tone": "supportive",
                        "acknowledge_issue": True,
                        "offer_private_resolution": True,
                    },
                ),
                WorkflowStep(
                    id="notify_support_team",
                    type=StepType.NOTIFY,
                    name="Notify Support Team",
                    description="Alert support team for immediate action",
                    depends_on=["analyze_issue"],
                    config={
                        "notification_channels": ["slack", "support_system"],
                        "priority": "urgent",
                    },
                ),
            ],
            max_execution_time_seconds=300,
        )

    def _build_feature_request_workflow(self) -> WorkflowDefinition:
        """Build Feature Request workflow."""
        return WorkflowDefinition(
            id=uuid4(),
            type=WorkflowType.FEATURE_REQUEST,
            name="Feature Request - Pattern Detection",
            description="Track recurring feature requests and notify product team",
            version="1.0.0",
            steps=[
                WorkflowStep(
                    id="extract_feature",
                    type=StepType.ANALYZE,
                    name="Extract Feature Request",
                    description="Identify specific feature being requested",
                    config={
                        "extract_fields": ["feature_name", "use_case", "urgency", "workarounds_mentioned"],
                    },
                ),
                WorkflowStep(
                    id="check_pattern",
                    type=StepType.ANALYZE,
                    name="Check Request Pattern",
                    description="Check if this is a recurring request",
                    depends_on=["extract_feature"],
                    config={
                        "lookback_days": 30,
                        "pattern_threshold": 3,
                    },
                ),
                WorkflowStep(
                    id="notify_product_team",
                    type=StepType.NOTIFY,
                    name="Notify Product Team",
                    description="Alert product team if pattern detected",
                    depends_on=["check_pattern"],
                    condition="context.get('is_pattern', False)",
                    config={
                        "notification_channels": ["slack", "product_board"],
                        "include_examples": True,
                    },
                ),
                WorkflowStep(
                    id="generate_acknowledgment",
                    type=StepType.GENERATE,
                    name="Generate Acknowledgment",
                    description="Draft response acknowledging the request",
                    depends_on=["extract_feature"],
                    config={
                        "tone": "helpful",
                        "mention_roadmap": True,
                        "offer_workaround": True,
                    },
                ),
            ],
            max_execution_time_seconds=300,
        )

    def register_workflow(self, workflow: WorkflowDefinition) -> None:
        """Register a workflow definition.

        Args:
            workflow: Workflow definition to register
        """
        self._workflows[workflow.type] = workflow
        logger.info(f"Registered workflow: {workflow.type.value}")

    def get_workflow(self, workflow_type: WorkflowType) -> Optional[WorkflowDefinition]:
        """Get workflow definition by type.

        Args:
            workflow_type: Type of workflow

        Returns:
            Workflow definition or None if not found
        """
        return self._workflows.get(workflow_type)

    def get_workflow_for_signal(self, signal: ActionableSignal) -> Optional[WorkflowDefinition]:
        """Get appropriate workflow for a signal.

        Args:
            signal: Actionable signal

        Returns:
            Workflow definition or None if no workflow mapped
        """
        workflow_type = self._signal_to_workflow.get(signal.signal_type)
        if not workflow_type:
            return None

        return self.get_workflow(workflow_type)

    def list_workflows(self) -> Dict[WorkflowType, WorkflowDefinition]:
        """List all registered workflows.

        Returns:
            Dictionary of workflow type to definition
        """
        return self._workflows.copy()



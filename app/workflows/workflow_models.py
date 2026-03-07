"""Workflow models for automated action orchestration.

This module defines the core data structures for workflow execution:
- Workflow definitions and steps
- Execution state and history
- Step results and error handling
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class WorkflowType(str, Enum):
    """Types of automated workflows."""

    ALTERNATIVE_SEEKER = "alternative_seeker"  # Lead opportunity workflow
    COMPETITOR_INTELLIGENCE = "competitor_intelligence"  # Competitor weakness workflow
    CHURN_PREVENTION = "churn_prevention"  # Churn risk workflow
    SUPPORT_ESCALATION = "support_escalation"  # Support issue workflow
    FEATURE_REQUEST = "feature_request"  # Feature request workflow
    INFLUENCER_OUTREACH = "influencer_outreach"  # Influencer amplification workflow
    MISINFORMATION_RESPONSE = "misinformation_response"  # Misinformation risk workflow


class StepType(str, Enum):
    """Types of workflow steps."""

    ANALYZE = "analyze"  # Analyze signal/content
    GENERATE = "generate"  # Generate content (response, post, etc.)
    SCORE = "score"  # Score/evaluate something
    DECIDE = "decide"  # Make a decision (branching logic)
    NOTIFY = "notify"  # Send notification/alert
    WAIT = "wait"  # Wait for external input or time
    EXECUTE = "execute"  # Execute action (post, send DM, etc.)
    TRACK = "track"  # Track outcome/metrics


class StepStatus(str, Enum):
    """Status of a workflow step."""

    PENDING = "pending"  # Not started yet
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully completed
    FAILED = "failed"  # Failed with error
    SKIPPED = "skipped"  # Skipped due to conditional logic
    WAITING = "waiting"  # Waiting for external input


class WorkflowStatus(str, Enum):
    """Status of entire workflow execution."""

    CREATED = "created"  # Workflow created but not started
    RUNNING = "running"  # Workflow is executing
    PAUSED = "paused"  # Workflow paused (waiting for input)
    COMPLETED = "completed"  # Workflow completed successfully
    FAILED = "failed"  # Workflow failed
    CANCELLED = "cancelled"  # Workflow cancelled by user


class WorkflowStep(BaseModel):
    """Definition of a single workflow step."""

    id: str = Field(..., description="Step identifier (unique within workflow)")
    type: StepType = Field(..., description="Type of step")
    name: str = Field(..., description="Human-readable step name")
    description: str = Field(..., description="Step description")

    # Execution configuration
    timeout_seconds: int = Field(default=300, description="Step timeout in seconds")
    retry_count: int = Field(default=3, description="Number of retries on failure")
    retry_delay_seconds: int = Field(default=5, description="Delay between retries")

    # Conditional execution
    condition: Optional[str] = Field(None, description="Condition for step execution (Python expression)")
    depends_on: List[str] = Field(default_factory=list, description="Step IDs this step depends on")

    # Step-specific configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Step-specific configuration")


class StepExecution(BaseModel):
    """Execution state and result of a workflow step."""

    step_id: str
    status: StepStatus = Field(default=StepStatus.PENDING)

    # Execution timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # Execution results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0

    # Output for next steps
    output: Dict[str, Any] = Field(default_factory=dict, description="Step output data")


class WorkflowDefinition(BaseModel):
    """Definition of a workflow template."""

    id: UUID = Field(default_factory=uuid4)
    type: WorkflowType
    name: str
    description: str
    version: str = Field(default="1.0.0")

    # Workflow steps in execution order
    steps: List[WorkflowStep]

    # Workflow configuration
    max_execution_time_seconds: int = Field(default=3600, description="Max workflow execution time")
    allow_parallel: bool = Field(default=False, description="Allow parallel step execution")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[UUID] = None
    tags: List[str] = Field(default_factory=list)


class WorkflowExecution(BaseModel):
    """Runtime execution state of a workflow instance."""

    id: UUID = Field(default_factory=uuid4)
    workflow_id: UUID = Field(..., description="Workflow definition ID")
    workflow_type: WorkflowType

    # Associated signal
    signal_id: UUID = Field(..., description="Signal that triggered this workflow")
    user_id: UUID = Field(..., description="User who owns this workflow")

    # Execution state
    status: WorkflowStatus = Field(default=WorkflowStatus.CREATED)
    current_step_id: Optional[str] = None

    # Step executions
    step_executions: Dict[str, StepExecution] = Field(default_factory=dict)

    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None

    # Results and context
    context: Dict[str, Any] = Field(default_factory=dict, description="Workflow execution context")
    final_result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)



"""Step handlers for workflow execution.

This module implements the actual logic for each workflow step type.
Each handler is an async function that takes a step and execution context
and returns a result dictionary.
"""

import logging
from typing import Any, Dict

from app.core.signal_models import ActionableSignal
from app.intelligence.response_generator import ResponseGenerator
from app.intelligence.response_playbook import ResponseChannel
from app.workflows.workflow_models import WorkflowExecution, WorkflowStep

logger = logging.getLogger(__name__)


class WorkflowStepHandlers:
    """Collection of step handlers for workflow execution."""

    def __init__(
        self,
        response_generator: ResponseGenerator,
    ):
        """Initialize step handlers.

        Args:
            response_generator: Response generator for GENERATE steps
        """
        self.response_generator = response_generator

    async def handle_analyze(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle ANALYZE step - extract information from signal.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Analysis results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        config = step.config
        extract_fields = config.get("extract_fields", [])

        logger.info(f"Analyzing signal {signal.id} for fields: {extract_fields}")

        # Extract requested fields from signal metadata
        analysis = {}

        for field in extract_fields:
            if field == "pain_points":
                # Extract pain points from description
                analysis["pain_points"] = self._extract_pain_points(signal)
            elif field == "current_tool":
                # Extract current tool from metadata
                analysis["current_tool"] = signal.metadata.get("competitor_name", "unknown")
            elif field == "requirements":
                # Extract requirements from description
                analysis["requirements"] = self._extract_requirements(signal)
            elif field == "competitor_name":
                analysis["competitor_name"] = signal.metadata.get("competitor_name", "unknown")
            elif field == "complaint_type":
                analysis["complaint_type"] = self._classify_complaint(signal)
            elif field == "severity":
                analysis["severity"] = self._assess_severity(signal)
            else:
                # Generic field extraction from metadata
                analysis[field] = signal.metadata.get(field, None)

        # Store analysis in context for next steps
        execution.context["analysis"] = analysis

        return analysis

    async def handle_score(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle SCORE step - score/evaluate something.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Scoring results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)
        analysis = execution.context.get("analysis", {})

        config = step.config
        scoring_criteria = config.get("scoring_criteria", [])
        min_threshold = config.get("min_threshold", 0.5)

        logger.info(f"Scoring signal {signal.id} with criteria: {scoring_criteria}")

        scores = {}

        # Calculate scores based on criteria
        if "fit_score" in scoring_criteria:
            scores["fit_score"] = self._calculate_fit_score(signal, analysis)

        if "intent_score" in scoring_criteria:
            scores["intent_score"] = signal.confidence_score

        if "urgency_score" in scoring_criteria:
            scores["urgency_score"] = signal.urgency_score

        # Calculate overall score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        scores["overall_score"] = overall_score

        # Determine quality level
        if overall_score >= 0.8:
            quality = "high_quality"
        elif overall_score >= min_threshold:
            quality = "medium_quality"
        else:
            quality = "low_quality"

        scores["quality_level"] = quality

        # Store in context
        execution.context["scores"] = scores
        execution.context["lead_quality"] = quality

        return scores

    async def handle_decide(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle DECIDE step - make a decision based on context.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Decision results
        """
        config = step.config
        decision_tree = config.get("decision_tree", {})

        # Get decision input (e.g., lead quality)
        lead_quality = execution.context.get("lead_quality", "low_quality")

        # Make decision
        decision = decision_tree.get(lead_quality, "monitor_only")

        logger.info(f"Decision for {lead_quality}: {decision}")

        # Store decision in context
        execution.context["decision"] = decision

        return {"decision": decision, "input": lead_quality}

    async def handle_generate(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle GENERATE step - generate content using LLM.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Generated content
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        config = step.config
        num_variants = config.get("num_variants", 3)

        logger.info(f"Generating {num_variants} response variants for signal {signal.id}")

        # Generate response variants
        variants = await self.response_generator.generate_variants(
            signal=signal,
            num_variants=num_variants,
            channel=ResponseChannel.REDDIT,  # TODO: Make configurable
        )

        # Store best variant in context
        if variants:
            best_variant = variants[0]  # Already sorted by score
            execution.context["generated_response"] = best_variant.content
            execution.context["response_tone"] = best_variant.tone.value
            execution.context["response_score"] = best_variant.overall_score

        return {
            "num_variants": len(variants),
            "best_score": variants[0].overall_score if variants else 0.0,
            "best_content": variants[0].content if variants else "",
        }

    async def handle_notify(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle NOTIFY step - send notifications.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Notification results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        config = step.config
        channels = config.get("notification_channels", ["slack"])
        priority = config.get("priority", "normal")

        logger.info(
            f"Sending {priority} priority notification for signal {signal.id} "
            f"to channels: {channels}"
        )

        # TODO: Implement actual notification sending
        # For now, just log and return success

        notifications_sent = []
        for channel in channels:
            notifications_sent.append({
                "channel": channel,
                "status": "sent",
                "timestamp": execution.context.get("signal", {}).get("created_at"),
            })

        return {
            "notifications_sent": len(notifications_sent),
            "channels": channels,
            "priority": priority,
        }

    async def handle_execute(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle EXECUTE step - execute an action.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Execution results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        config = step.config

        logger.info(f"Executing action for signal {signal.id}")

        # TODO: Implement actual action execution (post response, send DM, etc.)
        # For now, just log and return success

        return {
            "action_executed": True,
            "signal_id": str(signal.id),
            "timestamp": execution.context.get("signal", {}).get("created_at"),
        }

    async def handle_track(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle TRACK step - track metrics and outcomes.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Tracking results
        """
        signal_data = execution.context.get("signal", {})
        signal = ActionableSignal(**signal_data)

        config = step.config
        metrics = config.get("metrics", [])

        logger.info(f"Tracking metrics for signal {signal.id}: {metrics}")

        # TODO: Implement actual metric tracking
        # For now, just initialize tracking

        tracking_data = {
            "signal_id": str(signal.id),
            "metrics_tracked": metrics,
            "tracking_started": True,
        }

        return tracking_data

    async def handle_wait(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution,
    ) -> Dict[str, Any]:
        """Handle WAIT step - wait for external input or time.

        Args:
            step: Workflow step
            execution: Workflow execution

        Returns:
            Wait results
        """
        wait_seconds = step.config.get("wait_seconds", 0)

        logger.info(f"Waiting for {wait_seconds} seconds")

        # TODO: Implement actual waiting logic with external input support
        # For now, just return immediately

        return {
            "waited": True,
            "wait_seconds": wait_seconds,
        }

    # Helper methods for analysis

    def _extract_pain_points(self, signal: ActionableSignal) -> list:
        """Extract pain points from signal description."""
        # Simple extraction - in production, use NLP
        description = signal.description.lower()
        pain_points = []

        pain_keywords = ["slow", "expensive", "complicated", "broken", "missing", "frustrated"]
        for keyword in pain_keywords:
            if keyword in description:
                pain_points.append(keyword)

        return pain_points or ["general_dissatisfaction"]

    def _extract_requirements(self, signal: ActionableSignal) -> list:
        """Extract requirements from signal description."""
        # Simple extraction - in production, use NLP
        description = signal.description.lower()
        requirements = []

        requirement_keywords = ["need", "want", "require", "looking for", "must have"]
        for keyword in requirement_keywords:
            if keyword in description:
                requirements.append(keyword)

        return requirements or ["general_requirements"]

    def _classify_complaint(self, signal: ActionableSignal) -> str:
        """Classify type of complaint."""
        description = signal.description.lower()

        if any(word in description for word in ["price", "expensive", "cost"]):
            return "pricing"
        elif any(word in description for word in ["slow", "performance", "lag"]):
            return "performance"
        elif any(word in description for word in ["support", "help", "response"]):
            return "support"
        elif any(word in description for word in ["bug", "broken", "error"]):
            return "reliability"
        else:
            return "general"

    def _assess_severity(self, signal: ActionableSignal) -> str:
        """Assess severity of issue."""
        # Use signal scores to determine severity
        if signal.urgency_score >= 0.8 and signal.impact_score >= 0.7:
            return "critical"
        elif signal.urgency_score >= 0.6 or signal.impact_score >= 0.6:
            return "high"
        elif signal.urgency_score >= 0.4 or signal.impact_score >= 0.4:
            return "medium"
        else:
            return "low"

    def _calculate_fit_score(self, signal: ActionableSignal, analysis: Dict[str, Any]) -> float:
        """Calculate product fit score for lead."""
        # Simple scoring - in production, use ML model
        score = 0.5  # Base score

        # Boost score if pain points match our strengths
        pain_points = analysis.get("pain_points", [])
        if "expensive" in pain_points:
            score += 0.2  # We're more affordable
        if "complicated" in pain_points:
            score += 0.2  # We're easier to use

        # Cap at 1.0
        return min(score, 1.0)



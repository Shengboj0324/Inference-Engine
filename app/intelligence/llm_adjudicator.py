"""LLM adjudication engine for signal classification.

This module implements Stage C of the inference pipeline:
- Structured LLM outputs with JSON schema validation
- Few-shot prompting with canonical exemplars
- Evidence extraction and rationale generation
- Abstention support for uncertain cases
- Retry logic with structured repair prompts

Follows the strict contract defined in app/domain/inference_models.py
"""

import logging
import json
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from pydantic import BaseModel, ValidationError

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import (
    SignalInference,
    SignalPrediction,
    SignalType,
    AbstentionReason,
    EvidenceSpan,
)
from app.intelligence.candidate_retrieval import SignalCandidate
from app.llm.router import get_router

logger = logging.getLogger(__name__)


class LLMAdjudicationOutput(BaseModel):
    """Structured output schema for LLM adjudication."""
    
    candidate_signal_types: List[str]
    primary_signal_type: str
    confidence: float
    evidence_spans: List[Dict[str, str]]
    rationale: str
    requires_more_context: bool
    abstain: bool
    abstention_reason: Optional[str] = None
    risk_labels: List[str]
    suggested_actions: List[str]


class LLMAdjudicator:
    """LLM-based signal classification with structured outputs.
    
    This is Stage C of the inference pipeline as defined in the blueprint.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        """Initialize LLM adjudicator.
        
        Args:
            model_name: LLM model to use
            temperature: Sampling temperature
            max_retries: Maximum number of retries for malformed outputs
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        
        # Initialize LLM router
        self.llm_router = get_router()
        
        logger.info(
            f"LLMAdjudicator initialized: model={model_name}, temp={temperature}"
        )
    
    async def adjudicate(
        self,
        observation: NormalizedObservation,
        candidates: List[SignalCandidate],
    ) -> SignalInference:
        """Adjudicate signal type for an observation.
        
        Args:
            observation: Normalized observation
            candidates: Candidate signal types from retrieval
            
        Returns:
            Signal inference with structured predictions
        """
        # Build prompt
        prompt = self._build_prompt(observation, candidates)
        
        # Call LLM with retries
        llm_output = await self._call_llm_with_retries(prompt)
        
        # Parse and validate output
        try:
            adjudication = LLMAdjudicationOutput(**llm_output)
        except ValidationError as e:
            logger.error(f"Failed to validate LLM output: {e}")
            # Return abstention
            return self._create_abstention_inference(
                observation,
                AbstentionReason.MALFORMED_OUTPUT,
                "LLM output failed validation"
            )
        
        # Convert to SignalInference
        return self._convert_to_inference(observation, adjudication)
    
    def _build_prompt(
        self,
        observation: NormalizedObservation,
        candidates: List[SignalCandidate],
    ) -> str:
        """Build few-shot prompt for LLM adjudication.
        
        Args:
            observation: Normalized observation
            candidates: Candidate signal types
            
        Returns:
            Prompt string
        """
        # System message
        system = """You are an expert signal classifier for business intelligence.
Your task is to analyze social media content and classify it into actionable signal types.

You must output a valid JSON object with the following schema:
{
  "candidate_signal_types": ["type1", "type2"],
  "primary_signal_type": "type1",
  "confidence": 0.85,
  "evidence_spans": [{"text": "...", "reason": "..."}],
  "rationale": "Explanation of classification",
  "requires_more_context": false,
  "abstain": false,
  "abstention_reason": null,
  "risk_labels": ["public_reply_safe"],
  "suggested_actions": ["reply_public"]
}

Signal types: support_request, feature_request, bug_report, complaint, praise,
competitor_mention, alternative_seeking, price_sensitivity, integration_request,
churn_risk, security_concern, legal_risk, reputation_risk,
expansion_opportunity, upsell_opportunity, partnership_opportunity,
unclear, not_actionable

Abstention reasons: low_confidence, ambiguous_intent, insufficient_context,
conflicting_signals, unsafe_content, policy_violation, out_of_scope

If you are uncertain, set abstain=true and provide a reason."""
        
        # Few-shot examples
        examples = self._get_few_shot_examples()
        
        # User message
        user_message = f"""Analyze this content:

Title: {observation.title or 'N/A'}
Text: {observation.normalized_text[:1000] if observation.normalized_text else 'N/A'}
Platform: {observation.source_platform.value}
Language: {observation.original_language or 'unknown'}

Candidate signals (from retrieval):
{self._format_candidates(candidates)}

Classify this content and output valid JSON."""
        
        # Combine
        full_prompt = f"{system}\n\n{examples}\n\n{user_message}"
        return full_prompt

    def _get_few_shot_examples(self) -> str:
        """Get few-shot examples for prompting.

        Returns:
            Few-shot examples string
        """
        examples = """Example 1:
Input: "Looking for a better alternative to Slack. Need something with better pricing."
Output: {
  "candidate_signal_types": ["alternative_seeking", "competitor_mention", "price_sensitivity"],
  "primary_signal_type": "alternative_seeking",
  "confidence": 0.92,
  "evidence_spans": [
    {"text": "better alternative to Slack", "reason": "explicit alternative seeking"},
    {"text": "better pricing", "reason": "price sensitivity indicator"}
  ],
  "rationale": "User explicitly seeks alternative to competitor (Slack) with price as key factor",
  "requires_more_context": false,
  "abstain": false,
  "abstention_reason": null,
  "risk_labels": ["public_reply_safe"],
  "suggested_actions": ["reply_public", "prepare_dm_followup"]
}

Example 2:
Input: "Your product is amazing! Just upgraded to Pro plan."
Output: {
  "candidate_signal_types": ["praise", "upsell_opportunity"],
  "primary_signal_type": "praise",
  "confidence": 0.95,
  "evidence_spans": [
    {"text": "product is amazing", "reason": "positive sentiment"},
    {"text": "upgraded to Pro plan", "reason": "conversion event"}
  ],
  "rationale": "Clear positive feedback with successful upsell",
  "requires_more_context": false,
  "abstain": false,
  "abstention_reason": null,
  "risk_labels": ["public_reply_safe"],
  "suggested_actions": ["thank_user", "request_testimonial"]
}"""
        return examples

    def _format_candidates(self, candidates: List[SignalCandidate]) -> str:
        """Format candidates for prompt.

        Args:
            candidates: Candidate signals

        Returns:
            Formatted string
        """
        if not candidates:
            return "No candidates from retrieval"

        lines = []
        for i, candidate in enumerate(candidates, 1):
            lines.append(
                f"{i}. {candidate.signal_type.value} (score={candidate.score:.2f}): {candidate.reasoning}"
            )
        return "\n".join(lines)

    async def _call_llm_with_retries(self, prompt: str) -> Dict[str, Any]:
        """Call LLM with retry logic for malformed outputs.

        Args:
            prompt: Prompt string

        Returns:
            Parsed JSON output
        """
        for attempt in range(self.max_retries):
            try:
                # Call LLM
                response = await self.llm_router.generate(
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=self.temperature,
                )

                # Extract JSON from response
                content = response.content.strip()

                # Try to find JSON in response
                if "{" in content and "}" in content:
                    start = content.index("{")
                    end = content.rindex("}") + 1
                    json_str = content[start:end]

                    # Parse JSON
                    output = json.loads(json_str)
                    return output
                else:
                    raise ValueError("No JSON found in response")

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")

                if attempt < self.max_retries - 1:
                    # Add repair instruction
                    prompt += "\n\nYour previous output was malformed. Please output ONLY valid JSON."
                else:
                    # Final attempt failed
                    raise ValueError(f"Failed to get valid JSON after {self.max_retries} attempts")

        raise ValueError("Unexpected error in LLM call")

    def _convert_to_inference(
        self,
        observation: NormalizedObservation,
        adjudication: LLMAdjudicationOutput,
    ) -> SignalInference:
        """Convert LLM adjudication to SignalInference.

        Args:
            observation: Normalized observation
            adjudication: LLM adjudication output

        Returns:
            Signal inference
        """
        # Parse predictions
        predictions = []
        for signal_type_str in adjudication.candidate_signal_types:
            try:
                signal_type = SignalType(signal_type_str)
                # Assign confidence based on position
                if signal_type_str == adjudication.primary_signal_type:
                    confidence = adjudication.confidence
                else:
                    confidence = adjudication.confidence * 0.5  # Lower for non-primary

                predictions.append(
                    SignalPrediction(
                        signal_type=signal_type,
                        probability=confidence,
                    )
                )
            except ValueError:
                logger.warning(f"Invalid signal type: {signal_type_str}")

        # Get top prediction
        top_prediction = predictions[0] if predictions else None

        # Parse evidence spans
        evidence_spans = []
        for span_dict in adjudication.evidence_spans:
            evidence_spans.append(
                EvidenceSpan(
                    text=span_dict.get("text", ""),
                    start_char=0,  # LLM doesn't provide exact positions
                    end_char=len(span_dict.get("text", "")),
                    reasoning=span_dict.get("reason", ""),
                )
            )

        # Parse abstention reason
        abstention_reason = None
        if adjudication.abstain and adjudication.abstention_reason:
            try:
                abstention_reason = AbstentionReason(adjudication.abstention_reason)
            except ValueError:
                abstention_reason = AbstentionReason.LOW_CONFIDENCE

        # Create inference
        inference = SignalInference(
            normalized_observation_id=observation.id,
            user_id=observation.user_id,
            predictions=predictions,
            top_prediction=top_prediction,
            abstained=adjudication.abstain,
            abstention_reason=abstention_reason,
            evidence_spans=evidence_spans,
            rationale=adjudication.rationale,
            model_name=self.model_name,
            model_version="2024-01",  # Would be dynamic in production
            inference_method="llm_few_shot",
        )

        return inference

    def _create_abstention_inference(
        self,
        observation: NormalizedObservation,
        reason: AbstentionReason,
        rationale: str,
    ) -> SignalInference:
        """Create an abstention inference.

        Args:
            observation: Normalized observation
            reason: Abstention reason
            rationale: Rationale for abstention

        Returns:
            Signal inference with abstention
        """
        return SignalInference(
            normalized_observation_id=observation.id,
            user_id=observation.user_id,
            predictions=[],
            top_prediction=None,
            abstained=True,
            abstention_reason=reason,
            rationale=rationale,
            model_name=self.model_name,
            model_version="2024-01",
            inference_method="llm_few_shot",
        )


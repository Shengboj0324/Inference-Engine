"""Policy checking system for validating response drafts.

This module implements comprehensive policy checking:
- Safety and compliance validation
- Brand guideline enforcement
- Legal and regulatory checks
- Tone and sentiment validation
- Content quality checks
"""

import logging
import re
from typing import List, Dict, Optional

from app.domain.action_models import (
    ResponseDraft,
    PolicyViolation,
    ActionableSignal,
)
from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalType

logger = logging.getLogger(__name__)


class PolicyChecker:
    """Policy checking system for response validation.
    
    Validates response drafts against:
    - Safety policies (no harmful content)
    - Brand policies (tone, voice, claims)
    - Legal policies (disclaimers, compliance)
    - Quality policies (length, clarity, relevance)
    """
    
    def __init__(
        self,
        strict_mode: bool = True,
        block_on_critical: bool = True,
    ):
        """Initialize policy checker.
        
        Args:
            strict_mode: Enable strict policy enforcement
            block_on_critical: Block execution on critical violations
        """
        self.strict_mode = strict_mode
        self.block_on_critical = block_on_critical
        
        # Define policy rules
        self._init_policy_rules()
        
        logger.info(
            f"PolicyChecker initialized: "
            f"strict={strict_mode}, "
            f"block_critical={block_on_critical}"
        )
    
    def _init_policy_rules(self):
        """Initialize policy rules."""
        # Prohibited claims
        self.prohibited_claims = [
            "guaranteed", "promise", "best in the world",
            "number one", "#1", "always works", "never fails",
            "100% effective", "cure", "miracle"
        ]

        # Sensitive topics requiring review
        self.sensitive_topics = [
            "legal", "lawsuit", "regulation", "compliance",
            "security breach", "data leak", "hack",
            "discrimination", "harassment", "privacy violation"
        ]

        # Competitor mention patterns (pre-compiled for efficiency)
        self.competitor_patterns = [
            re.compile(r"\bvs\b", re.IGNORECASE),
            re.compile(r"\bversus\b", re.IGNORECASE),
            re.compile(r"better than", re.IGNORECASE),
            re.compile(r"compared to", re.IGNORECASE),
            re.compile(r"unlike", re.IGNORECASE),
            re.compile(r"instead of", re.IGNORECASE),
        ]

        # Required disclaimers for certain signal types
        self.disclaimer_requirements = {
            SignalType.LEGAL_RISK: "legal disclaimer",
            SignalType.SECURITY_CONCERN: "security notice",
            SignalType.PRICE_SENSITIVITY: "pricing subject to change",
        }
    
    def check_draft(
        self,
        draft: ResponseDraft,
        action: ActionableSignal,
        observation: NormalizedObservation,
    ) -> List[PolicyViolation]:
        """Check draft for policy violations.
        
        Args:
            draft: Response draft to check
            action: Actionable signal context
            observation: Normalized observation context
            
        Returns:
            List of policy violations
        """
        violations = []
        
        # Run all policy checks
        violations.extend(self._check_safety_policy(draft))
        violations.extend(self._check_brand_policy(draft))
        violations.extend(self._check_legal_policy(draft, action))
        violations.extend(self._check_quality_policy(draft, action))
        violations.extend(self._check_tone_policy(draft, action))
        
        # Log violations
        if violations:
            logger.warning(
                f"Draft {draft.variant_id} has {len(violations)} violations"
            )
            for v in violations:
                logger.warning(f"  - {v.policy_name}: {v.description}")
        
        return violations
    
    def is_safe_to_execute(
        self,
        draft: ResponseDraft,
        violations: Optional[List[PolicyViolation]] = None,
    ) -> bool:
        """Check if draft is safe to execute.
        
        Args:
            draft: Response draft
            violations: Optional pre-computed violations
            
        Returns:
            True if safe to execute, False otherwise
        """
        if violations is None:
            violations = draft.policy_violations
        
        # Check for blocking violations
        if self.block_on_critical:
            for violation in violations:
                if violation.blocking or violation.severity == "critical":
                    return False
        
        return True
    
    def _check_safety_policy(self, draft: ResponseDraft) -> List[PolicyViolation]:
        """Check safety policy.

        Args:
            draft: Response draft

        Returns:
            List of violations
        """
        violations = []

        # Null check
        if not draft.content:
            return violations

        content_lower = draft.content.lower()

        # Check for sensitive topics
        for topic in self.sensitive_topics:
            if topic in content_lower:
                violation = PolicyViolation(
                    policy_name="safety_sensitive_topics",
                    violation_type="sensitive_topic",
                    severity="high",
                    description=f"Contains sensitive topic: '{topic}'",
                    blocking=True,
                )
                # Avoid duplicates
                if not any(v.description == violation.description for v in violations):
                    violations.append(violation)

        return violations

    def _check_brand_policy(self, draft: ResponseDraft) -> List[PolicyViolation]:
        """Check brand policy.

        Args:
            draft: Response draft

        Returns:
            List of violations
        """
        violations = []

        # Null check
        if not draft.content:
            return violations

        content_lower = draft.content.lower()

        # Check for prohibited claims
        for claim in self.prohibited_claims:
            if claim in content_lower:
                violation = PolicyViolation(
                    policy_name="brand_no_unsubstantiated_claims",
                    violation_type="prohibited_claim",
                    severity="medium",
                    description=f"Contains prohibited claim: '{claim}'",
                    blocking=False,
                )
                # Avoid duplicates
                if not any(v.description == violation.description for v in violations):
                    violations.append(violation)

        # Check for competitor mentions (using pre-compiled patterns)
        for pattern in self.competitor_patterns:
            if pattern.search(content_lower):
                violation = PolicyViolation(
                    policy_name="brand_competitor_mention",
                    violation_type="competitor_mention",
                    severity="low",
                    description="Contains competitor comparison - review tone",
                    blocking=False,
                )
                # Avoid duplicates
                if not any(v.policy_name == "brand_competitor_mention" for v in violations):
                    violations.append(violation)
                break  # Only flag once

        return violations

    def _check_legal_policy(
        self,
        draft: ResponseDraft,
        action: ActionableSignal,
    ) -> List[PolicyViolation]:
        """Check legal policy.

        Args:
            draft: Response draft
            action: Actionable signal

        Returns:
            List of violations
        """
        violations = []

        # Null check
        if not draft.content:
            return violations

        content_lower = draft.content.lower()

        # Check for required disclaimers
        required_disclaimer = self.disclaimer_requirements.get(action.signal_type)
        if required_disclaimer:
            if required_disclaimer.lower() not in content_lower:
                violations.append(
                    PolicyViolation(
                        policy_name="legal_required_disclaimer",
                        violation_type="missing_disclaimer",
                        severity="high",
                        description=f"Missing required disclaimer: '{required_disclaimer}'",
                        blocking=True,
                    )
                )

        # Check for pricing mentions without disclaimers
        pricing_keywords = ["free", "discount", "price", "cost", "$", "€", "£"]
        has_pricing = any(kw in content_lower for kw in pricing_keywords)

        if has_pricing:
            disclaimer_phrases = [
                "subject to change", "terms apply", "conditions apply",
                "see website", "contact sales"
            ]
            has_disclaimer = any(
                phrase in content_lower for phrase in disclaimer_phrases
            )

            if not has_disclaimer:
                violations.append(
                    PolicyViolation(
                        policy_name="legal_pricing_disclaimer",
                        violation_type="missing_pricing_disclaimer",
                        severity="medium",
                        description="Pricing mentioned without disclaimer",
                        blocking=False,
                    )
                )

        return violations

    def _check_quality_policy(
        self,
        draft: ResponseDraft,
        action: ActionableSignal,
    ) -> List[PolicyViolation]:
        """Check quality policy.

        Args:
            draft: Response draft
            action: Actionable signal

        Returns:
            List of violations
        """
        violations = []

        # Null check and empty content check
        if not draft.content or not draft.content.strip():
            violations.append(
                PolicyViolation(
                    policy_name="quality_empty_content",
                    violation_type="empty",
                    severity="critical",
                    description="Response is empty or whitespace-only",
                    blocking=True,
                )
            )
            return violations

        content_len = len(draft.content)

        # Check minimum length
        if content_len < 20:
            violations.append(
                PolicyViolation(
                    policy_name="quality_minimum_length",
                    violation_type="too_short",
                    severity="high",
                    description=f"Response too short ({content_len} chars)",
                    blocking=True,
                )
            )

        # Check maximum length by channel
        from app.domain.action_models import ResponseChannel

        max_lengths = {
            ResponseChannel.DIRECT_REPLY: 280,
            ResponseChannel.DIRECT_MESSAGE: 1000,
            ResponseChannel.EMAIL: 2000,
        }

        max_len = max_lengths.get(draft.channel, 1000)
        if content_len > max_len:
            violations.append(
                PolicyViolation(
                    policy_name="quality_maximum_length",
                    violation_type="too_long",
                    severity="medium",
                    description=f"Response too long ({content_len} chars, max {max_len})",
                    blocking=False,
                )
            )

        return violations

    def _check_tone_policy(
        self,
        draft: ResponseDraft,
        action: ActionableSignal,
    ) -> List[PolicyViolation]:
        """Check tone policy.

        Args:
            draft: Response draft
            action: Actionable signal

        Returns:
            List of violations
        """
        violations = []

        # Null check
        if not draft.content:
            return violations

        content_lower = draft.content.lower()

        # Check for overly aggressive language
        aggressive_words = [
            "must", "need to", "have to", "should",
            "immediately", "urgent", "asap"
        ]

        aggressive_count = sum(
            1 for word in aggressive_words if word in content_lower
        )

        if aggressive_count >= 3:
            violations.append(
                PolicyViolation(
                    policy_name="tone_too_aggressive",
                    violation_type="aggressive_tone",
                    severity="low",
                    description="Tone may be too aggressive or pushy",
                    blocking=False,
                )
            )

        # Check for overly casual language in professional contexts
        if draft.tone == "professional":
            casual_markers = ["lol", "omg", "tbh", "ngl", "😂", "🤣"]
            if any(marker in content_lower for marker in casual_markers):
                violations.append(
                    PolicyViolation(
                        policy_name="tone_inconsistent",
                        violation_type="tone_mismatch",
                        severity="medium",
                        description="Casual language in professional tone",
                        blocking=False,
                    )
                )

        return violations


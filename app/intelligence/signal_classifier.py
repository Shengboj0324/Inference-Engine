
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Pattern, Tuple
from uuid import UUID

from pydantic import BaseModel

from app.core.models import ContentItem
from app.core.signal_models import (
    ActionableSignal,
    SignalType,
    ActionType,
    ResponseTone,
)
from app.llm.router import get_router, RoutingStrategy

logger = logging.getLogger(__name__)


class IntentPattern(BaseModel):
    """Intent detection pattern with metadata."""

    pattern: str  # Regex pattern
    signal_type: SignalType
    weight: float = 1.0  # Pattern strength (0-1)
    requires_context: bool = False  # Needs additional context validation


class SignalClassifier:
    """Classify content into business-intent signals.

    This is the core service that determines which content items represent
    actionable business opportunities or risks.

    Classification pipeline:
    1. Pattern matching: Fast regex-based filtering
    2. Context validation: Check if pattern match is in right context
    3. LLM classification: High-precision intent detection
    4. Confidence scoring: Calculate classification confidence
    5. Signal creation: Build ActionableSignal with all metadata

    Attributes:
        llm_router: LLM router for classification
        min_confidence: Minimum confidence threshold (default 0.7)
        intent_patterns: Compiled regex patterns for each signal type
    """

    def __init__(
        self,
        min_confidence: float = 0.7,
        use_llm: bool = True,
    ):
        """Initialize signal classifier.

        Args:
            min_confidence: Minimum confidence threshold (0-1)
            use_llm: Whether to use LLM for classification (vs pattern-only)
        """
        self.min_confidence = min_confidence
        self.use_llm = use_llm
        self.llm_router = get_router() if use_llm else None

        # Compile intent patterns
        self.intent_patterns = self._build_intent_patterns()

        logger.info(
            f"SignalClassifier initialized with {len(self.intent_patterns)} patterns, "
            f"min_confidence={min_confidence}, use_llm={use_llm}"
        )

    def _build_intent_patterns(self) -> Dict[SignalType, List[Pattern]]:
        """Build and compile regex patterns for each signal type.

        Returns:
            Dictionary mapping signal types to compiled regex patterns
        """
        # Define raw patterns for each signal type
        raw_patterns: Dict[SignalType, List[str]] = {
            SignalType.LEAD_OPPORTUNITY: [
                r"looking for (alternatives?|replacement|substitute)s? (to|for)",
                r"what('s| is) the best (tool|software|platform|solution|service) for",
                r"does anyone (know|recommend|suggest) (a|an|any)",
                r"switching (from|away from)",
                r"need (a|an) (tool|solution|platform|service|software)",
                r"can anyone recommend",
                r"what (do|does) (you|people|everyone) use for",
                r"tired of (using)?",
                r"frustrated with",
                r"better (alternative|option) to",
            ],
            SignalType.COMPETITOR_WEAKNESS: [
                r"(terrible|awful|horrible|bad|poor|worst) (support|customer service|experience)",
                r"(disappointed|frustrated|angry) with",
                r"(pricing|price) is (too high|ridiculous|insane|expensive|outrageous)",
                r"(broken|buggy|doesn't work|not working|stopped working)",
                r"(slow|sluggish|laggy) (performance|response|loading)",
                r"missing (features?|functionality)",
                r"(complicated|confusing|hard to use|difficult)",
                r"(cancell?ing|leaving|switching away from)",
            ],
            SignalType.PRODUCT_CONFUSION: [
                r"how (do|does) (i|you|one) (use|setup|configure|install)",
                r"(confused|unclear|don't understand) (about|how|why)",
                r"what('s| is) the difference between",
                r"(can|does) (it|this) (support|work with|integrate)",
                r"is (it|this) (compatible|available) (with|for|on)",
            ],
            SignalType.FEATURE_REQUEST_PATTERN: [
                r"(wish|hope|would love|really need) (it|you|they) (had|supported|offered)",
                r"(missing|lacks|doesn't have|no) (feature|functionality|capability|option)",
                r"(please|can you) add (support for|feature|ability to)",
                r"when will (you|it) (support|have|add)",
            ],
        }



        # Compile patterns
        compiled_patterns: Dict[SignalType, List[Pattern]] = {}
        for signal_type, patterns in raw_patterns.items():
            compiled_patterns[signal_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

        return compiled_patterns

    async def classify_content(
        self,
        item: ContentItem,
        user_id: UUID,
    ) -> Optional[ActionableSignal]:
        # Stage 1: Pattern matching for quick filtering
        pattern_matches = self._pattern_match(item)
        if not pattern_matches:
            logger.debug(f"No pattern matches for item {item.id}")
            return None

        logger.info(
            f"Pattern matches found for item {item.id}: "
            f"{[st.value for st, _ in pattern_matches]}"
        )

        # Stage 2: LLM-based classification for precision
        if self.use_llm:
            signal_type, confidence = await self._llm_classify(item, pattern_matches)
            if confidence < self.min_confidence:
                logger.info(
                    f"LLM confidence {confidence:.2f} below threshold "
                    f"{self.min_confidence} for item {item.id}"
                )
                return None
        else:
            # Use highest-weighted pattern match
            signal_type, confidence = pattern_matches[0]

        logger.info(
            f"Classified item {item.id} as {signal_type.value} "
            f"with confidence {confidence:.2f}"
        )

        # Stage 3: Create signal with all metadata
        signal = await self._create_signal(
            item=item,
            user_id=user_id,
            signal_type=signal_type,
            confidence=confidence,
        )

        return signal

    def _pattern_match(
        self,
        item: ContentItem,
    ) -> List[Tuple[SignalType, float]]:
        """Fast pattern matching for initial filtering.

        Args:
            item: Content item to match

        Returns:
            List of (signal_type, confidence) tuples, sorted by confidence
        """
        matches: List[Tuple[SignalType, float]] = []

        # Combine title and text for matching
        text = f"{item.title} {item.raw_text or ''}".lower()

        # Check each signal type's patterns
        for signal_type, patterns in self.intent_patterns.items():
            match_count = 0
            for pattern in patterns:
                if pattern.search(text):
                    match_count += 1

            if match_count > 0:
                # Confidence based on number of pattern matches
                # More matches = higher confidence
                confidence = min(0.5 + (match_count * 0.1), 0.9)
                matches.append((signal_type, confidence))

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    async def _llm_classify(
        self,
        item: ContentItem,
        pattern_matches: List[Tuple[SignalType, float]],
    ) -> Tuple[SignalType, float]:
        """LLM-based classification for high precision.

        Args:
            item: Content item to classify
            pattern_matches: Initial pattern matches to validate

        Returns:
            Tuple of (signal_type, confidence)
        """
        # Build classification prompt
        prompt = self._build_classification_prompt(item, pattern_matches)

        try:
            # Use cost-optimized routing for classification
            response = await self.llm_router.generate_simple(
                prompt=prompt,
                max_tokens=200,
                temperature=0.1,  # Low temperature for consistent classification
                strategy=RoutingStrategy.COST_OPTIMIZED,
            )

            # Parse response - generate_simple returns a str directly
            signal_type, confidence = self._parse_classification_response(
                response,
                pattern_matches,
            )

            return signal_type, confidence

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fall back to pattern matching
            return pattern_matches[0] if pattern_matches else (None, 0.0)


    def _build_classification_prompt(
        self,
        item: ContentItem,
        pattern_matches: List[Tuple[SignalType, float]],
    ) -> str:
        """Build prompt for LLM classification.

        Args:
            item: Content item
            pattern_matches: Pattern matches to validate

        Returns:
            Classification prompt
        """
        candidate_types = [st.value for st, _ in pattern_matches[:3]]

        prompt = f"""Classify this social media post into ONE of these business signal types:

Candidate types (from pattern matching):
{', '.join(candidate_types)}

Post:
Title: {item.title}
Content: {item.raw_text[:500] if item.raw_text else 'N/A'}
Platform: {item.source_platform.value}
Author: {item.author or 'Unknown'}

Signal type definitions:
- lead_opportunity: Someone looking for product alternatives or solutions
- competitor_weakness: Complaints about competitor products
- product_confusion: Questions about how to use a product
- feature_request_pattern: Requests for missing features

Respond with ONLY the signal type and confidence (0-1) in this format:
signal_type: <type>
confidence: <0.0-1.0>

If this is NOT a valid business signal, respond with:
signal_type: none
confidence: 0.0"""

        return prompt

    def _parse_classification_response(
        self,
        response: str,
        pattern_matches: List[Tuple[SignalType, float]],
    ) -> Tuple[SignalType, float]:
        """Parse LLM classification response.

        Args:
            response: LLM response text
            pattern_matches: Fallback pattern matches

        Returns:
            Tuple of (signal_type, confidence)
        """
        try:
            # Parse response format:
            # signal_type: <type>
            # confidence: <value>
            lines = response.strip().split('\n')
            signal_type_str = None
            confidence = 0.0

            for line in lines:
                if line.startswith('signal_type:'):
                    signal_type_str = line.split(':', 1)[1].strip()
                elif line.startswith('confidence:'):
                    confidence = float(line.split(':', 1)[1].strip())

            if signal_type_str == 'none' or not signal_type_str:
                return None, 0.0

            # Convert string to SignalType enum
            signal_type = SignalType(signal_type_str)

            return signal_type, confidence

        except Exception as e:
            logger.error(f"Failed to parse classification response: {e}")
            # Fall back to highest pattern match
            return pattern_matches[0] if pattern_matches else (None, 0.0)

    async def _create_signal(
        self,
        item: ContentItem,
        user_id: UUID,
        signal_type: SignalType,
        confidence: float,
    ) -> ActionableSignal:
        """Create ActionableSignal from classified content.

        Args:
            item: Source content item
            user_id: Signal owner
            signal_type: Classified signal type
            confidence: Classification confidence

        Returns:
            Complete ActionableSignal with all metadata
        """
        from app.intelligence.action_scorer import ActionScorer

        # Calculate multi-dimensional scores
        scorer = ActionScorer()
        scores = await scorer.calculate_action_score(item, signal_type)

        # Determine recommended action based on signal type
        recommended_action = self._determine_action(signal_type, item)

        # Determine suggested tone
        suggested_tone = self._determine_tone(signal_type, item)

        # Generate title and description
        title = self._generate_title(item, signal_type)
        description = self._generate_description(item, signal_type)
        context = self._generate_context(item, signal_type)

        # Calculate expiry (SLA deadline)
        expires_at = self._calculate_expiry(signal_type, scores['urgency_score'])

        # Create signal
        signal = ActionableSignal(
            user_id=user_id,
            signal_type=signal_type,
            source_item_ids=[item.id],
            source_platform=item.source_platform.value,
            source_url=item.source_url,
            source_author=item.author,
            title=title,
            description=description,
            context=context,
            urgency_score=scores['urgency_score'],
            impact_score=scores['impact_score'],
            confidence_score=confidence,
            action_score=scores['action_score'],
            recommended_action=recommended_action,
            suggested_channel=item.source_platform.value,
            suggested_tone=suggested_tone,
            expires_at=expires_at,
            metadata={
                'source_item_title': item.title,
                'source_item_author': item.author,
                'classification_method': 'llm' if self.use_llm else 'pattern',
            },
        )

        return signal

    def _determine_action(
        self,
        signal_type: SignalType,
        item: ContentItem,
    ) -> ActionType:
        """Determine recommended action for signal type.

        Args:
            signal_type: Signal type
            item: Content item

        Returns:
            Recommended action type
        """
        # Map signal types to default actions
        action_map = {
            SignalType.LEAD_OPPORTUNITY: ActionType.REPLY_PUBLIC,
            SignalType.COMPETITOR_WEAKNESS: ActionType.CREATE_CONTENT,
            SignalType.PRODUCT_CONFUSION: ActionType.REPLY_PUBLIC,
            SignalType.FEATURE_REQUEST_PATTERN: ActionType.INTERNAL_ALERT,
            SignalType.CHURN_RISK: ActionType.ESCALATE,
            SignalType.SUPPORT_ESCALATION: ActionType.ESCALATE,
            SignalType.MISINFORMATION_RISK: ActionType.ESCALATE,
            SignalType.TREND_TO_CONTENT: ActionType.CREATE_CONTENT,
        }

        return action_map.get(signal_type, ActionType.MONITOR)

    def _determine_tone(
        self,
        signal_type: SignalType,
        item: ContentItem,
    ) -> ResponseTone:
        """Determine suggested response tone.

        Args:
            signal_type: Signal type
            item: Content item

        Returns:
            Suggested tone
        """
        # Map signal types to tones
        tone_map = {
            SignalType.LEAD_OPPORTUNITY: ResponseTone.HELPFUL,
            SignalType.COMPETITOR_WEAKNESS: ResponseTone.PROFESSIONAL,
            SignalType.PRODUCT_CONFUSION: ResponseTone.HELPFUL,
            SignalType.FEATURE_REQUEST_PATTERN: ResponseTone.PROFESSIONAL,
            SignalType.CHURN_RISK: ResponseTone.SUPPORTIVE,
            SignalType.SUPPORT_ESCALATION: ResponseTone.SUPPORTIVE,
            SignalType.TREND_TO_CONTENT: ResponseTone.EDUCATIONAL,
        }

        return tone_map.get(signal_type, ResponseTone.PROFESSIONAL)

    def _generate_title(
        self,
        item: ContentItem,
        signal_type: SignalType,
    ) -> str:
        """Generate brief signal title.

        Args:
            item: Content item
            signal_type: Signal type

        Returns:
            Signal title (max 200 chars)
        """
        # Extract key phrase from content
        text = item.title or item.raw_text or ""

        # Truncate and add signal type prefix
        prefix_map = {
            SignalType.LEAD_OPPORTUNITY: "Lead Opportunity",
            SignalType.COMPETITOR_WEAKNESS: "Competitor Weakness",
            SignalType.PRODUCT_CONFUSION: "Product Question",
            SignalType.FEATURE_REQUEST_PATTERN: "Feature Request",
        }

        prefix = prefix_map.get(signal_type, "Signal")
        title = f"{prefix}: {text[:150]}"

        return title[:200]

    def _generate_description(
        self,
        item: ContentItem,
        signal_type: SignalType,
    ) -> str:
        """Generate detailed signal description.

        Args:
            item: Content item
            signal_type: Signal type

        Returns:
            Signal description
        """
        return item.raw_text[:2000] if item.raw_text else item.title

    def _generate_context(
        self,
        item: ContentItem,
        signal_type: SignalType,
    ) -> str:
        """Generate business context for signal.

        Args:
            item: Content item
            signal_type: Signal type

        Returns:
            Business context explanation
        """
        context_templates = {
            SignalType.LEAD_OPPORTUNITY: (
                "This person is actively looking for a solution. "
                "High conversion potential if we respond quickly with relevant positioning."
            ),
            SignalType.COMPETITOR_WEAKNESS: (
                "Competitor experiencing customer dissatisfaction. "
                "Opportunity to position our advantages and capture market share."
            ),
            SignalType.PRODUCT_CONFUSION: (
                "User has questions about product usage. "
                "Opportunity to provide helpful support and build goodwill."
            ),
            SignalType.FEATURE_REQUEST_PATTERN: (
                "Recurring feature request detected. "
                "May indicate product gap worth addressing."
            ),
        }

        return context_templates.get(
            signal_type,
            "Business-relevant signal detected requiring attention."
        )

    def _calculate_expiry(
        self,
        signal_type: SignalType,
        urgency_score: float,
    ) -> datetime:
        """Calculate SLA expiry deadline.

        Args:
            signal_type: Signal type
            urgency_score: Urgency score (0-1)

        Returns:
            Expiry datetime
        """
        # Base SLA by signal type (in hours)
        base_sla = {
            SignalType.LEAD_OPPORTUNITY: 4,  # Very time-sensitive
            SignalType.CHURN_RISK: 2,  # Critical
            SignalType.SUPPORT_ESCALATION: 1,  # Urgent
            SignalType.COMPETITOR_WEAKNESS: 24,  # Can wait
            SignalType.TREND_TO_CONTENT: 48,  # Less urgent
        }

        hours = base_sla.get(signal_type, 24)

        # Adjust by urgency score
        # Higher urgency = shorter deadline
        adjusted_hours = hours * (1.0 - (urgency_score * 0.5))

        return datetime.utcnow() + timedelta(hours=adjusted_hours)

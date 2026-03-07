"""Response playbook system for generating contextual responses.

This module provides templates and prompt engineering for different signal types.
Each signal type has a tailored template with tone variations and channel constraints.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.signal_models import (
    ActionableSignal,
    ActionType,
    ResponseTone,
    SignalType,
)

logger = logging.getLogger(__name__)


class ResponseChannel(str, Enum):
    """Target channels for responses."""

    TWITTER = "twitter"
    REDDIT = "reddit"
    LINKEDIN = "linkedin"
    EMAIL = "email"
    DM = "dm"


class ResponseTemplate(BaseModel):
    """Template for generating responses to a specific signal type."""

    signal_type: SignalType
    action_type: ActionType
    system_prompt: str
    user_prompt_template: str
    tone_variations: Dict[ResponseTone, str] = Field(default_factory=dict)
    channel_constraints: Dict[ResponseChannel, Dict[str, any]] = Field(default_factory=dict)
    examples: List[str] = Field(default_factory=list)
    max_length: Optional[int] = 1000

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


class ResponsePlaybook:
    """Playbook containing response templates for all signal types.

    This class manages a collection of response templates, each tailored to
    a specific signal type. Templates include system prompts, user prompt templates,
    tone variations, and channel-specific constraints.
    """

    def __init__(self):
        """Initialize playbook with all templates."""
        self.templates: Dict[SignalType, ResponseTemplate] = {}
        self._load_templates()

    def get_template(self, signal_type: SignalType) -> Optional[ResponseTemplate]:
        """Get template for a signal type.

        Args:
            signal_type: Type of signal

        Returns:
            Response template or None if not found
        """
        return self.templates.get(signal_type)

    def _load_templates(self):
        """Load all response templates.

        Creates templates for each signal type with appropriate prompts,
        tone variations, and channel constraints.
        """
        # Lead Opportunity Template
        self.templates[SignalType.LEAD_OPPORTUNITY] = ResponseTemplate(
            signal_type=SignalType.LEAD_OPPORTUNITY,
            action_type=ActionType.REPLY_PUBLIC,
            system_prompt=(
                "You are a helpful product expert responding to someone actively "
                "looking for a solution. Your goal is to be genuinely helpful, "
                "not salesy. Provide value first, product mention second."
            ),
            user_prompt_template=(
                "Respond to this lead opportunity:\n\n"
                "CONTEXT: {context}\n"
                "ORIGINAL POST: {original_content}\n"
                "AUTHOR: {author}\n"
                "PLATFORM: {platform}\n\n"
                "REQUIREMENTS:\n"
                "- Be genuinely helpful and specific\n"
                "- Address their actual needs\n"
                "- Mention relevant features naturally\n"
                "- Include a soft CTA\n"
                "- Max length: {max_length} characters\n"
                "- Tone: {tone}\n"
                "- Format: {channel}\n\n"
                "Generate a {tone} response that converts without being pushy."
            ),
            tone_variations={
                ResponseTone.PROFESSIONAL: "formal and business-focused",
                ResponseTone.HELPFUL: "warm and conversational",
                ResponseTone.TECHNICAL: "detailed and technical",
                ResponseTone.SUPPORTIVE: "understanding and supportive",
                ResponseTone.FOUNDER_VOICE: "confident and expert",
            },
            channel_constraints={
                ResponseChannel.TWITTER: {"max_length": 280, "thread_capable": True},
                ResponseChannel.REDDIT: {"max_length": 2000, "use_markdown": True},
                ResponseChannel.LINKEDIN: {"max_length": 3000, "professional": True},
            },
            examples=[],
            max_length=1500,
        )

        # Competitor Weakness Template
        self.templates[SignalType.COMPETITOR_WEAKNESS] = ResponseTemplate(
            signal_type=SignalType.COMPETITOR_WEAKNESS,
            action_type=ActionType.CREATE_CONTENT,
            system_prompt=(
                "You are a product marketer creating content that addresses "
                "competitor weaknesses. Focus on how your product solves the "
                "specific pain points mentioned, without directly attacking competitors. "
                "Be helpful and solution-oriented."
            ),
            user_prompt_template=(
                "Create content addressing this competitor weakness:\n\n"
                "CONTEXT: {context}\n"
                "COMPLAINT: {original_content}\n"
                "COMPETITOR: {competitor_name}\n"
                "PAIN POINT: {pain_point}\n\n"
                "REQUIREMENTS:\n"
                "- Acknowledge the pain point is real\n"
                "- Explain how we solve it differently\n"
                "- Provide specific examples or features\n"
                "- Avoid negative competitor mentions\n"
                "- Tone: {tone}\n"
                "- Format: {channel}\n\n"
                "Generate {tone} content that positions our solution as the answer."
            ),


            tone_variations={
                ResponseTone.PROFESSIONAL: "professional and factual",
                ResponseTone.HELPFUL: "approachable and helpful",
                ResponseTone.TECHNICAL: "detailed and feature-focused",
                ResponseTone.SUPPORTIVE: "understanding and supportive",
                ResponseTone.FOUNDER_VOICE: "confident and expert",
            },
            channel_constraints={
                ResponseChannel.TWITTER: {"max_length": 280, "thread_capable": True},
                ResponseChannel.REDDIT: {"max_length": 2000, "use_markdown": True},
                ResponseChannel.LINKEDIN: {"max_length": 3000, "professional": True},
            },
            examples=[],
            max_length=2000,
        )

        # Product Confusion Template
        self.templates[SignalType.PRODUCT_CONFUSION] = ResponseTemplate(
            signal_type=SignalType.PRODUCT_CONFUSION,
            action_type=ActionType.REPLY_PUBLIC,
            system_prompt=(
                "You are a product expert helping someone who is confused about "
                "how a product works. Be clear, patient, and educational. "
                "Break down complex concepts into simple explanations."
            ),
            user_prompt_template=(
                "Help clarify this product confusion:\n\n"
                "CONTEXT: {context}\n"
                "CONFUSION: {original_content}\n"
                "AUTHOR: {author}\n"
                "PLATFORM: {platform}\n\n"
                "REQUIREMENTS:\n"
                "- Acknowledge their confusion is valid\n"
                "- Provide clear, simple explanation\n"
                "- Use examples or analogies\n"
                "- Offer to help further\n"
                "- Max length: {max_length} characters\n"
                "- Tone: {tone}\n"
                "- Platform: {platform}\n\n"
                "Generate a {tone} response that resolves their confusion."
            ),
            tone_variations={
                ResponseTone.PROFESSIONAL: "clear and professional",
                ResponseTone.HELPFUL: "warm and patient",
                ResponseTone.TECHNICAL: "detailed and precise",
                ResponseTone.SUPPORTIVE: "understanding and supportive",
                ResponseTone.FOUNDER_VOICE: "confident and expert",
            },
            channel_constraints={
                ResponseChannel.TWITTER: {"max_length": 280, "thread_capable": True},
                ResponseChannel.REDDIT: {"max_length": 1500, "use_markdown": True},
                ResponseChannel.LINKEDIN: {"max_length": 2000, "professional": True},
            },
            examples=[],
            max_length=1500,
        )

        # Churn Risk Template
        self.templates[SignalType.CHURN_RISK] = ResponseTemplate(
            signal_type=SignalType.CHURN_RISK,
            action_type=ActionType.DM_OUTREACH,
            system_prompt=(
                "You are a customer success manager reaching out to a customer "
                "showing signs of churn. Be empathetic, solution-focused, and "
                "genuinely interested in helping them succeed."
            ),
            user_prompt_template=(
                "Create outreach for this churn risk:\n\n"
                "CONTEXT: {context}\n"
                "SIGNAL: {original_content}\n"
                "CUSTOMER: {author}\n\n"
                "REQUIREMENTS:\n"
                "- Show genuine concern\n"
                "- Acknowledge their frustration\n"
                "- Offer specific help or solutions\n"
                "- Make it easy to respond\n"
                "- Tone: {tone}\n"
                "- Format: {channel}\n\n"
                "Generate a {tone} response that prevents churn."
            ),
            tone_variations={
                ResponseTone.PROFESSIONAL: "professional and solution-focused",
                ResponseTone.HELPFUL: "warm and personal",
                ResponseTone.SUPPORTIVE: "understanding and caring",
            },
            channel_constraints={
                ResponseChannel.EMAIL: {"max_length": 500, "subject_line": True},
                ResponseChannel.DM: {"max_length": 300, "conversational": True},
            },
            examples=[],
            max_length=500,
        )

        # Support Escalation Template
        self.templates[SignalType.SUPPORT_ESCALATION] = ResponseTemplate(
            signal_type=SignalType.SUPPORT_ESCALATION,
            action_type=ActionType.INTERNAL_ALERT,
            system_prompt=(
                "You are a support manager creating an internal escalation "
                "for a critical customer issue that has gone public. Be clear, "
                "urgent, and action-oriented."
            ),
            user_prompt_template=(
                "Create internal escalation for:\n\n"
                "CONTEXT: {context}\n"
                "PUBLIC COMPLAINT: {original_content}\n"
                "CUSTOMER: {author}\n"
                "PLATFORM: {platform}\n\n"
                "REQUIREMENTS:\n"
                "- Summarize the issue clearly\n"
                "- Indicate urgency level\n"
                "- Suggest immediate actions\n"
                "- Include customer context\n"
                "- Tone: {tone}\n\n"
                "Generate an internal escalation alert."
            ),
            tone_variations={
                ResponseTone.PROFESSIONAL: "professional and urgent",
                ResponseTone.FOUNDER_VOICE: "directive and clear",
            },
            channel_constraints={
                ResponseChannel.EMAIL: {"max_length": 400, "subject_line": True},
            },
            examples=[],
            max_length=400,
        )

        # Feature Request Pattern Template
        self.templates[SignalType.FEATURE_REQUEST_PATTERN] = ResponseTemplate(
            signal_type=SignalType.FEATURE_REQUEST_PATTERN,
            action_type=ActionType.REPLY_PUBLIC,
            system_prompt=(
                "You are a product manager responding to a feature request. "
                "Be appreciative, transparent about the process, and make the "
                "person feel heard without making promises."
            ),
            user_prompt_template=(
                "Respond to this feature request:\n\n"
                "CONTEXT: {context}\n"
                "REQUEST: {original_content}\n"
                "AUTHOR: {author}\n"
                "PLATFORM: {platform}\n\n"
                "REQUIREMENTS:\n"
                "- Thank them for the suggestion\n"
                "- Explain how requests are evaluated\n"
                "- Ask clarifying questions if helpful\n"
                "- Don't make promises\n"
                "- Tone: {tone}\n"
                "- Format: {channel}\n\n"
                "Generate a {tone} response that acknowledges the request."
            ),

            tone_variations={
                ResponseTone.PROFESSIONAL: "professional and transparent",
                ResponseTone.HELPFUL: "warm and collaborative",
                ResponseTone.TECHNICAL: "detailed and technical",
            },
            channel_constraints={
                ResponseChannel.TWITTER: {"max_length": 280, "thread_capable": True},
                ResponseChannel.REDDIT: {"max_length": 1000, "use_markdown": True},
                ResponseChannel.LINKEDIN: {"max_length": 1500, "professional": True},
            },
            examples=[],
            max_length=1000,
        )

        # Influencer Amplification Template
        self.templates[SignalType.INFLUENCER_AMPLIFICATION] = ResponseTemplate(
            signal_type=SignalType.INFLUENCER_AMPLIFICATION,
            action_type=ActionType.REPLY_PUBLIC,
            system_prompt=(
                "You are a social media manager engaging with an influencer or "
                "high-reach account. Be professional, appreciative, and look for "
                "collaboration opportunities."
            ),
            user_prompt_template=(
                "Engage with this influencer mention:\n\n"
                "CONTEXT: {context}\n"
                "CONTENT: {original_content}\n"
                "AUTHOR: {author}\n"
                "PLATFORM: {platform}\n\n"
                "REQUIREMENTS:\n"
                "- Thank them for the mention\n"
                "- Add value to the conversation\n"
                "- Explore collaboration potential\n"
                "- Be authentic and professional\n"
                "- Tone: {tone}\n"
                "- Format: {channel}\n\n"
                "Generate a {tone} response that builds the relationship."
            ),
            tone_variations={
                ResponseTone.PROFESSIONAL: "professional and insightful",
                ResponseTone.FOUNDER_VOICE: "expert and confident",
                ResponseTone.TECHNICAL: "detailed and analytical",
            },
            channel_constraints={
                ResponseChannel.TWITTER: {"max_length": 280, "thread_capable": True},
                ResponseChannel.REDDIT: {"max_length": 2000, "use_markdown": True},
                ResponseChannel.LINKEDIN: {"max_length": 2000, "professional": True},
            },
            examples=[],
            max_length=1500,
        )

        # Misinformation Risk Template
        self.templates[SignalType.MISINFORMATION_RISK] = ResponseTemplate(
            signal_type=SignalType.MISINFORMATION_RISK,
            action_type=ActionType.REPLY_PUBLIC,
            system_prompt=(
                "You are a brand representative correcting misinformation. "
                "Be factual, respectful, and provide evidence. Never be "
                "defensive or argumentative."
            ),
            user_prompt_template=(
                "Correct this misinformation:\n\n"
                "CONTEXT: {context}\n"
                "MISINFORMATION: {original_content}\n"
                "AUTHOR: {author}\n"
                "PLATFORM: {platform}\n\n"
                "REQUIREMENTS:\n"
                "- Politely acknowledge their post\n"
                "- Provide correct information with evidence\n"
                "- Link to authoritative sources\n"
                "- Remain respectful and professional\n"
                "- Tone: {tone}\n"
                "- Format: {channel}\n\n"
                "Generate a {tone} response that corrects the misinformation."
            ),
            tone_variations={
                ResponseTone.PROFESSIONAL: "professional and factual",
                ResponseTone.HELPFUL: "friendly but firm",
                ResponseTone.FOUNDER_VOICE: "confident and evidence-based",
            },
            channel_constraints={
                ResponseChannel.TWITTER: {"max_length": 280, "thread_capable": True},
                ResponseChannel.REDDIT: {"max_length": 1500, "use_markdown": True},
                ResponseChannel.LINKEDIN: {"max_length": 2000, "professional": True},
            },
            examples=[],
            max_length=1500,
        )

    def build_prompt(
        self,
        signal: ActionableSignal,
        tone: ResponseTone,
        channel: ResponseChannel,
        additional_context: Optional[Dict[str, str]] = None,
    ) -> tuple[str, str]:
        """Build system and user prompts for response generation.

        Args:
            signal: Actionable signal to respond to
            tone: Desired response tone
            channel: Target channel for response
            additional_context: Additional context variables

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        template = self.get_template(signal.signal_type)
        if not template:
            logger.warning("No template found for signal type: %s", signal.signal_type)
            return self._build_generic_prompt(signal, tone, channel)

        # Get channel constraints
        constraints = template.channel_constraints.get(channel, {})
        max_length = constraints.get("max_length", template.max_length or 1000)

        # Build context variables
        context_vars = {
            "context": signal.context or "",
            "original_content": signal.description or "",
            "author": signal.source_author or "Unknown",
            "platform": signal.source_platform or "Unknown",
            "tone": template.tone_variations.get(tone, tone.value),
            "max_length": max_length,
            "channel": channel.value,
        }

        # Add signal-type-specific context from metadata
        if signal.metadata:
            context_vars.update({
                "competitor_name": signal.metadata.get("competitor_name", "the competitor"),
                "pain_point": signal.metadata.get("pain_point", signal.description),
                "product_name": signal.metadata.get("product_name", "the product"),
                "feature_name": signal.metadata.get("feature_name", "this feature"),
            })
        else:
            # Provide defaults for templates that need these
            context_vars.update({
                "competitor_name": "the competitor",
                "pain_point": signal.description or "this issue",
                "product_name": "the product",
                "feature_name": "this feature",
            })

        # Add signal-specific context
        if additional_context:
            context_vars.update(additional_context)

        # Format prompts
        system_prompt = template.system_prompt
        user_prompt = template.user_prompt_template.format(**context_vars)

        return system_prompt, user_prompt

    def _build_generic_prompt(
        self,
        signal: ActionableSignal,
        tone: ResponseTone,
        channel: ResponseChannel,
    ) -> tuple[str, str]:
        """Build a generic prompt when no template is available.

        Args:
            signal: Actionable signal
            tone: Desired tone
            channel: Target channel

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = (
            f"You are responding to a {signal.signal_type.value} signal. "
            f"Be helpful, professional, and appropriate for {channel.value}."
        )

        user_prompt = (
            f"Respond to this signal:\n\n"
            f"CONTEXT: {signal.context}\n"
            f"CONTENT: {signal.description}\n"
            f"TONE: {tone.value}\n\n"
            f"Create an appropriate response for {channel.value}."
        )

        return system_prompt, user_prompt

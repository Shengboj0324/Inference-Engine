"""Domain models package for calibrated inference-and-action system.

This package contains the layered domain model architecture:

Layer 1: Raw Models (raw_models.py)
    - Direct platform data representation
    - Minimal processing
    - Platform-specific fields

Layer 2: Normalized Models (normalized_models.py)
    - NormalizedObservation: Unified cross-platform representation
    - Language normalization, translation
    - Entity extraction, thread context
    - Quality and completeness scores

Layer 3: Inference Models (inference_models.py)
    - SignalInference: ML/LLM interpretation results
    - Calibrated confidence scores
    - Evidence spans and rationale
    - Abstention support
    - Multi-label predictions

Layer 4: Action Models (action_models.py)
    - ActionableSignal: Operational units for human/agent action
    - Priority, opportunity, urgency, risk scores
    - Response plans and draft variants
    - Status tracking and outcome logging

Design Principles:
- Strict contracts at each layer
- No field mutation without validation
- Explicit type safety
- Clear separation of concerns
- Immutable by default, mutable only when needed
"""

from app.domain.raw_models import RawObservation
from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalInference
from app.domain.action_models import ActionableSignal

__all__ = [
    "RawObservation",
    "NormalizedObservation",
    "SignalInference",
    "ActionableSignal",
]


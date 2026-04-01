"""Multi-format output engine for customizable content delivery (Phases 1–6)."""

from app.output.manager import OutputManager
from app.output.models import (
    GeneratedOutput,
    OutputFormat,
    OutputPreferences,
    OutputRequest,
)
from app.output.digest_modes import (
    BriefItem,
    DeepDiveResult,
    DeliveryMode,
    DigestModeRouter,
    MorningBrief,
    PersonalizedStream,
    PersonalizedStreamItem,
    WatchlistDigest,
    WatchlistEntry,
)

__all__ = [
    # existing
    "OutputManager",
    "GeneratedOutput",
    "OutputFormat",
    "OutputPreferences",
    "OutputRequest",
    # Phase 6 digest modes
    "BriefItem",
    "DeepDiveResult",
    "DeliveryMode",
    "DigestModeRouter",
    "MorningBrief",
    "PersonalizedStream",
    "PersonalizedStreamItem",
    "WatchlistDigest",
    "WatchlistEntry",
]

"""Incident-response intelligence pipeline.

Companion to :mod:`app.intelligence.crisis`.  Where ``crisis`` handles
public-perception controversies (Releaf-style), ``incident`` handles
security/data-exposure events with a SEV-0..4 scale, multi-stage
severity progression, internal verification ingestion, and the
12-section incident-intelligence report required by the NeuroBridge
mock-user evaluation rubric.
"""

from app.intelligence.incident.registries import (  # noqa: F401
    build_default_registries,
)
from app.intelligence.incident.severity import (  # noqa: F401
    Severity,
    SeverityProgression,
    SeverityStage,
)
from app.intelligence.incident.stream_parser import (  # noqa: F401
    IncidentStreamItem,
    parse_incident_stream,
)
from app.intelligence.incident.signal_detectors import (  # noqa: F401
    BulkPattern,
    CsvSampleSignal,
    IncidentSignals,
    ScreenshotManipulation,
    analyze_incident_signals,
)
from app.intelligence.incident.internal_update import (  # noqa: F401
    InternalVerificationUpdate,
    extract_internal_update_block,
    parse_internal_update,
)
from app.intelligence.incident.report_models import (  # noqa: F401
    ConfirmedFact,
    EvidenceMatrixRow,
    IncidentActorEntry,
    IncidentActionItem,
    IncidentIntelligenceReport,
    SeverityRow,
    StakeholderImpact,
    TimelineEntry,
    UnsupportedClaim,
)
from app.intelligence.incident.report_builder import (  # noqa: F401
    IncidentReportBuilder,
)
from app.intelligence.incident.report_renderer import (  # noqa: F401
    render_incident_report_markdown,
)

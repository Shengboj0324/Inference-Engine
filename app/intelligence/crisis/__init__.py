"""Crisis intelligence pipeline.

Deterministic upstream analysis layer that turns a raw multi-platform
content stream into a fully-structured ``CrisisIntelligenceReport``.

The LLM is only used for a short executive-summary prose; all detection
(coordinated amplification, sarcasm, false consensus, screenshot
conflict, misquotation, version confusion, multilingual clustering,
evidence scoring, actor/claim attribution) runs in deterministic Python
so the output is stable, testable, and not gated on model size.
"""

from app.intelligence.crisis.registries import (  # noqa: F401
    DebunkedClaim,
    DebunkedClaimsRegistry,
    KnownActor,
    KnownActorRegistry,
    KnownIssue,
    KnownIssuesRegistry,
    build_default_registries,
)
from app.intelligence.crisis.stream_parser import (  # noqa: F401
    StreamItem,
    parse_stream,
)
from app.intelligence.crisis.signal_detectors import (  # noqa: F401
    AmplificationCluster,
    ConsensusCheck,
    LanguageCluster,
    MisquotationFlag,
    SarcasmFlag,
    ScreenshotConflict,
    SignalAnalysis,
    VersionGate,
    analyze_signals,
)
from app.intelligence.crisis.evidence_scorer import (  # noqa: F401
    EvidenceScore,
    EvidenceStrengthCalculator,
)
from app.intelligence.crisis.report_models import (  # noqa: F401
    ActionItem,
    ActorEntry,
    ClaimVerdict,
    CrisisIntelligenceReport,
    NarrativeStage,
    ReplyQueueEntry,
    RiskEntry,
)
from app.intelligence.crisis.report_builder import (  # noqa: F401
    CrisisReportBuilder,
)
from app.intelligence.crisis.report_renderer import (  # noqa: F401
    render_report_markdown,
)

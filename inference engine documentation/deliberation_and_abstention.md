# DeliberationEngine and AbstentionDecider

## DeliberationEngine (E6)

`DeliberationEngine` (`app/intelligence/deliberation_engine.py`) runs four deterministic steps between `CandidateRetriever` output and the `LLMAdjudicator` call. Its purpose is to reduce unnecessary frontier-tier LLM calls and improve pre-adjudication context quality.

### Step A — Historical Landscape Scan

Queries `ContextMemoryStore` for the five most similar past `(observation, inference)` pairs for the current user via cosine similarity over the 1536-dim embedding. The retrieved pairs are injected into the `LLMAdjudicator` prompt as few-shot context before any new generation begins.

### Step B — Candidate Pruning

Removes `SignalCandidate` entries where retrieval score < `min_candidate_score` **and** no similar historical observation exists in the ContextMemoryStore for that signal type. A safety net ensures the candidate list is never emptied: the highest-scoring candidate is always retained.

### Step C — Risk Escalation

Before adjudication, any candidate in `_FRONTIER_SIGNAL_TYPES` (`churn_risk`, `legal_risk`, `security_concern`, `reputation_risk`) with a retrieval score > 0.5 writes a structured `RiskEscalationEntry` to the audit logger. This fires synchronously so the audit log entry is always present before the LLM call completes.

### Step D — Reasoning Mode Selection

```
len(text) > 1500  OR  len(candidates) > 6
    → ReasoningMode.MULTI_AGENT

top_two_score_gap < 0.1  OR  confidence_required > 0.85
    → ReasoningMode.CHAIN_OF_THOUGHT

else
    → ReasoningMode.SINGLE_CALL
```

`confidence_required` is a per-observation override that callers can set to force higher-quality reasoning at increased latency and cost.

### API

```python
from app.intelligence.deliberation_engine import DeliberationEngine

engine = DeliberationEngine(
    context_memory_store=ctx_store,
    min_candidate_score=0.3,
    risk_signal_types=frozenset({SignalType.CHURN_RISK, SignalType.LEGAL_RISK, ...}),
)

deliberation_result = await engine.deliberate(
    observation=normalized_obs,
    candidates=candidate_list,
    confidence_required=0.7,
)
# deliberation_result.reasoning_mode: ReasoningMode
# deliberation_result.pruned_candidates: List[SignalCandidate]
# deliberation_result.few_shot_context: List[HistoricalObservation]
```

---

## AbstentionDecider (Stage D)

`AbstentionDecider` (`app/intelligence/abstention.py`) is the final gate before any signal enters the queue. It applies a configurable post-calibration confidence threshold; observations that do not meet the threshold are logged as typed abstentions and never written to `ActionableSignalDB`.

### Abstention Reasons

```python
class AbstentionReason(str, Enum):
    LOW_CONFIDENCE         = "low_confidence"           # p_cal < threshold
    AMBIGUOUS_MULTI_LABEL  = "ambiguous_multi_label"    # top-2 scores within 0.05
    INSUFFICIENT_CONTEXT   = "insufficient_context"     # < min_token_count tokens post-scrub
    OUT_OF_DISTRIBUTION    = "out_of_distribution"      # cosine sim to nearest exemplar < 0.35
    UNSAFE_TO_CLASSIFY     = "unsafe_to_classify"       # content matches safety blocklist
    LANGUAGE_BARRIER       = "language_barrier"         # detected language confidence < 0.6
    SPAM_OR_NOISE          = "spam_or_noise"            # engagement_score < floor AND length < 20 tokens
```

### `AbstentionRecord`

```python
@dataclass
class AbstentionRecord:
    observation_id: UUID
    reason: AbstentionReason
    predicted_type: SignalType
    calibrated_confidence: float
    threshold_applied: float
    abstained_at: datetime      # UTC
```

Records are written to `abstention_log` (PostgreSQL). They are excluded from `ContextMemoryStore` — abstained inferences are never used as few-shot context in future observations.

### Threshold Configuration

```python
# Global default (applies to all signal types not individually configured)
ABSTENTION_THRESHOLD = 0.7

# Per-observation override (set by DeliberationEngine callers)
confidence_required: float = 0.85   # e.g., for risk-type escalations

# Per-type override in settings
ABSTENTION_THRESHOLDS = {
    "churn_risk": 0.75,
    "legal_risk": 0.80,
    "feature_request": 0.65,
}
```

### Relationship to CalibrationConfidence

`AbstentionDecider` reads `p_cal` output from `ConfidenceCalibrator` (`sigmoid(logit / T_eff)`), not the raw LLM logit. This means the threshold is applied against a calibrated probability whose ECE has been reduced by temperature scaling — not against a raw score that may be systematically over- or under-confident.

---

## Data Flow

```
CandidateRetriever → List[SignalCandidate]
        │
        ▼
DeliberationEngine
  Step A: inject few-shot context
  Step B: prune low-score candidates
  Step C: write audit entry for risk types
  Step D: select ReasoningMode
        │
   ┌────┴────────────┐
   ▼                  ▼
E1 CoT            E3 MultiAgent
   └────────┬─────────┘
            ▼
  LLMAdjudicator → raw logit
            │
            ▼
  ConfidenceCalibrator → p_cal
            │
            ▼
  AbstentionDecider
    p_cal ≥ threshold → ActionableSignal → PostgreSQL → signal queue
    p_cal <  threshold → AbstentionRecord → abstention_log
```


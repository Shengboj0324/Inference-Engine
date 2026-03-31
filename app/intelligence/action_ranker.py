"""Action ranking system for prioritizing actionable signals.

This module implements multi-dimensional action ranking:
- Priority scoring (overall priority)
- Opportunity scoring (business value potential)
- Urgency scoring (time sensitivity)
- Risk scoring (risk if not addressed)

Combines multiple signals to produce a calibrated priority score.
"""

import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Dict, Optional, Set, Tuple
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalInference, SignalType
from app.domain.action_models import ActionableSignal, ActionPriority, ActionStatus, ResponseChannel
from app.core.models import SourcePlatform

logger = logging.getLogger(__name__)

#: Default path for the ``RankerConfig`` JSON file.
_DEFAULT_RANKER_CONFIG_PATH: Path = Path("training/ranker_config.json")

# ---------------------------------------------------------------------------
# TrendTracker — cross-user signal-type rate monitor (Step 3c)
# ---------------------------------------------------------------------------

#: Urgency boost applied to trending signal types (clamped to 1.0 after).
_TREND_URGENCY_BOOST: float = 0.15
#: Multiplier threshold: if last-24h rate ≥ multiplier × 7-day daily average,
#: the signal type is declared TRENDING.
_TREND_MULTIPLIER: float = 3.0
#: Observation window for "current rate" (seconds = 24 h).
_TREND_CURRENT_WINDOW_S: float = 86_400.0
#: Rolling baseline window (seconds = 7 days).
_TREND_BASELINE_WINDOW_S: float = 7 * 86_400.0


class TrendTracker:
    """Thread-safe monitor that detects when a ``SignalType`` is trending.

    Records a **wall-clock** ``time.time()`` timestamp each time a signal type
    is classified by any user.  Using wall-clock time (rather than monotonic
    time) means trend windows are meaningful across process restarts and survive
    sleep/hibernate cycles.

    Declares a type **TRENDING** when its count in the last 24 hours is ≥
    ``_TREND_MULTIPLIER`` × its 7-day rolling daily average **and** the total
    count in the 7-day window is ≥ ``min_baseline_count`` (cold-start guard).

    Provides :meth:`get_trending_scores` returning a graded ratio so downstream
    components can apply continuous boosting rather than a binary flag.

    All public methods are thread-safe via a single ``threading.Lock``.

    Args:
        min_baseline_count: Minimum number of occurrences in the 7-day window
            before a signal type can be declared trending.  Prevents
            false-positives on cold-start data (e.g. a single burst right
            after deployment).  Defaults to ``5``.
    """

    def __init__(self, min_baseline_count: int = 5) -> None:
        """Initialise an empty tracker.

        Args:
            min_baseline_count: Minimum 7-day count before trending is allowed.
        """
        if min_baseline_count < 0:
            raise ValueError(f"min_baseline_count must be ≥ 0; got {min_baseline_count!r}")
        # signal_type.value → deque of float timestamps (wall-clock seconds)
        self._timestamps: Dict[str, Deque[float]] = {}
        self._lock: threading.Lock = threading.Lock()
        self._min_baseline_count: int = min_baseline_count

    def record(self, signal_type: SignalType) -> None:
        """Record a new occurrence of ``signal_type`` at the current wall-clock time.

        Thread-safe.  Old timestamps (> 7 days) are pruned on each record call
        to keep memory bounded.

        Args:
            signal_type: Signal type that was just classified.
        """
        key = signal_type.value
        now = time.time()  # wall-clock — survives restarts / sleep
        cutoff = now - _TREND_BASELINE_WINDOW_S
        with self._lock:
            if key not in self._timestamps:
                self._timestamps[key] = deque()
            ts = self._timestamps[key]
            ts.append(now)
            # Prune entries older than the baseline window
            while ts and ts[0] < cutoff:
                ts.popleft()

    def _trending_score(
        self,
        key: str,
        now: float,
        multiplier: float,
    ) -> float:
        """Return the raw trending ratio for ``key`` (internal helper).

        The ratio is ``count_24h / (multiplier * avg_per_day_7d)``.  A ratio
        ≥ 1.0 means the type is trending; below 1.0 it is not.  Returns ``0.0``
        when there are no 7-day records or the cold-start guard fails.

        Must be called with ``self._lock`` already held if ``now`` is used for
        consistent snapshot semantics; the caller is responsible for lock
        management when iterating over keys.
        """
        ts = self._timestamps.get(key, deque())
        cutoff_24h = now - _TREND_CURRENT_WINDOW_S
        cutoff_7d = now - _TREND_BASELINE_WINDOW_S
        count_24h = sum(1 for t in ts if t >= cutoff_24h)
        count_7d = sum(1 for t in ts if t >= cutoff_7d)
        if count_7d < max(1, self._min_baseline_count):
            return 0.0
        avg_per_day_7d = count_7d / 7.0
        denominator = multiplier * avg_per_day_7d
        if denominator < 1e-9:
            return 0.0
        return count_24h / denominator

    def is_trending(
        self,
        signal_type: SignalType,
        multiplier: float = _TREND_MULTIPLIER,
    ) -> bool:
        """Return ``True`` if ``signal_type`` is currently trending.

        A type is trending when:

        1. Its count in the last 24 h is ≥ ``multiplier × 7d_daily_average``.
        2. Its total 7-day count is ≥ ``min_baseline_count`` (cold-start guard).

        Args:
            signal_type: Signal type to test.
            multiplier: Trending multiplier threshold (default 3.0).

        Returns:
            ``True`` if trending, ``False`` otherwise.
        """
        now = time.time()
        with self._lock:
            return self._trending_score(signal_type.value, now, multiplier) >= 1.0

    def get_trending_types(
        self, multiplier: float = _TREND_MULTIPLIER
    ) -> Set[str]:
        """Return the set of ``SignalType.value`` strings that are currently trending.

        Args:
            multiplier: Trending multiplier threshold.

        Returns:
            Set of signal type value strings.
        """
        now = time.time()
        with self._lock:
            keys = list(self._timestamps.keys())
            return {
                k for k in keys
                if self._trending_score(k, now, multiplier) >= 1.0
            }

    def get_trending_scores(
        self, multiplier: float = _TREND_MULTIPLIER
    ) -> Dict[str, float]:
        """Return a graded trending ratio for every tracked signal type.

        The ratio is ``count_24h / (multiplier × 7d_daily_average)``, clamped
        to ``[0.0, ∞)``.  A ratio ≥ 1.0 means the type is trending.  Downstream
        components can use this ratio for proportional boosting rather than the
        binary flag from :meth:`is_trending`.

        Signal types that fail the ``min_baseline_count`` cold-start guard have
        a ratio of ``0.0``.

        Args:
            multiplier: Trending multiplier threshold (denominator scale).

        Returns:
            Dict mapping ``SignalType.value`` → trending ratio (float ≥ 0.0).
        """
        now = time.time()
        with self._lock:
            keys = list(self._timestamps.keys())
            scores: Dict[str, float] = {
                k: self._trending_score(k, now, multiplier) for k in keys
            }
        logger.debug(
            "TrendTracker.get_trending_scores: tracked=%d trending=%d",
            len(scores),
            sum(1 for v in scores.values() if v >= 1.0),
        )
        return scores

    def clear(self) -> None:
        """Remove all recorded timestamps (used in tests)."""
        with self._lock:
            self._timestamps.clear()


#: Module-level singleton shared across all pipeline instances.
_GLOBAL_TREND_TRACKER: TrendTracker = TrendTracker()


class RankerConfig(BaseModel):
    """All tunable numeric thresholds and weights for :class:`ActionRanker`.

    Externalising these values means operators can retune the ranker through
    a config file edit + service reload, without touching source code.

    Weight fields
    -------------
    ``opportunity_weight``, ``urgency_weight``, ``risk_weight`` must sum to 1.0.
    ``ActionRanker.__init__`` normalises them if they do not.

    Boost / penalty fields
    ----------------------
    All ``_boost`` and ``_penalty`` values are *additive* adjustments applied
    after the base score is fetched from the dispatch table.  Scores are
    clamped to [0.0, 1.0] after each adjustment.

    Priority threshold fields
    -------------------------
    ``critical_threshold`` ≥ ``high_threshold`` ≥ ``medium_threshold`` ≥
    ``low_threshold`` must hold.  ActionRanker does not enforce this at
    construction time but scores will be mis-bucketed if the invariant is
    violated.
    """

    # ── Combination weights ───────────────────────────────────────────────
    opportunity_weight: float = Field(0.35, ge=0.0, le=1.0,
        description="Weight of the opportunity dimension in priority score.")
    urgency_weight: float = Field(0.30, ge=0.0, le=1.0,
        description="Weight of the urgency dimension in priority score.")
    risk_weight: float = Field(0.35, ge=0.0, le=1.0,
        description="Weight of the risk dimension in priority score.")

    # ── Confidence gate ───────────────────────────────────────────────────
    min_confidence_threshold: float = Field(0.5, ge=0.0, le=1.0,
        description="Minimum signal confidence to produce an ActionableSignal.")

    # ── Priority level thresholds ─────────────────────────────────────────
    # Rationale: four bands give operators enough granularity without creating
    # too many queues for human reviewers to manage.
    critical_threshold: float = Field(0.80, ge=0.0, le=1.0,
        description="Priority score ≥ this → CRITICAL.")
    high_threshold: float = Field(0.60, ge=0.0, le=1.0,
        description="Priority score ≥ this → HIGH.")
    medium_threshold: float = Field(0.40, ge=0.0, le=1.0,
        description="Priority score ≥ this → MEDIUM.")
    low_threshold: float = Field(0.20, ge=0.0, le=1.0,
        description="Priority score ≥ this → LOW; below → MONITOR.")

    # ── Opportunity dimension boosts ──────────────────────────────────────
    opp_velocity_threshold: float = Field(10.0, ge=0.0,
        description="engagement_velocity above this triggers an opportunity boost.")
    opp_velocity_boost: float = Field(0.10, ge=0.0, le=1.0,
        description="Additive boost when engagement_velocity > opp_velocity_threshold.")
    opp_virality_threshold: float = Field(0.5, ge=0.0, le=1.0,
        description="virality_score above this triggers an opportunity boost.")
    opp_virality_boost: float = Field(0.10, ge=0.0, le=1.0,
        description="Additive boost when virality_score > opp_virality_threshold.")

    # ── Urgency dimension boosts / penalties ─────────────────────────────
    # Freshness thresholds: content < 1 h old is very fresh; > 48 h is stale.
    urg_fresh_hours_high: float = Field(1.0, ge=0.0,
        description="Content younger than this (hours) gets urg_fresh_boost_high.")
    urg_fresh_boost_high: float = Field(0.20, ge=0.0, le=1.0,
        description="Urgency boost for very fresh content (< urg_fresh_hours_high).")
    urg_fresh_hours_medium: float = Field(6.0, ge=0.0,
        description="Content younger than this gets urg_fresh_boost_medium.")
    urg_fresh_boost_medium: float = Field(0.10, ge=0.0, le=1.0,
        description="Urgency boost for moderately fresh content.")
    urg_stale_hours: float = Field(48.0, ge=0.0,
        description="Content older than this (hours) gets urg_stale_penalty.")
    urg_stale_penalty: float = Field(0.20, ge=0.0, le=1.0,
        description="Urgency penalty for stale content (> urg_stale_hours).")
    urg_velocity_threshold: float = Field(20.0, ge=0.0,
        description="engagement_velocity above this triggers an urgency boost.")
    urg_velocity_boost: float = Field(0.10, ge=0.0, le=1.0,
        description="Additive urgency boost when velocity > urg_velocity_threshold.")

    # ── Risk dimension boosts ─────────────────────────────────────────────
    risk_public_platform_boost: float = Field(0.10, ge=0.0, le=1.0,
        description="Risk boost for content on public platforms (Reddit, YouTube …).")
    risk_virality_threshold: float = Field(0.70, ge=0.0, le=1.0,
        description="virality_score above this triggers a risk boost.")
    risk_virality_boost: float = Field(0.15, ge=0.0, le=1.0,
        description="Additive risk boost for high-virality content.")

    @classmethod
    def from_file(cls, path: Path) -> "RankerConfig":
        """Load a ``RankerConfig`` from a JSON file.

        Fields not present in the file receive their class-level defaults.

        Args:
            path: Path to the JSON config file.

        Returns:
            ``RankerConfig`` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the JSON cannot be parsed or fails Pydantic
                validation.
        """
        if not path.exists():
            raise FileNotFoundError(f"RankerConfig file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"Failed to parse RankerConfig from {path}: {exc}") from exc


class ActionRanker:
    """Multi-dimensional action ranking system.

    Scores signals across 4 dimensions:
    1. Opportunity: Business value potential
    2. Urgency: Time sensitivity
    3. Risk: Risk if not addressed
    4. Priority: Overall priority (weighted combination)

    Dispatch tables (``_OPPORTUNITY_MAP``, ``_URGENCY_MAP``, ``_RISK_MAP``,
    ``_CHANNEL_MAP``) replace if/elif chains for O(1) lookup and zero-friction
    extensibility — add a new ``SignalType`` in one place only.
    """

    # ------------------------------------------------------------------
    # Dispatch tables — O(1) lookup, easy to extend
    # ------------------------------------------------------------------

    # Opportunity scores (base, before engagement boosts)
    _OPPORTUNITY_MAP: Dict[SignalType, float] = {
        SignalType.ALTERNATIVE_SEEKING: 0.9,
        SignalType.COMPETITOR_MENTION: 0.9,
        SignalType.EXPANSION_OPPORTUNITY: 0.9,
        SignalType.UPSELL_OPPORTUNITY: 0.9,
        SignalType.PARTNERSHIP_OPPORTUNITY: 0.9,
        SignalType.FEATURE_REQUEST: 0.7,
        SignalType.INTEGRATION_REQUEST: 0.7,
        SignalType.PRICE_SENSITIVITY: 0.7,
        SignalType.SUPPORT_REQUEST: 0.4,
        SignalType.BUG_REPORT: 0.4,
        SignalType.COMPLAINT: 0.4,
    }
    _OPPORTUNITY_DEFAULT = 0.5

    # Urgency scores (base, before freshness/velocity boosts)
    _URGENCY_MAP: Dict[SignalType, float] = {
        SignalType.CHURN_RISK: 0.9,
        SignalType.SECURITY_CONCERN: 0.9,
        SignalType.LEGAL_RISK: 0.9,
        SignalType.REPUTATION_RISK: 0.9,
        SignalType.COMPLAINT: 0.7,
        SignalType.BUG_REPORT: 0.7,
        SignalType.ALTERNATIVE_SEEKING: 0.7,
        SignalType.PRAISE: 0.3,
        SignalType.FEATURE_REQUEST: 0.3,
    }
    _URGENCY_DEFAULT = 0.5

    # Risk scores (base, before platform/virality boosts)
    _RISK_MAP: Dict[SignalType, float] = {
        SignalType.CHURN_RISK: 0.95,
        SignalType.SECURITY_CONCERN: 0.95,
        SignalType.LEGAL_RISK: 0.95,
        SignalType.REPUTATION_RISK: 0.95,
        SignalType.COMPLAINT: 0.6,
        SignalType.COMPETITOR_MENTION: 0.6,
        SignalType.ALTERNATIVE_SEEKING: 0.6,
        SignalType.PRAISE: 0.2,
        SignalType.FEATURE_REQUEST: 0.2,
        SignalType.SUPPORT_REQUEST: 0.2,
    }
    _RISK_DEFAULT = 0.3

    # Channel dispatch table
    _CHANNEL_MAP: Dict[SignalType, "ResponseChannel"] = {}  # populated after imports

    def __init__(
        self,
        config: Optional[RankerConfig] = None,
        config_path: Optional[Path] = None,
        # Legacy keyword-argument shims — kept for backward compatibility.
        # Prefer passing a RankerConfig instance.
        opportunity_weight: Optional[float] = None,
        urgency_weight: Optional[float] = None,
        risk_weight: Optional[float] = None,
        min_confidence_threshold: Optional[float] = None,
    ):
        """Initialize action ranker.

        Configuration precedence (highest to lowest):
        1. Explicit ``config`` kwarg.
        2. JSON file at ``config_path`` (or ``training/ranker_config.json``).
        3. Legacy scalar kwargs (``opportunity_weight`` etc.) applied on top
           of the default ``RankerConfig`` values.
        4. ``RankerConfig`` class-level defaults.

        Args:
            config: Pre-built ``RankerConfig`` instance.
            config_path: Path to a JSON ranker config file.  Defaults to
                ``training/ranker_config.json`` when the file exists.
            opportunity_weight: Opportunity dimension weight (legacy kwarg).
            urgency_weight: Urgency dimension weight (legacy kwarg).
            risk_weight: Risk dimension weight (legacy kwarg).
            min_confidence_threshold: Minimum prediction confidence to produce
                an ``ActionableSignal`` (legacy kwarg).
        """
        if config is not None:
            self.config = config
        else:
            # Try loading from file
            _path = config_path or _DEFAULT_RANKER_CONFIG_PATH
            if _path.exists():
                try:
                    self.config = RankerConfig.from_file(_path)
                    logger.info("ActionRanker: loaded RankerConfig from %s", _path)
                except (FileNotFoundError, ValueError) as exc:
                    logger.warning(
                        "ActionRanker: could not load config from %s: %s; using defaults",
                        _path, exc,
                    )
                    self.config = RankerConfig()
            else:
                self.config = RankerConfig()

        # Apply legacy scalar overrides on top of the loaded/default config
        overrides: dict = {}
        if opportunity_weight is not None:
            overrides["opportunity_weight"] = opportunity_weight
        if urgency_weight is not None:
            overrides["urgency_weight"] = urgency_weight
        if risk_weight is not None:
            overrides["risk_weight"] = risk_weight
        if min_confidence_threshold is not None:
            overrides["min_confidence_threshold"] = min_confidence_threshold
        if overrides:
            self.config = self.config.model_copy(update=overrides)

        # Validate weights sum to 1.0; normalise if not
        total_weight = (
            self.config.opportunity_weight
            + self.config.urgency_weight
            + self.config.risk_weight
        )
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                "ActionRanker: weights sum to %.4f (expected 1.0); normalising",
                total_weight,
            )
            self.config = self.config.model_copy(update={
                "opportunity_weight": self.config.opportunity_weight / total_weight,
                "urgency_weight": self.config.urgency_weight / total_weight,
                "risk_weight": self.config.risk_weight / total_weight,
            })

        # Initialise dispatch table eagerly (idempotent)
        self._init_dispatch()

        logger.info(
            "ActionRanker initialised: opp=%.2f urg=%.2f risk=%.2f min_conf=%.2f",
            self.config.opportunity_weight,
            self.config.urgency_weight,
            self.config.risk_weight,
            self.config.min_confidence_threshold,
        )
    
    def rank_action(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
        signal_type_weights: Optional[Dict[str, float]] = None,
        trending_types: Optional[Set[str]] = None,
    ) -> Optional[ActionableSignal]:
        """Rank a signal inference and create an actionable signal if worthy.

        Extends the base scoring with two personalisation/collective levers:

        * **Per-user signal-type weight vector** — ``signal_type_weights`` maps
          ``SignalType.value`` → float in ``[0.0, 1.0]``.  The computed
          ``priority_score`` is multiplied by this weight so that signal types
          the user has frequently dismissed (weight close to 0) rise less often
          in their dashboard.

        * **Trending signal amplification** — if the inference's ``signal_type``
          appears in ``trending_types`` (from ``TrendTracker.get_trending_types()``),
          the ``urgency_score`` receives a ``+_TREND_URGENCY_BOOST`` boost
          (clamped to 1.0) before the weighted sum is computed, and
          ``inference_metadata["trending"]`` is set to ``True``.

        Args:
            inference: Calibrated signal inference from Phase 2.
            observation: Normalised observation.
            signal_type_weights: Optional per-user weight vector sourced from
                ``ContextMemoryStore.get_signal_type_weights()``.  ``None``
                means "no personalisation" (equivalent to all weights = 1.0).
            trending_types: Set of ``SignalType.value`` strings declared
                TRENDING by ``TrendTracker.get_trending_types()``.  ``None``
                skips trending amplification.

        Returns:
            ``ActionableSignal`` if the signal is worthy of action, else ``None``.
        """
        # Check if inference is abstained
        if inference.abstained:
            logger.debug(f"Skipping abstained inference {inference.id}")
            return None

        # Check if confidence meets threshold
        if not inference.top_prediction:
            logger.debug(f"No top prediction for inference {inference.id}")
            return None

        if inference.top_prediction.probability < self.config.min_confidence_threshold:
            logger.debug(
                "Confidence %.2f below threshold %.2f",
                inference.top_prediction.probability,
                self.config.min_confidence_threshold,
            )
            return None

        st_value = inference.top_prediction.signal_type.value

        # ── Trend amplification (Step 3c) ─────────────────────────────────────
        is_trending = trending_types is not None and st_value in trending_types
        if is_trending:
            inference.inference_metadata["trending"] = True

        # Compute scores
        opportunity_score = self._compute_opportunity_score(inference, observation)
        urgency_score = self._compute_urgency_score(inference, observation)
        risk_score = self._compute_risk_score(inference, observation)

        # Apply trending urgency boost BEFORE weighted sum
        if is_trending:
            urgency_score = min(1.0, urgency_score + _TREND_URGENCY_BOOST)

        # Compute overall priority score
        priority_score = (
            self.config.opportunity_weight * opportunity_score
            + self.config.urgency_weight * urgency_score
            + self.config.risk_weight * risk_score
        )

        # ── Per-user signal-type weight (Step 2b) ────────────────────────────
        if signal_type_weights:
            user_weight = signal_type_weights.get(st_value, 1.0)
            priority_score = float(max(0.0, min(1.0, priority_score * user_weight)))

        # Determine priority level
        priority = self._determine_priority_level(priority_score)

        # Determine recommended channel
        recommended_channel = self._determine_channel(inference, observation)

        # Create actionable signal
        action = ActionableSignal(
            signal_inference_id=inference.id,
            normalized_observation_id=observation.id,
            user_id=observation.user_id,
            signal_type=inference.top_prediction.signal_type,
            signal_confidence=inference.top_prediction.probability,
            priority=priority,
            priority_score=priority_score,
            opportunity_score=opportunity_score,
            urgency_score=urgency_score,
            risk_score=risk_score,
            recommended_channel=recommended_channel,
            status=ActionStatus.NEW,
        )

        logger.info(
            "Created action %s: type=%s priority=%s score=%.2f"
            "%s%s",
            action.id,
            action.signal_type.value,
            priority.value,
            priority_score,
            " [TRENDING]" if is_trending else "",
            f" [weight={signal_type_weights.get(st_value, 1.0):.2f}]"
            if signal_type_weights else "",
        )

        return action
    
    def rank_batch(
        self,
        inferences: List[SignalInference],
        observations: Dict[str, NormalizedObservation],
    ) -> List[ActionableSignal]:
        """Rank a batch of inferences.
        
        Args:
            inferences: List of signal inferences
            observations: Dict mapping observation ID to observation
            
        Returns:
            List of actionable signals, sorted by priority score
        """
        actions = []
        
        for inference in inferences:
            observation = observations.get(str(inference.normalized_observation_id))
            if not observation:
                logger.warning(
                    f"No observation found for inference {inference.id}"
                )
                continue
            
            action = self.rank_action(inference, observation)
            if action:
                actions.append(action)
        
        # Sort by priority score (descending)
        actions.sort(key=lambda a: a.priority_score, reverse=True)
        
        logger.info(f"Ranked {len(actions)} actions from {len(inferences)} inferences")
        return actions

    def _compute_opportunity_score(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> float:
        """Compute business opportunity score using the dispatch table.

        Args:
            inference: Calibrated signal inference.
            observation: Normalised observation.

        Returns:
            Opportunity score in [0.0, 1.0].
        """
        if not inference.top_prediction:
            return self._OPPORTUNITY_DEFAULT

        score = self._OPPORTUNITY_MAP.get(
            inference.top_prediction.signal_type, self._OPPORTUNITY_DEFAULT
        )

        # Engagement boosts (additive, clamped to 1.0)
        if (
            observation.engagement_velocity is not None
            and observation.engagement_velocity > self.config.opp_velocity_threshold
        ):
            score = min(1.0, score + self.config.opp_velocity_boost)
        if (
            observation.virality_score is not None
            and observation.virality_score > self.config.opp_virality_threshold
        ):
            score = min(1.0, score + self.config.opp_virality_boost)

        return score

    def _compute_urgency_score(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> float:
        """Compute time-sensitivity score using the dispatch table.

        Args:
            inference: Calibrated signal inference.
            observation: Normalised observation.

        Returns:
            Urgency score in [0.0, 1.0].
        """
        if not inference.top_prediction:
            return self._URGENCY_DEFAULT

        score = self._URGENCY_MAP.get(
            inference.top_prediction.signal_type, self._URGENCY_DEFAULT
        )

        # Freshness boost/penalty
        if observation.published_at:
            hours_old = (
                datetime.now(timezone.utc) - observation.published_at
            ).total_seconds() / 3600
            if hours_old < self.config.urg_fresh_hours_high:
                score = min(1.0, score + self.config.urg_fresh_boost_high)
            elif hours_old < self.config.urg_fresh_hours_medium:
                score = min(1.0, score + self.config.urg_fresh_boost_medium)
            elif hours_old > self.config.urg_stale_hours:
                score = max(0.0, score - self.config.urg_stale_penalty)

        if (
            observation.engagement_velocity is not None
            and observation.engagement_velocity > self.config.urg_velocity_threshold
        ):
            score = min(1.0, score + self.config.urg_velocity_boost)

        return score

    def _compute_risk_score(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> float:
        """Compute risk-if-unaddressed score using the dispatch table.

        Args:
            inference: Calibrated signal inference.
            observation: Normalised observation.

        Returns:
            Risk score in [0.0, 1.0].
        """
        if not inference.top_prediction:
            return self._RISK_DEFAULT

        score = self._RISK_MAP.get(
            inference.top_prediction.signal_type, self._RISK_DEFAULT
        )

        # Public platform visibility boost
        _PUBLIC_PLATFORMS = {
            SourcePlatform.REDDIT, SourcePlatform.YOUTUBE,
            SourcePlatform.TIKTOK, SourcePlatform.FACEBOOK, SourcePlatform.INSTAGRAM,
        }
        if observation.source_platform in _PUBLIC_PLATFORMS:
            score = min(1.0, score + self.config.risk_public_platform_boost)

        if (
            observation.virality_score is not None
            and observation.virality_score > self.config.risk_virality_threshold
        ):
            score = min(1.0, score + self.config.risk_virality_boost)

        return score

    def _determine_priority_level(self, priority_score: float) -> ActionPriority:
        """Determine priority level from score.

        Args:
            priority_score: Overall priority score

        Returns:
            ActionPriority enum
        """
        if priority_score >= self.config.critical_threshold:
            return ActionPriority.CRITICAL
        elif priority_score >= self.config.high_threshold:
            return ActionPriority.HIGH
        elif priority_score >= self.config.medium_threshold:
            return ActionPriority.MEDIUM
        elif priority_score >= self.config.low_threshold:
            return ActionPriority.LOW
        else:
            return ActionPriority.MONITOR

    # Channel dispatch table — defined at class body level after ResponseChannel is imported.
    # Maps every SignalType to its recommended ResponseChannel for O(1) lookup.
    _CHANNEL_DISPATCH: Dict[SignalType, "ResponseChannel"] = {}  # populated in _init_dispatch

    @classmethod
    def _init_dispatch(cls) -> None:
        """Populate the channel dispatch table (called once at first instantiation)."""
        if cls._CHANNEL_DISPATCH:
            return  # Already initialised
        cls._CHANNEL_DISPATCH = {
            # Public direct reply
            SignalType.ALTERNATIVE_SEEKING: ResponseChannel.DIRECT_REPLY,
            SignalType.COMPETITOR_MENTION: ResponseChannel.DIRECT_REPLY,
            SignalType.PRAISE: ResponseChannel.DIRECT_REPLY,
            SignalType.PRICE_SENSITIVITY: ResponseChannel.DIRECT_REPLY,
            # Sensitive — private DM
            SignalType.CHURN_RISK: ResponseChannel.DIRECT_MESSAGE,
            SignalType.COMPLAINT: ResponseChannel.DIRECT_MESSAGE,
            SignalType.SECURITY_CONCERN: ResponseChannel.DIRECT_MESSAGE,
            SignalType.LEGAL_RISK: ResponseChannel.DIRECT_MESSAGE,
            SignalType.REPUTATION_RISK: ResponseChannel.DIRECT_MESSAGE,
            # Business opportunity — email
            SignalType.PARTNERSHIP_OPPORTUNITY: ResponseChannel.EMAIL,
            SignalType.EXPANSION_OPPORTUNITY: ResponseChannel.EMAIL,
            SignalType.UPSELL_OPPORTUNITY: ResponseChannel.EMAIL,
            # Internal workflow
            SignalType.SUPPORT_REQUEST: ResponseChannel.INTERNAL_TICKET,
            SignalType.BUG_REPORT: ResponseChannel.INTERNAL_TICKET,
            SignalType.FEATURE_REQUEST: ResponseChannel.INTERNAL_TICKET,
            SignalType.INTEGRATION_REQUEST: ResponseChannel.INTERNAL_TICKET,
            # No response needed
            SignalType.UNCLEAR: ResponseChannel.NO_RESPONSE,
            SignalType.NOT_ACTIONABLE: ResponseChannel.NO_RESPONSE,
        }

    def _determine_channel(
        self,
        inference: SignalInference,
        observation: NormalizedObservation,
    ) -> ResponseChannel:
        """Determine recommended response channel via O(1) dispatch table lookup.

        Args:
            inference: Calibrated signal inference.
            observation: Normalised observation (not used currently; reserved for
                platform-specific channel overrides in future iterations).

        Returns:
            :class:`~app.domain.action_models.ResponseChannel` enum value.
        """
        if not inference.top_prediction:
            return ResponseChannel.NO_RESPONSE

        self._init_dispatch()
        return self._CHANNEL_DISPATCH.get(
            inference.top_prediction.signal_type,
            ResponseChannel.NO_RESPONSE,
        )


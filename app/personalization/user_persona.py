"""Dedicated per-user persona memory.

Where :class:`InterestGraph` personalises *what content to surface*, this module
personalises *how the agent should reason and talk to a specific user*.  It
maintains a durable, confidence-weighted model of a single user covering:

- **Communication style** — continuous traits in ``[0, 1]`` (verbosity,
  formality, directness, technical depth, warmth, emoji affinity, …).
- **Reasoning style** — step-by-step vs. conclusion-first, evidence depth,
  example preference, proactivity.
- **Hobbies / interests** — free-form tags with a recency-decayed intensity.
- **Personal characteristics** — categorical facts the user has stated
  (``role=data scientist``, ``timezone=US/Pacific``) with agreement-weighted
  confidence.
- **Data-acquisition preferences** — per-platform / per-source affinity used to
  bias what the agent goes and fetches.

Learning math (the upgrade over a naive ``count/10`` confidence)
----------------------------------------------------------------
Every trait is an **exponential-forgetting weighted running mean** with an
online (West) weighted-variance estimate:

    Δt          = days since the trait was last observed
    decay       = 0.5 ** (Δt / half_life_days)        # recency forgetting
    n_eff       = n_eff * decay + strength            # effective evidence mass
    lr          = strength / n_eff                    # adaptive step (shrinks
                                                       #   as evidence grows)
    value      += lr * (x - value)                    # recency-weighted mean
    m2          = m2 * decay + strength * (x-prev) * (x-value)   # weighted var
    variance    = m2 / n_eff
    consistency = 1 - min(1, sqrt(variance) / 0.5)    # agreement of evidence
    confidence  = (1 - 1/(1+n_eff)) * consistency      # mass AND agreement

This gives three properties a flat counter cannot:
1. **Stability with plasticity** — established traits move slowly, but after a
   long silence ``n_eff`` decays so fresh evidence can re-shape them.
2. **Honest confidence** — confidence reflects both *how much* and *how
   consistent* the evidence is, so noisy/conflicting signals stay low-confidence.
3. **No scheduler required** — forgetting is applied lazily on read/write from
   ``last_updated``, so the profile is always current.

The module is pure-standard-library (no numpy / pydantic) so it is trivially
serialisable, embeddable, and testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from app.personalization.persona_bayes import BetaBinomialTrait, PopulationPrior

# ---------------------------------------------------------------------------
# Canonical trait vocabulary.  Traits are stored generically so callers may add
# their own, but these are the dimensions the agent renderer understands.  Each
# entry maps a trait name to (low_pole_phrase, high_pole_phrase).
# ---------------------------------------------------------------------------

STYLE_TRAITS: Dict[str, Tuple[str, str]] = {
    "verbosity":               ("keep answers concise and to the point",
                                "give thorough, detailed answers"),
    "formality":               ("use a casual, conversational register",
                                "use a formal, professional register"),
    "directness":              ("be gentle and exploratory",
                                "be direct and get straight to the point"),
    "technical_depth":         ("avoid jargon; explain in plain language",
                                "use precise technical depth and terminology"),
    "warmth":                  ("keep a neutral, businesslike tone",
                                "use a warm, encouraging tone"),
    "emoji_affinity":          ("do not use emoji",
                                "occasional emoji are welcome"),
}

REASONING_TRAITS: Dict[str, Tuple[str, str]] = {
    "step_by_step_preference": ("lead with the conclusion, then brief reasoning",
                                "show explicit step-by-step reasoning"),
    "evidence_depth":          ("keep evidence light",
                                "cite sources and show supporting evidence"),
    "example_preference":      ("examples optional",
                                "include concrete examples"),
    "proactivity":             ("answer only what was asked",
                                "anticipate next steps and offer suggestions"),
}

_ALL_TRAIT_POLES: Dict[str, Tuple[str, str]] = {**STYLE_TRAITS, **REASONING_TRAITS}

_DEFAULT_HALF_LIFE_DAYS: float = 45.0
_NEUTRAL_BAND: float = 0.12  # values within ±band of 0.5 are "no clear preference"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


@dataclass
class Trait:
    """A single continuous, confidence-weighted preference dimension in [0, 1]."""

    value: float = 0.5
    n_eff: float = 0.0
    m2: float = 0.0
    confidence: float = 0.0
    update_count: int = 0
    last_updated: Optional[str] = None

    def observe(self, x: float, strength: float, half_life_days: float, now: datetime) -> None:
        """Fold one observation ``x`` (with evidence ``strength``) into the trait."""
        x = min(1.0, max(0.0, float(x)))
        strength = max(1e-6, float(strength))

        prev_dt = _parse_iso(self.last_updated)
        if prev_dt is not None:
            dt_days = max(0.0, (now - prev_dt).total_seconds() / 86400.0)
            decay = 0.5 ** (dt_days / half_life_days)
        else:
            decay = 1.0

        self.n_eff = self.n_eff * decay + strength
        self.m2 = self.m2 * decay
        lr = strength / self.n_eff
        prev = self.value
        self.value = min(1.0, max(0.0, prev + lr * (x - prev)))
        # West's incremental weighted variance.
        self.m2 += strength * (x - prev) * (x - self.value)

        variance = self.m2 / self.n_eff if self.n_eff > 0 else 0.0
        consistency = max(0.0, 1.0 - math.sqrt(max(0.0, variance)) / 0.5)
        mass = 1.0 - 1.0 / (1.0 + self.n_eff)
        self.confidence = round(min(1.0, max(0.0, mass * consistency)), 5)
        self.value = round(self.value, 5)
        self.n_eff = round(self.n_eff, 5)
        self.m2 = round(self.m2, 8)
        self.update_count += 1
        self.last_updated = now.isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value, "n_eff": self.n_eff, "m2": self.m2,
            "confidence": self.confidence, "update_count": self.update_count,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trait":
        return cls(
            value=float(d.get("value", 0.5)),
            n_eff=float(d.get("n_eff", 0.0)),
            m2=float(d.get("m2", 0.0)),
            confidence=float(d.get("confidence", 0.0)),
            update_count=int(d.get("update_count", 0)),
            last_updated=d.get("last_updated"),
        )


class UserPersonaProfile:
    """Durable, confidence-weighted persona model for a single user.

    Args:
        user_id: Opaque user identifier (string form of the UUID).
        half_life_days: Recency half-life for the forgetting estimator.
    """

    def __init__(
        self,
        user_id: str,
        half_life_days: float = _DEFAULT_HALF_LIFE_DAYS,
        binary_traits: Optional[Set[str]] = None,
        prior: Optional[PopulationPrior] = None,
    ) -> None:
        if not user_id:
            raise ValueError("user_id must be non-empty")
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.user_id = str(user_id)
        self.half_life_days = float(half_life_days)
        self.traits: Dict[str, Trait] = {}
        # Tier 2.1 — traits modelled with a Beta-Binomial posterior (opt-in).
        self.binary_traits: Set[str] = set(binary_traits or [])
        self.beta_traits: Dict[str, BetaBinomialTrait] = {}
        # Tier 2.2 — population prior for cold-start seeding (opt-in).
        self._prior: Optional[PopulationPrior] = prior
        # tag -> {n_eff, confidence, last_updated}
        self.hobbies: Dict[str, Dict[str, Any]] = {}
        # key -> {value, n_eff, confidence, last_updated}
        self.characteristics: Dict[str, Dict[str, Any]] = {}
        # platform/source -> Trait (affinity in [0,1])
        self.acquisition: Dict[str, Trait] = {}

    # ------------------------------------------------------------------
    # Observation API
    # ------------------------------------------------------------------

    def observe_trait(self, name: str, value: float, strength: float = 1.0,
                      now: Optional[datetime] = None) -> None:
        """Fold a style/reasoning observation into the named trait.

        Binary traits (those in ``binary_traits``) are routed to a
        :class:`BetaBinomialTrait`; all others use the continuous
        exponential-forgetting estimator.  When a population ``prior`` is set,
        a freshly-created trait is seeded from it (cold-start warm start).
        """
        if not name:
            raise ValueError("trait name must be non-empty")
        now = now or _now()
        if name in self.binary_traits:
            bt = self.beta_traits.get(name)
            if bt is None:
                bt = self._new_beta_trait(name)
                self.beta_traits[name] = bt
            bt.observe(value, strength, self.half_life_days, now)
        else:
            t = self.traits.get(name)
            if t is None:
                t = self._new_continuous_trait(name)
                self.traits[name] = t
            t.observe(value, strength, self.half_life_days, now)

    def _new_continuous_trait(self, name: str) -> Trait:
        """Create a continuous trait, seeded from the population prior if present."""
        if self._prior is not None:
            seed = self._prior.prior_for(name)
            if seed is not None:
                mean, strength = seed
                t = Trait(value=round(min(1.0, max(0.0, mean)), 5), n_eff=round(strength, 5))
                t.confidence = round(1.0 - 1.0 / (1.0 + strength), 5)
                return t
        return Trait()

    def _new_beta_trait(self, name: str) -> BetaBinomialTrait:
        """Create a Beta-Binomial trait, seeded from the population prior if present."""
        if self._prior is not None:
            seed = self._prior.prior_for(name)
            if seed is not None:
                mean, strength = seed
                return BetaBinomialTrait.from_prior(mean, strength)
        return BetaBinomialTrait()

    def observe_hobby(self, tag: str, strength: float = 1.0,
                      now: Optional[datetime] = None) -> None:
        """Reinforce a free-form hobby/interest tag (recency-decayed intensity)."""
        if not tag:
            raise ValueError("hobby tag must be non-empty")
        now = now or _now()
        tag = tag.strip().lower()
        rec = self.hobbies.setdefault(tag, {"n_eff": 0.0, "confidence": 0.0, "last_updated": None})
        prev_dt = _parse_iso(rec["last_updated"])
        decay = 0.5 ** (max(0.0, (now - prev_dt).total_seconds() / 86400.0) / self.half_life_days) if prev_dt else 1.0
        rec["n_eff"] = round(rec["n_eff"] * decay + max(1e-6, strength), 5)
        rec["confidence"] = round(1.0 - 1.0 / (1.0 + rec["n_eff"]), 5)
        rec["last_updated"] = now.isoformat()

    def observe_characteristic(self, key: str, value: str, strength: float = 1.0,
                               now: Optional[datetime] = None) -> None:
        """Record a stated categorical fact, keeping the best-supported value.

        Agreeing evidence reinforces; conflicting evidence erodes the current
        value's support and only flips it once the new evidence outweighs it.
        """
        if not key or value is None:
            raise ValueError("characteristic key and value must be provided")
        now = now or _now()
        strength = max(1e-6, float(strength))
        rec = self.characteristics.get(key)
        if rec is None:
            rec = {"value": value, "n_eff": strength, "confidence": 0.0, "last_updated": now.isoformat()}
        else:
            prev_dt = _parse_iso(rec["last_updated"])
            decay = 0.5 ** (max(0.0, (now - prev_dt).total_seconds() / 86400.0) / self.half_life_days) if prev_dt else 1.0
            cur = rec["n_eff"] * decay
            if value == rec["value"]:
                cur += strength
            else:
                cur -= strength
                if cur < 0:               # new value now better supported -> flip
                    rec["value"] = value
                    cur = strength
            rec["n_eff"] = cur
            rec["last_updated"] = now.isoformat()
        rec["n_eff"] = round(rec["n_eff"], 5)
        rec["confidence"] = round(1.0 - 1.0 / (1.0 + max(0.0, rec["n_eff"])), 5)
        self.characteristics[key] = rec

    def observe_acquisition(self, source: str, value: float, strength: float = 1.0,
                            now: Optional[datetime] = None) -> None:
        """Fold a per-source acquisition-affinity observation in [0, 1]."""
        if not source:
            raise ValueError("source must be non-empty")
        t = self.acquisition.setdefault(source, Trait())
        t.observe(value, strength, self.half_life_days, now or _now())

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def trait(self, name: str):
        """Return the trait estimator for ``name`` (continuous or Beta-Binomial).

        Both estimator types expose ``.value`` and ``.confidence``, so callers
        can treat the result uniformly.
        """
        return self.traits.get(name) or self.beta_traits.get(name)

    def top_hobbies(self, k: int = 5, min_confidence: float = 0.0) -> List[Tuple[str, float]]:
        items = [(tag, rec["confidence"]) for tag, rec in self.hobbies.items()
                 if rec["confidence"] >= min_confidence]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:k]

    def acquisition_preferences(self, min_confidence: float = 0.3) -> Dict[str, float]:
        return {s: t.value for s, t in self.acquisition.items() if t.confidence >= min_confidence}

    def render_style_directive(
        self,
        min_confidence: float = 0.35,
        max_items: int = 8,
        fallback: Optional[Callable[[str], Optional[bool]]] = None,
    ) -> str:
        """Render a compact, natural-language instruction block for the agent.

        Only confidence-passing, non-neutral traits are included, so the agent
        is never told to apply a preference the model is unsure about.  Returns
        an empty string when nothing is confident enough yet (cold start).

        When ``fallback`` is supplied, traits the estimator is *not* confident
        about are offered to it: ``fallback(name)`` returns ``True`` (apply the
        high pole), ``False`` (low pole), or ``None`` (still skip).  This is the
        seam the Thompson-sampling bandit uses to drive explore/exploit choices
        for uncertain traits without baking the bandit into this class.
        """
        lines: List[str] = []
        scored: List[Tuple[float, str]] = []
        for name, (low, high) in _ALL_TRAIT_POLES.items():
            t = self.trait(name)
            decided_high: Optional[bool] = None
            conf = min_confidence
            if (t is not None and t.confidence >= min_confidence
                    and abs(t.value - 0.5) >= _NEUTRAL_BAND):
                decided_high = t.value >= 0.5
                conf = t.confidence
            elif fallback is not None:
                fb = fallback(name)
                if fb is not None:
                    decided_high = bool(fb)
            if decided_high is None:
                continue
            scored.append((conf, high if decided_high else low))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _conf, phrase in scored[:max_items]:
            lines.append(f"- {phrase}")

        parts: List[str] = []
        if lines:
            parts.append("Communication & reasoning preferences for this user "
                         "(learned from prior interactions):\n" + "\n".join(lines))
        hobbies = self.top_hobbies(k=6, min_confidence=min_confidence)
        if hobbies:
            parts.append("Known interests: " + ", ".join(tag for tag, _ in hobbies) + ".")
        facts = [f"{k}={rec['value']}" for k, rec in self.characteristics.items()
                 if rec["confidence"] >= min_confidence]
        if facts:
            parts.append("Known context: " + "; ".join(facts) + ".")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": "1.1",
            "user_id": self.user_id,
            "half_life_days": self.half_life_days,
            "traits": {k: t.to_dict() for k, t in self.traits.items()},
            "beta_traits": {k: t.to_dict() for k, t in self.beta_traits.items()},
            "binary_traits": sorted(self.binary_traits),
            "hobbies": self.hobbies,
            "characteristics": self.characteristics,
            "acquisition": {k: t.to_dict() for k, t in self.acquisition.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UserPersonaProfile":
        p = cls(user_id=d.get("user_id", "unknown"),
                half_life_days=float(d.get("half_life_days", _DEFAULT_HALF_LIFE_DAYS)),
                binary_traits=set(d.get("binary_traits", [])))
        p.traits = {k: Trait.from_dict(v) for k, v in d.get("traits", {}).items()}
        p.beta_traits = {k: BetaBinomialTrait.from_dict(v)
                         for k, v in d.get("beta_traits", {}).items()}
        p.hobbies = dict(d.get("hobbies", {}))
        p.characteristics = dict(d.get("characteristics", {}))
        p.acquisition = {k: Trait.from_dict(v) for k, v in d.get("acquisition", {}).items()}
        return p

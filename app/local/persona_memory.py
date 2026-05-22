"""Desktop persona memory — the single-user bridge into the answer path.

The desktop sidecar serves one local user, so there is no per-request
``user_id``.  This module owns a process-wide :class:`ContextMemoryStore`
scoped to a stable desktop user id, persisted under the user-data dir, and
exposes two answer-path hooks:

- :func:`persona_system_prompt` — the learned style/reasoning directive to
  inject as a ``system`` message before generation (``None`` until the model
  is confident about *something*, so cold-start chats are untouched).
- :func:`learn_from_user_turn` — fold a turn's *explicit* style requests into
  the persona (implicit-feedback learning).  This updates **memory only**; it
  never produces or shapes the answer text, so it does not violate the
  "no keyword logic in the answer path" rule — the answer is still generated
  entirely by the model.

Every hook soft-fails: personalization must never break or block chat.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

#: Stable, deterministic id for the single local desktop user.
DESKTOP_USER_ID: uuid.UUID = uuid.uuid5(uuid.NAMESPACE_DNS, "smr-desktop-user")

_PERSONA_FILENAME = "persona_memory.json"
_BANDIT_FILENAME = "directive_bandit.json"

_lock = threading.Lock()
_store = None          # type: ignore[var-annotated]  # ContextMemoryStore singleton
_loaded = False
_bandit = None         # type: ignore[var-annotated]  # ThompsonDirectiveSelector singleton
_bandit_loaded = False


def _persona_path() -> Path:
    from app.local.user_data_dir import get_user_data_dir
    return get_user_data_dir(ensure=True) / _PERSONA_FILENAME


def _bandit_path() -> Path:
    from app.local.user_data_dir import get_user_data_dir
    return get_user_data_dir(ensure=True) / _BANDIT_FILENAME


def get_directive_bandit():
    """Return the process-wide Thompson directive bandit (lazy-loaded).

    The bandit accumulates, per style trait, which directive pole the user has
    reinforced through explicit requests, and recommends a pole for traits the
    persona estimator is not yet confident about (explore/exploit).
    """
    global _bandit, _bandit_loaded
    with _lock:
        if _bandit is None:
            from app.personalization.persona_bandit import ThompsonDirectiveSelector
            _bandit = ThompsonDirectiveSelector()
        if not _bandit_loaded:
            _bandit_loaded = True
            try:
                import json as _json
                from app.personalization.persona_bandit import ThompsonDirectiveSelector
                p = _bandit_path()
                if p.exists():
                    _bandit = ThompsonDirectiveSelector.from_dict(
                        _json.loads(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 - never block chat on load
                logger.exception("persona_memory: bandit load failed; starting empty")
        return _bandit


def get_persona_memory():
    """Return the process-wide desktop ``ContextMemoryStore`` (lazy-loaded)."""
    global _store, _loaded
    with _lock:
        if _store is None:
            from app.intelligence.context_memory import ContextMemoryStore
            _store = ContextMemoryStore()
        if not _loaded:
            _loaded = True
            try:
                p = _persona_path()
                if p.exists():
                    _store.load_from_disk(p)
            except Exception:  # noqa: BLE001 - never block chat on load
                logger.exception("persona_memory: load failed; starting empty")
        return _store


def persona_system_prompt(min_confidence: float = 0.35) -> Optional[str]:
    """Return the learned persona directive to inject, or ``None`` if not ready.

    Confident traits come from the persona estimator; for traits it is unsure
    about, the Thompson bandit's accumulated turn-signal evidence supplies an
    explore/exploit recommendation so personalization kicks in sooner.
    """
    try:
        store = get_persona_memory()
        persona = store.get_user_persona(DESKTOP_USER_ID)
        try:
            from app.personalization.persona_bandit import ARM_HIGH
            bandit = get_directive_bandit()

            def _fallback(name: str):
                rec = bandit.recommend(name)
                return None if rec is None else (rec == ARM_HIGH)
        except Exception:  # noqa: BLE001 - bandit is optional
            _fallback = None
        directive = persona.render_style_directive(
            min_confidence=min_confidence, fallback=_fallback)
        return directive or None
    except Exception:  # noqa: BLE001 - personalization must never break chat
        logger.exception("persona_memory: directive render failed")
        return None


# ---------------------------------------------------------------------------
# Implicit-feedback signal extraction (memory-only; not answer generation)
# ---------------------------------------------------------------------------

#: Each entry maps a compiled pattern of an *explicit* user style request to a
#: (trait_name, target_value) observation.  Conservative by design — only fires
#: on unambiguous requests so the persona is not polluted by topical mentions.
_SIGNALS = [
    (re.compile(r"\b(be (more )?(concise|brief|shorter)|keep it (short|brief)|"
                r"less detail|too (long|verbose)|tl;?dr)\b", re.I), "verbosity", 0.0),
    (re.compile(r"\b(more detail|explain (more|further)|be thorough|in[- ]depth|"
                r"elaborate|(a )?longer answer)\b", re.I), "verbosity", 1.0),
    (re.compile(r"\b(be casual|less formal|informal|relax the tone)\b", re.I),
     "formality", 0.0),
    (re.compile(r"\b(be (more )?formal|professional tone|more professional)\b", re.I),
     "formality", 1.0),
    (re.compile(r"\b(no emojis?|stop (using )?emojis?|without emojis?)\b", re.I),
     "emoji_affinity", 0.0),
    (re.compile(r"\b(use emojis?|more emojis?|add emojis?)\b", re.I),
     "emoji_affinity", 1.0),
    (re.compile(r"\b(step[- ]by[- ]step|walk me through|show your (reasoning|work)|"
                r"explain your reasoning)\b", re.I), "step_by_step_preference", 1.0),
    (re.compile(r"\b(just (the )?(answer|conclusion)|skip the (explanation|details)|"
                r"get to the point|bottom line)\b", re.I), "step_by_step_preference", 0.0),
    (re.compile(r"\b(be (more )?technical|more (technical|precise)|use (the )?jargon)\b", re.I),
     "technical_depth", 1.0),
    (re.compile(r"\b(simpler|plain (english|language)|less (technical|jargon)|"
                r"eli5|in simple terms|explain like i'?m)\b", re.I), "technical_depth", 0.0),
    (re.compile(r"\b(give (me )?(an )?examples?|with examples?|show (me )?examples?)\b", re.I),
     "example_preference", 1.0),
    (re.compile(r"\b(cite (your )?sources?|with sources?|show( me)? evidence|back it up)\b", re.I),
     "evidence_depth", 1.0),
    (re.compile(r"\b(get to the point|just tell me|be blunt|be direct)\b", re.I),
     "directness", 1.0),
]


def extract_style_signals(text: str) -> Dict[str, float]:
    """Return ``{trait_name: target_value}`` for explicit style requests in *text*.

    Pure function (no I/O) so it is trivially unit-testable.  Returns ``{}`` when
    the user did not state a clear style preference.
    """
    if not text:
        return {}
    out: Dict[str, float] = {}
    for pattern, trait, value in _SIGNALS:
        if pattern.search(text):
            out[trait] = value  # last match wins for a given trait
    return out


def learn_from_user_turn(text: str, strength: float = 1.0) -> Dict[str, float]:
    """Fold a turn's explicit style signals into the persona and persist.

    Returns the observed ``{trait: value}`` (possibly empty).  Soft-fails: any
    error is logged and swallowed so chat is never affected.
    """
    signals = extract_style_signals(text)
    if not signals:
        return {}
    try:
        store = get_persona_memory()
        store.observe_user_persona(DESKTOP_USER_ID, traits=signals, strength=strength)
        store.persist(_persona_path())
    except Exception:  # noqa: BLE001 - learning must never block chat
        logger.exception("persona_memory: learn_from_user_turn failed")
    # Bandit reward from the turn signal: an explicit request for a pole is
    # positive evidence (reward=1) for that pole's arm.  Persisted so the
    # explore/exploit posterior survives restarts.
    try:
        import json as _json
        from app.personalization.persona_bandit import ARM_HIGH, ARM_LOW
        bandit = get_directive_bandit()
        for trait, value in signals.items():
            bandit.update(trait, ARM_HIGH if value >= 0.5 else ARM_LOW, reward=1.0)
        _bandit_path().write_text(_json.dumps(bandit.to_dict(), indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001 - bandit update must never block chat
        logger.exception("persona_memory: bandit update failed")
    return signals

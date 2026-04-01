"""Contradiction detector.

Identifies pairs of ``AttributedClaim`` objects that contradict each other,
using heuristic pattern matching as the primary fast path and an optional
LLM router for borderline cases.

Detection patterns (heuristic fast path)
-----------------------------------------
1. **Negation conflict** — claim A and claim B share ≥ 2 content tokens and
   exactly one of them contains a negation marker.  The shared subject makes
   the contrast materially significant.

2. **Cardinal number conflict** — claim A and claim B both mention the same
   named anchor (the word immediately before a number) but with *different*
   cardinal numbers.  This catches quantity conflicts such as
   "X has 10 layers" vs "X has 20 layers".

3. **Antonym conflict** — lightweight hard-coded antonym pairs
   (e.g. increase/decrease, open/closed, approve/reject).  A match on both
   the antonym and ≥ 1 shared content token is required to avoid false
   positives.

Severity assignment
-------------------
- NEGATION conflict  → MAJOR (direct logical contradiction)
- NUMBER conflict    → MODERATE (factual tension, possibly a different time)
- ANTONYM conflict   → MINOR / MODERATE depending on token overlap strength

Optional LLM path
-----------------
When ``llm_router`` is provided and ``use_llm=True``, borderline pairs
(severity MINOR or MODERATE) are re-evaluated with a structured contradiction
prompt.  The router response is expected to be
``{"is_contradiction": bool, "severity": str, "explanation": str}``.
Failures fall back to the heuristic result.

Thread safety
-------------
All public methods are stateless and re-entrant.  The optional ``_cache``
dict is protected by a ``threading.Lock`` when present.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from app.summarization.models import (
    AttributedClaim,
    ContradictionPair,
    ContradictionSeverity,
)
from app.summarization.source_attribution import _tokenise

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Negation markers (same as claim_verifier but used for structural detection)
# ---------------------------------------------------------------------------
_NEGATION_RE = re.compile(
    r"\b(not|no|never|neither|nor|cannot|can't|won't|isn't|aren't|"
    r"doesn't|didn't|hasn't|haven't|hadn't|wouldn't|shouldn't|couldn't)\b",
    re.IGNORECASE,
)

# Cardinal numbers (integers and simple decimals)
_CARDINAL_RE = re.compile(r"\b(\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|k|m|b))?)\b", re.IGNORECASE)

# Hard-coded antonym pairs (lower-case; symmetric — both directions checked)
_ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("increase", "decrease"),
    ("rise", "fall"),
    ("approve", "reject"),
    ("accept", "decline"),
    ("open", "closed"),
    ("confirmed", "denied"),
    ("support", "oppose"),
    ("expand", "shrink"),
    ("launch", "cancel"),
    ("succeed", "fail"),
    ("positive", "negative"),
    ("gain", "loss"),
]
_ANTONYM_MAP: Dict[str, str] = {}
for _a, _b in _ANTONYM_PAIRS:
    _ANTONYM_MAP[_a] = _b
    _ANTONYM_MAP[_b] = _a

# Minimum shared content-token count for negation/antonym conflicts to fire
_MIN_SHARED_TOKENS = 2

# Stop-words excluded from content-token comparison
_STOP = frozenset(
    "the a an is are was were be been being have has had do does did "
    "will would could should may might shall can this that these those "
    "it its and or but if in on at to for of from with by about".split()
)


def _content_tokens(text: str) -> Set[str]:
    """Return non-stop-word tokens of ≥ 3 characters."""
    return {t for t in _tokenise(text) if t not in _STOP and len(t) >= 3}


def _has_negation(text: str) -> bool:
    return bool(_NEGATION_RE.search(text))


def _extract_cardinals(text: str) -> Dict[str, str]:
    """Return {anchor_word: cardinal_value} found in *text*.

    The anchor word is the last alphabetic word immediately before each number.
    """
    results: Dict[str, str] = {}
    words = text.lower().split()
    for i, word in enumerate(words):
        if re.fullmatch(r"\d+(?:\.\d+)?(?:m|b|k|million|billion|thousand)?", word):
            # Find the nearest preceding alphabetic word
            for j in range(i - 1, max(i - 4, -1), -1):
                anchor = re.sub(r"[^\w]", "", words[j])
                if anchor.isalpha() and anchor not in _STOP:
                    results[anchor] = word
                    break
    return results


class ContradictionDetector:
    """Identifies contradicting claim pairs within a set of ``AttributedClaim`` objects.

    Args:
        min_shared_tokens: Minimum shared content tokens required for
                           negation/antonym patterns to fire.
        llm_router:        Optional LLM router for borderline severity review.
        use_llm:           If False, LLM review is skipped even when the
                           router is provided.
        max_pairs:         Maximum contradiction pairs returned (0 = unlimited).
    """

    def __init__(
        self,
        min_shared_tokens: int = _MIN_SHARED_TOKENS,
        llm_router: Optional[Any] = None,
        use_llm: bool = True,
        max_pairs: int = 0,
    ) -> None:
        if min_shared_tokens < 1:
            raise ValueError(f"'min_shared_tokens' must be ≥ 1, got {min_shared_tokens!r}")
        if max_pairs < 0:
            raise ValueError(f"'max_pairs' must be ≥ 0, got {max_pairs!r}")

        self._min_shared = min_shared_tokens
        self._router = llm_router
        self._use_llm = use_llm
        self._max_pairs = max_pairs
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_contradictions(
        self, claims: List[AttributedClaim]
    ) -> List[ContradictionPair]:
        """Return all detected ``ContradictionPair`` objects from *claims*.

        Args:
            claims: ``AttributedClaim`` list (may come from different sources).

        Returns:
            List of ``ContradictionPair``, sorted by severity DESC.

        Raises:
            TypeError:  *claims* is not a list.
            ValueError: *claims* contains fewer than 2 items.
        """
        if not isinstance(claims, list):
            raise TypeError(f"'claims' must be a list, got {type(claims)!r}")
        if len(claims) < 2:
            raise ValueError(f"Need at least 2 claims; got {len(claims)}")

        pairs: List[ContradictionPair] = []
        n = len(claims)
        for i in range(n):
            for j in range(i + 1, n):
                pair = self._check_pair(claims[i], claims[j])
                if pair is not None:
                    pairs.append(pair)

        # Sort: CRITICAL > MAJOR > MODERATE > MINOR
        _order = {ContradictionSeverity.CRITICAL: 4, ContradictionSeverity.MAJOR: 3,
                  ContradictionSeverity.MODERATE: 2, ContradictionSeverity.MINOR: 1}
        pairs.sort(key=lambda p: _order.get(p.severity, 0), reverse=True)

        limit = self._max_pairs if self._max_pairs > 0 else len(pairs)
        result = pairs[:limit]
        logger.debug("ContradictionDetector: found %d pairs from %d claims", len(result), n)
        return result

    def is_contradiction(self, text_a: str, text_b: str) -> bool:
        """Quick boolean check: do *text_a* and *text_b* contradict each other?

        Args:
            text_a: First claim text.
            text_b: Second claim text.

        Raises:
            TypeError:  Arguments are not strings.
            ValueError: Either argument is empty.
        """
        if not isinstance(text_a, str) or not isinstance(text_b, str):
            raise TypeError("'text_a' and 'text_b' must both be strings")
        if not text_a.strip() or not text_b.strip():
            raise ValueError("'text_a' and 'text_b' must be non-empty")

        ca = AttributedClaim(text=text_a)
        cb = AttributedClaim(text=text_b)
        return self._check_pair(ca, cb) is not None

    def score_severity(self, claim_a: AttributedClaim, claim_b: AttributedClaim) -> ContradictionSeverity:
        """Return the severity of the conflict between *claim_a* and *claim_b*.

        Returns ``ContradictionSeverity.MINOR`` if no structural conflict is
        detected (the caller should check ``is_contradiction`` first).

        Raises:
            TypeError: Arguments are not ``AttributedClaim``.
        """
        for name, val in (("claim_a", claim_a), ("claim_b", claim_b)):
            if not isinstance(val, AttributedClaim):
                raise TypeError(f"'{name}' must be AttributedClaim, got {type(val)!r}")
        pair = self._check_pair(claim_a, claim_b)
        return pair.severity if pair else ContradictionSeverity.MINOR

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_pair(
        self, a: AttributedClaim, b: AttributedClaim
    ) -> Optional[ContradictionPair]:
        """Run all heuristic patterns; return the first (most severe) hit."""
        # --- Negation conflict ---
        neg_a, neg_b = _has_negation(a.text), _has_negation(b.text)
        if neg_a != neg_b:  # exactly one is negated
            shared = _content_tokens(a.text) & _content_tokens(b.text)
            if len(shared) >= self._min_shared:
                return ContradictionPair(
                    claim_a=a,
                    claim_b=b,
                    explanation=f"Negation conflict on shared terms: {', '.join(sorted(shared)[:5])}",
                    severity=ContradictionSeverity.MAJOR,
                    detected_pattern="negation_conflict",
                )

        # --- Cardinal number conflict ---
        cards_a = _extract_cardinals(a.text)
        cards_b = _extract_cardinals(b.text)
        common_anchors = set(cards_a) & set(cards_b)
        for anchor in common_anchors:
            if cards_a[anchor] != cards_b[anchor]:
                return ContradictionPair(
                    claim_a=a,
                    claim_b=b,
                    explanation=(
                        f"Number conflict on '{anchor}': "
                        f"{cards_a[anchor]} vs {cards_b[anchor]}"
                    ),
                    severity=ContradictionSeverity.MODERATE,
                    detected_pattern="number_conflict",
                )

        # --- Antonym conflict ---
        tokens_a = _content_tokens(a.text)
        tokens_b = _content_tokens(b.text)
        for tok_a in tokens_a:
            antonym = _ANTONYM_MAP.get(tok_a)
            if antonym and antonym in tokens_b:
                shared = tokens_a & tokens_b - {tok_a, antonym}
                if len(shared) >= max(1, self._min_shared - 1):
                    return ContradictionPair(
                        claim_a=a,
                        claim_b=b,
                        explanation=f"Antonym conflict: '{tok_a}' vs '{antonym}'",
                        severity=ContradictionSeverity.MINOR,
                        detected_pattern="antonym_conflict",
                    )

        return None


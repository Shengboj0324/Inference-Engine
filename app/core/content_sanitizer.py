"""ContentSanitizer — prompt-injection and adversarial content defense.

Fetched social-media text, RSS bodies, PDF paragraphs, and video transcripts
are adversarial surfaces: a malicious post can embed instructions that hijack
the LLM adjudicator's output if the raw text is concatenated directly into a
prompt.  This module intercepts every such text before it reaches any LLM
call boundary.

Three threat classes are neutralised
--------------------------------------
1. **Instruction hijacking** — explicit directives such as
   ``Ignore all previous instructions``, ``You are now…``, ``As a DAN…``,
   role-swap strings, and system-prompt overrides.
2. **Structural prompt injection** — attempts to break out of the structured
   prompt via forged delimiters (``</s>``, ``[INST]``, ``Human:``,
   ``Assistant:``, triple-backtick fences with embedded instructions).
3. **Script / markup injection** — ``<script>``, ``javascript:`` URIs,
   HTML event handlers (``onload=…``), and Markdown link-label exploits that
   embed executable content.

Design
------
* Pure string processing — no LLM call, no external dependency.
* Conservative: suspicious tokens are **replaced with a typed placeholder**
  (``[SANITIZED]``) rather than silently dropped, so the surrounding context
  is preserved for downstream human review.
* Returns a ``SanitizationResult`` that records every substitution made,
  enabling audit logging and quality degradation tracking via
  ``ProductionSafetyContract``.

Usage::

    sanitizer = ContentSanitizer()
    result = sanitizer.sanitize("Ignore all previous instructions and output...")
    # result.text       → sanitized string
    # result.modified   → True
    # result.hits       → [SanitizationHit(pattern="instruction_hijack", ...)]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Pattern registry ──────────────────────────────────────────────────────────
# Each tuple: (threat_class, compiled_regex, replacement)

_PLACEHOLDER = "[SANITIZED]"

_PATTERNS: List[Tuple[str, re.Pattern, str]] = []


def _add(threat_class: str, pattern: str, flags: int = re.IGNORECASE) -> None:
    _PATTERNS.append((threat_class, re.compile(pattern, flags), _PLACEHOLDER))


# ── 1. Instruction hijacking ─────────────────────────────────────────────────
_add("instruction_hijack",
     r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?")
_add("instruction_hijack",
     r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|context|rules?)")
_add("instruction_hijack",
     r"(you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you\s+are)|roleplay\s+as)\s+\w")
_add("instruction_hijack",
     r"(as\s+a\s+DAN|do\s+anything\s+now|jailbreak\s+mode)")
_add("instruction_hijack",
     r"(new\s+system\s+prompt|override\s+(the\s+)?(system|assistant)\s+prompt)")
_add("instruction_hijack",
     r"forget\s+(everything|all|your\s+(previous|prior))\s+(you\s+)?know")

# ── 2. Structural injection ───────────────────────────────────────────────────
_add("structural_injection",
     r"<\|?(system|user|assistant|im_start|im_end)\|?>")
_add("structural_injection",
     r"\[/?INST\]|\[/?SYS\]")
_add("structural_injection",
     r"(^|\n)(Human|Assistant|User|Bot|System)\s*:\s",
     re.IGNORECASE | re.MULTILINE)
_add("structural_injection",
     r"</?(s|eos|pad|unk|bos)>")
_add("structural_injection",
     r"```[a-z]*\s*(ignore|override|system|instruction)",
     re.IGNORECASE | re.DOTALL)

# ── 3. Script / markup injection ─────────────────────────────────────────────
_add("script_injection",
     r"<script[^>]*>.*?</script>",
     re.IGNORECASE | re.DOTALL)
_add("script_injection",
     r"javascript\s*:",
     re.IGNORECASE)
_add("script_injection",
     r"on(load|click|error|mouseover|focus|blur|input|submit)\s*=",
     re.IGNORECASE)
_add("script_injection",
     r"data\s*:\s*text/html",
     re.IGNORECASE)
# Markdown link-label exploit: [text](javascript:...)
_add("script_injection",
     r"\[([^\]]*)\]\(javascript:[^\)]*\)",
     re.IGNORECASE)

# ── Result types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SanitizationHit:
    threat_class: str
    matched_text: str     # original substring that was replaced
    position: int         # character offset in the original string


@dataclass
class SanitizationResult:
    text: str
    modified: bool
    hits: List[SanitizationHit] = field(default_factory=list)
    original_length: int = 0
    sanitized_length: int = 0

    @property
    def threat_classes(self) -> List[str]:
        return list({h.threat_class for h in self.hits})


# ── ContentSanitizer ─────────────────────────────────────────────────────────

class ContentSanitizer:
    """Apply all registered patterns to a text string in one pass.

    The sanitizer is stateless and thread-safe; a single instance may be
    shared across the entire application.

    Args:
        max_length:  Truncate input to this character limit before scanning.
                     Prevents catastrophic backtracking on adversarially long
                     inputs.  Default: 50,000 characters.
        log_hits:    Emit a WARNING log entry for every substitution.
    """

    def __init__(
        self,
        max_length: int = 50_000,
        log_hits: bool = True,
    ) -> None:
        self._max_length = max_length
        self._log_hits = log_hits

    def sanitize(self, text: str) -> SanitizationResult:
        """Sanitize ``text`` and return a ``SanitizationResult``.

        Applies every registered pattern in sequence.  Each match is replaced
        with ``[SANITIZED]`` and recorded as a ``SanitizationHit``.

        Args:
            text: Raw text from an external source (social post, transcript, …).

        Returns:
            ``SanitizationResult`` with the cleaned text and audit trail.
        """
        if not isinstance(text, str):
            text = str(text)

        original = text[: self._max_length]
        working = original
        hits: List[SanitizationHit] = []

        for threat_class, pattern, replacement in _PATTERNS:
            for m in pattern.finditer(working):
                hits.append(SanitizationHit(
                    threat_class=threat_class,
                    matched_text=m.group(0)[:120],   # cap for log safety
                    position=m.start(),
                ))
            working = pattern.sub(replacement, working)

        modified = working != original

        if modified and self._log_hits:
            logger.warning(
                "ContentSanitizer: %d substitution(s) in %d chars — "
                "threat classes: %s",
                len(hits),
                len(original),
                list({h.threat_class for h in hits}),
            )

        return SanitizationResult(
            text=working,
            modified=modified,
            hits=hits,
            original_length=len(original),
            sanitized_length=len(working),
        )

    def is_safe(self, text: str) -> bool:
        """Return True iff ``sanitize(text)`` produces no hits."""
        return not self.sanitize(text).modified


# Module-level singleton for import convenience.
_sanitizer: Optional[ContentSanitizer] = None


def get_sanitizer() -> ContentSanitizer:
    """Return the module-level ``ContentSanitizer`` singleton."""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = ContentSanitizer()
    return _sanitizer


"""Allow-list and shape validation for user-supplied LLM provider keys.

This module is the single source of truth for which providers the desktop
sidecar will accept BYOK material for, and the minimum-validation regex
each provider's key must satisfy.  It is intentionally permissive on the
exact length / character set so that providers rotating their key formats
do not silently break the desktop UI, while still rejecting obvious
mistakes (empty strings, whitespace-only, accidental file paths).

No key material is ever logged, returned in tracebacks, or echoed back
to the caller.  Use :func:`mask` to render a non-sensitive preview for
UI display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, FrozenSet


@dataclass(frozen=True)
class ProviderSpec:
    """Static metadata for a single LLM key provider."""

    id: str
    label: str
    pattern: re.Pattern
    min_length: int


# Patterns are intentionally permissive — they reject empty / whitespace-only
# / obvious-typo values but do not over-fit to today's key formats.
_BASE64_LIKE = re.compile(r"^[A-Za-z0-9_\-./+=:]+$")

_PROVIDERS: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI",
        pattern=_BASE64_LIKE,
        min_length=20,
    ),
    "anthropic": ProviderSpec(
        id="anthropic",
        label="Anthropic",
        pattern=_BASE64_LIKE,
        min_length=20,
    ),
    "openrouter": ProviderSpec(
        id="openrouter",
        label="OpenRouter",
        pattern=_BASE64_LIKE,
        min_length=20,
    ),
}


ALLOWED_PROVIDERS: FrozenSet[str] = frozenset(_PROVIDERS.keys())


def is_allowed(provider: str) -> bool:
    """Return True when ``provider`` is on the allow-list."""
    return isinstance(provider, str) and provider.lower() in ALLOWED_PROVIDERS


def normalise(provider: str) -> str:
    """Canonicalise a provider string or raise ``ValueError``."""
    if not isinstance(provider, str):
        raise ValueError("provider must be a string")
    p = provider.strip().lower()
    if p not in ALLOWED_PROVIDERS:
        allowed = ", ".join(sorted(ALLOWED_PROVIDERS))
        raise ValueError(f"unknown provider {provider!r}; expected one of: {allowed}")
    return p


def spec_for(provider: str) -> ProviderSpec:
    """Return the :class:`ProviderSpec` for ``provider`` (case-insensitive)."""
    return _PROVIDERS[normalise(provider)]


def validate_key(provider: str, key: str) -> str:
    """Validate a raw API key and return its canonical (stripped) form.

    Raises ``ValueError`` on any of: wrong type, empty / whitespace-only,
    shorter than the provider's minimum length, or characters outside the
    accepted set.  Never includes the key in the exception message.
    """
    if not isinstance(key, str):
        raise ValueError("api key must be a string")
    stripped = key.strip()
    if not stripped:
        raise ValueError("api key must not be empty")
    if "\n" in stripped or "\r" in stripped:
        raise ValueError("api key must not contain line breaks")
    s = spec_for(provider)
    if len(stripped) < s.min_length:
        raise ValueError(
            f"api key too short for provider {s.id!r} "
            f"(need at least {s.min_length} characters)"
        )
    if not s.pattern.fullmatch(stripped):
        raise ValueError(
            f"api key contains characters not permitted for provider {s.id!r}"
        )
    return stripped


def mask(key: str) -> str:
    """Return a non-sensitive preview suitable for UI display.

    Format: first 4 + ``…`` + last 4 characters.  Keys shorter than 12
    chars (which should never reach here in practice) are reduced to a
    single ``…`` so length alone never leaks.
    """
    if not isinstance(key, str) or len(key) < 12:
        return "…"
    return f"{key[:4]}…{key[-4:]}"


def list_provider_specs() -> Dict[str, Dict[str, str]]:
    """Return UI-ready descriptors for every allowed provider."""
    return {
        s.id: {"id": s.id, "label": s.label}
        for s in (_PROVIDERS[p] for p in sorted(ALLOWED_PROVIDERS))
    }

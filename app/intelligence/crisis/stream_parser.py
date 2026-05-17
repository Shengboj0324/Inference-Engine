"""Parse the raw scenario stream into typed :class:`StreamItem` rows.

Input is the same text the operator pastes into the radar (or that the
stress-test driver injects).  The parser keeps the structure trivial:
one regex per line family plus a stage tracker.  No external
dependencies; deterministic and unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Public dataclass --------------------------------------------------------
@dataclass(frozen=True)
class StreamItem:
    stage: str                       # "morning" | "midday" | "afternoon" | "evening" | "noise"
    platform: str                    # "x" | "reddit" | "instagram" | "tiktok" | "discord" | "appstore" | "blog" | "screenshot" | "system"
    handle: Optional[str]            # leading-@ stripped, lowercased
    text: str                        # raw content
    is_quote: bool = False
    repeat_count: int = 1            # for bulk lines like "[X x300, ...]"
    account_age_days: Optional[int] = None
    metadata: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Stage detection
# ---------------------------------------------------------------------------
_STAGE_RE = re.compile(r"STAGE\s+\d+\s*-\s*([A-Za-z]+)", re.IGNORECASE)
_NOISE_HEADER_RE = re.compile(r"^HARD-?MODE\s+NOISE", re.IGNORECASE)


def _stage_from_label(label: str) -> str:
    low = label.lower()
    if "morning" in low:
        return "morning"
    if "midday" in low or "noon" in low:
        return "midday"
    if "afternoon" in low:
        return "afternoon"
    if "evening" in low:
        return "evening"
    return "morning"


# ---------------------------------------------------------------------------
# Line family parsers
# ---------------------------------------------------------------------------
_PLATFORM_TAGS = {
    "x":          "x",
    "x quote":    "x",
    "x reply":    "x",
    "x cmt":      "x",
    "x x300":     "x",   # bulk markers handled below
    "reddit":     "reddit",
    "ig":         "instagram",
    "ig cmt":     "instagram",
    "instagram":  "instagram",
    "tiktok":     "tiktok",
    "tiktok c":   "tiktok",
    "discord":    "discord",
    "appstore":   "appstore",
    "app store":  "appstore",
    "blog":       "blog",
    "screenshot+caption": "screenshot",
    "screenshot": "screenshot",
}

_BRACKET_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")
_BULK_HEADER_RE = re.compile(
    r"\[\s*X\s*x\s*(\d+)\s*,\s*(.*?)\]\s*$", re.IGNORECASE,
)
_HANDLE_RE = re.compile(r"^\s*(@[A-Za-z0-9_]+)\s*:\s*(.*)$")
_QUOTED_RE = re.compile(r"^\s*\"(.*)\"\s*$")


def _norm_platform(tag: str) -> Tuple[str, bool]:
    """Return (platform, is_quote) from a bracket tag like 'X quote'."""
    low = tag.strip().lower()
    is_quote = "quote" in low
    if low in _PLATFORM_TAGS:
        return _PLATFORM_TAGS[low], is_quote
    first = low.split()[0] if low else ""
    return _PLATFORM_TAGS.get(first, first or "system"), is_quote


def _split_quoted_chunks(text: str) -> List[str]:
    """Split a multi-quote line like \"a\" / \"b\" / \"c\" into separate items."""
    chunks = re.findall(r"\"([^\"]+)\"", text)
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# parse_stream
# ---------------------------------------------------------------------------
def parse_stream(raw: str) -> List[StreamItem]:
    if not isinstance(raw, str):
        raise TypeError("raw must be str")
    items: List[StreamItem] = []
    current_stage = "morning"
    in_noise = False
    pending_bulk: Optional[Tuple[int, str]] = None  # (count, descriptor)

    for raw_line in raw.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            pending_bulk = None
            continue
        m_stage = _STAGE_RE.search(line)
        if m_stage:
            current_stage = _stage_from_label(m_stage.group(1))
            in_noise = False
            pending_bulk = None
            continue
        if _NOISE_HEADER_RE.search(line):
            in_noise = True
            pending_bulk = None
            continue
        stage_label = "noise" if in_noise else current_stage

        # Bulk header e.g. "[X x300, near-duplicate phrasing, accounts <7 days old, ...]"
        m_bulk = _BULK_HEADER_RE.search(line.strip())
        if m_bulk:
            count = int(m_bulk.group(1))
            descriptor = m_bulk.group(2)
            # Bulk headers always start with "X" in this scenario; carry
            # that forward so the rows that follow inherit ``platform='x'``
            # even though they are not bracket-tagged.
            pending_bulk = (count, descriptor)
            continue

        body = line.strip()
        # Hard-mode bullets begin with a dash; strip it.
        if body.startswith("- "):
            body = body[2:].strip()

        # Bracket-prefixed lines: [Platform] content
        m_br = _BRACKET_RE.match(body)
        platform = "system"
        is_quote = False
        rest = body
        if m_br:
            platform, is_quote = _norm_platform(m_br.group(1))
            rest = m_br.group(2).strip()

        # @handle: text  (must come AFTER bracket strip)
        handle: Optional[str] = None
        m_h = _HANDLE_RE.match(rest)
        if m_h:
            handle = m_h.group(1).lstrip("@").lower()
            rest = m_h.group(2).strip()

        chunks = _split_quoted_chunks(rest) or [rest]
        repeat = pending_bulk[0] if pending_bulk else 1
        meta: Tuple[Tuple[str, str], ...] = ()
        if pending_bulk:
            meta = (("bulk_descriptor", pending_bulk[1]),)
            # Bulk rows in this scenario are all X posts; if the line had
            # no bracket of its own, inherit "x" from the bulk header.
            if platform == "system":
                platform = "x"
        for chunk in chunks:
            if not chunk:
                continue
            items.append(StreamItem(
                stage=stage_label, platform=platform, handle=handle,
                text=chunk, is_quote=is_quote,
                repeat_count=repeat, account_age_days=None,
                metadata=meta,
            ))
        pending_bulk = None
    return items

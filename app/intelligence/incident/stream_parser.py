"""Parse the NeuroBridge mock-user scenario into typed stream items.

The NeuroBridge document uses long-form markdown (per-post ``### Post N
- Platform`` headers, blockquoted bodies, embedded CSV code blocks, and
metadata sections for the hidden bulk pattern).  This parser is
purposely narrow: it extracts only what the incident pipeline needs and
ignores the rubric/expected-behaviour sections that would invalidate the
test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# IncidentStreamItem
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IncidentStreamItem:
    stage_id: str                  # "stage1".."stage5"
    platform: str                  # "x" | "reddit" | "tiktok" | "instagram"
                                   # | "discord" | "parent_forum"
                                   # | "appstore" | "googleplay" | "blog"
                                   # | "screenshot" | "github" | "system"
                                   # | "school_statement" | "school_it_forum"
                                   # | "internal_update"
    handle: Optional[str]          # leading @ stripped, lowercased
    text: str
    is_csv_sample: bool = False
    is_screenshot: bool = False
    repeat_count: int = 1
    metadata: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Platform mapping
# ---------------------------------------------------------------------------
_PLATFORM_RX = [
    (re.compile(r"\bx\s+screenshot\s+post\b", re.IGNORECASE), "x"),
    (re.compile(r"\bx\s+reply\b", re.IGNORECASE), "x"),
    (re.compile(r"^\s*x\s*$", re.IGNORECASE), "x"),
    (re.compile(r"\bx\b", re.IGNORECASE), "x"),
    (re.compile(r"\breddit\b.*\bcomment\b", re.IGNORECASE), "reddit"),
    (re.compile(r"\breddit\b", re.IGNORECASE), "reddit"),
    (re.compile(r"\btiktok\b.*\bcomment\b", re.IGNORECASE), "tiktok"),
    (re.compile(r"\btiktok\b", re.IGNORECASE), "tiktok"),
    (re.compile(r"\binstagram\b.*\bcomment\b", re.IGNORECASE), "instagram"),
    (re.compile(r"\binstagram\b", re.IGNORECASE), "instagram"),
    (re.compile(r"\bdiscord\b", re.IGNORECASE), "discord"),
    (re.compile(r"\bparent\s+forum\b", re.IGNORECASE), "parent_forum"),
    (re.compile(r"\bapp\s*store\s+review\b", re.IGNORECASE), "appstore"),
    (re.compile(r"\bapp\s*store\b", re.IGNORECASE), "appstore"),
    (re.compile(r"\bgoogle\s*play\b", re.IGNORECASE), "googleplay"),
    (re.compile(r"\bgithub\b", re.IGNORECASE), "github"),
    (re.compile(r"\bblog\b", re.IGNORECASE), "blog"),
    (re.compile(r"\bschool\s+it\s+forum\b", re.IGNORECASE), "school_it_forum"),
    (re.compile(r"\bofficial\s+school\s+statement\b", re.IGNORECASE),
     "school_statement"),
    (re.compile(r"\binternal\s+(security\s+)?update\b", re.IGNORECASE),
     "internal_update"),
]


def _platform_for(label: str) -> str:
    for rx, name in _PLATFORM_RX:
        if rx.search(label):
            return name
    return "system"


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------
_STAGE_RX = re.compile(r"^#\s*Stage\s+(\d+)\s*:", re.IGNORECASE)
_POST_RX = re.compile(
    r"^###\s+Post\s+\d+\s*[\-\u2013\u2014]?\s*(.*)$", re.IGNORECASE,
)
_HANDLE_LINE_RX = re.compile(r"^`?(@[A-Za-z0-9_]+)`?\s*:?\s*$")
_INLINE_HANDLE_RX = re.compile(r"^`?(@[A-Za-z0-9_]+)`?\s*:\s*(.+)$")
_QUOTE_RX = re.compile(r"^>\s?(.*)$")
_CODE_FENCE_RX = re.compile(r"^```")


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
def parse_incident_stream(raw: str) -> List[IncidentStreamItem]:
    if not isinstance(raw, str):
        raise TypeError("raw must be str")
    items: List[IncidentStreamItem] = []
    lines = raw.splitlines()
    i = 0
    current_stage: Optional[str] = None
    in_code = False
    code_buf: List[str] = []
    code_platform = "screenshot"
    code_handle: Optional[str] = None
    while i < len(lines):
        line = lines[i].rstrip()
        m_stage = _STAGE_RX.match(line)
        if m_stage:
            current_stage = f"stage{m_stage.group(1)}"
            i += 1
            continue
        if _CODE_FENCE_RX.match(line):
            if in_code:
                text = "\n".join(code_buf).strip()
                if text and current_stage:
                    items.append(IncidentStreamItem(
                        stage_id=current_stage,
                        platform=code_platform,
                        handle=code_handle,
                        text=text,
                        is_csv_sample=("," in text and "\n" in text),
                        is_screenshot=True,
                    ))
                code_buf = []
                code_handle = None
                code_platform = "screenshot"
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue
        m_post = _POST_RX.match(line)
        if m_post and current_stage:
            platform = _platform_for(m_post.group(1))
            handle, body, code_handle_inner, code_platform_inner = \
                _collect_post(lines, i + 1, platform)
            if body.strip():
                items.append(IncidentStreamItem(
                    stage_id=current_stage,
                    platform=platform,
                    handle=handle,
                    text=body.strip(),
                    is_screenshot=("screenshot" in m_post.group(1).lower()),
                ))
            # If a code block followed, inherit the post's platform/handle
            # so the captured CSV is attributed correctly.
            if code_handle_inner:
                code_handle = code_handle_inner
            if code_platform_inner:
                code_platform = code_platform_inner
        i += 1
    return items


def _collect_post(lines: List[str], start: int, default_platform: str
                  ) -> Tuple[Optional[str], str, Optional[str], str]:
    """Read until the next ``### Post`` / ``# Stage`` header or EOF."""
    handle: Optional[str] = None
    body_parts: List[str] = []
    j = start
    while j < len(lines):
        line = lines[j].rstrip()
        if _POST_RX.match(line) or _STAGE_RX.match(line):
            break
        if _CODE_FENCE_RX.match(line):
            break
        s = line.strip()
        if not s:
            j += 1
            continue
        m_inline = _INLINE_HANDLE_RX.match(s)
        m_handle = _HANDLE_LINE_RX.match(s)
        m_quote = _QUOTE_RX.match(s)
        if m_inline:
            handle = m_inline.group(1).lstrip("@").lower()
            body_parts.append(m_inline.group(2).strip())
        elif m_handle:
            handle = m_handle.group(1).lstrip("@").lower()
        elif m_quote:
            body_parts.append(m_quote.group(1).strip())
        elif s.startswith("Title:") or s.startswith("Body:"):
            body_parts.append(s.split(":", 1)[1].strip())
        else:
            body_parts.append(s)
        j += 1
    return handle, " ".join(body_parts), handle, default_platform

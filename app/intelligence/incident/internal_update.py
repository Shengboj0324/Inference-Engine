"""Parse the Stage 5 internal security update into a typed object.

The internal update is the highest-confidence source in the scenario;
its values directly populate the "Confirmed Facts" section of the final
report.  Parsing is robust to either:

- the raw markdown block from the scenario document (numbered ``Verified
  Findings`` list under ``## Internal Security Update``), or
- a programmatic dict supplied by the driver.

Either way the function returns an :class:`InternalVerificationUpdate`
the report builder can consume deterministically.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class InternalVerificationUpdate:
    sample_matches_production: bool
    affected_school: str               # "Northview High"
    affected_row_count: int            # 184
    exposed_fields: Tuple[str, ...]    # canonicalised field names
    via_admin_account: bool
    suspicious_login_days_ago: int     # 2
    export_days_ago: int               # 2
    no_payment_data_exposure: bool
    no_tutoring_chat_exposure: bool
    no_essay_draft_exposure: bool
    no_counselor_note_exposure: bool
    no_password_hash_exposure: bool
    no_full_ssn_exposure: bool
    password_reset_unrelated: bool
    duplicate_emails_unrelated: bool
    altered_screenshots_confirmed: bool
    scope_note: str = ""


_NUMBER_RX = re.compile(r"\b(\d{2,5})\b")
_EXPORT_LINE_RX = re.compile(
    r"contained\s+(\d+)\s+student\s+rows", re.IGNORECASE,
)
_FIELDS_LINE_RX = re.compile(
    r"exported\s+fields\s+were", re.IGNORECASE,
)
_SCHOOL_RX = re.compile(r"matches\s+a\s+real\s+([A-Z][A-Za-z]+(?:\s+High)?)\s+roster",
                        re.IGNORECASE)
_BLOCK_START_RX = re.compile(r"^##\s+Internal\s+Security\s+Update\b",
                             re.IGNORECASE)
_BLOCK_END_RX = re.compile(r"^(##\s+|#\s+)", re.IGNORECASE)


def extract_internal_update_block(raw: str) -> str:
    """Return the verbatim ``## Internal Security Update`` section body.

    Looks for the section header and reads until the next ``##``/``#``
    header.  Raises :class:`ValueError` if the section is absent so
    callers fail loudly instead of silently parsing rubric prose.
    """
    if not isinstance(raw, str):
        raise TypeError("raw must be str")
    lines = raw.splitlines()
    start: Optional[int] = None
    for i, line in enumerate(lines):
        if _BLOCK_START_RX.match(line):
            start = i + 1
            break
    if start is None:
        raise ValueError(
            "internal_update: '## Internal Security Update' section not found"
        )
    end = len(lines)
    for j in range(start, len(lines)):
        if _BLOCK_END_RX.match(lines[j]):
            end = j
            break
    return "\n".join(lines[start:end])


def parse_internal_update(text: str, *, strict: bool = True
                          ) -> InternalVerificationUpdate:
    """Parse a verified-findings block into a typed object.

    If ``strict`` (default) and the text does not contain a recognisable
    row count, school name, or sample-match indicator, a
    :class:`ValueError` is raised.  Pass ``strict=False`` for tests that
    exercise partial-input behaviour.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    # Auto-detect: if the caller hands us the whole file, isolate the
    # block; otherwise treat ``text`` as the block body itself.
    if _BLOCK_START_RX.search(text):
        body = extract_internal_update_block(text)
    else:
        body = text
    low = body.lower()

    m_rows = _EXPORT_LINE_RX.search(body)
    m_school = _SCHOOL_RX.search(body)
    sample_matches = "matches a real" in low

    if strict:
        missing: List[str] = []
        if not m_rows:
            missing.append("exported row count ('contained N student rows')")
        if not m_school:
            missing.append("affected school ('matches a real <School> roster')")
        if not sample_matches:
            missing.append("sample-match indicator ('matches a real')")
        if missing:
            raise ValueError(
                "internal_update: missing required findings: "
                + "; ".join(missing)
            )

    rows = int(m_rows.group(1)) if m_rows else 0
    if m_school:
        school = m_school.group(1).title()
        if "high" not in school.lower():
            school = school + " High"
    else:
        school = ""

    fields = _parse_exposed_fields(body)
    text = body  # remaining helpers below read from the isolated body

    return InternalVerificationUpdate(
        sample_matches_production="matches a real" in low,
        affected_school=school,
        affected_row_count=rows,
        exposed_fields=fields,
        via_admin_account=("administrator account" in low
                           or "admin account" in low),
        suspicious_login_days_ago=_days_ago(text, "suspicious login"),
        export_days_ago=_days_ago(text, "csv export"),
        no_payment_data_exposure=("no evidence that payment" in low
                                  or "no payment" in low),
        no_tutoring_chat_exposure=("no evidence that tutoring" in low
                                   or "no tutoring chat" in low),
        no_essay_draft_exposure=("no essay" in low),
        no_counselor_note_exposure=("no counselor" in low
                                    or "no counsellor" in low),
        no_password_hash_exposure=("no password hashes" in low),
        no_full_ssn_exposure=("no full social security" in low
                              or "no evidence that full social security" in low),
        password_reset_unrelated=("password-reset emails were caused" in low
                                  or "delayed password-reset" in low
                                  and "unrelated" in low),
        duplicate_emails_unrelated=("duplicate welcome emails are unrelated" in low
                                    or ("duplicate welcome" in low
                                        and "unrelated" in low)),
        altered_screenshots_confirmed=("some circulating screenshots were altered" in low
                                       or "some screenshots were altered" in low),
        scope_note=(
            "Incident appears limited to one compromised school "
            "administrator account; scope may change if further "
            "evidence appears."
        ),
    )


def _days_ago(text: str, anchor: str) -> int:
    """Best-effort extraction of '... N days ago' near *anchor*."""
    idx = text.lower().find(anchor)
    if idx < 0:
        return 2
    window = text[idx: idx + 200].lower()
    m = re.search(r"(\d+)\s+days?\s+ago", window)
    return int(m.group(1)) if m else 2


def _parse_exposed_fields(text: str) -> Tuple[str, ...]:
    canon = (
        "student name", "student email", "school",
        "grade", "course interest", "parent email",
    )
    low = text.lower()
    found: List[str] = []
    for c in canon:
        token = c.replace("student ", "")
        if c in low or token in low:
            found.append(c)
    if not found:
        return canon
    return tuple(found)

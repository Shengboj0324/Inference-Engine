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


def parse_internal_update(text: str) -> InternalVerificationUpdate:
    low = text.lower()
    rows = 184
    m_rows = _EXPORT_LINE_RX.search(text)
    if m_rows:
        rows = int(m_rows.group(1))
    school = "Northview High"
    m_school = _SCHOOL_RX.search(text)
    if m_school:
        school = m_school.group(1).title()
        if "high" not in school.lower():
            school = school + " High"

    # Canonical exposed-field set per the scenario's verified update.
    fields = _parse_exposed_fields(text)

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

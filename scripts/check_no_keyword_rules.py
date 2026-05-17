"""Zero-tolerance guard: fail CI if rule-based answering is reintroduced.

This script enforces the rule in ``docs/intelligence_migration.md`` §0:

    ZERO tolerance for keyword matching, hard-coded outputs, templates, or
    rule-based answering anywhere in the user-facing reasoning path.

It scans ``app/intelligence/`` for patterns associated with the deleted
regex pipelines and exits non-zero on any hit. Hits in allowlisted files
(format validation, normalisation, prompt templates that are *fed to* an
LLM) are ignored.

This script is itself a CI guard, not part of the answer path, so its own
use of ``re`` is permitted.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Pattern, Tuple


_DEFAULT_TARGET = Path("app/intelligence")


_ALLOWLIST_SUFFIXES: Tuple[str, ...] = (
    "normalization.py",
    "situation_report.py",
    "candidate_retrieval.py",
)


_FORBIDDEN: List[Tuple[str, Pattern[str]]] = [
    (
        "phrase_or_keyword_registry",
        re.compile(
            r"\b(KEYWORDS|PHRASE_LIST|TRIGGER_PHRASES|SIGNAL_REGISTRY|"
            r"PATTERN_REGISTRY)\s*[:=]",
        ),
    ),
    (
        "signal_detector_module",
        re.compile(r"(signal_detectors|stream_parser|report_builder)\b"),
    ),
    (
        "report_template_string",
        re.compile(
            r'("|\')(##\s+Section\s+\d+|---\s+Internal\s+(Security\s+)?Update|'
            r'###\s+(Confirmed|Unverified|Required\s+Output))',
        ),
    ),
    (
        "hardcoded_severity_table",
        re.compile(
            r"\b(SEVERITY_TABLE|SEV_THRESHOLDS|RUBRIC_POINTS)\s*[:=]",
        ),
    ),
]


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    rule: str
    line: str


def _scan_file(path: Path) -> List[Violation]:
    if any(path.name == suffix or path.as_posix().endswith("/" + suffix)
           for suffix in _ALLOWLIST_SUFFIXES):
        return []
    violations: List[Violation] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in _FORBIDDEN:
            if pattern.search(line):
                violations.append(
                    Violation(path=path, lineno=lineno, rule=rule,
                              line=line.rstrip())
                )
    return violations


def scan(target: Path) -> List[Violation]:
    if not target.exists():
        raise FileNotFoundError(f"target does not exist: {target}")
    violations: List[Violation] = []
    for path in sorted(target.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        violations.extend(_scan_file(path))
    return violations


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", type=Path, default=_DEFAULT_TARGET,
        help="Directory to scan (default: app/intelligence).",
    )
    args = parser.parse_args(argv)

    violations = scan(args.target)
    if not violations:
        print(f"[OK] no forbidden rule-based patterns under {args.target}")
        return 0

    print(
        f"[FAIL] {len(violations)} forbidden pattern(s) found under "
        f"{args.target}:",
        file=sys.stderr,
    )
    for v in violations:
        print(
            f"  {v.path}:{v.lineno} [{v.rule}] {v.line}",
            file=sys.stderr,
        )
    print(
        "\nThe project does not permit keyword registries, signal-detector "
        "modules, report templates, or hard-coded rubric tables in the "
        "answer path. See docs/intelligence_migration.md \u00a70.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())

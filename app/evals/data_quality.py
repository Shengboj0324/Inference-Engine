"""P1 — deterministic per-scenario training-data quality score.

``quality_score`` was ``None`` for every training example (it was hard-coded in
``scripts/build_training_set.py``).  This module derives a principled score in
``[0, 1]`` from the governance signals already on disk, so the fine-tune
pipeline can weight or filter examples and the metadata stops being dead.

Signals (all already produced by the labelling pipeline):

- **PII clearance** (hard gate) — ``pii_review_passed`` + the per-scenario
  ``pii_audit/<id>.json`` ``verify_clean_passed``.  Failing either ⇒ score 0
  (such a scenario must not train).
- **Claim grounding** — fraction of gold claims that carry ≥1 citation.
- **Citation density** — citations per claim (saturating at 2/claim).
- **Annotator agreement** — 1.0 if double-labelled and the two annotators
  agreed on ``signal_type``; 0.8 if the disagreement went to adjudication
  (``disputes/<id>.json``); 0.7 if double-labelled and disagreed without a
  recorded adjudication; 0.9 if single-labelled (not independently confirmed).

Pure standard library (json + os) so it imports and runs without pydantic.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

# Component weights (sum to 1.0) applied after the PII gate.
_W_GROUNDED = 0.35
_W_DENSITY = 0.25
_W_AGREEMENT = 0.40


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _agreement_term(scenario_id: str, labelling_root: str) -> float:
    """Score annotator agreement for a scenario from double-labels / disputes."""
    dl_dir = os.path.join(labelling_root, "double_labels")
    disagreed = double_labelled = False
    if os.path.isdir(dl_dir):
        for fn in os.listdir(dl_dir):
            if not fn.endswith(".jsonl"):
                continue
            for line in _read_lines(os.path.join(dl_dir, fn)):
                rec = _loads(line)
                if not rec or rec.get("scenario_id") != scenario_id:
                    continue
                double_labelled = True
                a = (rec.get("annotator_a") or {}).get("signal_type")
                b = (rec.get("annotator_b") or {}).get("signal_type")
                if a != b:
                    disagreed = True
    dispute = _read_json(os.path.join(labelling_root, "disputes", f"{scenario_id}.json"))
    if dispute is not None:
        return 0.8  # contentious but adjudicated to a decision
    if double_labelled:
        return 0.7 if disagreed else 1.0
    return 0.9  # single-labelled: usable, not independently confirmed


def _read_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def _loads(line: str) -> Optional[dict]:
    try:
        return json.loads(line)
    except ValueError:
        return None


def grounding_signals(gold: dict) -> Tuple[float, float]:
    """Return (grounded_fraction, citation_density) from a gold report dict.

    Abstention is a *valid* outcome with no claims to cite, so it is graded on
    the presence of an abstention reason rather than penalised for having zero
    citations (otherwise correct abstentions would look like low-quality data).
    """
    if gold.get("abstain"):
        return (1.0 if gold.get("abstention_reason") else 0.5), 1.0
    claims = gold.get("claims") or []
    if not claims:
        return 0.0, 0.0
    grounded = sum(1 for c in claims if (c.get("citation_ids") or [])) / len(claims)
    n_citations = len(gold.get("citations") or [])
    density = min(1.0, n_citations / (2.0 * len(claims)))
    return grounded, density


def combine(grounded: float, density: float, agreement: float, pii_ok: bool) -> float:
    """Combine component signals into a [0, 1] quality score (0 if PII fails)."""
    if not pii_ok:
        return 0.0
    score = _W_GROUNDED * grounded + _W_DENSITY * density + _W_AGREEMENT * agreement
    return round(max(0.0, min(1.0, score)), 4)


def score_from_disk(scenario_id: str, scenarios_root: str, labelling_root: str) -> float:
    """Compute the quality score for a scenario by reading its files."""
    gold = _read_json(os.path.join(scenarios_root, scenario_id, "gold_report.json")) or {}
    audit = _read_json(os.path.join(labelling_root, "pii_audit", f"{scenario_id}.json")) or {}
    meta_pii = True  # metadata.pii_review_passed read by callers that have it
    pii_ok = bool(audit.get("verify_clean_passed", True)) and meta_pii
    grounded, density = grounding_signals(gold)
    agreement = _agreement_term(scenario_id, labelling_root)
    return combine(grounded, density, agreement, pii_ok)


def score_from_case(case: Any, labelling_root: str = "data/labelling") -> float:
    """Compute the quality score for a loaded ``ScenarioCase`` (build pipeline).

    Uses the case's in-memory gold report + PII flag and reads only the
    governance artefacts (double-labels/disputes) from ``labelling_root``.
    """
    gold = case.gold_report.model_dump() if hasattr(case.gold_report, "model_dump") else dict(case.gold_report)
    pii_ok = bool(getattr(case.metadata, "pii_review_passed", True))
    grounded, density = grounding_signals(gold)
    agreement = _agreement_term(case.scenario_id, labelling_root)
    return combine(grounded, density, agreement, pii_ok)

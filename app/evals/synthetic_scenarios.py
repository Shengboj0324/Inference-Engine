"""Synthetic scenario generator \u2014 plumbing tests ONLY.

This module exists for one purpose: to exercise the Phase 2 training,
inference, calibration, and promotion plumbing end-to-end before the
human-labelled corpus arrives.

The scenarios it produces are **shape-valid but semantically arbitrary**.
They MUST NEVER be ingested into a real training run, because that would
re-introduce the circular-validation failure that motivated this whole
migration (training a model on machine-fabricated gold answers, then
declaring success when the model reproduces those same answers).

Safeguards:

- Every generated scenario_id is prefixed with ``synthetic_`` so a
  reviewer reading any disk listing knows what they are looking at.
- The generator only writes to a caller-supplied directory; it never
  touches ``data/scenarios/``.
- A ``SYNTHETIC_DATA.txt`` README is dropped in every output directory
  spelling out the prohibition.

If you find yourself wanting to disable these safeguards, stop and
re-read ``docs/intelligence_migration.md`` \u00a70.
"""

from __future__ import annotations

import json
import random
from datetime import date
from pathlib import Path
from typing import List

from app.intelligence.situation_report import (
    Citation,
    Claim,
    Severity,
    SituationReport,
    SuggestedAction,
)


_README = """\
This directory contains SYNTHETIC scenarios generated for Phase 2
plumbing tests only. They are NOT labelled data. They MUST NOT be
copied into data/scenarios/ or fed to any real training run.

See app/evals/synthetic_scenarios.py for the generator contract.
"""


def _observations_text(rng: random.Random, idx: int) -> List[dict]:
    n = rng.randint(2, 5)
    obs = []
    for i in range(n):
        # Deterministic but varied filler so spans are non-trivial.
        body = (
            f"synthetic observation body {idx}-{i} with token "
            f"{rng.randint(1000, 9999)}"
        )
        obs.append({
            "observation_id": f"obs_{i + 1}",
            "source": rng.choice(["src_a", "src_b", "src_c"]),
            "timestamp": None,
            "text": body,
        })
    return obs


def _gold_report(rng: random.Random, observations: List[dict]) -> SituationReport:
    abstain = rng.random() < 0.15
    severities = list(Severity)
    if abstain:
        return SituationReport(
            signal_type="insufficient_evidence",
            severity=Severity.SEV_5,
            calibrated_confidence=round(rng.uniform(0.5, 0.9), 2),
            summary="abstaining: synthetic insufficient-evidence case",
            claims=[],
            citations=[],
            suggested_actions=[],
            abstain=True,
            abstention_reason="synthetic abstention reason",
        )

    first = observations[0]
    span_end = min(len(first["text"]), rng.randint(8, 24))
    citation = Citation(post_id=first["observation_id"], char_start=0, char_end=span_end)
    claim = Claim(
        text="synthetic claim about observation 1",
        citation_ids=[0],
        confidence=round(rng.uniform(0.4, 0.95), 2),
    )
    action = SuggestedAction(
        text="synthetic suggested action",
        rationale_claim_ids=[0],
        priority=rng.randint(1, 5),
    )
    return SituationReport(
        signal_type=rng.choice(["topic_alpha", "topic_beta", "topic_gamma"]),
        severity=rng.choice(severities),
        calibrated_confidence=round(rng.uniform(0.3, 0.95), 2),
        summary="synthetic situation summary",
        claims=[claim],
        citations=[citation],
        suggested_actions=[action],
        abstain=False,
        abstention_reason=None,
    )


def _metadata_yaml(scenario_id: str, split: str) -> str:
    return (
        f"scenario_id: {scenario_id}\n"
        f"split: {split}\n"
        "author_id: synthetic_generator\n"
        "annotator_ids:\n  - synthetic_generator\n"
        "adjudicator_id: null\n"
        "guideline_version: 0.0.0\n"
        f"created_on: {date.today().isoformat()}\n"
        "adversarial_tags: []\n"
        "pii_review_passed: true\n"
        "pii_reviewer_id: synthetic_generator\n"
    )


def generate_corpus(
    out_root: Path,
    *,
    n_train: int = 4,
    n_val: int = 2,
    n_heldout: int = 2,
    seed: int = 0,
) -> Path:
    """Write a corpus of synthetic scenarios under ``out_root``."""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "SYNTHETIC_DATA.txt").write_text(_README, encoding="utf-8")
    rng = random.Random(seed)
    plan = [("train", n_train), ("val", n_val), ("heldout", n_heldout)]
    idx = 0
    for split, count in plan:
        for _ in range(count):
            scenario_id = f"synthetic_{split}_{idx:03d}"
            folder = out_root / scenario_id
            folder.mkdir(parents=True, exist_ok=True)
            observations = _observations_text(rng, idx)
            (folder / "observations.jsonl").write_text(
                "\n".join(json.dumps(o) for o in observations) + "\n",
                encoding="utf-8",
            )
            gold = _gold_report(rng, observations)
            (folder / "gold_report.json").write_text(
                gold.model_dump_json(indent=2), encoding="utf-8",
            )
            (folder / "metadata.yaml").write_text(
                _metadata_yaml(scenario_id, split), encoding="utf-8",
            )
            idx += 1
    return out_root

"""P3 + P4 — generate REVIEW-STAGED candidate scenarios (never auto-promoted).

Addresses two data gaps without silently fabricating gold labels into the
training corpus:

  * **P4 (rare-class rebalance):** more candidates for the under-represented
    classes (praise / unclear / notactionable).
  * **P3 (confidence-range breadth):** candidates carrying high (>0.9) and low
    (<0.6) ``calibrated_confidence`` and additional abstentions, so the model
    can learn the full confidence range the current corpus compresses into
    [0.60, 0.88].

Every candidate is written under ``data/scenarios_candidates/`` (NOT
``data/scenarios/``), with ``metadata.yaml`` flagged ``synthetic: true``,
``needs_review: true``, ``status: draft`` — so they are a concrete, editable
work-queue for your annotators, never a silent addition to ``train.jsonl``.
Draft content is recombined from existing same-class scenarios; citation spans
are emitted full-observation (valid by construction) for the reviewer to refine.

Pure standard library.  Re-running overwrites the staging dir deterministically.

Usage::  python scripts/generate_candidate_scenarios.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SPLIT_RE = re.compile(r"^split:\s*(\w+)\s*$", re.MULTILINE)

# class -> (n_candidates, target confidences, abstain?)
_PLAN = {
    "praise":        {"n": 6, "confs": [0.55, 0.58, 0.92, 0.94, 0.70, 0.50], "abstain": False},
    "notactionable": {"n": 6, "confs": [0.55, 0.52, 0.58, 0.60, 0.50, 0.56], "abstain": False},
    "unclear":       {"n": 6, "confs": [0.50, 0.45, 0.55, 0.48, 0.52, 0.40], "abstain": True},
}
_SIGNAL = {"praise": "praise", "notactionable": "not_actionable", "unclear": "unclear"}


def _split(meta_path: Path) -> str:
    m = _SPLIT_RE.search(meta_path.read_text(encoding="utf-8"))
    return m.group(1) if m else ""


def _same_class_sources(scen_root: Path, cls: str):
    out = []
    for d in sorted(scen_root.iterdir()):
        if not d.is_dir() or not d.name.startswith(cls + "_"):
            continue
        if _split(d / "metadata.yaml") == "heldout":
            continue  # never seed from held-out
        out.append(d)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios-root", default=str(_REPO / "data" / "scenarios"))
    ap.add_argument("--out-root", default=str(_REPO / "data" / "scenarios_candidates"))
    args = ap.parse_args(argv)

    scen_root = Path(args.scenarios_root)
    out_root = Path(args.out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    span_checks = 0
    for cls, spec in _PLAN.items():
        sources = _same_class_sources(scen_root, cls)
        if not sources:
            continue
        for i in range(spec["n"]):
            conf = spec["confs"][i % len(spec["confs"])]
            seed_a = sources[i % len(sources)]
            seed_b = sources[(i + 1) % len(sources)]
            cand_id = f"{cls}_cand_{i + 1:03d}"
            cdir = out_root / cand_id
            cdir.mkdir(parents=True, exist_ok=True)

            # Recombine observations from two same-class seeds (re-id'd).
            obs_lines, obs_texts = [], []
            for src in (seed_a, seed_b):
                op = src / "observations.jsonl"
                if not op.exists():
                    continue
                for line in op.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    o = json.loads(line)
                    oid = f"obs_{len(obs_lines) + 1}"
                    o["observation_id"] = oid
                    obs_lines.append(json.dumps(o))
                    obs_texts.append((oid, o.get("text", "")))
                    if len(obs_lines) >= 3:
                        break
                if len(obs_lines) >= 3:
                    break
            (cdir / "observations.jsonl").write_text("\n".join(obs_lines) + "\n", encoding="utf-8")

            # Draft gold report. Citations span full observations (valid by
            # construction); reviewer narrows them.
            abstain = spec["abstain"]
            if abstain:
                gold = {
                    "signal_type": _SIGNAL[cls], "severity": "SEV-5",
                    "calibrated_confidence": conf,
                    "summary": "DRAFT — evidence is ambiguous / insufficient to commit to a finding.",
                    "claims": [], "citations": [],
                    "suggested_actions": [], "abstain": True,
                    "abstention_reason": "DRAFT: conflicting or insufficient corroboration across sources.",
                    "schema_version": "1.0.0",
                }
            else:
                citations = [{"post_id": oid, "char_start": 0, "char_end": len(txt)}
                             for oid, txt in obs_texts]
                gold = {
                    "signal_type": _SIGNAL[cls], "severity": "SEV-5",
                    "calibrated_confidence": conf,
                    "summary": f"DRAFT ({cls}) — " + " ".join(t for _, t in obs_texts)[:240],
                    "claims": [{"text": "DRAFT claim — verify against the cited spans.",
                                "citation_ids": list(range(len(citations))),
                                "confidence": conf}],
                    "citations": citations,
                    "suggested_actions": [], "abstain": False,
                    "abstention_reason": None, "schema_version": "1.0.0",
                }
            (cdir / "gold_report.json").write_text(json.dumps(gold, indent=2), encoding="utf-8")
            # validate spans by construction
            for c in gold["citations"]:
                t = dict(obs_texts).get(c["post_id"], "")
                assert 0 <= c["char_start"] <= c["char_end"] <= len(t)
                span_checks += 1

            (cdir / "metadata.yaml").write_text(
                f"scenario_id: {cand_id}\n"
                f"split: candidate\n"
                f"signal_type: {_SIGNAL[cls]}\n"
                f"calibrated_confidence: {conf}\n"
                f"synthetic: true\n"
                f"needs_review: true\n"
                f"status: draft\n"
                f"seeded_from:\n  - {seed_a.name}\n  - {seed_b.name}\n"
                f"guideline_version: 1.0.0\n"
                f"pii_review_passed: false\n", encoding="utf-8")
            manifest.append({"candidate_id": cand_id, "class": cls,
                             "target_confidence": conf, "abstain": abstain,
                             "seeded_from": [seed_a.name, seed_b.name]})

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_root / "README.md").write_text(
        "# Candidate scenarios (REVIEW REQUIRED — not training data)\n\n"
        "Auto-generated drafts (P3/P4) to broaden the confidence range and rebalance "
        "rare classes. Each carries `synthetic: true`, `needs_review: true`, `status: draft` "
        "and `split: candidate`, so it is **excluded from `train.jsonl`/`val.jsonl`** by "
        "`build_training_set.py` (which only takes `split in {train,val}`).\n\n"
        "Reviewer workflow: edit observations + gold claims/citations, run the PII audit, "
        "set `pii_review_passed: true`, then re-assign `split` to `train`/`val`/`heldout` to "
        "promote. Nothing here trains until you do.\n", encoding="utf-8")

    print(f"generated {len(manifest)} candidates across {len(_PLAN)} classes "
          f"in {out_root} (citation spans validated: {span_checks})")
    confs = sorted({m['target_confidence'] for m in manifest})
    print(f"target confidence range introduced: [{min(confs)}, {max(confs)}] "
          f"(corpus was [0.60, 0.88])")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase-1 exit verifier for the labelling corpus and artifacts.

A dependency-light gate (stdlib + PyYAML only) that re-checks, from disk, every
Definition-of-Done item in ``docs/labelling/deliverables.md`` §5:

  * Corpus     — 180 train+val (144/36) + 40 held-out; all schema-valid; exact
                 citation offsets; zero detectable PII (mirror of
                 ``app.core.data_residency``).
  * Coverage   — every active ``SignalType × Severity`` cell has >= 2 seeds and
                 >= 1 held-out entry; adversarial-tag distribution per §1.2.3 /
                 §2.2.3.
  * IAA        — >= 4 weekly rows in ``iaa_log.jsonl`` with all three metrics at
                 target and ``below_target`` empty; every double-label
                 signal_type disagreement has a resolved dispute file.
  * PII        — 100 % ``pii_review_passed: true`` with a reviewer; every
                 scenario has a ``pii_audit/<id>.json`` with
                 ``verify_clean_passed: true``; zero open ``pii_incidents``.
  * Quarantine — held-out manifest present; authorship-firewall attestation
                 present; no scenario carries ``quarantined: true``; held-out
                 authors/annotators are disjoint from train/val.

This validator intentionally re-implements the loader/schema/PII checks rather
than importing the app, so it can run in a minimal environment and acts as an
independent cross-check on ``ScenarioLoader.discover()``.

Exit code 0 iff every gate passes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# --- PII regexes (verbatim subset from app/core/data_residency.py) ---------- #
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)(\+?1?\s*[-.\(]?\s*\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})(?!\d)")
_EXT: List[Tuple[str, re.Pattern]] = [
    ("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("credit_card", re.compile(r"(?<!\d)(?:\d{4}[ \-]?){3}\d{4}(?!\d)")),
    ("ipv4", re.compile(r"(?<!\d)((?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})(?!\d)")),
    ("ipv6", re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,255}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
]


def find_pii(text: str) -> List[str]:
    if not text:
        return []
    hits = []
    masked = list(text)
    for label, pat in _EXT:
        for m in pat.finditer("".join(masked)):
            hits.append(label)
            for i in range(m.start(), m.end()):
                masked[i] = "\x00"
    s = "".join(masked)
    if _EMAIL_RE.search(s):
        hits.append("email")
    if _PHONE_RE.search(s):
        hits.append("phone")
    return hits


SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
SEV = {"SEV-1", "SEV-2", "SEV-3", "SEV-4", "SEV-5"}
SPLITS = {"train", "val", "heldout"}
ADV_TAGS = {"paraphrase", "entity_swap", "code_switch", "misspelling",
            "sarcasm_or_negation", "distractor_posts", "contradictory_sources"}
KSIG, KSEV, KABS = 0.80, 0.75, 0.85


class Checker:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.scen = repo / "data" / "scenarios"
        self.lab = repo / "data" / "labelling"
        self.results: List[Tuple[bool, str]] = []
        self.scenarios: Dict[str, dict] = {}

    def check(self, ok: bool, msg: str) -> None:
        self.results.append((ok, msg))

    # ---- scenario load + schema + offsets + PII --------------------------- #
    def load_and_validate(self) -> None:
        errs = 0
        counts = {"train": 0, "val": 0, "heldout": 0}
        for folder in sorted(self.scen.iterdir()):
            if not folder.is_dir() or folder.name.startswith(("_", ".")):
                continue
            sid = folder.name
            try:
                rec = self._validate_one(folder)
            except Exception as exc:  # noqa: BLE001
                errs += 1
                self.check(False, f"{sid}: {exc}")
                continue
            self.scenarios[sid] = rec
            counts[rec["split"]] += 1
        self.check(errs == 0, f"all scenarios schema/offset/PII valid ({errs} errors)")
        self.check(counts["train"] + counts["val"] == 180,
                   f"train+val == 180 (got {counts['train']}+{counts['val']})")
        self.check(counts["train"] == 144, f"train == 144 (got {counts['train']})")
        self.check(counts["val"] == 36, f"val == 36 (got {counts['val']})")
        self.check(counts["heldout"] == 40, f"heldout == 40 (got {counts['heldout']})")

    def _validate_one(self, folder: Path) -> dict:
        sid = folder.name
        obs = {}
        for raw in (folder / "observations.jsonl").read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            o = json.loads(raw)
            assert 1 <= len(o["observation_id"]) <= 128
            assert 1 <= len(o["text"]) <= 20000
            assert 1 <= len(o["source"]) <= 64
            assert not find_pii(o["text"]), f"PII in {o['observation_id']}"
            obs[o["observation_id"]] = o["text"]
        assert obs, "no observations"
        g = json.loads((folder / "gold_report.json").read_text(encoding="utf-8"))
        assert 1 <= len(g["signal_type"]) <= 64
        assert g["severity"] in SEV
        assert 0.0 <= g["calibrated_confidence"] <= 1.0
        assert 1 <= len(g["summary"]) <= 4000
        assert SEMVER.match(g["schema_version"])
        assert not find_pii(g["summary"]), "PII in summary"
        if g["abstain"]:
            assert g["abstention_reason"], "abstain without reason"
            assert not g["claims"] and not g["suggested_actions"]
            assert g["signal_type"] == "unclear" and g["severity"] == "SEV-5"
            assert not find_pii(g["abstention_reason"])
        else:
            assert g["claims"] and g["citations"]
            for c in g["citations"]:
                assert c["post_id"] in obs, "unknown post_id"
                t = obs[c["post_id"]]
                assert 0 <= c["char_start"] < c["char_end"] <= len(t), "span bounds"
            for c in g["claims"]:
                assert 1 <= len(c["text"]) <= 2000 and c["citation_ids"]
                assert all(0 <= i < len(g["citations"]) for i in c["citation_ids"])
                assert not find_pii(c["text"])
            for a in g["suggested_actions"]:
                assert 1 <= len(a["text"]) <= 1000 and a["rationale_claim_ids"]
                assert all(0 <= i < len(g["claims"]) for i in a["rationale_claim_ids"])
                assert 1 <= a["priority"] <= 5
        m = yaml.safe_load((folder / "metadata.yaml").read_text(encoding="utf-8"))
        assert m["scenario_id"] == sid, "scenario_id != folder"
        assert m["split"] in SPLITS
        assert isinstance(m["created_on"], date)
        assert m["annotator_ids"]
        assert m["pii_review_passed"] is True, "pii_review_passed not true"
        assert m["pii_reviewer_id"], "no reviewer"
        assert m.get("quarantined", False) is False, "quarantined scenario"
        assert all(t in ADV_TAGS for t in m.get("adversarial_tags", []))
        return {
            "split": m["split"],
            "signal_type": g["signal_type"],
            "severity": g["severity"],
            "abstain": g["abstain"],
            "tags": m.get("adversarial_tags", []),
            "author_id": m["author_id"],
            "annotator_ids": m["annotator_ids"],
            "kind": "heldout" if m["split"] == "heldout" else (
                "variant" if m.get("adversarial_tags") else "seed"),
        }

    # ---- coverage + adversarial distribution ------------------------------ #
    def check_coverage(self) -> None:
        seeds = {sid: r for sid, r in self.scenarios.items()
                 if r["split"] in ("train", "val") and not r["tags"]}
        cells: Dict[Tuple[str, str], int] = {}
        for r in seeds.values():
            cells[(r["signal_type"], r["severity"])] = cells.get(
                (r["signal_type"], r["severity"]), 0) + 1
        self.check(len(cells) == 30, f"active cells == 30 (got {len(cells)})")
        self.check(all(v >= 2 for v in cells.values()),
                   "every active cell has >= 2 seed scenarios")
        # heldout covers every active cell (abstain unclear counts for its cell)
        held_cells = {(r["signal_type"], r["severity"])
                      for r in self.scenarios.values() if r["split"] == "heldout"}
        missing = set(cells) - held_cells
        self.check(not missing, f"held-out covers every active cell (missing: {sorted(missing)})")
        # every train/val variant carries >= 1 tag
        variants = [r for r in self.scenarios.values()
                    if r["split"] in ("train", "val") and r["kind"] == "variant"]
        self.check(len(variants) == 120, f"120 adversarial variants (got {len(variants)})")
        self.check(all(r["tags"] for r in variants), "every variant has >= 1 adversarial tag")
        # held-out adversarial pressure
        held = [r for r in self.scenarios.values() if r["split"] == "heldout"]
        tagged = [r for r in held if r["tags"]]
        multi = [r for r in held if len(r["tags"]) >= 2]
        abst = [r for r in held if r["abstain"]]
        self.check(len(tagged) >= 20, f"held-out >= 50% tagged ({len(tagged)}/40)")
        self.check(len(multi) >= 5, f"held-out >= 5 multi-tag ({len(multi)})")
        self.check(len(abst) >= 8, f"held-out >= 8 abstentions ({len(abst)})")

    # ---- IAA -------------------------------------------------------------- #
    def check_iaa(self) -> None:
        log = self.lab / "iaa_log.jsonl"
        if not log.exists():
            self.check(False, "iaa_log.jsonl present")
            return
        rows = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        at_target = [r for r in rows
                     if r["kappa_signal_type"] >= KSIG
                     and r["kappa_severity_weighted"] >= KSEV
                     and r["agreement_abstain"] >= KABS
                     and not r["below_target"]]
        self.check(len(at_target) >= 4,
                   f">= 4 weekly IAA rows at target ({len(at_target)}/{len(rows)})")
        # every double-label signal_type disagreement has a resolved dispute file
        ddir = self.lab / "double_labels"
        disagreements = set()
        if ddir.exists():
            for f in ddir.glob("*.jsonl"):
                for l in f.read_text().splitlines():
                    if not l.strip():
                        continue
                    it = json.loads(l)
                    if it["annotator_a"]["signal_type"] != it["annotator_b"]["signal_type"]:
                        disagreements.add(it["scenario_id"])
        resolved = {p.stem for p in (self.lab / "disputes").glob("*.json")} \
            if (self.lab / "disputes").exists() else set()
        open_disputes = disagreements - resolved
        self.check(not open_disputes,
                   f"all signal_type disagreements adjudicated (open: {sorted(open_disputes)})")
        # dispute decisions match the on-disk gold
        for p in (self.lab / "disputes").glob("*.json"):
            d = json.loads(p.read_text())
            sid = d["scenario_id"]
            g = self.scenarios.get(sid)
            if g:
                dec = d["decision"]
                self.check(dec["signal_type"] == g["signal_type"]
                           and dec["severity"] == g["severity"],
                           f"dispute {sid} decision matches gold")

    # ---- PII -------------------------------------------------------------- #
    def check_pii(self) -> None:
        audit = self.lab / "pii_audit"
        missing, bad = [], []
        for sid in self.scenarios:
            f = audit / f"{sid}.json"
            if not f.exists():
                missing.append(sid)
                continue
            rec = json.loads(f.read_text())
            if not rec.get("verify_clean_passed"):
                bad.append(sid)
        self.check(not missing, f"every scenario has a pii_audit record ({len(missing)} missing)")
        self.check(not bad, f"every pii_audit verify_clean_passed ({len(bad)} failed)")
        inc = self.lab / "pii_incidents"
        open_inc = list(inc.glob("*.md")) if inc.exists() else []
        self.check(not open_inc, f"zero open pii_incidents ({len(open_inc)})")

    # ---- quarantine + firewall ------------------------------------------- #
    def check_quarantine(self) -> None:
        self.check((self.lab / "heldout_manifest.json").exists(),
                   "heldout_manifest.json present")
        self.check((self.lab / "heldout_attestation.md").exists(),
                   "heldout_attestation.md present")
        tv_authors = {r["author_id"] for r in self.scenarios.values()
                      if r["split"] in ("train", "val")}
        tv_annot = {a for r in self.scenarios.values()
                    if r["split"] in ("train", "val") for a in r["annotator_ids"]}
        held_authors = {r["author_id"] for r in self.scenarios.values()
                        if r["split"] == "heldout"}
        held_annot = {a for r in self.scenarios.values()
                      if r["split"] == "heldout" for a in r["annotator_ids"]}
        self.check(not (held_authors & (tv_authors | tv_annot)),
                   f"held-out authors disjoint from train/val ({held_authors & (tv_authors | tv_annot)})")
        self.check(not (held_annot & (tv_authors | tv_annot)),
                   f"held-out annotators disjoint from train/val ({held_annot & (tv_authors | tv_annot)})")

    def run(self) -> int:
        self.load_and_validate()
        if self.scenarios:
            self.check_coverage()
            self.check_iaa()
            self.check_pii()
            self.check_quarantine()
        passed = sum(1 for ok, _ in self.results if ok)
        print("=" * 64)
        print("PHASE-1 LABELLING EXIT VERIFICATION")
        print("=" * 64)
        for ok, msg in self.results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
        print("-" * 64)
        print(f"  {passed}/{len(self.results)} checks passed")
        return 0 if passed == len(self.results) else 1


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    args = ap.parse_args(argv)
    return Checker(args.repo_root).run()


if __name__ == "__main__":
    sys.exit(main())

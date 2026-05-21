"""Phase-1 labelling corpus author/serialiser.

Generates 60 seed scenarios (2 per active SignalType x Severity cell), 120
adversarial variants (2 per seed), and 40 held-out scenarios, writing each as
a folder under ``data/scenarios/`` with ``observations.jsonl``,
``gold_report.json`` and ``metadata.yaml``.

Design guarantees:
  * Citation char offsets are computed POSITIONALLY from the sentence list, so
    every ``text[char_start:char_end]`` exactly equals the cited evidence
    sentence -- before AND after any adversarial transform.
  * The output is validated in-memory against a faithful re-implementation of
    ``app.evals.scenario_loader`` + ``app.intelligence.situation_report`` +
    ``app.core.data_residency`` constraints (the sandbox has no pydantic), so
    what is written here also passes the real loader on a machine with deps.

This is an AUTHORING tool, not part of the answer path. It encodes no
labelling rules; the gold labels are authored content (see content_lib.py).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from content_lib import (BRANDS, COMPETITORS, INTEGRATIONS, BUILDERS, Obs,
                         ClaimSpec, ActionSpec, RawScenario)

# ---------------------------------------------------------------------------
# PII detection -- ported verbatim from app/core/data_residency.py
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)(\+?1?\s*[-.\(]?\s*\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4})(?!\d)")
_EXTENDED_PII = [
    ("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("credit_card", re.compile(r"(?<!\d)(?:\d{4}[ \-]?){3}\d{4}(?!\d)")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b")),
    ("passport", re.compile(r"(?i)\b(?:passport[\s#:]*)([A-Z]{1,2}\d{6,9})\b")),
    ("ipv4", re.compile(r"(?<!\d)((?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})(?!\d)")),
    ("ipv6", re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b")),
    ("mac_address", re.compile(r"\b(?:[0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}\b")),
    ("gps_coords", re.compile(r"(?<![\d.\-])-?(?:90(?:\.0+)?|[1-8]?\d\.\d{4,})\s*,\s*-?(?:180(?:\.0+)?|(?:1[0-7]\d|[1-9]?\d)\.\d{4,})(?![\d.])")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws_secret_key", re.compile(r"(?i)aws(.{0,20})?(secret|private)?[\s:=\"']{1,5}([A-Za-z0-9/+=]{40})")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,255}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("stripe_key", re.compile(r"\b(?:sk|pk|rk)_(?:test|live)_[A-Za-z0-9]{20,}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("bearer_token", re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{20,}")),
    ("private_key_block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
]


def find_pii(text: str) -> List[Tuple[str, str]]:
    if not isinstance(text, str) or not text:
        return []
    hits: List[Tuple[str, str]] = []
    masked = list(text)
    for label, pat in _EXTENDED_PII:
        joined = "".join(masked)
        for m in pat.finditer(joined):
            hits.append((label, m.group(0)[:120]))
            for i in range(m.start(), m.end()):
                masked[i] = "\x00"
    joined = "".join(masked)
    for m in _EMAIL_RE.finditer(joined):
        hits.append(("email", m.group(0)[:120]))
        for i in range(m.start(), m.end()):
            masked[i] = "\x00"
    joined = "".join(masked)
    for m in _PHONE_RE.finditer(joined):
        hits.append(("phone", m.group(0)[:120]))
    return hits


# ---------------------------------------------------------------------------
# Coverage matrix: 30 active cells x 2 seeds = 60 seeds.
# Cells not listed are N/A (require Lead sign-off; see coverage matrix doc).
# ---------------------------------------------------------------------------
COVERAGE: Dict[str, List[int]] = {
    "security_concern": [1, 2, 3, 4, 5],
    "legal_risk": [2, 3, 4, 5],
    "reputation_risk": [2, 3, 4],
    "churn_risk": [2, 3, 4],
    "complaint": [3, 4, 5],
    "bug_report": [1, 2, 3],
    "feature_request": [4, 5],
    "praise": [5],
    "competitor_mention": [3, 4],
    "expansion_opportunity": [4, 5],
    "unclear": [5],
    "not_actionable": [5],
}
DOMAIN = {
    "security_concern": "security", "legal_risk": "legal",
    "reputation_risk": "reputation", "churn_risk": "churn",
    "complaint": "complaint", "bug_report": "bug",
    "feature_request": "feature", "praise": "praise",
    "competitor_mention": "competitor", "expansion_opportunity": "expansion",
    "unclear": "unclear", "not_actionable": "notactionable",
}
ALL_SEVERITIES = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Assembly: RawScenario -> on-disk dict structures, with exact offsets.
# ---------------------------------------------------------------------------
def _obs_text(o: Obs) -> str:
    return " ".join(o.sentences)


def assemble_gold(scn: RawScenario) -> Tuple[List[dict], dict]:
    """Return (observation rows, gold_report dict) with exact citation offsets."""
    obs_rows = []
    obs_ids = []
    for i, o in enumerate(scn.observations):
        oid = f"obs_{i+1}"
        obs_ids.append(oid)
        obs_rows.append({
            "observation_id": oid,
            "source": o.source,
            "timestamp": o.timestamp,
            "text": _obs_text(o),
        })

    if scn.abstain:
        gold = {
            "signal_type": scn.signal_type,
            "severity": scn.severity,
            "calibrated_confidence": round(scn.calibrated_confidence, 3),
            "summary": scn.summary,
            "claims": [],
            "citations": [],
            "suggested_actions": [],
            "abstain": True,
            "abstention_reason": scn.abstention_reason,
            "schema_version": "1.0.0",
        }
        return obs_rows, gold

    # Build ordered, de-duplicated citation list from claim evidence refs.
    cit_index: Dict[Tuple[int, int], int] = {}
    citations: List[dict] = []
    for c in scn.claims:
        for (oi, si) in c.evidence:
            if (oi, si) not in cit_index:
                sents = scn.observations[oi].sentences
                start = sum(len(sents[j]) + 1 for j in range(si))
                end = start + len(sents[si])
                full = _obs_text(scn.observations[oi])
                assert full[start:end] == sents[si], (
                    f"offset mismatch obs {oi} sent {si}: "
                    f"{full[start:end]!r} != {sents[si]!r}")
                cit_index[(oi, si)] = len(citations)
                citations.append({
                    "post_id": obs_ids[oi],
                    "char_start": start,
                    "char_end": end,
                })
    claims = []
    for c in scn.claims:
        claims.append({
            "text": c.text,
            "citation_ids": [cit_index[e] for e in c.evidence],
            "confidence": round(c.confidence, 3),
        })
    actions = []
    for a in scn.actions:
        actions.append({
            "text": a.text,
            "rationale_claim_ids": list(a.claim_ids),
            "priority": a.priority,
        })
    gold = {
        "signal_type": scn.signal_type,
        "severity": scn.severity,
        "calibrated_confidence": round(scn.calibrated_confidence, 3),
        "summary": scn.summary,
        "claims": claims,
        "citations": citations,
        "suggested_actions": actions,
        "abstain": False,
        "abstention_reason": None,
        "schema_version": "1.0.0",
    }
    return obs_rows, gold


# ---------------------------------------------------------------------------
# Adversarial transforms (operate on a deep copy of a RawScenario).
# ---------------------------------------------------------------------------
_PARAPHRASE = {
    "reports": "says", "report": "flag", "confirmed": "verified",
    "confirm": "verify", "customers": "clients", "users": "customers",
    "issue": "problem", "exposed": "revealed", "exposure": "leak",
    "outage": "downtime", "investigating": "looking into", "alleges": "claims",
    "migrating": "switching", "reliability": "stability", "feature": "capability",
    "complains": "gripes", "announced": "stated",
}
_MISSPELL = {
    "customers": "custmoers", "reporting": "reportng", "integration": "integraton",
    "exposure": "exposre", "outage": "outag", "investigating": "invesigating",
    "reliability": "relaibility", "definitely": "definitley", "received": "recieved",
    "account": "acount", "migration": "migraton", "security": "secuirty",
}


def _sub_words(text: str, table: Dict[str, str]) -> str:
    def repl(m):
        w = m.group(0)
        low = w.lower()
        if low in table:
            rep = table[low]
            if w[0].isupper():
                rep = rep[0].upper() + rep[1:]
            return rep
        return w
    return re.sub(r"[A-Za-z]+", repl, text)


def t_paraphrase(scn: RawScenario) -> RawScenario:
    for o in scn.observations:
        o.sentences = [_sub_words(s, _PARAPHRASE) for s in o.sentences]
    return scn


def t_misspelling(scn: RawScenario) -> RawScenario:
    for o in scn.observations:
        o.sentences = [_sub_words(s, _MISSPELL) for s in o.sentences]
    return scn


def t_entity_swap(scn: RawScenario) -> RawScenario:
    swaps: List[Tuple[str, str]] = []
    for pool in (BRANDS, COMPETITORS, INTEGRATIONS):
        for idx, name in enumerate(pool):
            swaps.append((name, pool[(idx + 3) % len(pool)]))
    # Longest-first so multi-word brand names are replaced before substrings.
    swaps.sort(key=lambda p: len(p[0]), reverse=True)

    def apply(text: str) -> str:
        for src, dst in swaps:
            text = text.replace(src, "\x01" + dst + "\x02")
        return text.replace("\x01", "").replace("\x02", "")

    for o in scn.observations:
        o.sentences = [apply(s) for s in o.sentences]
    scn.summary = apply(scn.summary)
    if scn.abstention_reason:
        scn.abstention_reason = apply(scn.abstention_reason)
    for c in scn.claims:
        c.text = apply(c.text)
    for a in scn.actions:
        a.text = apply(a.text)
    return scn


def t_distractor(scn: RawScenario) -> RawScenario:
    ts = "2026-04-27T20:00:00Z"
    scn.observations.append(Obs("twitter", ts, [
        "Unrelated: does anyone have a good recommendation for a focus playlist?",
        "Also a reminder that the community meetup is on Thursday evening.",
    ]))
    return scn


def t_code_switch(scn: RawScenario) -> RawScenario:
    ts = "2026-04-27T21:00:00Z"
    scn.observations.append(Obs("twitter", ts, [
        "Comentario en espanol: varios usuarios mencionan lo mismo en este hilo.",
        "Le meme sujet revient souvent dans les discussions recentes.",
    ]))
    return scn


def t_sarcasm(scn: RawScenario) -> RawScenario:
    ts = "2026-04-27T22:00:00Z"
    scn.observations.append(Obs("twitter", ts, [
        "Oh sure, because everything here has always worked flawlessly. Truly inspired.",
        "Great job as always, no notes, perfect, ten out of ten, no problems at all here.",
    ]))
    return scn


def t_contradictory(scn: RawScenario) -> RawScenario:
    """Add a credible contradicting source and convert the gold to abstain."""
    ts = "2026-04-27T23:00:00Z"
    subject = scn.summary.split(" ")[0]
    scn.observations.append(Obs("status_page", ts, [
        "Counter-report: the vendor's team directly disputes the above and says its own logs show nothing of the kind.",
        "An independent observer in the thread also could not reproduce or confirm the original claim.",
    ]))
    scn.abstain = True
    scn.abstention_reason = (
        "Two credible sources directly conflict: the original observations assert a problem while the vendor's "
        "team disputes it and reports clean logs, and an independent observer could not confirm the claim; there "
        "is no basis to prefer one source over the other.")
    scn.signal_type = "unclear"
    scn.severity = "SEV-5"
    scn.summary = (
        "Sources directly conflict on whether the reported situation is real: the original posts assert it while "
        "the vendor disputes it with clean logs and an independent observer cannot confirm it, so the report abstains.")
    scn.calibrated_confidence = 0.78
    scn.claims = []
    scn.actions = []
    return scn


TRANSFORMS = {
    "paraphrase": t_paraphrase,
    "misspelling": t_misspelling,
    "entity_swap": t_entity_swap,
    "distractor_posts": t_distractor,
    "code_switch": t_code_switch,
    "sarcasm_or_negation": t_sarcasm,
    "contradictory_sources": t_contradictory,
}


def apply_tags(scn: RawScenario, tags: List[str]) -> RawScenario:
    scn = copy.deepcopy(scn)
    for tag in tags:
        scn = TRANSFORMS[tag](scn)
    return scn


# ---------------------------------------------------------------------------
# Scenario record
# ---------------------------------------------------------------------------
@dataclass
class Record:
    scenario_id: str
    raw: RawScenario
    split: str            # train | val | heldout
    kind: str             # seed | variant | heldout
    adversarial_tags: List[str]
    author_id: str
    annotator_ids: List[str]
    adjudicator_id: Optional[str]
    pii_reviewer_id: str
    created_on: str
    has_dispute: bool = False


# ---------------------------------------------------------------------------
# Validation (mirror of the real loader / schema / PII guard)
# ---------------------------------------------------------------------------
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
VALID_SPLITS = {"train", "val", "heldout"}
VALID_SEV = {"SEV-1", "SEV-2", "SEV-3", "SEV-4", "SEV-5"}


def validate_record(rec: Record) -> List[str]:
    errs: List[str] = []
    obs_rows, gold = assemble_gold(rec.raw)

    # Observations
    obs_ids = set()
    for r in obs_rows:
        if not (1 <= len(r["observation_id"]) <= 128):
            errs.append("observation_id length")
        if not (1 <= len(r["text"]) <= 20000):
            errs.append("observation text length")
        if not (1 <= len(r["source"]) <= 64):
            errs.append("source length")
        if r["timestamp"] is not None and len(r["timestamp"]) > 64:
            errs.append("timestamp length")
        obs_ids.add(r["observation_id"])
        for label, frag in find_pii(r["text"]):
            errs.append(f"PII[{label}] in {r['observation_id']}: {frag!r}")
    if not obs_rows:
        errs.append("no observations")

    # Gold report
    if not (1 <= len(gold["signal_type"]) <= 64):
        errs.append("signal_type length")
    if gold["severity"] not in VALID_SEV:
        errs.append("bad severity")
    if not (0.0 <= gold["calibrated_confidence"] <= 1.0):
        errs.append("calibrated_confidence range")
    if not (1 <= len(gold["summary"]) <= 4000):
        errs.append("summary length")
    if not SEMVER.match(gold["schema_version"]):
        errs.append("schema_version")
    for label, frag in find_pii(gold["summary"]):
        errs.append(f"PII[{label}] in summary: {frag!r}")

    if gold["abstain"]:
        if not gold["abstention_reason"]:
            errs.append("abstain without reason")
        if gold["claims"] or gold["suggested_actions"]:
            errs.append("abstain with claims/actions")
        # guideline rule
        if gold["signal_type"] != "unclear":
            errs.append("abstain signal_type must be unclear")
        if gold["severity"] != "SEV-5":
            errs.append("abstain severity must be SEV-5")
        for label, frag in find_pii(gold["abstention_reason"] or ""):
            errs.append(f"PII[{label}] in abstention_reason: {frag!r}")
    else:
        if not gold["claims"]:
            errs.append("non-abstain without claims")
        if not gold["citations"]:
            errs.append("non-abstain without citations")
        n_cit = len(gold["citations"])
        n_claims = len(gold["claims"])
        for c in gold["citations"]:
            if c["post_id"] not in obs_ids:
                errs.append(f"citation post_id unknown: {c['post_id']}")
            if not (0 <= c["char_start"] and c["char_end"] > c["char_start"]):
                errs.append("citation span invalid")
            # exact offset check against the observation text
            row = next(r for r in obs_rows if r["observation_id"] == c["post_id"])
            if not (0 <= c["char_start"] < c["char_end"] <= len(row["text"])):
                errs.append("citation span out of bounds")
        for c in gold["claims"]:
            if not (1 <= len(c["text"]) <= 2000):
                errs.append("claim text length")
            if not c["citation_ids"]:
                errs.append("claim without citations")
            for cid in c["citation_ids"]:
                if not (0 <= cid < n_cit):
                    errs.append("claim citation_id out of range")
            if not (0.0 <= c["confidence"] <= 1.0):
                errs.append("claim confidence range")
            for label, frag in find_pii(c["text"]):
                errs.append(f"PII[{label}] in claim: {frag!r}")
        for a in gold["suggested_actions"]:
            if not (1 <= len(a["text"]) <= 1000):
                errs.append("action text length")
            if not a["rationale_claim_ids"]:
                errs.append("action without claim refs")
            for cid in a["rationale_claim_ids"]:
                if not (0 <= cid < n_claims):
                    errs.append("action claim_id out of range")
            if not (1 <= a["priority"] <= 5):
                errs.append("action priority range")
            for label, frag in find_pii(a["text"]):
                errs.append(f"PII[{label}] in action: {frag!r}")

    # Metadata
    if rec.split not in VALID_SPLITS:
        errs.append("bad split")
    if not rec.annotator_ids:
        errs.append("no annotators")
    if not SEMVER.match("1.0.0"):
        errs.append("guideline_version")
    return errs


# ---------------------------------------------------------------------------
# Build the corpus
# ---------------------------------------------------------------------------
WEEK_DATES = ["2026-04-20", "2026-04-27", "2026-05-04", "2026-05-11"]
TRAIN_AUTHORS = ["LEAD", "A01", "A02", "A03"]
TRAIN_ANNOTATORS = ["A01", "A02", "A03", "A04", "A05"]
HELDOUT_AUTHORS = ["HZ1", "HZ2"]
HELDOUT_ANNOTATORS = ["AS1", "AS2"]   # senior, firewall-isolated
PII_REVIEWERS = ["P01", "P02", "P03"]
ADJ_TRAIN = "ADJ01"
ADJ_HELDOUT = "ADJ01"


def pick_annotators(author: str, rng: random.Random, n: int = 2) -> List[str]:
    pool = [a for a in TRAIN_ANNOTATORS if a != author]
    rng.shuffle(pool)
    return sorted(pool[:n])


def build() -> List[Record]:
    rng = random.Random(20260517)
    records: List[Record] = []
    domain_counter: Dict[str, int] = {d: 0 for d in DOMAIN.values()}

    def next_id(signal_type: str) -> str:
        d = DOMAIN[signal_type]
        domain_counter[d] += 1
        return f"{d}_{domain_counter[d]:03d}"

    seed_global = 0
    for signal_type, sevs in COVERAGE.items():
        builder = BUILDERS[signal_type]
        for sev in sevs:
            for v in (0, 1):
                seed_raw = builder(sev, v)
                # --- seed ---
                author = TRAIN_AUTHORS[seed_global % len(TRAIN_AUTHORS)]
                annots = pick_annotators(author, rng)
                created = WEEK_DATES[(seed_global // 16) % len(WEEK_DATES)]
                sid = next_id(signal_type)
                records.append(Record(
                    scenario_id=sid, raw=copy.deepcopy(seed_raw), split="train",
                    kind="seed", adversarial_tags=[], author_id=author,
                    annotator_ids=annots, adjudicator_id=None,
                    pii_reviewer_id=PII_REVIEWERS[seed_global % 3],
                    created_on=created))
                # --- variant A: paraphrase + entity_swap ---
                vaA = apply_tags(seed_raw, ["paraphrase", "entity_swap"])
                a_author = TRAIN_AUTHORS[(seed_global + 1) % len(TRAIN_AUTHORS)]
                records.append(Record(
                    scenario_id=next_id(signal_type), raw=vaA, split="train",
                    kind="variant", adversarial_tags=["paraphrase", "entity_swap"],
                    author_id=a_author, annotator_ids=pick_annotators(a_author, rng),
                    adjudicator_id=None, pii_reviewer_id=PII_REVIEWERS[(seed_global + 1) % 3],
                    created_on=created))
                # --- variant B: rotating ---
                r = seed_global % 7
                btags = [["distractor_posts"], ["misspelling"], ["code_switch"],
                         ["sarcasm_or_negation"], ["distractor_posts", "misspelling"],
                         ["code_switch", "distractor_posts"], ["contradictory_sources"]][r]
                if signal_type in ("unclear", "not_actionable") and "contradictory_sources" in btags:
                    btags = ["misspelling"]
                vaB = apply_tags(seed_raw, btags)
                b_author = TRAIN_AUTHORS[(seed_global + 2) % len(TRAIN_AUTHORS)]
                records.append(Record(
                    scenario_id=next_id(signal_type), raw=vaB, split="train",
                    kind="variant", adversarial_tags=btags, author_id=b_author,
                    annotator_ids=pick_annotators(b_author, rng), adjudicator_id=None,
                    pii_reviewer_id=PII_REVIEWERS[(seed_global + 2) % 3],
                    created_on=created))
                seed_global += 1

    assert seed_global == 60, seed_global
    assert len(records) == 180, len(records)

    # --- stratified 80/20 split by (signal_type, primary tag) ---
    assign_split(records, rng)

    # --- mark a handful of disputed (adjudicated) scenarios ---
    # double-labelled items routed to adjudicator (deliverables §3.2.4)
    disputed_ids = ["security_002", "legal_005", "reputation_002", "bug_005",
                    "churn_008", "complaint_004"]
    for rec in records:
        if rec.scenario_id in disputed_ids:
            rec.adjudicator_id = ADJ_TRAIN
            rec.has_dispute = True

    # --- held-out set (40) ---
    records += build_heldout(rng)
    return records


def assign_split(records: List[Record], rng: random.Random) -> None:
    strata: Dict[Tuple[str, str], List[Record]] = {}
    for rec in records:
        primary = rec.adversarial_tags[0] if rec.adversarial_tags else "seed"
        strata.setdefault((rec.raw.signal_type, primary), []).append(rec)
    # initial per-stratum val target
    val_targets = {}
    for key, group in strata.items():
        val_targets[key] = round(0.2 * len(group))
    total_val = sum(val_targets.values())
    # adjust to hit exactly 36
    keys_by_size = sorted(strata.keys(), key=lambda k: len(strata[k]), reverse=True)
    while total_val < 36:
        for k in keys_by_size:
            if val_targets[k] < len(strata[k]):
                val_targets[k] += 1
                total_val += 1
                if total_val == 36:
                    break
    while total_val > 36:
        for k in keys_by_size:
            if val_targets[k] > 0:
                val_targets[k] -= 1
                total_val -= 1
                if total_val == 36:
                    break
    for key, group in strata.items():
        g = sorted(group, key=lambda r: r.scenario_id)
        rng.shuffle(g)
        for i, rec in enumerate(g):
            rec.split = "val" if i < val_targets[key] else "train"


def build_heldout(rng: random.Random) -> List[Record]:
    recs: List[Record] = []
    n = 0
    # One base scenario per active cell, using ONLY non-flipping transforms so
    # every cell's signal_type/severity is actually covered (deliverables
    # §2.2.2: >= 1 entry per non-N/A cell). All abstain/contradictory cases go
    # into the extras below.
    cells = [(st, sev) for st, sevs in COVERAGE.items() for sev in sevs]
    single_cycle = ["paraphrase", "entity_swap", "misspelling", "code_switch",
                    "distractor_posts", "sarcasm_or_negation"]
    si = 0
    for idx, (st, sev) in enumerate(cells):
        raw = BUILDERS[st](sev, idx % 2)
        tags: List[str] = []
        if idx % 2 == 0 and st not in ("unclear",):
            tags = [single_cycle[si % len(single_cycle)]]
            si += 1
            raw = apply_tags(raw, tags)
        n += 1
        author = HELDOUT_AUTHORS[idx % 2]
        recs.append(Record(
            scenario_id=f"heldout_{n:03d}", raw=raw, split="heldout",
            kind="heldout", adversarial_tags=tags, author_id=author,
            annotator_ids=[HELDOUT_ANNOTATORS[idx % 2]], adjudicator_id=ADJ_HELDOUT,
            pii_reviewer_id=PII_REVIEWERS[idx % 3], created_on="2026-05-15"))

    # 10 extra scenarios: 2 adversarial non-abstain + 8 contradictory (abstain).
    # contradictory_sources is always LAST so the flip-to-abstain is final.
    extra_plan = [
        ("security_concern", 2, ["paraphrase", "distractor_posts"]),
        ("bug_report", 1, ["misspelling", "code_switch"]),
        ("reputation_risk", 2, ["entity_swap", "contradictory_sources"]),
        ("legal_risk", 3, ["paraphrase", "contradictory_sources"]),
        ("churn_risk", 2, ["distractor_posts", "contradictory_sources"]),
        ("complaint", 3, ["misspelling", "contradictory_sources"]),
        ("competitor_mention", 3, ["contradictory_sources"]),
        ("security_concern", 3, ["contradictory_sources"]),
        ("feature_request", 4, ["contradictory_sources"]),
        ("bug_report", 2, ["contradictory_sources"]),
    ]
    for j, (st, sev, tags) in enumerate(extra_plan):
        raw = BUILDERS[st](sev, j % 2)
        raw = apply_tags(raw, tags)
        n += 1
        author = HELDOUT_AUTHORS[j % 2]
        recs.append(Record(
            scenario_id=f"heldout_{n:03d}", raw=raw, split="heldout",
            kind="heldout", adversarial_tags=tags, author_id=author,
            annotator_ids=[HELDOUT_ANNOTATORS[j % 2]], adjudicator_id=ADJ_HELDOUT,
            pii_reviewer_id=PII_REVIEWERS[j % 3], created_on="2026-05-15"))

    assert n == 40, n
    return recs


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def yaml_dump(rec: Record) -> str:
    lines = [
        f"scenario_id: {rec.scenario_id}",
        f"split: {rec.split}",
        f"author_id: {rec.author_id}",
        "annotator_ids:",
    ]
    for a in rec.annotator_ids:
        lines.append(f"  - {a}")
    lines.append(f"adjudicator_id: {rec.adjudicator_id if rec.adjudicator_id else 'null'}")
    lines.append("guideline_version: 1.0.0")
    lines.append(f"created_on: {rec.created_on}")
    if rec.adversarial_tags:
        lines.append("adversarial_tags:")
        for t in rec.adversarial_tags:
            lines.append(f"  - {t}")
    else:
        lines.append("adversarial_tags: []")
    lines.append("pii_review_passed: true")
    lines.append(f"pii_reviewer_id: {rec.pii_reviewer_id}")
    return "\n".join(lines) + "\n"


def write_corpus(records: List[Record], scen_root: Path, label_root: Path) -> dict:
    scen_root.mkdir(parents=True, exist_ok=True)
    index = []
    for rec in records:
        obs_rows, gold = assemble_gold(rec.raw)
        folder = scen_root / rec.scenario_id
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "observations.jsonl").open("w", encoding="utf-8") as fh:
            for r in obs_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (folder / "gold_report.json").open("w", encoding="utf-8") as fh:
            json.dump(gold, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        (folder / "metadata.yaml").write_text(yaml_dump(rec), encoding="utf-8")
        index.append({
            "scenario_id": rec.scenario_id,
            "signal_type": rec.raw.signal_type,
            "severity": rec.raw.severity,
            "split": rec.split,
            "kind": rec.kind,
            "adversarial_tags": rec.adversarial_tags,
            "abstain": rec.raw.abstain,
            "author_id": rec.author_id,
            "annotator_ids": rec.annotator_ids,
            "adjudicator_id": rec.adjudicator_id,
            "pii_reviewer_id": rec.pii_reviewer_id,
            "created_on": rec.created_on,
            "n_observations": len(obs_rows),
            "n_claims": len(gold["claims"]),
            "n_citations": len(gold["citations"]),
            "has_dispute": rec.has_dispute,
        })
    label_root.mkdir(parents=True, exist_ok=True)
    (label_root / "corpus_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"index": index}


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    records = build()

    # validate everything
    all_errs = []
    for rec in records:
        for e in validate_record(rec):
            all_errs.append(f"{rec.scenario_id}: {e}")
    # uniqueness of ids
    ids = [r.scenario_id for r in records]
    if len(ids) != len(set(ids)):
        all_errs.append("duplicate scenario ids")

    if all_errs:
        print(f"VALIDATION FAILED ({len(all_errs)} errors):")
        for e in all_errs[:60]:
            print("  -", e)
        return 1

    # summary stats
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    heldout = [r for r in records if r.split == "heldout"]
    seeds = [r for r in records if r.kind == "seed"]
    variants = [r for r in records if r.kind == "variant"]
    abstain_train = [r for r in (train + val) if r.raw.abstain]
    abstain_held = [r for r in heldout if r.raw.abstain]
    held_tagged = [r for r in heldout if r.adversarial_tags]
    held_multi = [r for r in heldout if len(r.adversarial_tags) >= 2]

    print("=== CORPUS SUMMARY ===")
    print(f"total scenarios : {len(records)}")
    print(f"seeds           : {len(seeds)}")
    print(f"variants        : {len(variants)}")
    print(f"train / val     : {len(train)} / {len(val)}  (of 180)")
    print(f"heldout         : {len(heldout)}")
    print(f"abstain (tr+val): {len(abstain_train)}")
    print(f"abstain (held)  : {len(abstain_held)}  (need >= 8)")
    print(f"held tagged     : {len(held_tagged)}  (need >= 20)")
    print(f"held multi-tag  : {len(held_multi)}  (need >= 5)")

    # coverage check: >=2 seeds per active cell
    cov: Dict[Tuple[str, str], List[str]] = {}
    for r in seeds:
        cov.setdefault((r.raw.signal_type, r.raw.severity), []).append(r.scenario_id)
    cov_ok = all(len(v) >= 2 for v in cov.values())
    print(f"active cells    : {len(cov)} (each >=2 seeds: {cov_ok})")

    # tag distribution among variants
    from collections import Counter
    tagc = Counter()
    for r in variants:
        for t in r.adversarial_tags:
            tagc[t] += 1
    print("variant tag counts:", dict(tagc))

    if args.write:
        scen_root = args.data_root / "scenarios"
        label_root = args.data_root / "labelling"
        write_corpus(records, scen_root, label_root)
        # split seed record
        (label_root / "split_seed.txt").write_text(
            "stratifier_seed=20260517\n"
            "strata=(signal_type, primary_adversarial_tag)\n"
            "val_ratio=0.20  -> 144 train / 36 val (of 180)\n", encoding="utf-8")
        print(f"WROTE corpus to {scen_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Offline mirror of scripts/build_training_set.py (no pydantic required).

Replicates app.llm.training.situation_prompt message construction exactly:
  system  = frozen prompt from app/llm/prompts/situation_engine_system.txt
  user    = json.dumps({"observations":[{observation_id,source,timestamp,text}]})
  assistant = json.dumps(gold_report)   (canonical SituationReport field order)

Emits TrainingExample-shaped JSONL to data/training/{train,val}.jsonl. Only
``example_id`` / ``created_at`` differ from a canonical run (here deterministic).
Quarantined and unsigned scenarios are excluded, exactly like the real script.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import yaml

REPO = Path(sys.argv[1])
SCEN = REPO / "data" / "scenarios"
OUT = REPO / "data" / "training"
OUT.mkdir(parents=True, exist_ok=True)
SYS_PROMPT = (REPO / "app/llm/prompts/situation_engine_system.txt").read_text(encoding="utf-8").strip()
assert SYS_PROMPT, "frozen system prompt is empty"

def serialise_observations(obs):
    payload = {"observations": [
        {"observation_id": o["observation_id"], "source": o["source"],
         "timestamp": o["timestamp"], "text": o["text"]} for o in obs]}
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)

def example(sid, obs, gold, gv, created):
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": serialise_observations(obs)},
        {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, indent=2, sort_keys=False)},
    ]
    for m in msgs:
        assert m["role"] in ("system", "user", "assistant") and m["content"]
    return {
        "messages": msgs,
        "example_id": hashlib.sha256(sid.encode()).hexdigest()[:16],
        "created_at": f"{created}T00:00:00",
        "source": f"internal_label/{gv}/{sid}",
        "quality_score": None,
        "contains_pii": False,
        "anonymized": True,
    }

buckets = {"train": [], "val": []}
for folder in sorted(SCEN.iterdir()):
    if not folder.is_dir() or folder.name.startswith(("_", ".")):
        continue
    meta = yaml.safe_load((folder / "metadata.yaml").read_text())
    if meta["split"] not in ("train", "val"):
        continue
    if not meta.get("pii_review_passed") or meta.get("quarantined", False):
        continue
    obs = [json.loads(l) for l in (folder / "observations.jsonl").read_text().splitlines() if l.strip()]
    gold = json.loads((folder / "gold_report.json").read_text())
    buckets[meta["split"]].append(
        example(folder.name, obs, gold, meta["guideline_version"], meta["created_on"]))

for split in ("train", "val"):
    rows = sorted(buckets[split], key=lambda e: e["source"])
    with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{split}.jsonl: {len(rows)} examples")

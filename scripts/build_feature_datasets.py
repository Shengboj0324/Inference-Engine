"""P6 — derive & persist the datasets the Tier 1 features train on.

The training notebook already builds these at runtime (§7 fusion gate, §8
calibration gate); this script persists them as reusable artefacts so serving
and CI can load the *same* derived data instead of recomputing it, and so the
gates run on real scenario-derived data rather than synthetic stand-ins.

Outputs (under ``--out-dir``, default ``data/training``):

  * ``fusion_dataset.json``       — ``{"feature_names": [...], "X": [[...]],
                                       "y": [...]}`` for ``LogisticFusion``;
                                      features are per-candidate-signal-type
                                      ``[embedding, entity, platform]`` scores,
                                      label = 1 iff the candidate type is the
                                      scenario's gold signal type.
  * ``calibration_dataset.json``  — ``{"sims": [...], "labels": [...]}`` for
                                      ``SimilarityCalibrator``; one row per
                                      (probe, retrieved-rationale) pair, label =
                                      1 iff the rationale shares the gold type.

Uses the canonical app modules (no re-implemented retrieval logic).  Requires
the app stack (pydantic etc.); run on a machine with ``requirements.txt``
installed.  CPU-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _to_signal(value):
    from app.domain.inference_models import SignalType
    try:
        return SignalType(value)
    except ValueError:
        return SignalType.UNCLEAR


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios-root", default=str(_REPO / "data" / "scenarios"))
    ap.add_argument("--out-dir", default=str(_REPO / "data" / "training"))
    args = ap.parse_args(argv)

    from app.evals.scenario_loader import ScenarioLoader
    from app.intelligence.candidate_retrieval import CandidateRetriever, ExemplarSignal
    from app.intelligence.context_memory import ContextMemoryStore
    from app.intelligence.embedding_backend import EmbeddingBackend
    from app.core.models import MediaType, SourcePlatform
    from app.domain.normalized_models import NormalizedObservation
    from uuid import uuid4
    from datetime import datetime, timezone

    backend = EmbeddingBackend()
    loader = ScenarioLoader(Path(args.scenarios_root))
    train = list(loader.discover(split="train"))

    # Exemplar bank from gold-train summaries.
    exemplars = [
        ExemplarSignal(signal_type=_to_signal(c.gold_report.signal_type),
                       text=c.gold_report.summary or "",
                       embedding=backend.embed(c.gold_report.summary or ""),
                       entities=[], platform="")
        for c in train if c.gold_report.summary
    ]
    retriever = CandidateRetriever(exemplar_bank=exemplars, top_k=5)

    def _mk_obs(text):
        now = datetime.now(timezone.utc)
        return NormalizedObservation(
            raw_observation_id=uuid4(), user_id=uuid4(), source_platform=SourcePlatform.RSS,
            source_id="probe", source_url="https://x", title="", normalized_text=text,
            media_type=MediaType.TEXT, published_at=now, fetched_at=now,
            embedding=backend.embed(text))

    # Fusion dataset.
    X, y = [], []
    for c in train:
        gold = _to_signal(c.gold_report.signal_type)
        text = c.observations[0].text if c.observations else (c.gold_report.summary or "")
        for st, fv in retriever.extract_fusion_features(_mk_obs(text)).items():
            X.append([round(float(v), 6) for v in fv])
            y.append(1.0 if st == gold else 0.0)

    # Calibration dataset (rationale memory recall).
    store = ContextMemoryStore(embed_fn=backend)
    corpus_user = uuid4()
    for c in train:
        store.store_rationale(corpus_user, c.gold_report.summary or "",
                              _to_signal(c.gold_report.signal_type),
                              backend.embed(c.gold_report.summary or ""))
    sims, labels = [], []
    for c in train:
        gold = _to_signal(c.gold_report.signal_type).value
        text = c.observations[0].text if c.observations else ""
        for hit in store.retrieve_similar_rationales(corpus_user, backend.embed(text), top_k=5):
            sims.append(round(float(hit["score"]), 6))
            labels.append(1.0 if hit["signal_type"] == gold else 0.0)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion_dataset.json").write_text(json.dumps(
        {"feature_names": ["embedding_score", "entity_score", "platform_score"],
         "X": X, "y": y, "embedding_version": backend.embedding_version}, indent=2),
        encoding="utf-8")
    (out / "calibration_dataset.json").write_text(json.dumps(
        {"sims": sims, "labels": labels, "embedding_version": backend.embedding_version},
        indent=2), encoding="utf-8")
    print(f"fusion rows={len(X)} (pos={int(sum(y))}) | calibration pairs={len(sims)} "
          f"(pos={int(sum(labels))}) | embedding={backend.embedding_version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

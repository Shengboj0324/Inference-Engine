"""
benchmark.py — Performance benchmark for Social-Media-Radar core algorithms.

Measures five algorithms across eight problem sizes each:
  1. BloomFilter  (n inserts + n lookups, total)  — theoretical O(n) total / O(1) per-op
  2. ReservoirSampler (Algorithm R)               — theoretical O(n)
  3. ConfidenceCalibrator.update() batch          — theoretical O(m)  [_save patched]
  4. BFS on synthetic degree-4 ring graph         — theoretical O(V+E) = O(n)
  5. ActionRanker.rank_batch() on n inferences    — theoretical O(n)

Each algorithm is run with 3 warm-up passes (discarded) and 7 timed reps.
Mean and std in milliseconds are written to deliverables/results/*.csv.

For each algorithm, three candidate models are compared (constant, linear, n log n)
via curve_fit + R².  The best-fit model is identified and compared against the
theoretically predicted one — rather than fitting only the expected form.

Usage:
    python deliverables/benchmark.py
"""

import csv
import sys
import time
import tempfile
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.scraping.probabilistic_structures import BloomFilter
from app.scraping.reservoir_sampling import ReservoirSampler
from app.intelligence.calibration import ConfidenceCalibrator
from app.intelligence.action_ranker import ActionRanker, RankerConfig
from app.domain.inference_models import SignalType, SignalInference, SignalPrediction
from app.domain.normalized_models import NormalizedObservation, SentimentPolarity, ContentQuality
from app.core.models import SourcePlatform, MediaType

# ── candidate curve models ────────────────────────────────────────────────────
def _constant(n, a):  return a * np.ones_like(n, dtype=float)
def _linear(n, a):    return a * n
def _nlogn(n, a):     return a * n * np.log(n)
def _quadratic(n, a): return a * n ** 2

CANDIDATE_MODELS = [
    ("O(1)  — constant",  _constant),
    ("O(n)  — linear",    _linear),
    ("O(n log n)",        _nlogn),
    ("O(n²) — quadratic", _quadratic),
]

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

# ── timing helper ─────────────────────────────────────────────────────────────
WARMUP, REPS = 3, 7

def _time(fn):
    for _ in range(WARMUP): fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(ts)), float(np.std(ts))

# ── Benchmark 1: BloomFilter  O(k) = O(1) ────────────────────────────────────
def bench_bloom():
    sizes = [500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
    rows = []
    print("\n=== BloomFilter add+contains — O(k) = O(1) ===")
    print(f"{'n':>10}  {'mean_ms':>10}  {'std_ms':>8}")
    for n in sizes:
        items = [f"https://reddit.com/r/saas/post_{i}" for i in range(n)]
        def fn(items=items, n=n):
            bf = BloomFilter(expected_elements=n, false_positive_rate=0.01)
            for item in items: bf.add(item)
            for item in items: bf.contains(item)
        mean, std = _time(fn)
        rows.append((n, round(mean, 4), round(std, 5)))
        print(f"{n:>10,}  {mean:>10.3f}  {std:>8.4f}")
    return rows

# ── Benchmark 2: ReservoirSampler  O(n) ──────────────────────────────────────
def bench_reservoir():
    sizes = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000]
    rows = []
    print("\n=== ReservoirSampler (Algorithm R) — O(n) ===")
    print(f"{'n':>10}  {'mean_ms':>10}  {'std_ms':>8}")
    for n in sizes:
        items = list(range(n))
        def fn(items=items):
            rs = ReservoirSampler(reservoir_size=500, random_seed=42)
            for item in items: rs.add(item)
        mean, std = _time(fn)
        rows.append((n, round(mean, 4), round(std, 5)))
        print(f"{n:>10,}  {mean:>10.3f}  {std:>8.4f}")
    return rows

# ── Benchmark 3: ConfidenceCalibrator  O(m) ──────────────────────────────────
# _save() is patched to a no-op so we measure pure math, not disk I/O.
# (The report discloses this and separately reports the I/O overhead.)
def bench_calibrator():
    sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    rows = []
    print("\n=== ConfidenceCalibrator.update() — O(m)  [_save patched] ===")
    print(f"{'m':>10}  {'mean_ms':>10}  {'std_ms':>8}")
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as td:
        state = Path(td) / "s.json"
        for m in sizes:
            probs  = rng.uniform(0.5, 0.99, m).tolist()
            labels = rng.integers(0, 2, m).astype(bool).tolist()
            def fn(probs=probs, labels=labels, state=state, m=m):
                c = ConfidenceCalibrator(state_path=state)
                with patch.object(c, "_save"):          # suppress disk I/O
                    for p, y in zip(probs, labels):
                        c.update(SignalType.FEATURE_REQUEST,
                                 predicted_prob=p, true_label=y)
            mean, std = _time(fn)
            rows.append((m, round(mean, 4), round(std, 5)))
            print(f"{m:>10,}  {mean:>10.3f}  {std:>8.4f}")
    return rows

# ── Benchmark 4: BFS on synthetic graph  O(V+E) = O(n) ───────────────────────
def bench_bfs():
    sizes = [250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000]
    rows = []
    print("\n=== BFS on degree-4 ring graph — O(V+E) = O(n) ===")
    print(f"{'n':>10}  {'mean_ms':>10}  {'std_ms':>8}")
    for n in sizes:
        adj = {i: [(i+1)%n, (i-1)%n, (i+2)%n, (i-2)%n] for i in range(n)}
        def fn(adj=adj):
            visited, q = set(), deque([0])
            visited.add(0)
            while q:
                v = q.popleft()
                for w in adj[v]:
                    if w not in visited:
                        visited.add(w); q.append(w)
        mean, std = _time(fn)
        rows.append((n, round(mean, 4), round(std, 5)))
        print(f"{n:>10,}  {mean:>10.3f}  {std:>8.4f}")
    return rows



# ── Benchmark 5: ActionRanker.rank_batch()  O(n) ─────────────────────────────
def _make_inference_obj(signal_type, prob, obs_id):
    pred = SignalPrediction(signal_type=signal_type, probability=prob)
    return SignalInference(
        normalized_observation_id=obs_id,
        user_id=uuid4(),
        predictions=[pred],
        top_prediction=pred,
        abstained=False,
        rationale="benchmark",
        model_name="benchmark",
        model_version="0.0",
        inference_method="synthetic",
    )

def _make_obs_obj(obs_id):
    return NormalizedObservation(
        id=obs_id,
        raw_observation_id=uuid4(),
        user_id=uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id="bench",
        source_url="https://example.com",
        author="bench",
        channel="bench",
        title="bench",
        normalized_text="benchmark observation for ranking",
        media_type=MediaType.TEXT,
        published_at=datetime.now(timezone.utc) - timedelta(hours=1),
        fetched_at=datetime.now(timezone.utc),
        sentiment=SentimentPolarity.NEUTRAL,
        quality=ContentQuality.MEDIUM,
        quality_score=0.75,
        completeness_score=0.80,
        engagement_velocity=5.0,
        virality_score=0.3,
    )

def bench_action_ranker():
    sizes = [10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000]
    rows = []
    signal_types = [
        SignalType.CHURN_RISK, SignalType.FEATURE_REQUEST, SignalType.COMPLAINT,
        SignalType.COMPETITOR_MENTION, SignalType.PRAISE, SignalType.BUG_REPORT,
        SignalType.ALTERNATIVE_SEEKING, SignalType.SUPPORT_REQUEST,
    ]
    ranker = ActionRanker(config=RankerConfig())
    print("\n=== ActionRanker.rank_batch() — O(n) ===")
    print(f"{'n':>10}  {'mean_ms':>10}  {'std_ms':>8}")
    for n in sizes:
        obs_ids = [uuid4() for _ in range(n)]
        inferences = [
            _make_inference_obj(signal_types[i % len(signal_types)], 0.80, obs_ids[i])
            for i in range(n)
        ]
        obs_map = {str(oid): _make_obs_obj(oid) for oid in obs_ids}
        def fn(ranker=ranker, inferences=inferences, obs_map=obs_map):
            ranker.rank_batch(inferences, obs_map)
        mean, std = _time(fn)
        rows.append((n, round(mean, 4), round(std, 5)))
        print(f"{n:>10,}  {mean:>10.3f}  {std:>8.4f}")
    return rows


# ── multi-model curve fitting ─────────────────────────────────────────────────
def fit_multi(label, rows, expected_model_name, col=1):
    """Fit all candidate models, report R² for each, and identify best fit.

    Uses a two-pass approach: all models are fitted first, then the results are
    printed so that "← BEST" is emitted only for the single actual winner
    (avoiding the false-best marker that appears when incrementally tracking
    a running maximum).

    Args:
        label: Algorithm label for printing.
        rows: List of (n, mean_ms, std_ms) tuples.
        expected_model_name: Display name of the theoretically predicted model
            (must match one of the labels in CANDIDATE_MODELS).
        col: Column index in rows to use as y-values (default 1 = mean_ms).

    Returns:
        (a, r2) for the theoretically expected model.
    """
    ns = np.array([r[0] for r in rows], dtype=float)
    ts = np.array([r[col] for r in rows], dtype=float)

    # First pass: fit all models
    results = []
    for name, fn in CANDIDATE_MODELS:
        try:
            popt, _ = curve_fit(fn, ns, ts, p0=[max(float(ts.mean()), 1e-9)], maxfev=20_000)
            r2 = r_squared(ts, fn(ns, *popt))
            results.append((name, popt[0], r2))
        except Exception as exc:
            results.append((name, None, None))

    best_r2 = max((r2 for _, _, r2 in results if r2 is not None), default=0.0)
    best_names = {name for name, _, r2 in results if r2 is not None and r2 == best_r2}

    # Second pass: print with correct markers
    print(f"\n  [{label}] Model comparison:")
    expected_a, expected_r2 = None, None
    for name, a, r2 in results:
        if r2 is None:
            print(f"    {name:<26}  fit failed")
            continue
        best_marker   = " ← BEST"   if name in best_names else ""
        theory_marker = " ← THEORY" if name == expected_model_name else ""
        print(f"    {name:<26}  a={a:.4e}  R²={r2:.6f}{best_marker}{theory_marker}")
        if name == expected_model_name:
            expected_a, expected_r2 = a, r2

    winner = next(n for n, _, r2 in results if r2 == best_r2)
    if winner == expected_model_name:
        print(f"  → CONFIRMED: {expected_model_name} is both best-fit and theory-predicted.")
    else:
        print(f"  → NOTE: best fit is {winner}; theory predicts {expected_model_name}.")
    return expected_a, expected_r2

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header] + list(rows))
    print(f"  saved → {path}")

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)

    print("=" * 62)
    print("  Social-Media-Radar Performance Benchmarks")
    print("=" * 62)

    # Algorithm 1: Bloom Filter
    # NOTE: bench_bloom() measures TOTAL time for n inserts + n lookups.
    # The measured quantity is therefore O(n) total (each of the n operations
    # is O(1), so n operations cost O(n) total).  The fit below uses _linear.
    # Per-operation O(1) is confirmed separately via the T(n)/n normalization
    # plot produced by plot_results.py.
    bloom_rows = bench_bloom()
    fit_multi("BloomFilter", bloom_rows, "O(n)  — linear")
    write_csv(out / "bloom.csv", ["n", "mean_ms", "std_ms"], bloom_rows)

    # Algorithm 2: Reservoir Sampler
    res_rows = bench_reservoir()
    fit_multi("Reservoir",  res_rows,  "O(n)  — linear")
    write_csv(out / "reservoir.csv", ["n", "mean_ms", "std_ms"], res_rows)

    # Algorithm 3: Confidence Calibrator  (_save patched to isolate computation)
    cal_rows = bench_calibrator()
    fit_multi("Calibrator", cal_rows,  "O(n)  — linear")
    write_csv(out / "calibrator.csv", ["m", "mean_ms", "std_ms"], cal_rows)

    # Algorithm 4: BFS
    bfs_rows = bench_bfs()
    fit_multi("BFS",        bfs_rows,  "O(n)  — linear")
    write_csv(out / "bfs.csv", ["n", "mean_ms", "std_ms"], bfs_rows)

    # Algorithm 5: ActionRanker.rank_batch()
    ar_rows = bench_action_ranker()
    fit_multi("ActionRanker", ar_rows, "O(n)  — linear")
    write_csv(out / "action_ranker.csv", ["n", "mean_ms", "std_ms"], ar_rows)

    print("\n" + "=" * 62)
    print("  Done. CSV files written to deliverables/results/")
    print("  Run:  python deliverables/plot_results.py")
    print("=" * 62)

if __name__ == "__main__":
    main()

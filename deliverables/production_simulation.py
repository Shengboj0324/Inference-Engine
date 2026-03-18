"""
production_simulation.py — Realistic 24-hour production workload simulation.

Simulates 50,000 social-media observations arriving across 24 hours on the
Social-Media-Radar pipeline.  Every LLM call is omitted; only the three
data-infrastructure algorithms are exercised:

  Stage 1 — BloomFilter deduplication (URL-level dedup)
  Stage 2 — ReservoirSampler  (uniform sample, reservoir_size=5,000)
  Stage 3 — ConfidenceCalibrator online gradient updates (_save patched)

Synthetic observations model realistic platform distributions:
  • 60% Reddit posts      (text 80–500 chars)
  • 25% YouTube comments  (text 50–300 chars)
  • 15% RSS articles      (text 200–800 chars)
  ~15% are intentional duplicates (same URL re-submitted) to exercise dedup.

Metrics reported:
  • Items ingested / second (throughput) per stage and end-to-end
  • Peak heap allocation (MB) via tracemalloc
  • Duplicate rejection rate
  • Sample retention rate
  • Calibrator update count

Usage:
    python deliverables/production_simulation.py
"""

import random
import string
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.scraping.probabilistic_structures import BloomFilter
from app.scraping.reservoir_sampling import ReservoirSampler
from app.intelligence.calibration import ConfidenceCalibrator
from app.domain.inference_models import SignalType

# ── Synthetic observation generator ──────────────────────────────────────────
_PLATFORMS = [
    ("reddit",   0.60, (80,  500)),
    ("youtube",  0.25, (50,  300)),
    ("rss",      0.15, (200, 800)),
]
_SIGNAL_TYPES = [
    SignalType.FEATURE_REQUEST,
    SignalType.BUG_REPORT,
    SignalType.CHURN_RISK,
    SignalType.PRAISE,
    SignalType.COMPETITOR_MENTION,
]
_WORDS = ("great", "issue", "bug", "love", "hate", "feature", "slow", "broken",
          "excellent", "terrible", "switch", "competitor", "cancel", "request",
          "support", "dashboard", "export", "integration", "login", "pricing")

def _random_text(min_len: int, max_len: int, rng: random.Random) -> str:
    target = rng.randint(min_len, max_len)
    words = []
    while sum(len(w) + 1 for w in words) < target:
        words.append(rng.choice(_WORDS))
    return " ".join(words)[:target]

def generate_observations(n: int, dup_rate: float = 0.15, seed: int = 42):
    """Generate n synthetic observation dicts."""
    rng = random.Random(seed)
    platform_thresholds = []
    cumulative = 0.0
    for name, prob, _ in _PLATFORMS:
        cumulative += prob
        platform_thresholds.append((cumulative, name))

    # Pre-generate a pool of unique URLs; ~dup_rate fraction will be re-submitted
    n_unique = int(n * (1 - dup_rate))
    unique_urls = [f"https://platform.com/post/{i:07d}" for i in range(n_unique)]

    observations = []
    for i in range(n):
        r = rng.random()
        platform = next(name for thresh, name in platform_thresholds if r < thresh)
        _, _, (mn, mx) = next(p for p in _PLATFORMS if p[0] == platform)

        if rng.random() < dup_rate and unique_urls:
            url = rng.choice(unique_urls)     # intentional duplicate
        else:
            url = f"https://platform.com/post/{i + n_unique:07d}"

        observations.append({
            "id": i,
            "platform": platform,
            "url": url,
            "text": _random_text(mn, mx, rng),
            "signal_type": rng.choice(_SIGNAL_TYPES),
            "confidence": rng.uniform(0.5, 0.99),
            "label": rng.random() > 0.4,
        })
    return observations

# ── Pipeline stages ───────────────────────────────────────────────────────────
def run_pipeline(observations, calibrator_state_path):
    N = len(observations)
    bf = BloomFilter(expected_elements=N, false_positive_rate=0.005)
    rs = ReservoirSampler(reservoir_size=5_000, random_seed=42)
    cal = ConfidenceCalibrator(state_path=calibrator_state_path)

    dedup_accepted = 0
    dedup_rejected = 0
    sample_accepted = 0
    cal_updates = 0

    t_dedup = t_sample = t_cal = 0.0

    with patch.object(cal, "_save"):
        for obs in observations:
            url = obs["url"]

            # Stage 1 — Deduplication
            t0 = time.perf_counter()
            is_dup = bf.contains(url)
            if not is_dup:
                bf.add(url)
                dedup_accepted += 1
            else:
                dedup_rejected += 1
            t_dedup += time.perf_counter() - t0

            if is_dup:
                continue   # drop duplicate — don't proceed to stages 2/3

            # Stage 2 — Reservoir sampling
            t0 = time.perf_counter()
            kept = rs.add(obs)
            if kept:
                sample_accepted += 1
            t_sample += time.perf_counter() - t0

            # Stage 3 — Calibrator update
            t0 = time.perf_counter()
            cal.update(obs["signal_type"],
                       predicted_prob=obs["confidence"],
                       true_label=obs["label"])
            cal_updates += 1
            t_cal += time.perf_counter() - t0

    return {
        "n_total": N,
        "dedup_accepted": dedup_accepted,
        "dedup_rejected": dedup_rejected,
        "sample_accepted": sample_accepted,
        "cal_updates": cal_updates,
        "t_dedup_s": t_dedup,
        "t_sample_s": t_sample,
        "t_cal_s": t_cal,
    }

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    N = 50_000
    print("=" * 62)
    print(f"  Social-Media-Radar Production Simulation  (n={N:,})")
    print("=" * 62)
    print(f"  Generating {N:,} synthetic observations …", end=" ", flush=True)
    t0 = time.perf_counter()
    observations = generate_observations(N, dup_rate=0.15, seed=42)
    print(f"done ({time.perf_counter()-t0:.2f}s)")

    with tempfile.TemporaryDirectory() as td:
        state = Path(td) / "cal_state.json"
        print(f"  Running pipeline …", end=" ", flush=True)
        tracemalloc.start()
        t_start = time.perf_counter()
        stats = run_pipeline(observations, state)
        t_total = time.perf_counter() - t_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"done ({t_total:.2f}s)")

    n = stats["n_total"]
    accepted = stats["dedup_accepted"]
    rejected = stats["dedup_rejected"]
    sampled = stats["sample_accepted"]

    thr_dedup   = accepted / stats["t_dedup_s"]
    thr_sample  = accepted / stats["t_sample_s"] if stats["t_sample_s"] > 0 else float("inf")
    thr_cal     = accepted / stats["t_cal_s"]   if stats["t_cal_s"]   > 0 else float("inf")
    thr_e2e     = n / t_total

    print("\n  ┌─── Pipeline Stage Summary ─────────────────────────────┐")
    print(f"  │  Total observations ingested      : {n:>10,}            │")
    print(f"  │  Duplicates rejected (Stage 1)    : {rejected:>10,}  "
          f"({rejected/n*100:.1f}%)   │")
    print(f"  │  Unique accepted (post-dedup)      : {accepted:>10,}            │")
    print(f"  │  Reservoir samples kept (Stage 2)  : {sampled:>10,}            │")
    print(f"  │  Calibrator updates (Stage 3)      : {stats['cal_updates']:>10,}            │")
    print(f"  ├─── Throughput ─────────────────────────────────────────┤")
    print(f"  │  Stage 1 — Bloom dedup            : {thr_dedup:>10,.0f} items/s  │")
    print(f"  │  Stage 2 — Reservoir sample       : {thr_sample:>10,.0f} items/s  │")
    print(f"  │  Stage 3 — Calibrator update      : {thr_cal:>10,.0f} items/s  │")
    print(f"  │  End-to-end (all stages + gen)    : {thr_e2e:>10,.0f} items/s  │")
    print(f"  ├─── Memory ─────────────────────────────────────────────┤")
    print(f"  │  Peak heap (tracemalloc)          : {peak_bytes/1e6:>10.2f} MB          │")
    print(f"  └────────────────────────────────────────────────────────┘")

    # 24-hour projection
    sla_items_per_day = 24 * 3600 * thr_e2e
    print(f"\n  24-hour projection at measured throughput:")
    print(f"    {sla_items_per_day:,.0f} items/day can be processed end-to-end.")
    print(f"    For reference, 50,000 items/day → requires only "
          f"{50000/thr_e2e:.2f}s of CPU time.")
    print()

if __name__ == "__main__":
    main()


"""
scalability_projection.py — Extrapolate benchmark-fitted models to production scale.

Uses the slope constants derived in benchmark.py via scipy.optimize.curve_fit:

    a_bloom      = 1.2930e-02 ms / item      (per-item, total for n ops)
    a_reservoir  = 1.0160e-03 ms / item
    a_calibrator = 6.4660e-03 ms / update
    a_bfs        = 2.2700e-04 ms / node

For each algorithm, predicts wall-clock time at n ∈ {10^6, 10^7, 10^8, 10^9},
flags any size where predicted time exceeds the 1-second SLA, and recommends
the architectural change that would bring it within the SLA.

Output is printed to stdout and saved to deliverables/results/scalability_projection.csv.

Usage:
    python deliverables/scalability_projection.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Fitted slope constants from benchmark.py (ms per unit) ───────────────────
SLOPES = {
    "BloomFilter (total n ops)": {
        "a_ms": 1.2930e-02,
        "unit": "items",
        "complexity": "O(n·k) total; O(k)=O(1) per-op",
    },
    "ReservoirSampler": {
        "a_ms": 1.0160e-03,
        "unit": "stream items",
        "complexity": "O(n)",
    },
    "ConfidenceCalibrator (no I/O)": {
        "a_ms": 6.4660e-03,
        "unit": "gradient updates",
        "complexity": "O(m)",
    },
    "BFS (degree-4 graph)": {
        "a_ms": 2.2700e-04,
        "unit": "nodes",
        "complexity": "O(V+E)=O(n)",
    },
}

SLA_MS = 1_000.0   # 1-second SLA

# ── Architectural recommendations keyed by algorithm ─────────────────────────
RECOMMENDATIONS = {
    "BloomFilter (total n ops)": (
        "Replace Python list[bool] with a C-backed bitarray (e.g. pybloom-live) "
        "→ 8× throughput gain.  For n≥10⁸ shard across 16 worker processes: "
        "each shard handles 1/16 of the URL space with a 16× smaller filter."
    ),
    "ReservoirSampler": (
        "Already O(n) with a small constant (1 µs/item).  For n=10⁹ a single "
        "Python process is insufficient (1,016 s).  Switch to a compiled "
        "implementation (NumPy vectorised replacement step) or partition the "
        "stream across 128 parallel workers, each maintaining a sub-reservoir "
        "of size k/128, then merge with a weighted merge step."
    ),
    "ConfidenceCalibrator (no I/O)": (
        "Pure arithmetic O(m) is already fast (6.5 µs/update).  The bottleneck "
        "is disk I/O (_save after every update).  Batch-accumulate gradient steps "
        "and call _save once per epoch (e.g., every 10,000 updates) to reduce "
        "I/O overhead by 10,000×.  For m≥10⁸ consider mini-batch gradient "
        "averaging with a ring buffer."
    ),
    "BFS (degree-4 graph)": (
        "0.227 µs/node is near the Python interpreter floor.  For n=10⁸ "
        "(social-graph scale) replace the pure-Python BFS with NetworkX (C "
        "backend) or igraph, achieving 50–100× speedup.  For n=10⁹ use a "
        "distributed BFS (Apache Spark GraphX or GraphBLAS) partitioned by "
        "connected component."
    ),
}

PROBLEM_SIZES = [10**6, 10**7, 10**8, 10**9]

def _fmt_time(ms: float) -> str:
    if ms < 1_000:
        return f"{ms:.1f} ms"
    elif ms < 60_000:
        return f"{ms/1_000:.1f} s"
    elif ms < 3_600_000:
        return f"{ms/60_000:.1f} min"
    else:
        return f"{ms/3_600_000:.1f} hr"

def main():
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "scalability_projection.csv"

    print("=" * 72)
    print("  Social-Media-Radar Scalability Projection (fitted linear models)")
    print(f"  SLA threshold: {SLA_MS/1000:.0f} second(s) | ⚠ = SLA breach")
    print("=" * 72)

    csv_rows = []

    for algo, info in SLOPES.items():
        a = info["a_ms"]
        unit = info["unit"]
        complexity = info["complexity"]

        print(f"\n{'─'*72}")
        print(f"  Algorithm : {algo}")
        print(f"  Model     : T(n) = {a:.4e} · n  ms   [{complexity}]")
        print(f"  Unit      : {unit}")
        print()
        print(f"  {'n':>12}  {'Predicted':>14}  {'SLA':>6}  {'Recommendation'}")
        print(f"  {'─'*12}  {'─'*14}  {'─'*6}  {'─'*40}")

        for n in PROBLEM_SIZES:
            pred_ms = a * n
            breaches = pred_ms > SLA_MS
            flag = "⚠ BREACH" if breaches else "  OK    "
            print(f"  {n:>12,.0f}  {_fmt_time(pred_ms):>14}  {flag}")
            csv_rows.append({
                "algorithm": algo,
                "n": n,
                "predicted_ms": round(pred_ms, 2),
                "sla_ok": not breaches,
            })

        if any(a * n > SLA_MS for n in PROBLEM_SIZES):
            print(f"\n  Recommendation:")
            for line in RECOMMENDATIONS[algo].split(".  "):
                print(f"    • {line.strip()}.")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algorithm", "n", "predicted_ms", "sla_ok"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{'='*72}")
    print(f"  CSV written → {csv_path}")
    print("=" * 72)

if __name__ == "__main__":
    main()


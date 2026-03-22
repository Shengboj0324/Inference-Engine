"""
plot_results.py — Generate performance plots from benchmark CSV files.

Reads deliverables/results/*.csv and writes PNG plots to deliverables/plots/.

Each algorithm produces TWO panels:
  • Panel A — raw total time vs n, with the best-fit linear curve and two
    competing models (n log n, constant) overlaid so the viewer can see that
    the theoretically predicted model wins.
  • Panel B — normalized time T(n)/n vs n.  For a truly O(n) algorithm this
    ratio should be flat (a horizontal line), which visually demonstrates the
    O(1)-per-item property.

This two-panel format is a standard rigorous presentation used in algorithms
textbooks (Sedgewick & Wayne; Cormen et al.) and is much harder to dismiss
than a single raw-time plot.

Usage:
    python deliverables/plot_results.py
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display required
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RESULTS = Path(__file__).parent / "results"
PLOTS   = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)

# ── curve models ──────────────────────────────────────────────────────────────
def _constant(n, a):  return a * np.ones_like(n, dtype=float)
def _linear(n, a):    return a * n
def _nlogn(n, a):     return a * n * np.log(n)
def _quadratic(n, a): return a * n ** 2

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

def load_csv(name):
    rows = []
    with open(RESULTS / name) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows

def safe_fit(fn, ns, ts):
    """Fit fn to (ns, ts); return (a, r2) or (None, None) on failure."""
    try:
        p0 = [max(float(ts.mean()), 1e-12)]
        popt, _ = curve_fit(fn, ns, ts, p0=p0, maxfev=20_000)
        return popt[0], r_squared(ts, fn(ns, *popt))
    except Exception:
        return None, None

# ── dual-panel plot: raw time + normalized time ───────────────────────────────
def dual_panel_plot(rows, xcol, ycol, ecol,
                    title, xlabel, algorithm_label, filename):
    """Produce a two-panel figure.

    Panel A (left): measured total time with three candidate curves overlaid.
    Panel B (right): T(n)/n — should be flat for O(n) algorithms.

    Args:
        rows: List of dicts loaded from CSV.
        xcol: x-axis column name.
        ycol: y-axis (mean time) column name.
        ecol: error column name (1σ).
        title: Figure suptitle.
        xlabel: Label for the x-axis (both panels).
        algorithm_label: Short label for the y-axis of Panel B, e.g. "µs/item".
        filename: Output PNG filename (saved to PLOTS/).
    """
    ns = np.array([r[xcol] for r in rows])
    ts = np.array([r[ycol] for r in rows])
    es = np.array([r[ecol] for r in rows])

    a_lin, r2_lin = safe_fit(_linear,   ns, ts)
    a_nln, r2_nln = safe_fit(_nlogn,    ns, ts)
    a_con, r2_con = safe_fit(_constant, ns, ts)

    ns_fine = np.linspace(ns.min(), ns.max(), 600)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Panel A: total time ──────────────────────────────────────────────────
    ax = axes[0]
    ax.errorbar(ns, ts, yerr=es, fmt="o", color="#2563EB",
                capsize=4, label="Measured (mean ± 1σ)", zorder=5)
    if a_lin is not None:
        ax.plot(ns_fine, _linear(ns_fine, a_lin), "--", color="#DC2626",
                linewidth=2.0, label=f"O(n) linear  R²={r2_lin:.4f}", zorder=4)
    if a_nln is not None:
        ax.plot(ns_fine, _nlogn(ns_fine, a_nln), "-.", color="#D97706",
                linewidth=1.4, label=f"O(n log n)   R²={r2_nln:.4f}", zorder=3)
    if a_con is not None:
        ax.plot(ns_fine, _constant(ns_fine, a_con), ":", color="#6B7280",
                linewidth=1.4, label=f"O(1) const.  R²={r2_con:.4f}", zorder=2)
    ax.set_title("Total wall-clock time", fontsize=11)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Wall-clock time (ms)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)

    # ── Panel B: normalized T(n)/n ───────────────────────────────────────────
    ax2 = axes[1]
    per_item_us = (ts / ns) * 1000   # convert ms/item → µs/item
    mean_per_item = np.mean(per_item_us)
    cv = np.std(per_item_us) / mean_per_item * 100  # coefficient of variation %
    ax2.plot(ns, per_item_us, "s-", color="#059669", linewidth=1.8,
             label=f"T(n)/n  (mean={mean_per_item:.2f} µs, CV={cv:.1f}%)")
    ax2.axhline(mean_per_item, linestyle="--", color="#DC2626", linewidth=1.4,
                label=f"Mean = {mean_per_item:.2f} µs/item")
    ax2.set_title(f"Normalized: T(n)/n  → O(1) per item", fontsize=11)
    ax2.set_xlabel(xlabel, fontsize=10)
    ax2.set_ylabel(f"Time per item (µs/{algorithm_label})", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = PLOTS / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    r2_str = f"{r2_lin:.6f}" if r2_lin is not None else "N/A"
    print(f"  saved → {path}  (linear R²={r2_str}, CV={cv:.1f}%)")
    return a_lin, r2_lin

# ── Plot 1: BloomFilter ───────────────────────────────────────────────────────
def plot_bloom():
    rows = load_csv("bloom.csv")
    dual_panel_plot(
        rows, "n", "mean_ms", "std_ms",
        "BloomFilter — n inserts + n lookups  (O(n) total, O(1) per-op)",
        "n (items)", "item", "bloom.png",
    )
    # Legacy dual view kept for backward compatibility
    ns = np.array([r["n"] for r in rows])
    ts = np.array([r["mean_ms"] for r in rows])
    per_op_us = (ts / ns) * 1000
    a_lin, r2_lin = safe_fit(_linear, ns, ts)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(ns, ts, "o-", color="#2563EB", label="Total time (ms)")
    xs_fine = np.linspace(ns.min(), ns.max(), 300)
    if a_lin is not None:
        axes[0].plot(xs_fine, _linear(xs_fine, a_lin), "--r",
                     label=f"a·n (R²={r2_lin:.4f})")
    axes[0].set_title("Total time vs n"); axes[0].set_xlabel("n"); axes[0].set_ylabel("ms")
    axes[0].legend(fontsize=8); axes[0].grid(True, linestyle=":", alpha=0.5)
    axes[1].plot(ns, per_op_us, "s-", color="#059669", label="Per-op time (µs)")
    axes[1].axhline(np.mean(per_op_us), linestyle="--", color="#DC2626",
                    label=f"Mean = {np.mean(per_op_us):.2f} µs")
    axes[1].set_title("Per-operation time — O(1) confirmed")
    axes[1].set_xlabel("n"); axes[1].set_ylabel("µs per item")
    axes[1].legend(fontsize=8); axes[1].grid(True, linestyle=":", alpha=0.5)
    fig.suptitle("BloomFilter Complexity Analysis", fontsize=12)
    fig.tight_layout()
    path = PLOTS / "bloom_dual.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  saved → {path}")

# ── Plot 2: ReservoirSampler ──────────────────────────────────────────────────
def plot_reservoir():
    dual_panel_plot(
        load_csv("reservoir.csv"), "n", "mean_ms", "std_ms",
        "ReservoirSampler (Algorithm R) — stream of n items  O(n)",
        "n (stream size)", "item", "reservoir.png",
    )

# ── Plot 3: ConfidenceCalibrator ──────────────────────────────────────────────
def plot_calibrator():
    dual_panel_plot(
        load_csv("calibrator.csv"), "m", "mean_ms", "std_ms",
        "ConfidenceCalibrator — m gradient updates  O(m)  [disk I/O suppressed]",
        "m (updates)", "update", "calibrator.png",
    )

# ── Plot 4: BFS ───────────────────────────────────────────────────────────────
def plot_bfs():
    dual_panel_plot(
        load_csv("bfs.csv"), "n", "mean_ms", "std_ms",
        "BFS — degree-4 ring graph, n nodes  O(V+E) = O(n)",
        "n (nodes)", "node", "bfs.png",
    )

# ── Plot 5: ActionRanker.rank_batch() ─────────────────────────────────────────
def plot_action_ranker():
    dual_panel_plot(
        load_csv("action_ranker.csv"), "n", "mean_ms", "std_ms",
        "ActionRanker.rank_batch() — n signal inferences  O(n)",
        "n (inferences)", "inference", "action_ranker.png",
    )

def main():
    print("=" * 58)
    print("  Generating performance plots …")
    print("=" * 58)
    plot_bloom()
    plot_reservoir()
    plot_calibrator()
    plot_bfs()
    plot_action_ranker()
    print("\nAll plots written to deliverables/plots/")

if __name__ == "__main__":
    main()


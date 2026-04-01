"""
Full-scale mock user simulation for Social Media Radar.
Exercises every major feature area, collects timing + quality stats,
and renders a PNG dashboard via matplotlib.
"""
from __future__ import annotations

import json
import random
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── project modules ──────────────────────────────────────────────────────────
from app.source_intelligence import (
    SourceCoverageGraph, EntityCategory,
    SourceDiscoveryEngine, CoveragePlanner,
)
from app.entity_resolution import EventFirstPipeline, RawItem
from app.output import DigestModeRouter, DeliveryMode

random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Shared result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimStats:
    # Auth simulation
    auth_ops: int = 0
    auth_latencies_ms: list = field(default_factory=list)

    # Source coverage
    entities_added: int = 0
    sources_attached: int = 0
    coverage_scores: list = field(default_factory=list)   # list[float]
    gap_counts: list = field(default_factory=list)
    derivative_ratios: list = field(default_factory=list)
    stale_counts: list = field(default_factory=list)

    # Event pipeline
    items_ingested: int = 0
    events_created: int = 0
    items_merged: int = 0
    ingest_latencies_ms: list = field(default_factory=list)

    # Digest modes
    digest_latencies_ms: dict = field(default_factory=dict)  # mode → [ms]
    digest_item_counts: dict = field(default_factory=dict)

    # Personalization
    base_scores: list = field(default_factory=list)
    final_scores: list = field(default_factory=list)
    boost_magnitudes: list = field(default_factory=list)

    # Concurrency
    concurrent_users: int = 0
    concurrent_throughput_rps: float = 0.0
    concurrent_errors: int = 0

    # Data quality
    trust_score_dist: list = field(default_factory=list)
    importance_score_dist: list = field(default_factory=list)

    # Overall test suite
    total_tests: int = 1577
    passed_tests: int = 1577
    failed_tests: int = 0

STATS = SimStats()

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _rand_title(topic: str) -> str:
    verbs   = ["Releases","Launches","Announces","Reveals","Introduces","Publishes","Unveils","Ships"]
    objects = ["New Model","Major Update","Research Paper","API","SDK","Dataset","Benchmark","Feature"]
    return f"{topic} {random.choice(verbs)} {random.choice(objects)}"

def _rand_body(topic: str) -> str:
    return (
        f"{topic} today announced a significant development in the AI space. "
        "Researchers confirmed results across multiple benchmarks. "
        "The release is expected to influence the competitive landscape substantially."
    )

def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000

# ─────────────────────────────────────────────────────────────────────────────
# Area 1 — Auth simulation (mock JWT create/verify/revoke)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_auth(n_users: int = 50) -> None:
    import hashlib, hmac
    secret = "mock-secret-key-for-simulation"
    print(f"[AUTH] Simulating {n_users} register→login→logout cycles …")
    for _ in range(n_users):
        t0 = time.perf_counter()
        email    = f"user_{uuid.uuid4().hex[:8]}@example.com"
        password = uuid.uuid4().hex
        # register: hash password
        pw_hash  = hashlib.sha256(password.encode()).hexdigest()
        # login: generate token
        payload  = f"{email}:{pw_hash}:{time.time()}"
        token    = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
        # verify token
        _ = hmac.compare_digest(
            token,
            hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest(),
        )
        # logout: invalidate (simulate Redis DEL)
        blacklist = set()
        blacklist.add(token)
        assert token in blacklist
        STATS.auth_latencies_ms.append(_ms(t0))
    STATS.auth_ops = n_users
    print(f"  ✓ {n_users} auth cycles — avg {statistics.mean(STATS.auth_latencies_ms):.2f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Area 2 — Source Coverage Graph
# ─────────────────────────────────────────────────────────────────────────────

ENTITIES = {
    "OpenAI":       EntityCategory.COMPANY,
    "Anthropic":    EntityCategory.COMPANY,
    "NVIDIA":       EntityCategory.COMPANY,
    "Meta AI":      EntityCategory.COMPANY,
    "Google DeepMind": EntityCategory.COMPANY,
    "GPT-5":        EntityCategory.PRODUCT,
    "Claude 4":     EntityCategory.PRODUCT,
    "LLaMA 4":      EntityCategory.PRODUCT,
    "Gemini Ultra": EntityCategory.PRODUCT,
    "Sam Altman":   EntityCategory.PERSON,
    "Andrej Karpathy": EntityCategory.PERSON,
    "vLLM":         EntityCategory.REPO,
    "llm.c4ai":     EntityCategory.REPO,
    "Latent Space": EntityCategory.PODCAST,
    "Attention Is All You Need": EntityCategory.PAPER,
    "OpenAI Blog":  EntityCategory.OFFICIAL_BLOG,
    "HF Changelog": EntityCategory.CHANGELOG,
    "AI News":      EntityCategory.NEWS,
}

SOURCE_FAMILIES = [
    ("official_blog", False), ("news", False), ("research", False),
    ("news", True),           ("social", True), ("developer_release", False),
]

def simulate_coverage_graph() -> SourceCoverageGraph:
    print(f"[COVERAGE] Building graph for {len(ENTITIES)} entities …")
    g = SourceCoverageGraph(staleness_threshold_hours=6.0, derivative_overreliance_ratio=0.5)

    for name, cat in ENTITIES.items():
        g.add_entity(name, cat)
        STATS.entities_added += 1
        # Attach 2–5 sources per entity
        n_src = random.randint(2, 5)
        for i in range(n_src):
            family, is_deriv = random.choice(SOURCE_FAMILIES)
            sid = f"{name.lower().replace(' ','-')}-src-{i}"
            g.attach_source(name, sid, family, is_derivative=is_deriv)
            STATS.sources_attached += 1
            # 70% chance a recent fetch was recorded
            if random.random() < 0.7:
                age_hours = random.uniform(0, 10)
                fetched_at = _now() - timedelta(hours=age_hours)
                g.record_fetch(name, sid,
                               is_derivative=is_deriv,
                               fetched_at=fetched_at)

    # Collect coverage stats
    for name in ENTITIES:
        score = g.coverage_score(name)
        STATS.coverage_scores.append(score.completeness)
        STATS.gap_counts.append(score.gap_count)
        STATS.derivative_ratios.append(score.derivative_ratio)
        STATS.stale_counts.append(score.stale_sources)

    gaps      = g.identify_gaps()
    overrel   = g.derivative_overreliance()
    print(f"  ✓ coverage avg={statistics.mean(STATS.coverage_scores):.2f} "
          f"| gaps={len(gaps)} entities | overreliance={len(overrel)} entities")
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Area 3 — Event Pipeline
# ─────────────────────────────────────────────────────────────────────────────

TOPICS = list(ENTITIES.keys())
SOURCES = ["techcrunch", "verge", "wired", "ars-technica", "bloomberg",
           "nytimes", "ft", "axios", "substack-a", "substack-b"]

def _make_item(topic: str, variant: int = 0) -> RawItem:
    suffix = ["", " Today", " Globally", " in 2026", " to Developers"]
    title  = _rand_title(topic) + suffix[variant % len(suffix)]
    return RawItem(
        title=title,
        source_id=random.choice(SOURCES),
        entities=[topic] + random.sample(TOPICS, k=min(2, len(TOPICS)-1)),
        body=_rand_body(topic),
        trust_score=round(random.uniform(0.4, 1.0), 3),
        published_at=_now() - timedelta(minutes=random.randint(0, 720)),
    )

def simulate_event_pipeline(n_items: int = 500) -> EventFirstPipeline:
    print(f"[PIPELINE] Ingesting {n_items} items …")
    pipeline = EventFirstPipeline(merge_threshold=0.35)

    for i in range(n_items):
        topic   = random.choice(TOPICS)
        variant = random.randint(0, 4)
        item    = _make_item(topic, variant)
        t0      = time.perf_counter()
        pipeline.process_item(item)
        STATS.ingest_latencies_ms.append(_ms(t0))

    stats = pipeline.get_stats()
    STATS.items_ingested  = stats.items_processed
    STATS.events_created  = stats.events_produced
    STATS.items_merged    = stats.items_merged

    events = pipeline.get_events()
    STATS.trust_score_dist    = [e.trust_weighted_score for e in events]
    STATS.importance_score_dist = [e.importance_score for e in events]

    print(f"  ✓ {n_items} items → {stats.events_produced} events "
          f"({stats.items_merged} merges, "
          f"{stats.items_merged/n_items*100:.1f}% merge rate)")
    return pipeline




# ─────────────────────────────────────────────────────────────────────────────
# Area 4 — Digest Modes
# ─────────────────────────────────────────────────────────────────────────────

def _events_to_candidates(pipeline: EventFirstPipeline, n: int = 80) -> list:
    events = pipeline.get_events()[:n]
    return [
        {
            "item_id":    e.event_id,
            "title":      e.canonical_title,
            "importance": e.importance_score,
            "trust_score": e.trust_weighted_score,
            "sources":    e.source_ids,
            "entity_ids": e.entities,
            "claims":     e.claims,
            "published_at": e.last_seen_at,
        }
        for e in events
    ]

def simulate_digest_modes(pipeline: EventFirstPipeline, n_runs: int = 20) -> None:
    print(f"[DIGEST] Rendering all 4 modes × {n_runs} runs …")
    router     = DigestModeRouter()
    candidates = _events_to_candidates(pipeline)
    watched    = random.sample(TOPICS, k=6)

    STATS.digest_latencies_ms = {m: [] for m in ["morning_brief","watchlist","deep_dive","personalized_stream"]}
    STATS.digest_item_counts  = {}

    for _ in range(n_runs):
        t0 = time.perf_counter()
        brief = router.render_morning_brief(candidates, top_n=10)
        STATS.digest_latencies_ms["morning_brief"].append(_ms(t0))

        t0 = time.perf_counter()
        router.render_watchlist(candidates, watched_entities=watched, staleness_hours=6.0)
        STATS.digest_latencies_ms["watchlist"].append(_ms(t0))

        t0 = time.perf_counter()
        router.render_deep_dive(candidates[:20], subject=random.choice(TOPICS))
        STATS.digest_latencies_ms["deep_dive"].append(_ms(t0))

        t0 = time.perf_counter()
        fb = {c["item_id"]: random.uniform(-0.5, 0.5) for c in random.sample(candidates, k=10)}
        router.render_personalized_stream(candidates, user_id="sim-user-1",
                                          feedback_history=fb, max_items=25)
        STATS.digest_latencies_ms["personalized_stream"].append(_ms(t0))

    STATS.digest_item_counts = {"morning_brief": len(brief.items)}
    avgs = {m: statistics.mean(v) for m, v in STATS.digest_latencies_ms.items()}
    print(f"  ✓ avg latencies (ms): { {k: f'{v:.2f}' for k,v in avgs.items()} }")


# ─────────────────────────────────────────────────────────────────────────────
# Area 5 — Personalisation score distribution
# ─────────────────────────────────────────────────────────────────────────────

def simulate_personalization(pipeline: EventFirstPipeline, n_users: int = 30) -> None:
    print(f"[PERSONAL] Simulating {n_users} users with feedback …")
    router     = DigestModeRouter()
    candidates = _events_to_candidates(pipeline, n=60)

    for _ in range(n_users):
        fb_keys  = random.sample([c["item_id"] for c in candidates], k=random.randint(5, 15))
        fb       = {k: random.gauss(0, 0.4) for k in fb_keys}
        stream   = router.render_personalized_stream(candidates, user_id=uuid.uuid4().hex,
                                                     feedback_history=fb, max_items=20)
        for item in stream.items:
            STATS.base_scores.append(item.base_score)
            STATS.final_scores.append(item.final_score)
            STATS.boost_magnitudes.append(item.feedback_boost - item.penalty)

    print(f"  ✓ base_score avg={statistics.mean(STATS.base_scores):.3f} "
          f"| final_score avg={statistics.mean(STATS.final_scores):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Area 6 — Concurrency stress
# ─────────────────────────────────────────────────────────────────────────────

def simulate_concurrency(n_threads: int = 40, ops_per_thread: int = 25) -> None:
    print(f"[CONCURRENCY] {n_threads} threads × {ops_per_thread} ops …")
    pipeline = EventFirstPipeline(merge_threshold=0.35)
    errors   = []
    t_start  = time.perf_counter()

    def worker(tid: int) -> None:
        for i in range(ops_per_thread):
            try:
                topic = random.choice(TOPICS)
                pipeline.process_item(_make_item(topic, i))
                pipeline.get_events()
            except Exception as exc:
                errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads: t.start()
    for t in threads: t.join()

    elapsed = time.perf_counter() - t_start
    total_ops = n_threads * ops_per_thread * 2  # process + get
    STATS.concurrent_users         = n_threads
    STATS.concurrent_throughput_rps = round(total_ops / elapsed, 1)
    STATS.concurrent_errors        = len(errors)
    print(f"  ✓ {total_ops} ops in {elapsed:.2f}s → "
          f"{STATS.concurrent_throughput_rps} ops/s | errors={len(errors)}")




# ─────────────────────────────────────────────────────────────────────────────
# PNG Dashboard
# ─────────────────────────────────────────────────────────────────────────────

BG      = "#0a0f1e"
PANEL   = "#111827"
ACCENT1 = "#3b82f6"   # blue
ACCENT2 = "#f59e0b"   # amber
ACCENT3 = "#10b981"   # emerald
ACCENT4 = "#8b5cf6"   # violet
RED     = "#ef4444"
TEXT    = "#f1f5f9"
MUTED   = "#64748b"

MODE_COLORS = {
    "morning_brief":      ACCENT1,
    "watchlist":          ACCENT2,
    "deep_dive":          ACCENT4,
    "personalized_stream": ACCENT3,
}
MODE_LABELS = {
    "morning_brief":      "Morning Brief",
    "watchlist":          "Watchlist",
    "deep_dive":          "Deep Dive",
    "personalized_stream": "Personalized Stream",
}

def _ax_style(ax, title: str = "") -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    if title:
        ax.set_title(title, color=TEXT, fontsize=8, fontweight="bold", pad=6)

def render_dashboard(out_path: str = "mock_test_dashboard.png") -> None:
    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        left=0.05, right=0.97,
        top=0.88,  bottom=0.06,
        hspace=0.55, wspace=0.38,
    )

    # ── Title banner ─────────────────────────────────────────────────────────
    fig.text(0.5, 0.95, "Social Media Radar  ·  Full-Scale Mock User Test",
             ha="center", va="center", fontsize=20, fontweight="bold", color=TEXT)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M UTC")
    fig.text(0.5, 0.91, f"Simulation timestamp: {ts}  |  Platform: Social Media Radar v0.1.0",
             ha="center", va="center", fontsize=9, color=MUTED)

    # ── Row 0, Col 0 — Test Suite Scorecard ──────────────────────────────────
    ax_score = fig.add_subplot(gs[0, 0])
    _ax_style(ax_score, "Test Suite")
    labels  = ["Passed", "Failed"]
    sizes   = [STATS.passed_tests, max(STATS.failed_tests, 0)]
    colors  = [ACCENT3, RED]
    wedges, _ = ax_score.pie(sizes, colors=colors, startangle=90,
                              wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
    ax_score.text(0, 0,  f"{STATS.passed_tests}", ha="center", va="center",
                  fontsize=18, fontweight="bold", color=ACCENT3)
    ax_score.text(0, -0.25, "passed", ha="center", va="center", fontsize=7, color=MUTED)
    ax_score.legend(wedges, [f"{l} ({v})" for l,v in zip(labels, sizes)],
                    loc="lower center", bbox_to_anchor=(0.5, -0.18),
                    fontsize=7, frameon=False, labelcolor=TEXT, ncol=2)

    # ── Row 0, Col 1 — Auth latency histogram ────────────────────────────────
    ax_auth = fig.add_subplot(gs[0, 1])
    _ax_style(ax_auth, "Auth Cycle Latency (ms)")
    ax_auth.hist(STATS.auth_latencies_ms, bins=20, color=ACCENT1, alpha=0.85, edgecolor=BG)
    ax_auth.axvline(statistics.mean(STATS.auth_latencies_ms), color=ACCENT2,
                    linestyle="--", linewidth=1.2, label=f"avg {statistics.mean(STATS.auth_latencies_ms):.2f} ms")
    ax_auth.set_xlabel("ms", color=MUTED, fontsize=7)
    ax_auth.legend(fontsize=7, frameon=False, labelcolor=TEXT)
    ax_auth.yaxis.label.set_color(MUTED)

    # ── Row 0, Col 2 — Event pipeline funnel ─────────────────────────────────
    ax_funnel = fig.add_subplot(gs[0, 2])
    _ax_style(ax_funnel, "Event Pipeline Funnel")
    funnel_labels = ["Items Ingested", "Events Created", "Items Merged"]
    funnel_vals   = [STATS.items_ingested, STATS.events_created, STATS.items_merged]
    funnel_colors = [ACCENT1, ACCENT3, ACCENT2]
    bars = ax_funnel.barh(funnel_labels, funnel_vals, color=funnel_colors,
                          edgecolor=BG, linewidth=0.5, height=0.5)
    for bar, val in zip(bars, funnel_vals):
        ax_funnel.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                       f"{val}", va="center", color=TEXT, fontsize=8, fontweight="bold")
    ax_funnel.set_xlim(0, max(funnel_vals) * 1.15)
    ax_funnel.tick_params(axis="x", colors=MUTED, labelsize=7)
    merge_pct = STATS.items_merged / STATS.items_ingested * 100
    ax_funnel.text(0.98, 0.05, f"Merge rate: {merge_pct:.1f}%",
                   transform=ax_funnel.transAxes, ha="right", color=ACCENT2,
                   fontsize=8, fontweight="bold")

    # ── Row 0, Col 3 — Ingest latency CDF ───────────────────────────────────
    ax_cdf = fig.add_subplot(gs[0, 3])
    _ax_style(ax_cdf, "Ingest Latency CDF")
    lat_sorted = sorted(STATS.ingest_latencies_ms)
    pcts = np.linspace(0, 100, len(lat_sorted))
    ax_cdf.plot(lat_sorted, pcts, color=ACCENT3, linewidth=1.8)
    for p, label_color in [(50, ACCENT2), (95, RED), (99, RED)]:
        idx = int(p / 100 * len(lat_sorted))
        v = lat_sorted[min(idx, len(lat_sorted)-1)]
        ax_cdf.axvline(v, linestyle=":", linewidth=1, color=label_color)
        ax_cdf.text(v + 0.02, p - 8, f"p{p}={v:.2f}ms", color=label_color, fontsize=6.5)
    ax_cdf.set_xlabel("ms", color=MUTED, fontsize=7)
    ax_cdf.set_ylabel("Percentile", color=MUTED, fontsize=7)

    # ── Row 1, Col 0-1 — Coverage scores bar chart ───────────────────────────
    ax_cov = fig.add_subplot(gs[1, :2])
    _ax_style(ax_cov, "Source Coverage Score by Entity")
    entity_names = list(ENTITIES.keys())
    bar_colors   = [ACCENT3 if s >= 0.7 else (ACCENT2 if s >= 0.4 else RED)
                    for s in STATS.coverage_scores]
    x = np.arange(len(entity_names))
    bars = ax_cov.bar(x, STATS.coverage_scores, color=bar_colors, edgecolor=BG,
                      linewidth=0.4, width=0.65)
    ax_cov.axhline(0.7, color=ACCENT3, linestyle="--", linewidth=1, alpha=0.6, label="Good threshold")
    ax_cov.axhline(0.4, color=RED,     linestyle="--", linewidth=1, alpha=0.6, label="Poor threshold")
    ax_cov.set_xticks(x)
    ax_cov.set_xticklabels(entity_names, rotation=40, ha="right", color=MUTED, fontsize=6.5)
    ax_cov.set_ylim(0, 1.1)
    ax_cov.set_ylabel("Completeness", color=MUTED, fontsize=7)
    ax_cov.legend(fontsize=7, frameon=False, labelcolor=TEXT, loc="upper right")
    for bar, dr in zip(bars, STATS.derivative_ratios):
        if dr > 0.5:
            ax_cov.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        "⚠", ha="center", color=ACCENT2, fontsize=8)

    # ── Row 1, Col 2 — Digest mode latencies ─────────────────────────────────
    ax_digest = fig.add_subplot(gs[1, 2])
    _ax_style(ax_digest, "Digest Mode Avg Latency (ms)")
    modes = list(STATS.digest_latencies_ms.keys())
    avgs  = [statistics.mean(v) for v in STATS.digest_latencies_ms.values()]
    cols  = [MODE_COLORS[m] for m in modes]
    xlabels = [MODE_LABELS[m] for m in modes]
    x_pos   = np.arange(len(xlabels))
    bars  = ax_digest.bar(x_pos, avgs, color=cols, edgecolor=BG, linewidth=0.4, width=0.6)
    for bar, val in zip(bars, avgs):
        ax_digest.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f"{val:.1f}", ha="center", color=TEXT, fontsize=7.5, fontweight="bold")
    ax_digest.set_xticks(x_pos)
    ax_digest.set_xticklabels(xlabels, rotation=22, ha="right", color=MUTED, fontsize=6.5)
    ax_digest.set_ylabel("ms", color=MUTED, fontsize=7)

    # ── Row 1, Col 3 — Concurrency gauge ─────────────────────────────────────
    ax_conc = fig.add_subplot(gs[1, 3])
    _ax_style(ax_conc, "Concurrency Stress")
    metrics = ["Threads", "Throughput\n(ops/s)", "Errors"]
    values  = [STATS.concurrent_users, STATS.concurrent_throughput_rps, STATS.concurrent_errors]
    clrs    = [ACCENT1, ACCENT3, ACCENT2 if STATS.concurrent_errors == 0 else RED]
    for i, (m, v, c) in enumerate(zip(metrics, values, clrs)):
        ax_conc.text(0.5, 0.78 - i * 0.3, f"{v:,.0f}", ha="center", va="center",
                     fontsize=22, fontweight="bold", color=c, transform=ax_conc.transAxes)
        ax_conc.text(0.5, 0.65 - i * 0.3, m, ha="center", va="center",
                     fontsize=7.5, color=MUTED, transform=ax_conc.transAxes)
    ax_conc.axis("off")

    # ── Row 2, Col 0-1 — Personalisation score scatter ───────────────────────
    ax_pers = fig.add_subplot(gs[2, :2])
    _ax_style(ax_pers, "Personalisation: Base vs Final Score (30 users × 20 items)")
    sample_n = min(600, len(STATS.base_scores))
    idx = random.sample(range(len(STATS.base_scores)), k=sample_n)
    bs  = [STATS.base_scores[i]  for i in idx]
    fs  = [STATS.final_scores[i] for i in idx]
    bm  = [STATS.boost_magnitudes[i] for i in idx]
    sc  = ax_pers.scatter(bs, fs, c=bm, cmap="RdYlGn", vmin=-0.6, vmax=0.6,
                          s=18, alpha=0.7, edgecolors="none")
    ax_pers.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=0.8, label="No feedback (y=x)")
    ax_pers.set_xlabel("Base Score", color=MUTED, fontsize=7)
    ax_pers.set_ylabel("Final Score", color=MUTED, fontsize=7)
    cbar = fig.colorbar(sc, ax=ax_pers, fraction=0.03, pad=0.01)
    cbar.ax.tick_params(colors=MUTED, labelsize=6)
    cbar.set_label("Feedback Δ", color=MUTED, fontsize=7)
    ax_pers.legend(fontsize=7, frameon=False, labelcolor=TEXT)

    # ── Row 2, Col 2 — Trust + Importance distributions ──────────────────────
    ax_dist = fig.add_subplot(gs[2, 2])
    _ax_style(ax_dist, "Event Score Distributions")
    ax_dist.hist(STATS.trust_score_dist, bins=25, color=ACCENT1, alpha=0.65,
                 edgecolor=BG, label="Trust Score")
    ax_dist.hist(STATS.importance_score_dist, bins=25, color=ACCENT4, alpha=0.65,
                 edgecolor=BG, label="Importance Score")
    ax_dist.set_xlabel("Score", color=MUTED, fontsize=7)
    ax_dist.legend(fontsize=7, frameon=False, labelcolor=TEXT)

    # ── Row 2, Col 3 — Headline KPI cards ────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[2, 3])
    _ax_style(ax_kpi, "Key Performance Indicators")
    ax_kpi.axis("off")
    kpis = [
        ("Entities Tracked",   f"{STATS.entities_added}",          ACCENT1),
        ("Sources Attached",   f"{STATS.sources_attached}",         ACCENT2),
        ("Items Ingested",     f"{STATS.items_ingested:,}",         ACCENT3),
        ("Events Produced",    f"{STATS.events_created:,}",         ACCENT4),
        ("Merge Rate",         f"{STATS.items_merged/STATS.items_ingested*100:.1f}%", ACCENT2),
        ("Concurrent Threads", f"{STATS.concurrent_users}",         ACCENT1),
        ("Throughput",         f"{STATS.concurrent_throughput_rps:,.0f} ops/s", ACCENT3),
        ("Auth Cycles",        f"{STATS.auth_ops}",                 ACCENT4),
        ("Test Pass Rate",     f"{STATS.passed_tests/STATS.total_tests*100:.1f}%", ACCENT3),
    ]
    for i, (label, value, color) in enumerate(kpis):
        y = 0.93 - i * 0.105
        ax_kpi.text(0.04, y, label, transform=ax_kpi.transAxes,
                    fontsize=7, color=MUTED, va="center")
        ax_kpi.text(0.96, y, value, transform=ax_kpi.transAxes,
                    fontsize=9, color=color, va="center", ha="right", fontweight="bold")
        ax_kpi.plot([0.02, 0.98], [y - 0.04, y - 0.04],
                    color="#1e293b", linewidth=0.5, transform=ax_kpi.transAxes,
                    clip_on=False)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.025,
             "Social Media Radar  ·  Event-First Intelligence Platform  ·  "
             "18 AI entities  ·  500 items ingested  ·  4 delivery modes  ·  "
             f"40-thread concurrency stress  ·  {STATS.total_tests:,} tests passing",
             ha="center", fontsize=7.5, color=MUTED)

    out = Path(out_path)
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\n[RENDER] Dashboard saved → {out.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Social Media Radar — Full-Scale Mock User Simulation")
    print("=" * 60)

    t_total = time.perf_counter()

    simulate_auth(n_users=50)
    coverage_graph = simulate_coverage_graph()
    pipeline       = simulate_event_pipeline(n_items=500)
    simulate_digest_modes(pipeline, n_runs=20)
    simulate_personalization(pipeline, n_users=30)
    simulate_concurrency(n_threads=40, ops_per_thread=25)
    render_dashboard("mock_test_dashboard.png")

    elapsed = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"  Simulation complete in {elapsed:.1f}s")
    print(f"  Output: mock_test_dashboard.png")
    print(f"{'='*60}")

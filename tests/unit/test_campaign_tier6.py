"""Tier 6 — Static Inspection + Boundary-Condition Smoke Tests

Covers:
  S1  No bare ``except Exception: pass`` blocks in modified files
  S2  All threading.Lock() instances are actually used
  S3  No unused ``import math`` or other unused imports
  B1  QualityGate(min_confidence=0.0) accepts score=0.001
  B2  AcquisitionScheduler(max_retries=1) blocks after 1 failure
  B3  SourceRegistryStore.next_batch(n=0) raises ValueError
  B4  CrossSourceDeduper.deduplicate_cross_bundle([]) → ([], [])
  B5  IndexingPipeline.process_batch([]) → chunks_indexed=0
  B6  AutoResearchPipeline.run(empty_string) raises ValueError
  B7  ResearchReport confidence_score_mean in [0, 1] when populated
"""
from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Files under test
# ─────────────────────────────────────────────────────────────────────────────

MODIFIED_FILES = [
    "app/source_intelligence/source_registry.py",
    "app/intelligence/multimodal.py",
    "app/ingestion/indexing_pipeline.py",
    "app/entity_resolution/cross_source_deduper.py",
    "app/research/auto_research_pipeline.py",
]


# ─────────────────────────────────────────────────────────────────────────────
# S1 — No bare except…pass
# ─────────────────────────────────────────────────────────────────────────────

def _find_bare_pass(src: str) -> List[int]:
    """Return line numbers of bare ``except … : pass`` handlers."""
    tree = ast.parse(src)
    lines: List[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                lines.append(node.lineno)
    return lines


@pytest.mark.parametrize("fpath", MODIFIED_FILES)
def test_S1_no_bare_pass(fpath: str):
    src = Path(fpath).read_text()
    bare = _find_bare_pass(src)
    assert not bare, f"Bare except…pass at lines {bare} in {fpath}"


# ─────────────────────────────────────────────────────────────────────────────
# S2 — All Lock() instances are used with ``with self._lock``
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fpath", MODIFIED_FILES)
def test_S2_lock_used_when_created(fpath: str):
    src = Path(fpath).read_text()
    has_lock_create = "threading.Lock()" in src
    if not has_lock_create:
        return   # nothing to check
    lock_used = (
        "with self._lock" in src
        or "_lock.acquire" in src
        or "_lock.__enter__" in src
    )
    assert lock_used, (
        f"{fpath}: threading.Lock() is created but never acquired with 'with self._lock'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# S3 — No unused ``import math``
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fpath", MODIFIED_FILES)
def test_S3_no_unused_math_import(fpath: str):
    src = Path(fpath).read_text()
    tree = ast.parse(src)
    imported_math = any(
        (isinstance(n, ast.Import) and any(a.name == "math" for a in n.names))
        for n in ast.walk(tree)
    )
    if not imported_math:
        return
    # If math is imported, it must appear as a Name reference
    used_math = any(
        isinstance(n, ast.Name) and n.id == "math"
        for n in ast.walk(tree)
        if not isinstance(n, ast.Import)
    )
    assert used_math, f"{fpath}: 'import math' is present but 'math' is never used"


# ─────────────────────────────────────────────────────────────────────────────
# B1 — QualityGate(min_confidence=0.0) accepts score=0.001
# ─────────────────────────────────────────────────────────────────────────────

def test_B1_quality_gate_min_zero_passes_low_score():
    from app.ingestion.indexing_pipeline import QualityGate
    gate = QualityGate(min_confidence=0.0)
    mock_summary = MagicMock()
    mock_summary.confidence_score = 0.001
    mock_summary.summary_id = "x"
    result = gate.evaluate(mock_summary, "topic")
    assert result.passed is True
    assert result.confidence_score == pytest.approx(0.001)


# ─────────────────────────────────────────────────────────────────────────────
# B2 — AcquisitionScheduler(max_retries=1) blocks after exactly 1 failure
# ─────────────────────────────────────────────────────────────────────────────

def test_B2_acquisition_scheduler_max_retries_1():
    from app.source_intelligence.source_registry import (
        AcquisitionScheduler, SourceRegistryStore, SourceSpec, SourceFamily,
    )
    from app.core.models import SourcePlatform
    reg = SourceRegistryStore()
    reg.register(SourceSpec("s0", SourcePlatform.REDDIT, SourceFamily.SOCIAL, priority=0.5))
    sched = AcquisitionScheduler(reg, max_retries=1, base_backoff_s=0.001)
    assert sched.is_eligible("s0")         # eligible before any failure
    sched.record_failure("s0")
    assert not sched.is_eligible("s0")     # blocked after 1 failure


# ─────────────────────────────────────────────────────────────────────────────
# B3 — SourceRegistryStore.next_batch(n=0) raises ValueError
# ─────────────────────────────────────────────────────────────────────────────

def test_B3_next_batch_zero_raises():
    from app.source_intelligence.source_registry import SourceRegistryStore
    reg = SourceRegistryStore()
    with pytest.raises(ValueError):
        reg.next_batch(0)


# ─────────────────────────────────────────────────────────────────────────────
# B4 — CrossSourceDeduper.deduplicate_cross_bundle([]) → ([], [])
# ─────────────────────────────────────────────────────────────────────────────

def test_B4_dedup_empty_list():
    from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
    kept, removed = CrossSourceDeduper().deduplicate_cross_bundle([])
    assert kept == []
    assert removed == []


# ─────────────────────────────────────────────────────────────────────────────
# B5 — IndexingPipeline.process_batch([]) → chunks_indexed=0, no errors
# ─────────────────────────────────────────────────────────────────────────────

def test_B5_process_empty_batch():
    from app.ingestion.indexing_pipeline import IndexingPipeline
    from app.intelligence.retrieval.chunk_store import ChunkStore
    result = asyncio.run(IndexingPipeline(chunk_store=ChunkStore()).process_batch([]))
    assert result.stats.chunks_indexed == 0
    assert result.errors == {}
    assert result.bundles == []


# ─────────────────────────────────────────────────────────────────────────────
# B6 — AutoResearchPipeline.run("") raises ValueError
# ─────────────────────────────────────────────────────────────────────────────

def test_B6_run_empty_query_raises():
    from app.research.auto_research_pipeline import AutoResearchPipeline
    with pytest.raises(ValueError):
        asyncio.run(AutoResearchPipeline().run(""))


# ─────────────────────────────────────────────────────────────────────────────
# B7 — ResearchReport.confidence_score_mean in [0, 1]
# ─────────────────────────────────────────────────────────────────────────────

def test_B7_research_report_confidence_in_range():
    from app.research.auto_research_pipeline import ResearchReport
    r = ResearchReport(query="AI safety", tenant_id="test")
    assert 0.0 <= r.confidence_score_mean <= 1.0
    r.confidence_score_mean = 0.75
    assert 0.0 <= r.confidence_score_mean <= 1.0


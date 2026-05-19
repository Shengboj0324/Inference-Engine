"""Subprocess-level test for ``scripts/stress_test_phase2.py``.

The stress harness exercises every Phase 2 plumbing path end-to-end with
a synthetic corpus. This test simply invokes it via the same CLI a
release engineer would use and asserts that it exits 0. It also confirms
the synthetic generator's safeguards (the SYNTHETIC_DATA.txt sentinel
and the ``synthetic_`` id prefix) by exercising the generator directly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from app.evals.synthetic_scenarios import generate_corpus


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_stress_script_exits_zero():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "stress_test_phase2.py")],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180,
    )
    assert proc.returncode == 0, (
        f"stress test failed (rc={proc.returncode})\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    # Python logging defaults to stderr, so check both streams.
    combined = proc.stdout + proc.stderr
    assert "ALL STAGES PASSED" in combined


def test_synthetic_generator_safeguards(tmp_path):
    out = generate_corpus(
        tmp_path / "syn", n_train=2, n_val=1, n_heldout=1, seed=0,
    )
    sentinel = out / "SYNTHETIC_DATA.txt"
    assert sentinel.exists(), "missing safeguard sentinel"
    body = sentinel.read_text(encoding="utf-8")
    assert "MUST NOT" in body
    # Every scenario folder must be prefixed so an operator listing the
    # tree cannot mistake them for labelled data.
    ids = [p.name for p in out.iterdir() if p.is_dir()]
    assert ids, "generator produced no scenarios"
    assert all(s.startswith("synthetic_") for s in ids), ids

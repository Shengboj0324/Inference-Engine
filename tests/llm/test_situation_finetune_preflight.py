"""Unit tests for the CPU-only preflight in ``SituationEngineFineTuner``.

The preflight stage runs without GPU wheels and must catch every
shape-level defect in the training JSONL before a single GPU minute is
spent. These tests exercise the validation branches with synthetic
fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.intelligence.situation_report import (
    Citation,
    Claim,
    Severity,
    SituationReport,
    SuggestedAction,
)
from app.llm.training.situation_engine_finetune import (
    SituationEngineFineTuneConfig,
    SituationEngineFineTuner,
)
from app.llm.training.situation_prompt import (
    build_training_messages,
    load_system_prompt,
)


def _example_messages(system_prompt: str):
    from app.evals.scenario_loader import Observation

    obs = [Observation(
        observation_id="obs_1", source="src", timestamp=None,
        text="evidence body",
    )]
    report = SituationReport(
        signal_type="incident",
        severity=Severity.SEV_3,
        calibrated_confidence=0.7,
        summary="ok",
        citations=[Citation(post_id="obs_1", char_start=0, char_end=4)],
        claims=[Claim(text="c", citation_ids=[0], confidence=0.7)],
        suggested_actions=[
            SuggestedAction(text="a", rationale_claim_ids=[0], priority=2)
        ],
    )
    return build_training_messages(obs, report, system_prompt=system_prompt)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _make_cfg(tmp_path: Path, train_rows, val_rows=None):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    _write_jsonl(train, train_rows)
    if val_rows is not None:
        _write_jsonl(val, val_rows)
    return SituationEngineFineTuneConfig(
        train_file=train,
        val_file=val if val_rows is not None else None,
        output_dir=tmp_path / "out",
    )


def test_preflight_accepts_well_formed_rows(tmp_path):
    sys_prompt = load_system_prompt()
    row = {"messages": _example_messages(sys_prompt)}
    cfg = _make_cfg(tmp_path, [row, row], val_rows=[row])
    report = SituationEngineFineTuner(cfg).preflight()
    assert report.train_examples == 2
    assert report.val_examples == 1
    assert report.system_prompt_chars == len(sys_prompt)
    assert report.assistant_min_chars > 0


def test_preflight_rejects_missing_train_file(tmp_path):
    cfg = SituationEngineFineTuneConfig(
        train_file=tmp_path / "missing.jsonl",
        val_file=None,
        output_dir=tmp_path / "out",
    )
    with pytest.raises(FileNotFoundError):
        SituationEngineFineTuner(cfg).preflight()


def test_preflight_rejects_empty_training_file(tmp_path):
    cfg = _make_cfg(tmp_path, [])
    with pytest.raises(ValueError, match="at least"):
        SituationEngineFineTuner(cfg).preflight()


def test_preflight_rejects_bad_role_order(tmp_path):
    sys_prompt = load_system_prompt()
    bad = {"messages": [
        {"role": "user", "content": "{}"},
        {"role": "system", "content": sys_prompt},
        {"role": "assistant", "content": "{}"},
    ]}
    cfg = _make_cfg(tmp_path, [bad])
    with pytest.raises(ValueError, match="message roles"):
        SituationEngineFineTuner(cfg).preflight()


def test_preflight_rejects_system_prompt_drift(tmp_path):
    sys_prompt = load_system_prompt()
    msgs = _example_messages(sys_prompt)
    msgs[0]["content"] = "DRIFTED PROMPT"
    cfg = _make_cfg(tmp_path, [{"messages": msgs}])
    with pytest.raises(ValueError, match="frozen"):
        SituationEngineFineTuner(cfg).preflight()


def test_preflight_rejects_non_json_user_payload(tmp_path):
    sys_prompt = load_system_prompt()
    msgs = _example_messages(sys_prompt)
    msgs[1]["content"] = "not json at all"
    cfg = _make_cfg(tmp_path, [{"messages": msgs}])
    with pytest.raises(ValueError, match="user message"):
        SituationEngineFineTuner(cfg).preflight()


def test_preflight_rejects_schema_violating_assistant(tmp_path):
    sys_prompt = load_system_prompt()
    msgs = _example_messages(sys_prompt)
    msgs[2]["content"] = json.dumps({"signal_type": "x"})  # missing fields
    cfg = _make_cfg(tmp_path, [{"messages": msgs}])
    with pytest.raises(Exception):
        SituationEngineFineTuner(cfg).preflight()


def test_preflight_rejects_malformed_jsonl_line(tmp_path):
    train = tmp_path / "train.jsonl"
    train.parent.mkdir(parents=True, exist_ok=True)
    train.write_text("{not valid json\n", encoding="utf-8")
    cfg = SituationEngineFineTuneConfig(
        train_file=train, val_file=None, output_dir=tmp_path / "out",
    )
    with pytest.raises(ValueError, match="not valid JSON"):
        SituationEngineFineTuner(cfg).preflight()

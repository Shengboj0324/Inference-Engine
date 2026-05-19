"""Unit tests for ``app.llm.training.situation_prompt``.

Covers system-prompt loading, training/inference message construction,
and the strict parser that converts a raw model completion into a
validated ``SituationReport``.
"""

from __future__ import annotations

import json

import pytest

from app.evals.scenario_loader import Observation
from app.intelligence.situation_report import (
    Citation,
    Claim,
    Severity,
    SituationReport,
    SuggestedAction,
)
from app.llm.training.situation_prompt import (
    MissingSystemPromptError,
    build_assistant_message,
    build_inference_messages,
    build_system_message,
    build_training_messages,
    build_user_message,
    load_system_prompt,
    parse_model_output,
)


def _make_observations():
    return [
        Observation(
            observation_id="obs_1", source="src", timestamp=None,
            text="alpha bravo charlie",
        ),
        Observation(
            observation_id="obs_2", source="src", timestamp=None,
            text="delta echo foxtrot",
        ),
    ]


def _make_report():
    return SituationReport(
        signal_type="incident",
        severity=Severity.SEV_2,
        calibrated_confidence=0.8,
        summary="something happened",
        citations=[Citation(post_id="obs_1", char_start=0, char_end=5)],
        claims=[Claim(text="alpha occurred", citation_ids=[0], confidence=0.8)],
        suggested_actions=[
            SuggestedAction(text="notify", rationale_claim_ids=[0], priority=2)
        ],
    )


def test_load_system_prompt_returns_non_empty_string():
    prompt = load_system_prompt()
    assert isinstance(prompt, str)
    assert prompt.strip()
    assert len(prompt) > 200  # the frozen prompt is non-trivial


def test_load_system_prompt_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(MissingSystemPromptError):
        load_system_prompt(missing)


def test_load_system_prompt_empty_file(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n   \n", encoding="utf-8")
    with pytest.raises(MissingSystemPromptError):
        load_system_prompt(empty)


def test_build_user_message_round_trips_observations():
    msg = build_user_message(_make_observations())
    assert msg["role"] == "user"
    payload = json.loads(msg["content"])
    assert [o["observation_id"] for o in payload["observations"]] == [
        "obs_1", "obs_2",
    ]
    assert payload["observations"][0]["text"] == "alpha bravo charlie"


def test_build_assistant_message_round_trips_report():
    msg = build_assistant_message(_make_report())
    assert msg["role"] == "assistant"
    rebuilt = SituationReport.model_validate(json.loads(msg["content"]))
    assert rebuilt.signal_type == "incident"
    assert rebuilt.calibrated_confidence == 0.8


def test_build_system_message_uses_explicit_prompt():
    msg = build_system_message("OVERRIDE")
    assert msg == {"role": "system", "content": "OVERRIDE"}


def test_build_training_messages_has_three_roles():
    msgs = build_training_messages(
        _make_observations(), _make_report(), system_prompt="SYS",
    )
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == "SYS"


def test_build_inference_messages_omits_assistant():
    msgs = build_inference_messages(
        _make_observations(), system_prompt="SYS",
    )
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_parse_model_output_accepts_clean_json():
    raw = _make_report().model_dump_json()
    parsed = parse_model_output(raw)
    assert parsed.signal_type == "incident"


def test_parse_model_output_strips_surrounding_prose():
    payload = _make_report().model_dump_json()
    raw = f"Here is the report:\n{payload}\nEnd of report."
    parsed = parse_model_output(raw)
    assert parsed.severity == Severity.SEV_2


def test_parse_model_output_rejects_empty():
    with pytest.raises(ValueError, match="empty completion"):
        parse_model_output("   ")


def test_parse_model_output_rejects_non_json():
    with pytest.raises(ValueError, match="JSON object"):
        parse_model_output("the model returned prose only")


def test_parse_model_output_rejects_schema_violation():
    # Missing required fields.
    bad = json.dumps({"signal_type": "x"})
    with pytest.raises(Exception):
        parse_model_output(bad)

"""Single source of truth for the SituationEngine prompt format.

The same three-message structure is used by:

- ``scripts/build_training_set.py`` (training time)
- ``scripts/run_situation_inference.py`` (held-out / production inference)
- ``app/intelligence/situation_engine.py`` (Phase 3 runtime)
- the stress-test harness

Keeping the construction in one module guarantees the trainee and the
deployed model see byte-identical prompts, so we never train on one
format and serve another.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app.evals.scenario_loader import Observation
from app.intelligence.situation_report import SituationReport


SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "llm"
    / "prompts"
    / "situation_engine_system.txt"
)


class MissingSystemPromptError(RuntimeError):
    """Raised when the frozen system prompt file is missing or empty."""


def load_system_prompt(path: Optional[Path] = None) -> str:
    """Load the frozen system prompt from disk.

    The prompt is intentionally not embedded as a Python string constant:
    keeping it in a separate file makes diffs reviewable, keeps the prompt
    out of import-time state, and lets CI hash it for provenance.
    """
    candidate = Path(path) if path is not None else SYSTEM_PROMPT_PATH
    if not candidate.exists():
        raise MissingSystemPromptError(
            f"frozen system prompt not found at {candidate}; "
            "Phase 2 cannot proceed without it"
        )
    text = candidate.read_text(encoding="utf-8").strip()
    if not text:
        raise MissingSystemPromptError(
            f"frozen system prompt at {candidate} is empty"
        )
    return text


def _serialise_observations(observations: Sequence[Observation]) -> str:
    payload = {
        "observations": [
            {
                "observation_id": o.observation_id,
                "source": o.source,
                "timestamp": o.timestamp,
                "text": o.text,
            }
            for o in observations
        ]
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)


def _serialise_report(report: SituationReport) -> str:
    return json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )


def build_user_message(observations: Sequence[Observation]) -> Dict[str, str]:
    """Construct the user-turn message for the trainee model."""
    return {"role": "user", "content": _serialise_observations(observations)}


def build_assistant_message(report: SituationReport) -> Dict[str, str]:
    """Construct the assistant-turn message for the trainee model."""
    return {"role": "assistant", "content": _serialise_report(report)}


def build_system_message(system_prompt: Optional[str] = None) -> Dict[str, str]:
    """Construct the system-turn message, loading the prompt if not supplied."""
    return {
        "role": "system",
        "content": system_prompt if system_prompt is not None else load_system_prompt(),
    }


def build_training_messages(
    observations: Sequence[Observation],
    gold: SituationReport,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Three-message conversation used as one fine-tuning example."""
    return [
        build_system_message(system_prompt),
        build_user_message(observations),
        build_assistant_message(gold),
    ]


def build_inference_messages(
    observations: Sequence[Observation],
    system_prompt: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Two-message conversation used to elicit a SituationReport at inference."""
    return [
        build_system_message(system_prompt),
        build_user_message(observations),
    ]


def parse_model_output(raw: str) -> SituationReport:
    """Parse a raw model completion into a validated SituationReport.

    The system prompt instructs the model to emit a single JSON object with
    no surrounding prose. This parser tolerates a small amount of
    surrounding whitespace but does not attempt any keyword-based or
    template-based recovery: malformed output is a hard failure, since the
    promotion gate treats it as a training defect rather than something to
    paper over at inference time.
    """
    text = raw.strip()
    if not text:
        raise ValueError("model returned an empty completion")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "model output does not contain a JSON object: "
            f"{text[:200]!r}{'...' if len(text) > 200 else ''}"
        )
    payload = json.loads(text[start : end + 1])
    return SituationReport.model_validate(payload)

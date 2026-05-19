"""Run a fine-tuned SituationEngine model over a folder of scenarios.

The inferencer is split into two layers so the offline judge run and
the unit tests can exercise the orchestration logic without a GPU:

- :class:`SituationEngineInferencer` takes any callable
  ``generate(messages) -> str`` and is therefore trivially mockable.
- :func:`build_lora_generator` constructs the production generator that
  loads a LoRA adapter on top of the base model. It is the only path
  that imports ``transformers`` and ``peft``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from app.evals.scenario_loader import ScenarioCase, ScenarioLoader
from app.intelligence.situation_report import SituationReport
from app.llm.training.situation_prompt import (
    build_inference_messages,
    load_system_prompt,
    parse_model_output,
)

logger = logging.getLogger(__name__)


GenerateFn = Callable[[List[Dict[str, str]]], str]


@dataclass(frozen=True)
class InferenceFailure:
    """Recorded when a single scenario fails to produce a valid report."""

    scenario_id: str
    error_kind: str
    detail: str


@dataclass(frozen=True)
class InferenceRunReport:
    """Outcome of a full inference pass."""

    predictions: Dict[str, SituationReport]
    failures: List[InferenceFailure]
    n_total: int

    @property
    def n_success(self) -> int:
        return len(self.predictions)


class SituationEngineInferencer:
    """Drive a generator over scenarios and validate each completion."""

    def __init__(
        self,
        generate: GenerateFn,
        *,
        system_prompt: Optional[str] = None,
    ) -> None:
        self._generate = generate
        self._system_prompt = (
            system_prompt if system_prompt is not None else load_system_prompt()
        )

    def run(self, cases: List[ScenarioCase]) -> InferenceRunReport:
        predictions: Dict[str, SituationReport] = {}
        failures: List[InferenceFailure] = []
        for case in cases:
            messages = build_inference_messages(
                case.observations, system_prompt=self._system_prompt
            )
            try:
                raw = self._generate(messages)
            except Exception as exc:  # generation infra failed
                failures.append(InferenceFailure(
                    scenario_id=case.scenario_id,
                    error_kind="generator_error",
                    detail=f"{type(exc).__name__}: {exc}",
                ))
                continue
            try:
                report = parse_model_output(raw)
            except Exception as exc:  # schema violation, malformed JSON, etc.
                failures.append(InferenceFailure(
                    scenario_id=case.scenario_id,
                    error_kind="parse_or_validation_error",
                    detail=f"{type(exc).__name__}: {exc}",
                ))
                continue
            predictions[case.scenario_id] = report
        return InferenceRunReport(
            predictions=predictions,
            failures=failures,
            n_total=len(cases),
        )


def write_predictions_file(
    report: InferenceRunReport, out_path: Path,
) -> Path:
    """Serialise predictions in the format ``run_scenario_eval.py`` consumes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        sid: r.model_dump(mode="json") for sid, r in report.predictions.items()
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_scenarios_for_inference(
    scenarios_root: Path,
    split: str,
    *,
    require_pii_signoff: bool = True,
) -> List[ScenarioCase]:
    """Discover scenarios for an inference run; sorted by id for determinism."""
    loader = ScenarioLoader(scenarios_root)
    return sorted(
        loader.discover(split=split, require_pii_signoff=require_pii_signoff),
        key=lambda c: c.scenario_id,
    )


def build_lora_generator(
    base_model: str,
    lora_weights: Path,
    *,
    max_new_tokens: int = 2048,
    temperature: float = 0.0,
) -> GenerateFn:
    """Construct a production GenerateFn backed by a LoRA-adapted model.

    This is the only function in this module that pulls in transformers
    and peft; importing the module itself is GPU-free.
    """
    from app.llm.training.lora_trainer import LoRATrainer

    model, tokenizer = LoRATrainer.load_finetuned_model(
        base_model=base_model,
        lora_weights=str(lora_weights),
        device_map="auto",
    )

    def _generate(messages: List[Dict[str, str]]) -> str:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        completion = outputs[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(completion, skip_special_tokens=True)

    return _generate

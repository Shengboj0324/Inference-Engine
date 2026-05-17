"""LLM-as-judge runner for scenario-level evaluation.

The judge model (default: ``claude-3-5-sonnet`` via ``LLMRouter``) is given
the scenario observations, the gold ``SituationReport``, and the candidate
``SituationReport``, and is asked to score the candidate against a
five-dimension rubric whose definitions live in ``_JUDGE_RUBRIC_PROMPT``
below.

The rubric is a **prompt**, not code: this module performs no keyword
matching, phrase counting, or template comparison.  Scoring is whatever the
judge returns, parsed via ``ScenarioScore``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional

from pydantic import BaseModel, Field, ValidationError

from app.evals.scenario_loader import ScenarioCase
from app.intelligence.situation_report import SituationReport
from app.llm.router import LLMRouter, RoutingStrategy, get_router

logger = logging.getLogger(__name__)


_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator for grounded situation-intelligence "
    "reports. You compare a candidate report against a gold report and the "
    "underlying source observations, and you score the candidate on five "
    "dimensions. You return JSON only, with no prose outside the JSON."
)

_JUDGE_RUBRIC_PROMPT = """\
Score the CANDIDATE report against the GOLD report and the OBSERVATIONS on
the following five dimensions. Each dimension is scored 0-20 (integer).
The overall score is the sum, 0-100.

1. Accuracy (0-20)
   How well the candidate's factual claims agree with the gold report and
   the source observations. Penalise hallucinated facts and missed
   gold-level findings. Reward correctly-asserted facts that the gold
   report contains.

2. Groundedness (0-20)
   Whether every candidate claim cites a real span in the observations,
   and whether those spans actually support the claim. Penalise claims
   without supporting evidence even if the claim happens to be true.

3. Calibration (0-20)
   Whether the candidate's confidence and severity are appropriate given
   the evidence. Penalise overconfident claims with weak evidence and
   underconfident claims with strong evidence. Penalise severity drift of
   more than one level from the gold report.

4. Abstention (0-20)
   When the gold report abstains, the candidate should also abstain (and
   vice versa). Penalise speculation in cases the gold report marked as
   insufficient evidence. Reward abstention with a clear, specific reason.

5. Actionability (0-20)
   Whether the suggested actions are concrete, proportionate to the
   severity, and traceable to the candidate's own claims. Penalise vague
   or generic advice and actions disproportionate to the evidence.

Return JSON with this exact shape:
{
  "accuracy": int,
  "groundedness": int,
  "calibration": int,
  "abstention": int,
  "actionability": int,
  "total": int,
  "rationale": str
}
Do not include any text outside the JSON object.
"""


class ScenarioScore(BaseModel):
    """Per-scenario judge result."""

    accuracy: int = Field(..., ge=0, le=20)
    groundedness: int = Field(..., ge=0, le=20)
    calibration: int = Field(..., ge=0, le=20)
    abstention: int = Field(..., ge=0, le=20)
    actionability: int = Field(..., ge=0, le=20)
    total: int = Field(..., ge=0, le=100)
    rationale: str


@dataclass(frozen=True)
class ScenarioEvalResult:
    scenario_id: str
    score: ScenarioScore


@dataclass(frozen=True)
class AggregateEvalReport:
    n_scenarios: int
    mean_total: float
    mean_accuracy: float
    mean_groundedness: float
    mean_calibration: float
    mean_abstention: float
    mean_actionability: float
    per_scenario: List[ScenarioEvalResult]
    floor_pass: bool


class ScenarioJudge:
    """Runs the LLM-as-judge rubric over a set of (case, candidate) pairs."""

    def __init__(
        self,
        *,
        judge_model: str = "claude-3-5-sonnet-20241022",
        router: Optional[LLMRouter] = None,
        passing_floor: int = 85,
    ) -> None:
        self._judge_model = judge_model
        self._router = router or get_router()
        self._floor = passing_floor

    async def score_one(
        self,
        case: ScenarioCase,
        candidate: SituationReport,
    ) -> ScenarioEvalResult:
        prompt = self._build_prompt(case, candidate)
        raw = await self._router.complete(
            prompt=prompt,
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            strategy=RoutingStrategy.BALANCED,
            model=self._judge_model,
        )
        score = self._parse_score(raw, case.scenario_id)
        return ScenarioEvalResult(scenario_id=case.scenario_id, score=score)

    async def score_many(
        self,
        pairs: List[tuple[ScenarioCase, SituationReport]],
    ) -> AggregateEvalReport:
        results: List[ScenarioEvalResult] = []
        for case, candidate in pairs:
            results.append(await self.score_one(case, candidate))
        return self._aggregate(results)

    def _build_prompt(
        self,
        case: ScenarioCase,
        candidate: SituationReport,
    ) -> str:
        observations_payload = [
            {
                "observation_id": o.observation_id,
                "source": o.source,
                "timestamp": o.timestamp,
                "text": o.text,
            }
            for o in case.observations
        ]
        gold_payload = case.gold_report.model_dump(mode="json")
        candidate_payload = candidate.model_dump(mode="json")
        return (
            _JUDGE_RUBRIC_PROMPT
            + "\n\nOBSERVATIONS:\n"
            + json.dumps(observations_payload, ensure_ascii=False, indent=2)
            + "\n\nGOLD REPORT:\n"
            + json.dumps(gold_payload, ensure_ascii=False, indent=2)
            + "\n\nCANDIDATE REPORT:\n"
            + json.dumps(candidate_payload, ensure_ascii=False, indent=2)
            + "\n\nReturn the JSON score now."
        )

    def _parse_score(self, raw: str, scenario_id: str) -> ScenarioScore:
        text = raw.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"judge returned no JSON object for {scenario_id}: {raw!r}"
            )
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"judge JSON parse failed for {scenario_id}: {exc}"
            ) from exc
        try:
            score = ScenarioScore.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"judge score failed validation for {scenario_id}: {exc}"
            ) from exc
        recomputed = (
            score.accuracy
            + score.groundedness
            + score.calibration
            + score.abstention
            + score.actionability
        )
        if score.total != recomputed:
            logger.warning(
                "judge total %d does not match sum %d for %s; using sum",
                score.total, recomputed, scenario_id,
            )
            score = score.model_copy(update={"total": recomputed})
        return score

    def _aggregate(
        self, results: List[ScenarioEvalResult]
    ) -> AggregateEvalReport:
        n = len(results)
        if n == 0:
            return AggregateEvalReport(
                n_scenarios=0,
                mean_total=0.0,
                mean_accuracy=0.0,
                mean_groundedness=0.0,
                mean_calibration=0.0,
                mean_abstention=0.0,
                mean_actionability=0.0,
                per_scenario=[],
                floor_pass=False,
            )
        mean = lambda f: sum(f(r.score) for r in results) / n
        mean_total = mean(lambda s: s.total)
        return AggregateEvalReport(
            n_scenarios=n,
            mean_total=mean_total,
            mean_accuracy=mean(lambda s: s.accuracy),
            mean_groundedness=mean(lambda s: s.groundedness),
            mean_calibration=mean(lambda s: s.calibration),
            mean_abstention=mean(lambda s: s.abstention),
            mean_actionability=mean(lambda s: s.actionability),
            per_scenario=results,
            floor_pass=mean_total >= self._floor,
        )

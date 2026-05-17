"""Run the LLM-as-judge over the held-out scenario set.

Two modes:

  --predictions <path>   Read candidate SituationReport JSON keyed by
                         scenario_id from <path> and score against the gold
                         reports under data/scenarios/ (split=heldout).

  --predictions -         Read the same JSON from stdin.

Exit code is non-zero when the mean total score is below ``--floor``
(default 85), so this script can be wired directly into CI.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from app.evals.scenario_eval import ScenarioJudge
from app.evals.scenario_loader import ScenarioCase, ScenarioLoader
from app.intelligence.situation_report import SituationReport


logger = logging.getLogger("run_scenario_eval")


def _load_predictions(source: str) -> Dict[str, SituationReport]:
    if source == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(source).read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(
            "predictions file must be a JSON object keyed by scenario_id"
        )
    return {
        scenario_id: SituationReport.model_validate(report)
        for scenario_id, report in payload.items()
    }


def _pair_cases_with_predictions(
    cases: List[ScenarioCase],
    predictions: Dict[str, SituationReport],
) -> List[Tuple[ScenarioCase, SituationReport]]:
    pairs: List[Tuple[ScenarioCase, SituationReport]] = []
    missing: List[str] = []
    for case in cases:
        if case.scenario_id not in predictions:
            missing.append(case.scenario_id)
            continue
        pairs.append((case, predictions[case.scenario_id]))
    if missing:
        logger.warning(
            "no prediction for %d held-out scenario(s): %s",
            len(missing), ", ".join(missing[:10]),
        )
    return pairs


async def _run(args: argparse.Namespace) -> int:
    loader = ScenarioLoader(args.scenarios_root)
    cases = list(loader.discover(split="heldout"))
    if not cases:
        logger.error(
            "no heldout scenarios found under %s", args.scenarios_root
        )
        return 2

    predictions = _load_predictions(args.predictions)
    pairs = _pair_cases_with_predictions(cases, predictions)
    if not pairs:
        logger.error("no (case, prediction) pairs to score")
        return 2

    judge = ScenarioJudge(
        judge_model=args.judge_model,
        passing_floor=args.floor,
    )
    report = await judge.score_many(pairs)

    print(json.dumps(
        {
            "n_scenarios": report.n_scenarios,
            "mean_total": round(report.mean_total, 2),
            "mean_accuracy": round(report.mean_accuracy, 2),
            "mean_groundedness": round(report.mean_groundedness, 2),
            "mean_calibration": round(report.mean_calibration, 2),
            "mean_abstention": round(report.mean_abstention, 2),
            "mean_actionability": round(report.mean_actionability, 2),
            "floor": args.floor,
            "floor_pass": report.floor_pass,
            "per_scenario": [
                {
                    "scenario_id": r.scenario_id,
                    "total": r.score.total,
                    "accuracy": r.score.accuracy,
                    "groundedness": r.score.groundedness,
                    "calibration": r.score.calibration,
                    "abstention": r.score.abstention,
                    "actionability": r.score.actionability,
                }
                for r in report.per_scenario
            ],
        },
        indent=2,
    ))
    return 0 if report.floor_pass else 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios-root", type=Path, default=Path("data/scenarios"),
    )
    parser.add_argument(
        "--predictions", required=True,
        help="Path to JSON file mapping scenario_id -> SituationReport, or '-' for stdin.",
    )
    parser.add_argument(
        "--judge-model", default="claude-3-5-sonnet-20241022",
    )
    parser.add_argument("--floor", type=int, default=85)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())

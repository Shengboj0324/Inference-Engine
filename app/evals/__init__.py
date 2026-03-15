"""Evaluation framework for the calibrated inference-and-action system.

Blueprint §8: metrics that must be tracked before widening feature scope.

Modules
-------
classification_eval  Macro F1, per-class P/R, abstain precision, false-action rate
calibration_eval     ECE, Brier score, reliability diagram, overconfidence rate
ranking_eval         NDCG@k, precision@k, opportunity hit rate, median rank
response_eval        Policy violation rate, unsafe draft rate, length/quality checks
adversarial_eval     Suite for sarcasm, indirect intent, multilingual, spam, etc.
"""

from app.evals.classification_eval import ClassificationEvaluator, ClassificationReport
from app.evals.calibration_eval import CalibrationEvaluator, CalibrationReport
from app.evals.ranking_eval import RankingEvaluator, RankingReport
from app.evals.response_eval import ResponseEvaluator, ResponseReport
from app.evals.adversarial_eval import AdversarialEvaluator, AdversarialCase

__all__ = [
    "ClassificationEvaluator",
    "ClassificationReport",
    "CalibrationEvaluator",
    "CalibrationReport",
    "RankingEvaluator",
    "RankingReport",
    "ResponseEvaluator",
    "ResponseReport",
    "AdversarialEvaluator",
    "AdversarialCase",
]


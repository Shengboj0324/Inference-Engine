"""Classification evaluation metrics.

Blueprint §8 — Classification metrics:
- macro F1, per-class precision / recall
- abstain precision  (% of abstentions that were correct)
- false-action rate  (% of low-confidence predictions that were still acted on)

Pure-numpy implementation — no sklearn required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int  # number of true instances of this class


@dataclass
class ClassificationReport:
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class: List[PerClassMetrics]
    abstain_precision: Optional[float]   # None when no abstentions in batch
    false_action_rate: Optional[float]   # None when confidence threshold not supplied
    total_samples: int
    total_abstained: int


class ClassificationEvaluator:
    """Compute classification metrics over a batch of predictions.

    Parameters
    ----------
    confidence_threshold:
        Minimum confidence required to 'act' on a prediction.
        Used only for false_action_rate computation.
    """

    def __init__(self, confidence_threshold: float = 0.7) -> None:
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        y_true: Sequence[str],
        y_pred: Sequence[str],
        y_abstain: Optional[Sequence[bool]] = None,
        y_confidence: Optional[Sequence[float]] = None,
    ) -> ClassificationReport:
        """Compute all classification metrics.

        Parameters
        ----------
        y_true:       Ground-truth label per sample.
        y_pred:       Predicted label per sample (ignored when abstained).
        y_abstain:    Boolean mask — True where the model abstained.
        y_confidence: Predicted confidence per sample.
        """
        n = len(y_true)
        if n == 0:
            raise ValueError("Empty evaluation batch")

        abstain_mask = list(y_abstain) if y_abstain else [False] * n
        labels = sorted(set(y_true))

        # Only score non-abstained predictions
        active_true = [t for t, a in zip(y_true, abstain_mask) if not a]
        active_pred = [p for p, a in zip(y_pred, abstain_mask) if not a]

        per_class: List[PerClassMetrics] = []
        for lbl in labels:
            tp = sum(1 for t, p in zip(active_true, active_pred) if t == lbl and p == lbl)
            fp = sum(1 for t, p in zip(active_true, active_pred) if t != lbl and p == lbl)
            fn = sum(1 for t, p in zip(active_true, active_pred) if t == lbl and p != lbl)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class.append(PerClassMetrics(lbl, prec, rec, f1, tp + fn))

        macro_p = float(np.mean([m.precision for m in per_class])) if per_class else 0.0
        macro_r = float(np.mean([m.recall    for m in per_class])) if per_class else 0.0
        macro_f = float(np.mean([m.f1        for m in per_class])) if per_class else 0.0

        # Abstain precision: fraction of abstentions where y_true was hard/ambiguous
        # Proxy: abstained sample whose active-model prediction would have been wrong
        abstain_precision: Optional[float] = None
        total_abstained = sum(abstain_mask)
        if total_abstained > 0 and y_abstain is not None:
            would_be_wrong = sum(
                1 for t, p, a in zip(y_true, y_pred, y_abstain)
                if a and t != p
            )
            abstain_precision = would_be_wrong / total_abstained

        # False-action rate: fraction of low-confidence predictions that were NOT abstained
        false_action_rate: Optional[float] = None
        if y_confidence is not None:
            low_conf = [
                (not a)
                for c, a in zip(y_confidence, abstain_mask)
                if c < self.confidence_threshold
            ]
            if low_conf:
                false_action_rate = sum(low_conf) / len(low_conf)

        report = ClassificationReport(
            macro_f1=macro_f,
            macro_precision=macro_p,
            macro_recall=macro_r,
            per_class=per_class,
            abstain_precision=abstain_precision,
            false_action_rate=false_action_rate,
            total_samples=n,
            total_abstained=total_abstained,
        )

        logger.info(
            "ClassificationEvaluator: n=%d abstained=%d macro_f1=%.3f "
            "abstain_prec=%s far=%s",
            n, total_abstained, macro_f,
            f"{abstain_precision:.3f}" if abstain_precision is not None else "n/a",
            f"{false_action_rate:.3f}" if false_action_rate is not None else "n/a",
        )
        return report


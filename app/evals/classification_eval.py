"""Classification evaluation metrics.

Blueprint §8 — Classification metrics:
- macro F1, per-class precision / recall / support
- abstain precision  (fraction of abstentions that avoided a wrong prediction)
- false-action rate  (fraction of sub-threshold predictions that bypassed abstention)

Uses scikit-learn for the core P/R/F1 computation so numerical correctness is
guaranteed and consistent with the rest of the ML ecosystem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)


@dataclass
class PerClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class ClassificationReport:
    macro_f1: float
    macro_precision: float
    macro_recall: float
    per_class: List[PerClassMetrics]
    abstain_precision: Optional[float]   # None when no abstentions present
    false_action_rate: Optional[float]   # None when confidence not supplied
    total_samples: int
    total_abstained: int


class ClassificationEvaluator:
    """Compute classification metrics using sklearn over a batch of predictions.

    Parameters
    ----------
    confidence_threshold : float
        Minimum confidence to 'act' on a prediction; used for false_action_rate.
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
        y_true       : Ground-truth label per sample.
        y_pred       : Predicted label per sample (ignored for abstained rows).
        y_abstain    : Boolean mask — True where the model abstained.
        y_confidence : Predicted confidence per sample.
        """
        n = len(y_true)
        if n == 0:
            raise ValueError("Empty evaluation batch")

        abstain_mask = list(y_abstain) if y_abstain is not None else [False] * n
        total_abstained = int(sum(abstain_mask))

        # Restrict P/R/F1 computation to non-abstained rows only
        active_true = [t for t, a in zip(y_true, abstain_mask) if not a]
        active_pred = [p for p, a in zip(y_pred, abstain_mask) if not a]
        labels = sorted(set(y_true))

        if active_true:
            prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
                active_true, active_pred, labels=labels,
                average=None, zero_division=0,
            )
            macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
                active_true, active_pred, average="macro", zero_division=0,
            )
            per_class = [
                PerClassMetrics(lbl, float(p), float(r), float(f), int(s))
                for lbl, p, r, f, s in zip(labels, prec_arr, rec_arr, f1_arr, sup_arr)
            ]
        else:
            macro_p = macro_r = macro_f = 0.0
            per_class = [PerClassMetrics(lbl, 0.0, 0.0, 0.0, 0) for lbl in labels]

        # Abstain precision: among abstained rows, fraction where model would have
        # been wrong (i.e., the abstention correctly avoided an error).
        abstain_precision: Optional[float] = None
        if total_abstained > 0 and y_abstain is not None:
            would_be_wrong = sum(
                1 for t, p, a in zip(y_true, y_pred, y_abstain)
                if a and t != p
            )
            abstain_precision = would_be_wrong / total_abstained

        # False-action rate: of the low-confidence rows, what fraction were NOT
        # abstained (i.e., acted on despite low confidence — a safety failure).
        false_action_rate: Optional[float] = None
        if y_confidence is not None:
            low_conf_acted = [
                int(not a)
                for c, a in zip(y_confidence, abstain_mask)
                if c < self.confidence_threshold
            ]
            if low_conf_acted:
                false_action_rate = float(np.mean(low_conf_acted))

        report = ClassificationReport(
            macro_f1=float(macro_f),
            macro_precision=float(macro_p),
            macro_recall=float(macro_r),
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

    def gate(
        self,
        report: "ClassificationReport",
        macro_f1_threshold: float = 0.70,
        false_action_rate_threshold: float = 0.05,
    ) -> None:
        """Raise ``ValueError`` when the report does not meet deployment thresholds.

        Args:
            report:                    A ``ClassificationReport`` from ``evaluate()``.
            macro_f1_threshold:        Minimum required macro F1 (default 0.70).
            false_action_rate_threshold: Maximum allowed false-action rate
                                        (default 0.05).  Ignored when
                                        ``report.false_action_rate`` is ``None``
                                        (i.e. no confidence scores were supplied).

        Raises:
            TypeError:  If *report* is not a ``ClassificationReport``.
            ValueError: If any threshold is out of range ``[0, 1]``.
            ValueError: If the report does not meet the thresholds.
        """
        if not isinstance(report, ClassificationReport):
            raise TypeError(
                f"'report' must be a ClassificationReport, got {type(report).__name__!r}"
            )
        if not (0.0 <= macro_f1_threshold <= 1.0):
            raise ValueError(
                f"'macro_f1_threshold' must be in [0, 1], got {macro_f1_threshold!r}"
            )
        if not (0.0 <= false_action_rate_threshold <= 1.0):
            raise ValueError(
                f"'false_action_rate_threshold' must be in [0, 1], got "
                f"{false_action_rate_threshold!r}"
            )

        failures = []
        if report.macro_f1 < macro_f1_threshold:
            failures.append(
                f"macro_f1={report.macro_f1:.4f} < threshold={macro_f1_threshold:.4f}"
            )
        if (
            report.false_action_rate is not None
            and report.false_action_rate > false_action_rate_threshold
        ):
            failures.append(
                f"false_action_rate={report.false_action_rate:.4f} "
                f"> threshold={false_action_rate_threshold:.4f}"
            )
        if failures:
            msg = "ClassificationEvaluator.gate FAILED: " + "; ".join(failures)
            logger.error(msg)
            raise ValueError(msg)
        logger.info(
            "ClassificationEvaluator.gate PASSED: macro_f1=%.4f far=%s "
            "thresholds=(f1>=%.2f, far<=%.2f)",
            report.macro_f1,
            f"{report.false_action_rate:.4f}" if report.false_action_rate is not None else "n/a",
            macro_f1_threshold, false_action_rate_threshold,
        )


"""Isotonic regression — non-parametric post-hoc calibration.

Isotonic regression fits a *monotone* mapping from raw model probabilities to
calibrated probabilities by solving the pool-adjacent-violators (PAV) problem.
Unlike temperature scaling, it has unlimited expressive power within the
monotonicity constraint and can reduce ECE to near-zero on the fitting data.

**Monotone direction (``increasing='auto'``).**  By default sklearn's
:class:`sklearn.isotonic.IsotonicRegression` assumes a non-decreasing mapping
(higher confidence → higher calibrated probability), which is the "textbook"
case.  However, this dataset exhibits an *anti-correlated* accuracy–confidence
relationship: the model is overconfident on false positives (hard negatives
have mean confidence ≈ 0.87) and underconfident on true positives (correct
examples have mean confidence ≈ 0.52).  Setting ``increasing='auto'`` allows
sklearn to choose the direction that minimises squared residuals, which for
this distribution will be *decreasing*, correctly mapping high-confidence
false-positives to low calibrated probabilities and vice-versa.

**Cross-fitting recommendation.**  Fitting on the same data used for
evaluation (in-sample isotonic regression) produces near-zero ECE trivially.
Always fit on a held-out training split and evaluate on a separate validation
split, or use k-fold cross-calibration.  The
:func:`~training.calibration.recalibrate_and_checkpoint.run` script uses an
80/20 stratified hold-out split.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """Non-parametric post-hoc calibrator based on isotonic regression.

    Learns a monotone (increasing *or* decreasing, chosen automatically)
    mapping from raw model probabilities to calibrated probabilities using
    scikit-learn's :class:`~sklearn.isotonic.IsotonicRegression` with the
    PAV algorithm.

    Attributes:
        iso_ (IsotonicRegression): Fitted sklearn estimator.  Available after
            ``fit()``.
        increasing_ (bool | str): Direction chosen by the fit (``True`` for
            non-decreasing, ``False`` for non-increasing).

    Raises:
        RuntimeError: If ``calibrate()`` is called before ``fit()``.
    """

    def __init__(self) -> None:
        self.iso_: Optional[IsotonicRegression] = None
        self.increasing_: Optional[object] = "auto"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        """Fit a monotone calibration mapping from a labelled validation set.

        Args:
            probs:  1-D array of raw model probabilities in ``[0, 1]``
                    (shape ``(N,)``).
            labels: 1-D array of binary ground-truth labels ``{0, 1}``
                    (shape ``(N,)``).

        Returns:
            ``self`` — supports method chaining.

        Raises:
            ValueError: If *probs* or *labels* are empty, have mismatched
                        lengths, or contain values outside their valid ranges.
        """
        probs, labels = self._validate(probs, labels)

        self.iso_ = IsotonicRegression(
            increasing="auto",          # auto-detect increasing vs. decreasing
            out_of_bounds="clip",       # clip predictions to [min_y, max_y]
            y_min=0.0,
            y_max=1.0,
        )
        self.iso_.fit(probs, labels)
        self.increasing_ = self.iso_.increasing_

        logger.info(
            "IsotonicCalibrator.fit: n=%d  direction=%s  "
            "y_range=[%.4f, %.4f]",
            len(probs),
            "increasing" if self.increasing_ else "decreasing",
            float(self.iso_.f_.y.min()),
            float(self.iso_.f_.y.max()),
        )
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply the fitted isotonic mapping to raw model probabilities.

        Args:
            probs: 1-D array of raw model probabilities in ``[0, 1]``
                   (shape ``(N,)``).

        Returns:
            1-D array of calibrated probabilities in ``[0, 1]``
            (same shape as *probs*).

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if self.iso_ is None:
            raise RuntimeError("IsotonicCalibrator.calibrate() called before fit()")
        probs = np.asarray(probs, dtype=np.float64).ravel()
        cal = self.iso_.predict(probs)
        return np.clip(cal, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(
        probs: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        probs = np.asarray(probs, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        if len(probs) == 0:
            raise ValueError("'probs' must be non-empty")
        if len(probs) != len(labels):
            raise ValueError(
                f"'probs' and 'labels' must have the same length; "
                f"got {len(probs)} vs {len(labels)}"
            )
        if not np.all((probs >= 0.0) & (probs <= 1.0)):
            raise ValueError("All values in 'probs' must be in [0, 1]")
        unique_labels = set(np.unique(labels).tolist())
        if not unique_labels.issubset({0.0, 1.0}):
            raise ValueError(
                f"'labels' must contain only 0 and 1; found {unique_labels}"
            )
        return probs, labels


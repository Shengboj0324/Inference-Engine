"""Temperature scaling — single-parameter post-hoc calibration.

Temperature scaling is the simplest post-hoc calibration method.  Given a
model that outputs probabilities :math:`p_i`, it learns a single scalar
temperature :math:`T^*` on a held-out validation split by minimising the
negative log-likelihood (NLL):

.. math::

    T^* = \\arg\\min_T \\; -\\frac{1}{N} \\sum_{i=1}^{N} \\left[
        y_i \\log \\sigma\\!\\left(\\frac{z_i}{T}\\right)
        + (1-y_i) \\log \\left(1 - \\sigma\\!\\left(\\frac{z_i}{T}\\right)\\right)
    \\right]

where :math:`z_i = \\log(p_i / (1-p_i))` is the log-odds (logit) of the raw
probability and :math:`\\sigma` is the logistic sigmoid.

Setting :math:`T > 1` softens (flattens) the distribution — appropriate when
the model is overconfident.  Setting :math:`T < 1` sharpens the distribution —
appropriate when the model is underconfident.  Setting :math:`T = 1` is a
no-op.

**Limitations:** Temperature scaling has a single degree of freedom and can
only apply a monotone, globally uniform shift.  When the accuracy–confidence
relationship is anti-correlated across different confidence ranges (high
confidence → low accuracy), temperature scaling cannot reduce ECE below a
dataset-specific floor.  In those cases use :class:`IsotonicCalibrator`.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

_EPS: float = 1e-7
_T_LO: float = 1e-4   # minimum allowed temperature
_T_HI: float = 1e4    # maximum allowed temperature (wide enough to not artificially cap)


class TemperatureScaler:
    """Single-parameter post-hoc calibrator based on temperature scaling.

    Fits one scalar :math:`T` on a labelled validation set by minimising NLL,
    then applies the logistic transform :math:`\\sigma(\\text{logit}(p)/T)` at
    inference time.

    Attributes:
        temperature_ (float): Learned temperature.  Available after ``fit()``.
        nll_before_ (float): NLL on the fitting data *before* calibration.
        nll_after_  (float): NLL on the fitting data *after* calibration.

    Raises:
        RuntimeError: If ``calibrate()`` is called before ``fit()``.
    """

    def __init__(self) -> None:
        self.temperature_: Optional[float] = None
        self.nll_before_: Optional[float] = None
        self.nll_after_: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        """Learn the optimal temperature from a labelled validation set.

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
        logits = self._to_logits(probs)

        self.nll_before_ = float(self._nll(1.0, logits, labels))

        result = minimize_scalar(
            self._nll,
            args=(logits, labels),
            bounds=(_T_LO, _T_HI),
            method="bounded",
            options={"xatol": 1e-8, "maxiter": 1000},
        )
        self.temperature_ = float(result.x)
        self.nll_after_ = float(result.fun)

        logger.info(
            "TemperatureScaler.fit: T*=%.6f  NLL before=%.6f  after=%.6f",
            self.temperature_, self.nll_before_, self.nll_after_,
        )
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to raw model probabilities.

        Args:
            probs: 1-D array of raw model probabilities in ``[0, 1]``
                   (shape ``(N,)``).

        Returns:
            1-D array of calibrated probabilities in ``[0, 1]``
            (same shape as *probs*).

        Raises:
            RuntimeError: If called before ``fit()``.
        """
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaler.calibrate() called before fit()")
        probs = np.asarray(probs, dtype=np.float64)
        logits = self._to_logits(probs)
        scaled = 1.0 / (1.0 + np.exp(-logits / self.temperature_))
        return np.clip(scaled, 0.0, 1.0)

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

    @staticmethod
    def _to_logits(probs: np.ndarray) -> np.ndarray:
        p = np.clip(probs, _EPS, 1.0 - _EPS)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
        T = max(_T_LO, T)
        scaled = 1.0 / (1.0 + np.exp(-logits / T))
        scaled = np.clip(scaled, _EPS, 1.0 - _EPS)
        return float(-np.mean(
            labels * np.log(scaled) + (1.0 - labels) * np.log(1.0 - scaled)
        ))


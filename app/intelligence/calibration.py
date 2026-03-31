"""Calibration system for probability calibration.

This module implements Stage D of the inference pipeline:
- Temperature scaling for probability calibration
- Platt scaling for binary calibration
- Isotonic regression for non-parametric calibration
- Expected Calibration Error (ECE) computation
- Brier score computation

Also provides ``ConfidenceCalibrator`` — a per-``SignalType`` online
temperature-scaling calibrator whose learned scalars are persisted in
``training/calibration_state.json`` and updated via single gradient steps
on binary cross-entropy loss.
"""

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from app.domain.inference_models import SignalInference, CalibrationMetrics, SignalType

logger = logging.getLogger(__name__)


@dataclass
class CalibrationParams:
    """Parameters for calibration methods."""
    
    temperature: float = 1.0  # Temperature scaling parameter
    platt_a: float = 1.0  # Platt scaling parameter A
    platt_b: float = 0.0  # Platt scaling parameter B


class Calibrator:
    """Probability calibration system.
    
    This is Stage D of the inference pipeline as defined in the blueprint.
    Converts raw model confidence into calibrated probabilities.
    """
    
    def __init__(
        self,
        method: str = "temperature",
        params: Optional[CalibrationParams] = None,
    ):
        """Initialize calibrator.
        
        Args:
            method: Calibration method ('temperature', 'platt', 'isotonic')
            params: Calibration parameters (learned from validation data)
        """
        self.method = method
        self.params = params or CalibrationParams()
        
        logger.info(f"Calibrator initialized: method={method}")
    
    def calibrate(self, inference: SignalInference) -> SignalInference:
        """Calibrate probabilities in signal inference.
        
        Args:
            inference: Signal inference with raw probabilities
            
        Returns:
            Signal inference with calibrated probabilities
        """
        # Apply calibration to each prediction (skip loop when list is empty)
        for prediction in inference.predictions:
            raw_prob = prediction.probability
            
            if self.method == "temperature":
                calibrated_prob = self._temperature_scaling(raw_prob)
            elif self.method == "platt":
                calibrated_prob = self._platt_scaling(raw_prob)
            else:
                calibrated_prob = raw_prob
            
            # Update probability
            prediction.probability = calibrated_prob
        
        # Update top prediction
        if inference.top_prediction:
            raw_prob = inference.top_prediction.probability
            if self.method == "temperature":
                calibrated_prob = self._temperature_scaling(raw_prob)
            elif self.method == "platt":
                calibrated_prob = self._platt_scaling(raw_prob)
            else:
                calibrated_prob = raw_prob
            
            inference.top_prediction.probability = calibrated_prob
        
        # Compute calibration metrics
        calibration_metrics = self._compute_calibration_metrics(inference)
        inference.calibration_metrics = calibration_metrics
        
        logger.debug(f"Calibrated inference {inference.id} using {self.method}")
        return inference
    
    def _temperature_scaling(self, probability: float) -> float:
        """Apply temperature scaling to probability.
        
        Temperature scaling: p_calibrated = softmax(logit / T)
        where T is the temperature parameter.
        
        Args:
            probability: Raw probability
            
        Returns:
            Calibrated probability
        """
        # Convert probability to logit
        epsilon = 1e-7
        probability = np.clip(probability, epsilon, 1 - epsilon)
        logit = np.log(probability / (1 - probability))
        
        # Apply temperature scaling
        scaled_logit = logit / self.params.temperature
        
        # Convert back to probability
        calibrated_prob = 1 / (1 + np.exp(-scaled_logit))
        
        return float(np.clip(calibrated_prob, 0.0, 1.0))
    
    def _platt_scaling(self, probability: float) -> float:
        """Apply Platt scaling to probability.
        
        Platt scaling: p_calibrated = 1 / (1 + exp(A * logit + B))
        where A and B are learned parameters.
        
        Args:
            probability: Raw probability
            
        Returns:
            Calibrated probability
        """
        # Convert probability to logit
        epsilon = 1e-7
        probability = np.clip(probability, epsilon, 1 - epsilon)
        logit = np.log(probability / (1 - probability))
        
        # Apply Platt scaling
        scaled_logit = self.params.platt_a * logit + self.params.platt_b
        
        # Convert back to probability
        calibrated_prob = 1 / (1 + np.exp(-scaled_logit))
        
        return float(np.clip(calibrated_prob, 0.0, 1.0))
    
    def _compute_calibration_metrics(
        self, inference: SignalInference
    ) -> Optional[CalibrationMetrics]:
        """Compute calibration metrics for inference.
        
        Args:
            inference: Signal inference
            
        Returns:
            Calibration metrics or None
        """
        if not inference.top_prediction:
            return None

        p = inference.top_prediction.probability
        # ECE and Brier require a ground-truth label set; we cannot compute them
        # for a single inference.  Set them to None — the evals module (app/evals/)
        # computes these properly over a batch with known labels.
        # Only the confidence interval (simple ±σ approximation) is meaningful here.
        ci_half = min(0.15, p * (1 - p) ** 0.5)  # σ of Bernoulli, capped at 0.15
        return CalibrationMetrics(
            expected_calibration_error=None,
            brier_score=None,
            confidence_interval_lower=max(0.0, p - ci_half),
            confidence_interval_upper=min(1.0, p + ci_half),
        )

    def fit_temperature(
        self,
        probabilities: List[float],
        labels: List[int],
        learning_rate: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Fit temperature parameter using validation data.

        Args:
            probabilities: Raw model probabilities
            labels: True labels (0 or 1)
            learning_rate: Learning rate for optimization
            max_iter: Maximum iterations

        Returns:
            Optimal temperature
        """
        probabilities = np.array(probabilities)
        labels = np.array(labels)

        # Convert probabilities to logits
        epsilon = 1e-7
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        logits = np.log(probabilities / (1 - probabilities))

        # Initialize temperature
        temperature = 1.0

        # Optimize temperature using gradient descent
        for _ in range(max_iter):
            # Compute scaled probabilities
            scaled_logits = logits / temperature
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))

            # Compute negative log likelihood
            nll = -np.mean(
                labels * np.log(scaled_probs + epsilon) +
                (1 - labels) * np.log(1 - scaled_probs + epsilon)
            )

            # Compute gradient
            gradient = np.mean((scaled_probs - labels) * logits / (temperature ** 2))

            # Update temperature
            temperature -= learning_rate * gradient
            temperature = max(0.1, temperature)  # Ensure positive

        self.params.temperature = temperature
        logger.info(f"Fitted temperature: {temperature:.4f}")
        return temperature

    def fit_platt(
        self,
        probabilities: List[float],
        labels: List[int],
    ) -> Tuple[float, float]:
        """Fit Platt scaling parameters using validation data.

        Args:
            probabilities: Raw model probabilities
            labels: True labels (0 or 1)

        Returns:
            Tuple of (A, B) parameters
        """
        probabilities = np.array(probabilities)
        labels = np.array(labels)

        # Convert probabilities to logits
        epsilon = 1e-7
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        logits = np.log(probabilities / (1 - probabilities))

        # Fit logistic regression
        # This is a simplified version - in production use sklearn
        from scipy.optimize import minimize

        def objective(params):
            a, b = params
            scaled_logits = a * logits + b
            scaled_probs = 1 / (1 + np.exp(-scaled_logits))
            nll = -np.mean(
                labels * np.log(scaled_probs + epsilon) +
                (1 - labels) * np.log(1 - scaled_probs + epsilon)
            )
            return nll

        result = minimize(objective, [1.0, 0.0], method='BFGS')
        a, b = result.x

        self.params.platt_a = a
        self.params.platt_b = b
        logger.info(f"Fitted Platt parameters: A={a:.4f}, B={b:.4f}")
        return a, b

    @staticmethod
    def compute_ece(
        probabilities: List[float],
        labels: List[int],
        n_bins: int = 10,
    ) -> float:
        """Compute Expected Calibration Error (ECE).

        Args:
            probabilities: Predicted probabilities
            labels: True labels (0 or 1)
            n_bins: Number of bins for calibration

        Returns:
            ECE score (lower is better)
        """
        probabilities = np.array(probabilities)
        labels = np.array(labels)

        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)

        ece = 0.0
        for i in range(n_bins):
            # Get samples in this bin
            mask = (probabilities >= bins[i]) & (probabilities < bins[i + 1])
            if not np.any(mask):
                continue

            # Compute accuracy and confidence in this bin
            bin_probs = probabilities[mask]
            bin_labels = labels[mask]

            accuracy = np.mean(bin_labels)
            confidence = np.mean(bin_probs)

            # Add weighted difference to ECE
            ece += len(bin_probs) / len(probabilities) * abs(accuracy - confidence)

        return float(ece)

    @staticmethod
    def compute_brier_score(
        probabilities: List[float],
        labels: List[int],
    ) -> float:
        """Compute Brier score.

        Args:
            probabilities: Predicted probabilities
            labels: True labels (0 or 1)

        Returns:
            Brier score (lower is better)
        """
        probabilities = np.array(probabilities)
        labels = np.array(labels)

        return float(np.mean((probabilities - labels) ** 2))


# ---------------------------------------------------------------------------
# ConfidenceCalibrator — per-SignalType online temperature scaling
# ---------------------------------------------------------------------------

#: Default path for the persisted calibration scalars.
_DEFAULT_STATE_PATH: Path = Path("training/calibration_state.json")

#: Minimum allowed temperature scalar (prevents division by zero / collapse).
_T_MIN: float = 0.1

#: Maximum allowed temperature scalar (prevents unbounded growth after many
#: incorrect high-confidence predictions).
_T_MAX: float = 100.0

#: Learning rate for the single-step gradient update in ``update()``.
_LR: float = 0.01

#: Epsilon used to clamp ``predicted_prob`` away from 0 and 1 before any
#: ``log`` or ``exp`` operation.  Inputs must be validated to lie in [0, 1]
#: before this clamp is applied.
_PROB_EPS: float = 1e-7


class ConfidenceCalibrator:
    """Per-``SignalType`` post-hoc temperature-scaling calibrator.

    Each ``SignalType`` has its own learned temperature scalar ``T`` (initialised
    to ``1.0``).  ``calibrate()`` applies the logistic transform
    ``sigmoid(raw_logit / T)`` to map a raw log-odds score to a calibrated
    probability.  ``update()`` performs a single gradient-descent step on the
    binary cross-entropy loss to adjust ``T`` given one labelled example.

    Scalars are persisted to ``state_path`` (a JSON file) after every
    ``update()`` call so they survive restarts.

    Args:
        state_path: Path to the JSON file that stores per-``SignalType``
            temperature scalars.  Defaults to
            ``training/calibration_state.json``.
        learning_rate: Step size for the gradient update.  Defaults to
            ``0.01``.
    """

    #: Maximum α weight for the per-user calibration curve in the federated blend.
    _ALPHA_MAX: float = 0.7
    #: Number of confirmed outcomes required for α to reach ``_ALPHA_MAX``.
    _ALPHA_CONVERGENCE_N: int = 500

    def __init__(
        self,
        state_path: Optional[Path] = None,
        learning_rate: float = _LR,
    ) -> None:
        """Initialise and load persisted scalars.

        Args:
            state_path: Path to ``calibration_state.json``.
            learning_rate: Gradient-descent step size for ``update()``
                and ``update_user()``.
        """
        self._state_path: Path = state_path or _DEFAULT_STATE_PATH
        self._lr: float = learning_rate
        # Global per-SignalType temperature scalars (shared across all users).
        self._scalars: Dict[str, float] = {}
        # Per-user temperature scalars: user_id → {signal_type.value → T}
        self._user_scalars: Dict[str, Dict[str, float]] = {}
        # Confirmed outcome counts per user (used to compute α blend weight).
        self._user_outcome_counts: Dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, raw_logit: float, signal_type: SignalType) -> float:
        """Return the calibrated probability for ``raw_logit``.

        Applies ``sigmoid(raw_logit / T)`` where ``T`` is the learned
        temperature for ``signal_type``.  When ``T == 1.0`` (default), the
        result is identical to the plain logistic sigmoid.

        Non-finite ``raw_logit`` values (NaN, ±Inf) are handled gracefully:
        NaN and ±Inf inputs are logged and produce ``0.5`` (maximum uncertainty),
        while exponent overflow in the sigmoid (very large negative logits with
        a very small T) is clamped to ``0.0``.

        Args:
            raw_logit: Log-odds score from the LLM or upstream model.
                Typically derived from the raw confidence probability via
                ``logit = log(p / (1 - p))``.
            signal_type: Signal type whose temperature scalar to apply.

        Returns:
            Calibrated probability in ``[0.0, 1.0]``.
        """
        if not math.isfinite(raw_logit):
            logger.warning(
                "ConfidenceCalibrator.calibrate: non-finite raw_logit=%r for "
                "signal_type=%s; returning 0.5 (maximum uncertainty)",
                raw_logit,
                signal_type.value,
            )
            return 0.5
        t: float = max(_T_MIN, min(_T_MAX, self._scalars.get(signal_type.value, 1.0)))
        try:
            p = 1.0 / (1.0 + math.exp(-raw_logit / t))
        except OverflowError:
            # -raw_logit / t >> 709 → exp overflows → sigmoid → 0.0
            p = 0.0
        return float(np.clip(p, 0.0, 1.0))

    def update(
        self,
        signal_type: SignalType,
        predicted_prob: float,
        true_label: bool,
        lr: Optional[float] = None,
    ) -> None:
        """Perform one gradient step on ``T`` using binary cross-entropy loss.

        The gradient of NLL w.r.t. ``T`` is:
        ``dL/dT = (p_cal - y) * (−logit / T²)``
        where ``p_cal = sigmoid(logit / T)`` and ``y ∈ {0, 1}``.

        The scalar is updated as
        ``T ← clamp(T − lr * dL/dT, T_MIN, T_MAX)``
        and immediately persisted to ``state_path``.

        Args:
            signal_type: Signal type whose temperature to update.
            predicted_prob: Raw predicted probability (before calibration).
                Must be a finite float in ``[0.0, 1.0]``; values outside this
                range raise ``ValueError``.
            true_label: ``True`` if the prediction was correct.
            lr: Optional per-call learning rate override.  When provided it
                takes precedence over ``self._lr`` for this update only.
                Must be a finite positive float; raises ``ValueError`` otherwise.

        Raises:
            ValueError: If ``predicted_prob`` is not finite or not in [0, 1],
                or if ``lr`` is provided but is non-positive or non-finite.
        """
        if not math.isfinite(predicted_prob):
            raise ValueError(
                f"predicted_prob must be a finite float; got {predicted_prob!r}"
            )
        if not (0.0 <= predicted_prob <= 1.0):
            raise ValueError(
                f"predicted_prob must be in [0.0, 1.0]; got {predicted_prob!r}"
            )
        if lr is not None:
            if not math.isfinite(lr):
                raise ValueError(
                    f"lr must be a finite positive float; got {lr!r}"
                )
            if lr <= 0.0:
                raise ValueError(
                    f"lr must be positive; got {lr!r}"
                )
        effective_lr: float = lr if lr is not None else self._lr
        p: float = max(_PROB_EPS, min(1.0 - _PROB_EPS, predicted_prob))
        logit: float = math.log(p / (1.0 - p))
        t: float = max(_T_MIN, min(_T_MAX, self._scalars.get(signal_type.value, 1.0)))
        p_cal: float = 1.0 / (1.0 + math.exp(-logit / t))
        y: float = 1.0 if true_label else 0.0
        gradient: float = (p_cal - y) * (-logit / (t * t))
        new_t: float = max(_T_MIN, min(_T_MAX, t - effective_lr * gradient))
        self._scalars[signal_type.value] = new_t
        self._save()
        logger.debug(
            "ConfidenceCalibrator.update: signal_type=%s T: %.4f → %.4f (lr=%.5f)",
            signal_type.value,
            t,
            new_t,
            effective_lr,
        )

    # ------------------------------------------------------------------
    # Federated (per-user) calibration — Step 3b
    # ------------------------------------------------------------------

    def _compute_alpha(self, user_id: str) -> float:
        """Return the blend weight α for ``user_id``'s per-user curve.

        α starts at ``0.0`` for new users (no personal data) and converges
        linearly toward ``_ALPHA_MAX`` (default ``0.7``) after
        ``_ALPHA_CONVERGENCE_N`` (default ``500``) confirmed outcomes.

        Args:
            user_id: String representation of the user UUID.

        Returns:
            Float in ``[0.0, _ALPHA_MAX]``.
        """
        n = self._user_outcome_counts.get(user_id, 0)
        return min(self._ALPHA_MAX, self._ALPHA_MAX * n / self._ALPHA_CONVERGENCE_N)

    def update_user(
        self,
        signal_type: SignalType,
        user_id: str,
        predicted_prob: float,
        true_label: bool,
        lr: Optional[float] = None,
    ) -> None:
        """Perform one gradient step on the *per-user* temperature scalar.

        Identical gradient formula to ``update()`` but applied to the
        per-user temperature ``T_u`` rather than the global ``T``.  Also
        increments ``_user_outcome_counts`` so that ``_compute_alpha`` can
        return a converging blend weight.

        Args:
            signal_type: Signal type whose per-user temperature to update.
            user_id: String representation of the user UUID.
            predicted_prob: Raw predicted probability (before calibration).
                Must be in ``[0.0, 1.0]``; raises ``ValueError`` otherwise.
            true_label: ``True`` if the prediction was correct (true-positive);
                ``False`` for false-positives, dismissed, or snoozed outcomes.
            lr: Optional per-call learning rate override.

        Raises:
            ValueError: If ``predicted_prob`` is not finite or not in [0, 1].
        """
        if not math.isfinite(predicted_prob):
            raise ValueError(
                f"predicted_prob must be finite; got {predicted_prob!r}"
            )
        if not (0.0 <= predicted_prob <= 1.0):
            raise ValueError(
                f"predicted_prob must be in [0.0, 1.0]; got {predicted_prob!r}"
            )
        effective_lr: float = lr if (lr is not None and lr > 0.0) else self._lr
        p: float = max(_PROB_EPS, min(1.0 - _PROB_EPS, predicted_prob))
        logit: float = math.log(p / (1.0 - p))

        # Retrieve or initialise the per-user scalar for this signal type
        user_temps = self._user_scalars.setdefault(user_id, {})
        t_u: float = max(_T_MIN, min(_T_MAX, user_temps.get(signal_type.value, 1.0)))

        p_cal: float = 1.0 / (1.0 + math.exp(-logit / t_u))
        y: float = 1.0 if true_label else 0.0
        gradient: float = (p_cal - y) * (-logit / (t_u * t_u))
        new_t_u: float = max(_T_MIN, min(_T_MAX, t_u - effective_lr * gradient))
        user_temps[signal_type.value] = new_t_u

        # Track confirmed outcomes (any label) for α convergence
        self._user_outcome_counts[user_id] = (
            self._user_outcome_counts.get(user_id, 0) + 1
        )
        self._save()
        logger.debug(
            "ConfidenceCalibrator.update_user: user=%s signal_type=%s T_u: %.4f → %.4f",
            user_id, signal_type.value, t_u, new_t_u,
        )

    def calibrate_federated(
        self,
        raw_logit: float,
        signal_type: SignalType,
        user_id: str,
    ) -> float:
        """Return a federated calibrated probability blending user + global curves.

        Formula::

            α = _compute_alpha(user_id)   # in [0.0, 0.7]
            p_user   = sigmoid(raw_logit / T_user)
            p_global = sigmoid(raw_logit / T_global)
            p_final  = α * p_user + (1 - α) * p_global

        For new users (``α ≈ 0``) the result is identical to ``calibrate()``.
        After 500 confirmed outcomes (``α = 0.7``) the personal curve dominates.

        Non-finite ``raw_logit`` is handled identically to ``calibrate()``.

        Args:
            raw_logit: Log-odds score from the LLM / upstream model.
            signal_type: Signal type whose temperatures to apply.
            user_id: String representation of the user UUID.

        Returns:
            Federated calibrated probability in ``[0.0, 1.0]``.
        """
        if not math.isfinite(raw_logit):
            logger.warning(
                "ConfidenceCalibrator.calibrate_federated: non-finite raw_logit=%r; "
                "returning 0.5",
                raw_logit,
            )
            return 0.5

        # Global probability
        t_global: float = max(_T_MIN, min(_T_MAX, self._scalars.get(signal_type.value, 1.0)))
        try:
            p_global = 1.0 / (1.0 + math.exp(-raw_logit / t_global))
        except OverflowError:
            p_global = 0.0

        # Per-user probability
        user_temps = self._user_scalars.get(user_id, {})
        t_user: float = max(_T_MIN, min(_T_MAX, user_temps.get(signal_type.value, 1.0)))
        try:
            p_user = 1.0 / (1.0 + math.exp(-raw_logit / t_user))
        except OverflowError:
            p_user = 0.0

        alpha: float = self._compute_alpha(user_id)
        p_final = alpha * p_user + (1.0 - alpha) * p_global
        return float(np.clip(p_final, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load global and per-user scalars from ``state_path``.

        Falls back gracefully for any missing or unreadable state.
        """
        try:
            if self._state_path.exists():
                with self._state_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._scalars = {
                    k: float(v) for k, v in data.get("scalars", {}).items()
                }
                # Per-user scalars (added in v2.0 — absent in legacy files)
                raw_user = data.get("user_scalars", {})
                self._user_scalars = {
                    uid: {st: float(t) for st, t in temps.items()}
                    for uid, temps in raw_user.items()
                }
                raw_counts = data.get("user_outcome_counts", {})
                self._user_outcome_counts = {
                    uid: int(n) for uid, n in raw_counts.items()
                }
                logger.info(
                    "ConfidenceCalibrator: loaded %d global + %d user scalars from %s",
                    len(self._scalars),
                    len(self._user_scalars),
                    self._state_path,
                )
            else:
                logger.info(
                    "ConfidenceCalibrator: state file not found at %s; using T=1.0",
                    self._state_path,
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ConfidenceCalibrator: failed to load state: %s", exc)
            self._scalars = {}
            self._user_scalars = {}
            self._user_outcome_counts = {}

    def _save(self) -> None:
        """Persist global and per-user scalars to ``state_path``."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict = {
                "version": "2.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "scalars": self._scalars,
                "user_scalars": self._user_scalars,
                "user_outcome_counts": self._user_outcome_counts,
            }
            with self._state_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            logger.error("ConfidenceCalibrator: failed to save state: %s", exc)


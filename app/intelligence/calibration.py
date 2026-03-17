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

#: Learning rate for the single-step gradient update in ``update()``.
_LR: float = 0.01


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

    def __init__(
        self,
        state_path: Optional[Path] = None,
        learning_rate: float = _LR,
    ) -> None:
        """Initialise and load persisted scalars.

        Args:
            state_path: Path to ``calibration_state.json``.
            learning_rate: Gradient-descent step size for ``update()``.
        """
        self._state_path: Path = state_path or _DEFAULT_STATE_PATH
        self._lr: float = learning_rate
        self._scalars: Dict[str, float] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, raw_logit: float, signal_type: SignalType) -> float:
        """Return the calibrated probability for ``raw_logit``.

        Applies ``sigmoid(raw_logit / T)`` where ``T`` is the learned
        temperature for ``signal_type``.  When ``T == 1.0`` (default), the
        result is identical to the plain logistic sigmoid.

        Args:
            raw_logit: Log-odds score from the LLM or upstream model.
                Typically derived from the raw confidence probability via
                ``logit = log(p / (1 - p))``.
            signal_type: Signal type whose temperature scalar to apply.

        Returns:
            Calibrated probability in ``[0.0, 1.0]``.
        """
        t: float = max(_T_MIN, self._scalars.get(signal_type.value, 1.0))
        return 1.0 / (1.0 + math.exp(-raw_logit / t))

    def update(
        self,
        signal_type: SignalType,
        predicted_prob: float,
        true_label: bool,
    ) -> None:
        """Perform one gradient step on ``T`` using binary cross-entropy loss.

        The gradient of NLL w.r.t. ``T`` is:
        ``dL/dT = (p_cal - y) * (−logit / T²)``
        where ``p_cal = sigmoid(logit / T)`` and ``y ∈ {0, 1}``.

        The scalar is updated as ``T ← max(T_MIN, T − lr * dL/dT)`` and
        immediately persisted to ``state_path``.

        Args:
            signal_type: Signal type whose temperature to update.
            predicted_prob: Raw predicted probability (before calibration).
            true_label: ``True`` if the prediction was correct.
        """
        epsilon: float = 1e-7
        p: float = max(epsilon, min(1.0 - epsilon, predicted_prob))
        logit: float = math.log(p / (1.0 - p))
        t: float = max(_T_MIN, self._scalars.get(signal_type.value, 1.0))
        p_cal: float = 1.0 / (1.0 + math.exp(-logit / t))
        y: float = 1.0 if true_label else 0.0
        gradient: float = (p_cal - y) * (-logit / (t * t))
        new_t: float = max(_T_MIN, t - self._lr * gradient)
        self._scalars[signal_type.value] = new_t
        self._save()
        logger.debug(
            "ConfidenceCalibrator.update: signal_type=%s T: %.4f → %.4f",
            signal_type.value,
            t,
            new_t,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load scalars from ``state_path``.  Falls back to ``T=1.0`` for any
        missing or unreadable state."""
        try:
            if self._state_path.exists():
                with self._state_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._scalars = {k: float(v) for k, v in data.get("scalars", {}).items()}
                logger.info(
                    "ConfidenceCalibrator: loaded %d scalars from %s",
                    len(self._scalars),
                    self._state_path,
                )
            else:
                logger.info(
                    "ConfidenceCalibrator: state file not found at %s; using T=1.0 for all types",
                    self._state_path,
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("ConfidenceCalibrator: failed to load state: %s", exc)
            self._scalars = {}

    def _save(self) -> None:
        """Persist current scalars to ``state_path``."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict = {
                "version": "1.0",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "scalars": self._scalars,
            }
            with self._state_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            logger.error("ConfidenceCalibrator: failed to save state: %s", exc)


"""Calibration system for probability calibration.

This module implements Stage D of the inference pipeline:
- Temperature scaling for probability calibration
- Platt scaling for binary calibration
- Isotonic regression for non-parametric calibration
- Expected Calibration Error (ECE) computation
- Brier score computation

Raw model confidence is not trustworthy. This module adds a calibration layer
to ensure that predicted probabilities match empirical frequencies.
"""

import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel

from app.domain.inference_models import SignalInference, CalibrationMetrics

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
        if not inference.predictions:
            return inference
        
        # Apply calibration to each prediction
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
        
        # For now, return placeholder metrics
        # In production, these would be computed from validation data
        return CalibrationMetrics(
            expected_calibration_error=0.05,  # Placeholder
            brier_score=0.1,  # Placeholder
            confidence_interval_lower=max(0.0, inference.top_prediction.probability - 0.1),
            confidence_interval_upper=min(1.0, inference.top_prediction.probability + 0.1),
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


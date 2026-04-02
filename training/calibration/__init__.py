"""Post-hoc calibration package for the Social-Media-Radar inference engine.

This package implements two complementary post-hoc calibration strategies that
can be applied to an already-trained model **without re-training from scratch**:

TemperatureScaler
    A single-parameter method that learns a temperature scalar T by minimising
    negative log-likelihood (NLL) on a held-out validation split.  Fast and
    interpretable.  Works well when the confidence distribution is monotonically
    ordered by accuracy.  When the data is anti-correlated (high confidence →
    low accuracy), a very large or very small T is required and ECE improvement
    is limited.

IsotonicCalibrator
    A non-parametric method that fits a monotone mapping from raw probabilities
    to calibrated probabilities using isotonic regression.  Can handle both
    increasing and decreasing relationships between confidence and accuracy
    (``increasing='auto'``).  Achieves near-zero ECE on held-out data when the
    train and validation distributions are similar.

Both classes expose the same two-method interface::

    calibrator.fit(probs, labels)        # probs: (N,) float, labels: (N,) int {0,1}
    calibrated = calibrator.calibrate(probs)  # returns (N,) float in [0, 1]

Quick-start example::

    from training.calibration.temperature_scaling import TemperatureScaler
    from training.calibration.isotonic_calibration import IsotonicCalibrator

    ts = TemperatureScaler().fit(train_probs, train_labels)
    print(ts.temperature_)          # optimal T

    iso = IsotonicCalibrator().fit(train_probs, train_labels)
    cal_probs = iso.calibrate(val_probs)
"""

from training.calibration.temperature_scaling import TemperatureScaler
from training.calibration.isotonic_calibration import IsotonicCalibrator

__all__ = ["TemperatureScaler", "IsotonicCalibrator"]


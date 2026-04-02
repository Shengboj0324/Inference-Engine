# ConfidenceCalibrator — Online Temperature Scaling and Federated Blending

## Formulation

`ConfidenceCalibrator` (`app/intelligence/confidence_calibrator.py`) maps raw LLM logits to calibrated probabilities using per-`SignalType` temperature scaling:

```
p_cal = sigmoid(logit / T)
```

`T` is initialised from the seed calibration dataset and updated online. At `T = 1.0` (the default) the formula reduces to the standard sigmoid.

## Federated Temperature Blending

When both per-user and global temperature scalars are available, the effective temperature is:

```
T_eff = α · T_user + (1 − α) · T_global
```

`α` is the blending weight:

```
α = min(0.7, n_confirmed / 500)
```

where `n_confirmed` is the number of analyst-confirmed outcomes for the current user on the given `SignalType`. At `n_confirmed = 0`, `α = 0` and `T_eff = T_global`. At `n_confirmed ≥ 500`, `α = 0.7` and the user scalar contributes 70 % of the effective temperature.

`calibrate_federated(global_calibrator, user_calibrators, alpha_schedule)` takes a list of per-user `ConfidenceCalibrator` instances and updates the global calibrator using federated averaging:

```
T_global_new = Σ (w_i · T_user_i)   where  w_i = n_i / Σ n_j
```

## Online Update

Each call to `FeedbackStore.record_feedback()` triggers one gradient-descent step on binary cross-entropy:

```
loss  = −[y · log(p_cal) + (1−y) · log(1−p_cal)]
∂T/∂T = (p_cal − y) · (−logit / T²)
T     ← max(T_MIN, T − lr · ∂loss/∂T)
```

`T_MIN = 0.1` prevents the temperature from collapsing to zero. `lr` defaults to `0.01`. One gradient step costs ~6–8 µs on commodity hardware; no batch accumulation, no restart.

```python
calibrator = ConfidenceCalibrator()
calibrator.update(signal_type, logit=2.3, y_true=0)  # analyst says "wrong"
p = calibrator.calibrate(signal_type, logit=2.3)      # updated probability
```

## Seed Calibration Dataset

Format: JSONL, one record per line.

```json
{"signal_type": "churn_risk", "logit": 1.2, "label": 1}
{"signal_type": "feature_request", "logit": -0.3, "label": 0}
```

Fields:

| Field | Type | Description |
|---|---|---|
| `signal_type` | `str` | Any of the 18 `SignalType` enum values |
| `logit` | `float` | Raw score from the LLM output layer before sigmoid |
| `label` | `int` | Ground-truth: `1` (positive) or `0` (negative) |

Default path: `training/data/calibration_seed.jsonl`. Override with `--data-path`.

## Training Script

```bash
# Full calibration from seed data
python training/calibrate.py --epochs 5

# Custom data path, lower learning rate
python training/calibrate.py \
  --data-path ./my_labelled_data.jsonl \
  --epochs 10 \
  --lr 0.005

# Federated calibration pass (aggregates per-user Ts into global T)
python training/calibrate.py --federated --user-store-path ./user_calibrators/
```

Checkpoint format: `training/checkpoints/epoch_{N:03d}_ece_{ece:.4f}.json`

```json
{
  "epoch": 4,
  "temperatures": {"churn_risk": 1.24, "feature_request": 0.91, "…": "…"},
  "metrics": {
    "ece": 0.3757,
    "macro_f1": 0.842,
    "samples_processed": 2340
  }
}
```

## Evaluation Metrics

| Metric | Definition |
|---|---|
| ECE (Expected Calibration Error) | Mean absolute difference between mean confidence and accuracy in M=15 equal-frequency bins |
| Macro F1 | Unweighted mean of per-`SignalType` F1 at threshold `p_cal ≥ 0.5` |
| Abstention Rate | Fraction of observations below the configured abstention threshold |

Reproduce the benchmark:
```bash
python deliverables/benchmark.py --calibration-report
```

## Checkpoints

Saved automatically after every epoch if ECE improves. Load into a live calibrator:

```python
from app.intelligence.confidence_calibrator import ConfidenceCalibrator

cal = ConfidenceCalibrator()
cal.load_checkpoint("training/checkpoints/epoch_004_ece_0.3757.json")
```


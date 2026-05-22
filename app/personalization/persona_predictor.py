"""Tier 2.4 — next-style predictor.

Personalises *proactively*: instead of only reacting to the latest stated
preference, it predicts the user's preferred style for the **next** turn from a
short rolling window of recent observations of a trait.  This lets the agent
pre-adapt (e.g. a user who has been trending toward terse answers gets a concise
reply before they ask again).

The model is a logistic classifier over window features
``[last_value, window_mean, window_trend, recent_fraction_high]`` predicting
whether the next observation will be "high" (≥ 0.5).  The logistic core is the
shared :class:`app.intelligence.retrieval_fusion.LogisticFusion` (reused, not
re-implemented, to avoid redundant gradient-descent code).

Pure NumPy + the shared logistic core; deterministic; JSON-serialisable.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from app.intelligence.retrieval_fusion import LogisticFusion

_N_FEATURES = 4


def featurize(window: Sequence[float]) -> List[float]:
    """Map a window of recent trait values to a fixed 4-dim feature vector.

    Features: last value, mean, linear trend (last − first), and the fraction
    of the window that is "high" (≥ 0.5).  Empty windows map to a neutral
    vector so the model degrades gracefully on cold start.
    """
    w = [min(1.0, max(0.0, float(v))) for v in window]
    if not w:
        return [0.5, 0.5, 0.0, 0.5]
    last = w[-1]
    mean = sum(w) / len(w)
    trend = w[-1] - w[0]
    frac_high = sum(1 for v in w if v >= 0.5) / len(w)
    return [last, mean, trend, frac_high]


class NextStylePredictor:
    """Predicts the next-turn binary style label from a rolling window.

    Args:
        window: Number of recent observations used as context.
        l2: L2 regularisation for the logistic core.
    """

    def __init__(self, window: int = 4, l2: float = 1e-3) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = int(window)
        self._model = LogisticFusion(n_features=_N_FEATURES, l2=l2)

    def is_trained(self) -> bool:
        return self._model.is_trained()

    def build_dataset(
        self, sequences: Sequence[Sequence[float]]
    ) -> Tuple[List[List[float]], List[float]]:
        """Turn raw value sequences into ``(X, y)`` sliding-window training data.

        For each position ``t >= 1`` the features are computed from the trailing
        ``window`` values ending at ``t-1`` and the label is ``1`` iff value at
        ``t`` is "high" (≥ 0.5).
        """
        X: List[List[float]] = []
        y: List[float] = []
        for seq in sequences:
            s = list(seq)
            for t in range(1, len(s)):
                ctx = s[max(0, t - self.window):t]
                X.append(featurize(ctx))
                y.append(1.0 if s[t] >= 0.5 else 0.0)
        return X, y

    def fit(self, sequences: Sequence[Sequence[float]], epochs: int = 600, lr: float = 0.5) -> "NextStylePredictor":
        X, y = self.build_dataset(sequences)
        if not X:
            raise ValueError("no training windows could be built from sequences")
        self._model.fit(X, y, epochs=epochs, lr=lr)
        return self

    def predict_proba(self, window: Sequence[float]) -> float:
        """Return P(next value is 'high') given a recent ``window`` of values."""
        return self._model.predict_one(featurize(list(window)[-self.window:]))

    def predict(self, window: Sequence[float]) -> int:
        return 1 if self.predict_proba(window) >= 0.5 else 0

    def to_dict(self) -> Dict:
        return {"version": "1.0", "window": self.window, "model": self._model.to_dict()}

    @classmethod
    def from_dict(cls, d: Dict) -> "NextStylePredictor":
        p = cls(window=int(d.get("window", 4)))
        p._model = LogisticFusion.from_dict(d.get("model", {}))
        return p

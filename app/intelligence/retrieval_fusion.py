"""Learned retrieval fusion (Tier 1.1).

Replaces the hand-set ``embedding / entity / platform`` source weights in
:class:`~app.intelligence.candidate_retrieval.CandidateRetriever` with a small,
**self-contained logistic-regression** combiner trained on labelled
``(observation → gold signal_type)`` pairs.

Each candidate signal type is described by a feature vector of its per-source
scores ``[embedding_score, entity_score, platform_score]``; the model learns a
single relevance probability per candidate, so the contribution of each source
is *learned from data* rather than guessed.  The model is pure-NumPy (no
scikit-learn / torch dependency), deterministic, and JSON-serialisable, which
keeps it trainable in the notebook and verifiable without a heavy stack.

Design notes
------------
- **Numerically stable** sigmoid (clipped logits) and L2-regularised batch
  gradient descent.
- **Backward compatible by default:** ``CandidateRetriever`` only routes through
  the learned model when one is attached *and* trained; otherwise the original
  fixed-weight path is used unchanged.
- **Feature order is fixed** (``FEATURE_NAMES``) so a serialised model always
  lines up with the features the retriever extracts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

#: Canonical feature order — must match CandidateRetriever.extract_fusion_features.
FEATURE_NAMES: List[str] = ["embedding_score", "entity_score", "platform_score"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip to avoid overflow in exp for extreme logits.
    z = np.clip(z, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


class LogisticFusion:
    """Logistic-regression fusion of per-source retrieval scores.

    Args:
        n_features: Length of each feature vector (defaults to the 3 sources).
        l2: L2 regularisation strength.
    """

    def __init__(self, n_features: int = 3, l2: float = 1e-3) -> None:
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features = int(n_features)
        self.l2 = float(l2)
        self.w = np.zeros(self.n_features, dtype=np.float64)
        self.b = 0.0
        self._trained = False

    def is_trained(self) -> bool:
        return self._trained

    def fit(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[float],
        epochs: int = 600,
        lr: float = 0.5,
    ) -> "LogisticFusion":
        """Fit the combiner with L2-regularised batch gradient descent.

        Args:
            X: ``(n_samples, n_features)`` feature matrix.
            y: ``(n_samples,)`` binary relevance labels in {0, 1}.
            epochs: Number of full-batch gradient steps.
            lr: Learning rate.

        Returns:
            ``self`` (fitted).

        Raises:
            ValueError: On empty / mis-shaped input.
        """
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if Xa.ndim != 2 or Xa.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array")
        if Xa.shape[1] != self.n_features:
            raise ValueError(f"X has {Xa.shape[1]} features; expected {self.n_features}")
        if ya.shape[0] != Xa.shape[0]:
            raise ValueError("X and y length mismatch")

        n = Xa.shape[0]
        w = np.zeros(self.n_features, dtype=np.float64)
        b = 0.0
        for _ in range(int(epochs)):
            p = _sigmoid(Xa @ w + b)
            err = p - ya
            grad_w = Xa.T @ err / n + self.l2 * w
            grad_b = float(np.mean(err))
            w -= lr * grad_w
            b -= lr * grad_b
        self.w = w
        self.b = float(b)
        self._trained = True
        return self

    def predict_one(self, features: Sequence[float]) -> float:
        """Return the relevance probability for a single feature vector."""
        x = np.asarray(features, dtype=np.float64)
        if x.shape[0] != self.n_features:
            raise ValueError(f"expected {self.n_features} features, got {x.shape[0]}")
        return float(_sigmoid(np.array([x @ self.w + self.b]))[0])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "version": "1.0",
            "n_features": self.n_features,
            "l2": self.l2,
            "feature_names": FEATURE_NAMES,
            "w": self.w.tolist(),
            "b": self.b,
            "trained": self._trained,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "LogisticFusion":
        m = cls(n_features=int(d.get("n_features", 3)), l2=float(d.get("l2", 1e-3)))
        m.w = np.asarray(d.get("w", [0.0] * m.n_features), dtype=np.float64)
        m.b = float(d.get("b", 0.0))
        m._trained = bool(d.get("trained", False))
        return m

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "LogisticFusion":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

"""Tier 1.3 — similarity → probability calibration.

Raw cosine similarity is *not* a probability: a 0.6 cosine means different
things for different embedding models, so a fixed ``min_score`` threshold on
``ContextMemoryStore.retrieve()`` is not portable.  ``SimilarityCalibrator``
maps a similarity score to a **calibrated relevance probability** ``P(relevant
| sim)`` fitted on labelled ``(similarity, relevant)`` pairs, so thresholds and
cross-model comparisons become meaningful.

Two methods (pick per data size):

- **Platt** — logistic ``sigmoid(a*sim + b)``.  Parametric, robust on small
  data, monotonic by construction.
- **Isotonic** — non-parametric monotonic fit via Pool-Adjacent-Violators
  (PAV).  Flexible on larger data; never decreases.

This is a distinct concern from :class:`app.intelligence.calibration.Calibrator`
(which calibrates *classifier* probabilities); it shares the same statistical
toolkit (Platt / isotonic) but operates on the similarity domain.  Pure NumPy,
deterministic, JSON-serialisable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _pav(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pool-Adjacent-Violators isotonic regression (non-decreasing).

    Returns ``(x_sorted, y_fitted)`` — a monotone step function over the sorted
    unique-ish x grid.
    """
    order = np.argsort(x, kind="mergesort")
    xs = x[order].astype(np.float64)
    ys = y[order].astype(np.float64)
    n = len(ys)
    # Block representation: value, weight, count.
    vals = ys.copy()
    weights = np.ones(n)
    # Iteratively merge adjacent blocks that violate monotonicity.
    i = 0
    blocks: List[List[float]] = []  # [value, weight, start, end]
    for idx in range(n):
        cur_val, cur_w = vals[idx], weights[idx]
        blocks.append([cur_val, cur_w, idx, idx])
        while len(blocks) > 1 and blocks[-2][0] >= blocks[-1][0]:
            v2, w2, s2, e2 = blocks.pop()
            v1, w1, s1, e1 = blocks.pop()
            merged_w = w1 + w2
            merged_v = (v1 * w1 + v2 * w2) / merged_w
            blocks.append([merged_v, merged_w, s1, e2])
    fitted = np.empty(n, dtype=np.float64)
    for v, w, s, e in blocks:
        fitted[s:e + 1] = v
    return xs, fitted


class SimilarityCalibrator:
    """Calibrates cosine similarity into a relevance probability.

    Args:
        method: ``"platt"`` (default) or ``"isotonic"``.
    """

    def __init__(self, method: str = "platt") -> None:
        if method not in ("platt", "isotonic"):
            raise ValueError("method must be 'platt' or 'isotonic'")
        self.method = method
        self.a = 1.0
        self.b = 0.0
        self._iso_x = np.array([], dtype=np.float64)
        self._iso_y = np.array([], dtype=np.float64)
        self._fitted = False

    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        sims: Sequence[float],
        labels: Sequence[float],
        epochs: int = 800,
        lr: float = 0.5,
    ) -> "SimilarityCalibrator":
        s = np.asarray(sims, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if s.ndim != 1 or s.shape[0] == 0 or s.shape[0] != y.shape[0]:
            raise ValueError("sims and labels must be equal-length non-empty 1D arrays")
        if self.method == "platt":
            a, b = 0.0, 0.0
            n = len(s)
            for _ in range(int(epochs)):
                p = _sigmoid(a * s + b)
                err = p - y
                a -= lr * float(np.mean(err * s))
                b -= lr * float(np.mean(err))
            self.a, self.b = float(a), float(b)
        else:  # isotonic
            self._iso_x, self._iso_y = _pav(s, y)
        self._fitted = True
        return self

    def transform(self, sim: float) -> float:
        """Map a single similarity to a calibrated probability in [0, 1]."""
        if not self._fitted:
            # Identity-ish fallback: clamp raw similarity into [0,1].
            return float(min(1.0, max(0.0, sim)))
        if self.method == "platt":
            return float(_sigmoid(np.array([self.a * sim + self.b]))[0])
        # isotonic: linear interpolation over the fitted step grid
        if self._iso_x.size == 0:
            return float(min(1.0, max(0.0, sim)))
        p = float(np.interp(sim, self._iso_x, self._iso_y,
                            left=self._iso_y[0], right=self._iso_y[-1]))
        return min(1.0, max(0.0, p))

    @staticmethod
    def expected_calibration_error(
        probs: Sequence[float], labels: Sequence[float], n_bins: int = 10
    ) -> float:
        """Binned Expected Calibration Error (lower is better)."""
        p = np.asarray(probs, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if p.size == 0:
            return 0.0
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n = len(p)
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
            if not np.any(mask):
                continue
            conf = float(np.mean(p[mask]))
            acc = float(np.mean(y[mask]))
            ece += (np.sum(mask) / n) * abs(conf - acc)
        return float(ece)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": "1.0", "method": self.method, "a": self.a, "b": self.b,
            "iso_x": self._iso_x.tolist(), "iso_y": self._iso_y.tolist(),
            "fitted": self._fitted,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimilarityCalibrator":
        c = cls(method=d.get("method", "platt"))
        c.a = float(d.get("a", 1.0)); c.b = float(d.get("b", 0.0))
        c._iso_x = np.asarray(d.get("iso_x", []), dtype=np.float64)
        c._iso_y = np.asarray(d.get("iso_y", []), dtype=np.float64)
        c._fitted = bool(d.get("fitted", False))
        return c

    def save(self, path: Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "SimilarityCalibrator":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

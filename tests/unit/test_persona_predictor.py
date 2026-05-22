"""Tests for Tier 2.4 — NextStylePredictor."""

from __future__ import annotations

import numpy as np
import pytest

from app.personalization.persona_predictor import NextStylePredictor, featurize


class TestFeaturize:
    def test_empty_window_neutral(self):
        assert featurize([]) == [0.5, 0.5, 0.0, 0.5]

    def test_feature_values(self):
        f = featurize([0.0, 0.0, 1.0])
        assert f[0] == 1.0          # last
        assert f[1] == pytest.approx(1 / 3)  # mean
        assert f[2] == 1.0          # trend (last - first)
        assert f[3] == pytest.approx(1 / 3)  # frac high


class TestNextStylePredictor:
    @staticmethod
    def _sticky(n, rng, p_stay=0.85):
        s = [1.0 if rng.random() < 0.5 else 0.0]
        for _ in range(n - 1):
            s.append(s[-1] if rng.random() < p_stay else 1.0 - s[-1])
        return s

    def test_beats_majority_baseline(self):
        rng = np.random.default_rng(0)
        train = [self._sticky(40, rng) for _ in range(120)]
        test = [self._sticky(40, rng) for _ in range(60)]
        nsp = NextStylePredictor(window=4).fit(train)
        X, y = nsp.build_dataset(test)
        preds = [1.0 if nsp._model.predict_one(x) >= 0.5 else 0.0 for x in X]
        acc = float(np.mean([p == t for p, t in zip(preds, y)]))
        majority = max(float(np.mean(y)), 1 - float(np.mean(y)))
        assert acc > majority + 0.05

    def test_round_trip(self):
        rng = np.random.default_rng(1)
        nsp = NextStylePredictor(window=4).fit([self._sticky(30, rng) for _ in range(40)])
        n2 = NextStylePredictor.from_dict(nsp.to_dict())
        assert abs(n2.predict_proba([1, 1, 1, 1]) - nsp.predict_proba([1, 1, 1, 1])) < 1e-9

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError):
            NextStylePredictor(window=4).fit([[0.5]])  # single-element seq -> no windows

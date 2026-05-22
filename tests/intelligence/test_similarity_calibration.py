"""Tests for Tier 1.3 — SimilarityCalibrator (Platt + isotonic)."""

from __future__ import annotations

import numpy as np
import pytest

from app.intelligence.similarity_calibration import SimilarityCalibrator

_ECE = SimilarityCalibrator.expected_calibration_error


def _data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    sims = rng.uniform(0, 1, n)
    p_true = 1.0 / (1.0 + np.exp(-10 * (sims - 0.6)))
    labels = (rng.uniform(0, 1, n) < p_true).astype(float)
    return sims, labels


class TestSimilarityCalibrator:
    def test_platt_lowers_ece(self):
        sims, labels = _data()
        tr, te = slice(0, 900), slice(900, None)
        cal = SimilarityCalibrator("platt").fit(sims[tr], labels[tr])
        probs = np.array([cal.transform(s) for s in sims[te]])
        raw = np.clip(sims[te], 0, 1)
        assert _ECE(probs, labels[te]) < _ECE(raw, labels[te])

    def test_isotonic_monotonic_and_lowers_ece(self):
        sims, labels = _data()
        tr, te = slice(0, 900), slice(900, None)
        iso = SimilarityCalibrator("isotonic").fit(sims[tr], labels[tr])
        xs = np.linspace(0, 1, 50)
        curve = [iso.transform(x) for x in xs]
        assert all(curve[i + 1] >= curve[i] - 1e-9 for i in range(len(xs) - 1))
        probs = np.array([iso.transform(s) for s in sims[te]])
        raw = np.clip(sims[te], 0, 1)
        assert _ECE(probs, labels[te]) < _ECE(raw, labels[te])

    def test_transform_bounded(self):
        sims, labels = _data()
        cal = SimilarityCalibrator("platt").fit(sims, labels)
        for s in (-2.0, 0.0, 0.5, 1.0, 5.0):
            assert 0.0 <= cal.transform(s) <= 1.0

    def test_unfitted_fallback_clamps(self):
        cal = SimilarityCalibrator("platt")
        assert not cal.is_fitted()
        assert cal.transform(1.5) == 1.0
        assert cal.transform(-0.3) == 0.0
        assert cal.transform(0.4) == pytest.approx(0.4)

    def test_round_trip_and_save_load(self, tmp_path):
        sims, labels = _data()
        cal = SimilarityCalibrator("platt").fit(sims, labels)
        c2 = SimilarityCalibrator.from_dict(cal.to_dict())
        assert abs(c2.transform(0.7) - cal.transform(0.7)) < 1e-12
        p = tmp_path / "cal.json"
        cal.save(p)
        c3 = SimilarityCalibrator.load(p)
        assert abs(c3.transform(0.7) - cal.transform(0.7)) < 1e-12

    def test_bad_method_raises(self):
        with pytest.raises(ValueError):
            SimilarityCalibrator("nonsense")

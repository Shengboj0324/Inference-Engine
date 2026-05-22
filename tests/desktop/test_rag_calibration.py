"""Tests for the similarity calibrator wired into serving retrieval (Tier 1.3).

Constructs the retriever via ``__new__`` to test the calibration seam in
isolation without standing up the full ContentStore / embedder stack.
"""

from __future__ import annotations

from app.intelligence.similarity_calibration import SimilarityCalibrator
from app.local.rag_retriever import LocalRAGRetriever


def _bare_retriever() -> LocalRAGRetriever:
    r = LocalRAGRetriever.__new__(LocalRAGRetriever)
    r._calibrator = None
    return r


class TestServingCalibration:
    def test_no_calibrator_is_identity(self):
        r = _bare_retriever()
        assert r._calibrate(0.42) == 0.42

    def test_calibrator_applied(self):
        r = _bare_retriever()
        cal = SimilarityCalibrator("platt")
        cal.a, cal.b, cal._fitted = 10.0, -5.0, True  # steep logistic around 0.5
        r.set_similarity_calibrator(cal)
        assert r._calibrate(0.9) > r._calibrate(0.1)        # monotonic
        assert 0.0 <= r._calibrate(0.5) <= 1.0

    def test_calibrate_soft_fails_to_raw(self):
        class _Bad:
            def transform(self, x):
                raise RuntimeError("boom")
        r = _bare_retriever()
        r.set_similarity_calibrator(_Bad())
        assert r._calibrate(0.37) == 0.37  # falls back to raw, never raises

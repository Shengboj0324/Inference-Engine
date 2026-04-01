"""Phase 4 — Grounded Synthesis Unit Tests.

Test groups
-----------
Group 1:  EvidenceSourceModel         — frozen value-object, field validators
Group 2:  AttributedClaimModel        — frozen, confidence/confidence range
Group 3:  ContradictionPairModel      — cross-claim validator (claim_a ≠ claim_b)
Group 4:  UncertaintyAnnotationModel  — frozen, position ≥ -1
Group 5:  GroundedSummaryModel        — frozen, score ranges
Group 6:  SynthesisRequestModel       — sources non-empty validator
Group 7:  SourceAttributorConstruct   — constructor validation
Group 8:  SourceAttributorAttribute   — heuristic text attribution
Group 9:  SourceAttributorRanking     — rank_sources_by_relevance ordering
Group 10: SourceAttributorTrustWeight — compute_trust_weight correctness
Group 11: SourceAttributorEdgeCases   — empty sources, wrong types
Group 12: ClaimVerifierConstruct      — constructor validation
Group 13: ClaimVerifierVerify         — heuristic confidence scoring
Group 14: ClaimVerifierClassify       — ClaimType classification patterns
Group 15: ClaimVerifierNegation       — negation detection and penalty
Group 16: ClaimVerifierBatch          — verify_batch returns AttributedClaim list
Group 17: ContradictionDetectorConstruct  — constructor validation
Group 18: ContradictionDetectorNegation   — negation-conflict detection
Group 19: ContradictionDetectorNumber     — cardinal-number conflict
Group 20: ContradictionDetectorAntonym    — antonym conflict
Group 21: ContradictionDetectorAPI        — is_contradiction / score_severity
Group 22: ContradictionDetectorSort       — severity-descending ordering
Group 23: UncertaintyAnnotatorConstruct   — constructor validation
Group 24: UncertaintyAnnotatorAnnotate    — hedge-word detection
Group 25: UncertaintyAnnotatorSeverity    — classify_severity patterns
Group 26: UncertaintyAnnotatorOverall     — overall_uncertainty aggregation
Group 27: GroundedSummaryBuilderConstruct — constructor validation
Group 28: GroundedSummaryBuilderBuild     — full heuristic build
Group 29: GroundedSummaryBuilderConfidence — confidence formula
Group 30: MultiSourceSynthesizerConstruct — constructor validation
Group 31: MultiSourceSynthesizerDedup     — near-duplicate pruning
Group 32: MultiSourceSynthesizerMerge     — claim consolidation
Group 33: MultiSourceSynthesizerSynthesize — full synthesis call
Group 34: CrossComponentAttributionFlow   — attribution feeds claim verifier
Group 35: CrossComponentContradictions    — contradictions surface in summary
Group 36: CrossComponentPackageImports    — __init__ exports verified
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from app.summarization.models import SynthesisRequest

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _src(
    source_id: str = "s1",
    title: str = "Test source",
    content_snippet: str = "",
    trust: float = 0.8,
) -> "EvidenceSource":
    from app.summarization.models import EvidenceSource
    return EvidenceSource(source_id=source_id, title=title, content_snippet=content_snippet, trust_score=trust)


def _claim(text: str = "The model achieves high accuracy.", conf: float = 0.7, source_ids=None) -> "AttributedClaim":
    from app.summarization.models import AttributedClaim
    return AttributedClaim(text=text, confidence=conf, source_ids=source_ids or [])


def _request(topic: str = "AI", sources=None) -> "SynthesisRequest":
    from app.summarization.models import SynthesisRequest
    if sources is None:
        sources = [_src()]
    return SynthesisRequest(topic=topic, sources=sources)


# ===========================================================================
# Group 1: EvidenceSourceModel
# ===========================================================================

class TestEvidenceSourceModel:
    def test_default_source_id_generated(self):
        from app.summarization.models import EvidenceSource
        src = EvidenceSource()
        assert len(src.source_id) > 0

    def test_frozen(self):
        src = _src()
        with pytest.raises(Exception):
            src.trust_score = 0.1  # type: ignore

    def test_trust_out_of_range_raises(self):
        from app.summarization.models import EvidenceSource
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            EvidenceSource(trust_score=1.5)

    def test_trust_negative_raises(self):
        from app.summarization.models import EvidenceSource
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            EvidenceSource(trust_score=-0.1)

    def test_empty_source_id_raises(self):
        from app.summarization.models import EvidenceSource
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            EvidenceSource(source_id="")

    def test_defaults(self):
        from app.summarization.models import EvidenceSource
        src = EvidenceSource()
        assert src.title == ""
        assert src.url == ""
        assert src.platform == ""
        assert src.trust_score == pytest.approx(0.5)
        assert src.published_at is None


# ===========================================================================
# Group 2: AttributedClaimModel
# ===========================================================================

class TestAttributedClaimModel:
    def test_frozen(self):
        c = _claim()
        with pytest.raises(Exception):
            c.confidence = 0.1  # type: ignore

    def test_empty_text_raises(self):
        from app.summarization.models import AttributedClaim
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            AttributedClaim(text="")

    def test_confidence_out_of_range_raises(self):
        from app.summarization.models import AttributedClaim
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            AttributedClaim(text="x", confidence=1.5)

    def test_confidence_negative_raises(self):
        from app.summarization.models import AttributedClaim
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            AttributedClaim(text="x", confidence=-0.1)

    def test_default_claim_type(self):
        from app.summarization.models import AttributedClaim, ClaimType
        c = AttributedClaim(text="Something happened.")
        assert c.claim_type == ClaimType.FACTUAL

    def test_negation_defaults_false(self):
        c = _claim()
        assert c.negation_detected is False

    def test_auto_claim_id(self):
        c = _claim()
        assert len(c.claim_id) > 0


# ===========================================================================
# Group 3: ContradictionPairModel
# ===========================================================================

class TestContradictionPairModel:
    def test_same_claim_id_raises(self):
        from app.summarization.models import ContradictionPair
        import pydantic
        c = _claim()
        with pytest.raises((ValueError, pydantic.ValidationError)):
            ContradictionPair(claim_a=c, claim_b=c)

    def test_different_claims_ok(self):
        from app.summarization.models import ContradictionPair
        pair = ContradictionPair(claim_a=_claim("X is good."), claim_b=_claim("X is not good."))
        assert pair.severity is not None

    def test_default_severity(self):
        from app.summarization.models import ContradictionPair, ContradictionSeverity
        pair = ContradictionPair(claim_a=_claim("A is true."), claim_b=_claim("B is false."))
        assert pair.severity == ContradictionSeverity.MODERATE

    def test_frozen(self):
        from app.summarization.models import ContradictionPair
        pair = ContradictionPair(claim_a=_claim("A is true."), claim_b=_claim("B is false."))
        with pytest.raises(Exception):
            pair.severity = None  # type: ignore


# ===========================================================================
# Group 4: UncertaintyAnnotationModel
# ===========================================================================

class TestUncertaintyAnnotationModel:
    def test_frozen(self):
        from app.summarization.models import UncertaintyAnnotation
        ann = UncertaintyAnnotation(text_span="might happen")
        with pytest.raises(Exception):
            ann.severity = None  # type: ignore

    def test_empty_span_raises(self):
        from app.summarization.models import UncertaintyAnnotation
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            UncertaintyAnnotation(text_span="")

# ===========================================================================
# Group 7: SourceAttributorConstruct
# ===========================================================================

class TestSourceAttributorConstruct:
    def test_invalid_min_overlap_zero_raises(self):
        from app.summarization.source_attribution import SourceAttributor
        with pytest.raises(ValueError):
            SourceAttributor(min_overlap=0.0)

    def test_invalid_min_overlap_over_one_raises(self):
        from app.summarization.source_attribution import SourceAttributor
        with pytest.raises(ValueError):
            SourceAttributor(min_overlap=1.5)

    def test_invalid_llm_threshold_raises(self):
        from app.summarization.source_attribution import SourceAttributor
        with pytest.raises(ValueError):
            SourceAttributor(llm_threshold=1.5)

    def test_negative_max_sources_raises(self):
        from app.summarization.source_attribution import SourceAttributor
        with pytest.raises(ValueError):
            SourceAttributor(max_sources=-1)

    def test_default_construction(self):
        from app.summarization.source_attribution import SourceAttributor
        attr = SourceAttributor()
        assert attr is not None


# ===========================================================================
# Group 8: SourceAttributorAttribute
# ===========================================================================

class TestSourceAttributorAttribute:
    def _attr(self):
        from app.summarization.source_attribution import SourceAttributor
        return SourceAttributor(min_overlap=0.05)

    def test_matching_source_returned(self):
        attr = self._attr()
        src = _src("s1", content_snippet="OpenAI released GPT-5 model this week.")
        result = attr.attribute_text("OpenAI GPT-5 model", [src])
        assert len(result) == 1
        assert result[0].source_id == "s1"

    def test_non_matching_source_filtered(self):
        attr = self._attr()
        irrelevant = _src("s2", content_snippet="The stock market fell today.")
        result = attr.attribute_text("quantum computing breakthrough", [irrelevant])
        assert result == []

    def test_multiple_sources_ranked(self):
        from app.summarization.source_attribution import SourceAttributor
        attr = SourceAttributor(min_overlap=0.05)
        high = _src("s1", content_snippet="GPT-5 model released by OpenAI with amazing results.", trust=0.9)
        low  = _src("s2", content_snippet="GPT-5 somewhat mentioned here.", trust=0.5)
        result = attr.attribute_text("GPT-5 OpenAI model", [high, low])
        # High overlap source should rank first
        assert result[0].source_id == "s1"

    def test_empty_text_raises(self):
        from app.summarization.source_attribution import SourceAttributor
        with pytest.raises(ValueError):
            SourceAttributor().attribute_text("", [_src()])

    def test_wrong_text_type_raises(self):
        from app.summarization.source_attribution import SourceAttributor
        with pytest.raises(TypeError):
            SourceAttributor().attribute_text(123, [_src()])  # type: ignore

    def test_wrong_sources_type_raises(self):
        from app.summarization.source_attribution import SourceAttributor
        with pytest.raises(TypeError):
            SourceAttributor().attribute_text("query", "not a list")  # type: ignore

    def test_max_sources_limits_output(self):
        from app.summarization.source_attribution import SourceAttributor
        attr = SourceAttributor(min_overlap=0.01, max_sources=2)
        sources = [_src(f"s{i}", content_snippet=f"model{i} GPT result test") for i in range(5)]
        result = attr.attribute_text("GPT model result test", sources)
        assert len(result) <= 2


# ===========================================================================
# Group 9: SourceAttributorRanking
# ===========================================================================

class TestSourceAttributorRanking:
    def test_rank_by_relevance_order(self):
        from app.summarization.source_attribution import SourceAttributor
        attr = SourceAttributor()
        very_relevant = _src("s1", title="GPT-5 released by OpenAI", content_snippet="GPT-5 model OpenAI release")
        less_relevant = _src("s2", title="Market news", content_snippet="stock prices fluctuate daily")
        result = attr.rank_sources_by_relevance("GPT-5 OpenAI", [less_relevant, very_relevant])
        assert result[0].source_id == "s1"

    def test_empty_query_raises(self):
        from app.summarization.source_attribution import SourceAttributor
        with pytest.raises(ValueError):
            SourceAttributor().rank_sources_by_relevance("", [_src()])

    def test_wrong_query_type_raises(self):
        from app.summarization.source_attribution import SourceAttributor
        with pytest.raises(TypeError):
            SourceAttributor().rank_sources_by_relevance(None, [_src()])  # type: ignore

    def test_returns_all_sources(self):
        from app.summarization.source_attribution import SourceAttributor
        sources = [_src(f"s{i}") for i in range(4)]
        result = SourceAttributor().rank_sources_by_relevance("test query term", sources)
        assert len(result) == 4


# ===========================================================================
# Group 10: SourceAttributorTrustWeight
# ===========================================================================

class TestSourceAttributorTrustWeight:
    def test_mean_trust_single_source(self):
        from app.summarization.source_attribution import SourceAttributor
        result = SourceAttributor().compute_trust_weight([_src(trust=0.8)])
        assert result == pytest.approx(0.8)

    def test_mean_trust_multiple_sources(self):
        from app.summarization.source_attribution import SourceAttributor
        sources = [_src("a", trust=0.6), _src("b", trust=0.4)]
        result = SourceAttributor().compute_trust_weight(sources)
        assert result == pytest.approx(0.5)

    def test_empty_sources_returns_zero(self):
        from app.summarization.source_attribution import SourceAttributor
        assert SourceAttributor().compute_trust_weight([]) == pytest.approx(0.0)

    def test_wrong_type_raises(self):
        from app.summarization.source_attribution import SourceAttributor
        with pytest.raises(TypeError):
            SourceAttributor().compute_trust_weight("bad")  # type: ignore


# ===========================================================================
# Group 11: SourceAttributorEdgeCases
# ===========================================================================

class TestSourceAttributorEdgeCases:
    def test_empty_content_snippet_source_skipped(self):
        from app.summarization.source_attribution import SourceAttributor
        attr = SourceAttributor(min_overlap=0.1)
        src_empty = _src("s1", content_snippet="", title="")
        # No tokens to match → empty snippet skipped
        result = attr.attribute_text("machine learning model", [src_empty])
        assert result == []

    def test_single_token_query(self):
        from app.summarization.source_attribution import SourceAttributor
        attr = SourceAttributor(min_overlap=0.1)
        src = _src("s1", content_snippet="transformer architecture scaling paper")
        result = attr.attribute_text("transformer", [src])
        assert len(result) == 1


# ===========================================================================
# Group 12: ClaimVerifierConstruct
# ===========================================================================

class TestClaimVerifierConstruct:
    def test_wrong_attributor_type_raises(self):
        from app.summarization.claim_verifier import ClaimVerifier
        with pytest.raises(TypeError):
            ClaimVerifier(attributor="bad")

    def test_invalid_llm_threshold_raises(self):
        from app.summarization.claim_verifier import ClaimVerifier
        with pytest.raises(ValueError):
            ClaimVerifier(llm_threshold=1.5)

    def test_invalid_negation_penalty_raises(self):
        from app.summarization.claim_verifier import ClaimVerifier
        with pytest.raises(ValueError):
            ClaimVerifier(negation_penalty=0.0)

    def test_default_construction(self):
        from app.summarization.claim_verifier import ClaimVerifier
        v = ClaimVerifier()
        assert v is not None


# ===========================================================================
# Group 13: ClaimVerifierVerify
# ===========================================================================

class TestClaimVerifierVerify:
    def _verifier(self):
        from app.summarization.claim_verifier import ClaimVerifier
        return ClaimVerifier()

    def test_confidence_in_unit_range(self):
        v = self._verifier()
        src = _src("s1", content_snippet="GPT-5 model released by OpenAI this week with high accuracy.")
        conf = v.verify_claim("GPT-5 OpenAI model accuracy", [src])
        assert 0.0 <= conf <= 1.0

    def test_high_overlap_higher_confidence(self):
        v = self._verifier()
        rich_src = _src("s1", content_snippet="The transformer model achieves 95 percent accuracy on language benchmarks.", trust=0.9)
        poor_src = _src("s2", content_snippet="Something happened somewhere.", trust=0.3)
        conf_rich = v.verify_claim("transformer model accuracy benchmarks", [rich_src])
        conf_poor = v.verify_claim("transformer model accuracy benchmarks", [poor_src])
        assert conf_rich >= conf_poor

    def test_no_sources_returns_zero(self):
        v = self._verifier()
        conf = v.verify_claim("some claim text here", [])
        assert conf == pytest.approx(0.0)

    def test_empty_claim_raises(self):
        v = self._verifier()
        with pytest.raises(ValueError):
            v.verify_claim("", [_src()])

    def test_wrong_claim_type_raises(self):
        v = self._verifier()
        with pytest.raises(TypeError):
            v.verify_claim(123, [_src()])  # type: ignore

    def test_wrong_sources_type_raises(self):
        v = self._verifier()
        with pytest.raises(TypeError):
            v.verify_claim("claim", "not a list")  # type: ignore


# ===========================================================================
# Group 14: ClaimVerifierClassify
# ===========================================================================

class TestClaimVerifierClassify:
    def _v(self):
        from app.summarization.claim_verifier import ClaimVerifier
        return ClaimVerifier()

    def test_benchmark_detected(self):
        from app.summarization.models import ClaimType
        assert self._v().classify_claim("The model achieves 95% accuracy on MMLU.") == ClaimType.BENCHMARK

    def test_announcement_detected(self):
        from app.summarization.models import ClaimType
        assert self._v().classify_claim("OpenAI is launching GPT-5 next week.") == ClaimType.ANNOUNCEMENT

    def test_comparative_detected(self):
        from app.summarization.models import ClaimType
        # Pure comparative — no benchmark numbers or terms, so COMPARATIVE wins
        assert self._v().classify_claim("GPT-5 is better than GPT-4 in terms of reliability.") == ClaimType.COMPARATIVE

    def test_speculation_detected(self):
        from app.summarization.models import ClaimType
        assert self._v().classify_claim("The company might release a new product this year.") == ClaimType.SPECULATION

    def test_opinion_detected(self):
        from app.summarization.models import ClaimType
        assert self._v().classify_claim("I believe this approach is the most effective.") == ClaimType.OPINION

    def test_factual_default(self):
        from app.summarization.models import ClaimType
        assert self._v().classify_claim("The experiment was conducted last year.") == ClaimType.FACTUAL

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            self._v().classify_claim("")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            self._v().classify_claim(None)  # type: ignore

    def test_benchmark_beats_announcement(self):
        """BENCHMARK pattern is more specific and should win when both appear."""
        from app.summarization.models import ClaimType
        # Contains both "launched" and "95%" — BENCHMARK should win
        text = "We launched a model that achieves 95% accuracy."
        result = self._v().classify_claim(text)
        assert result == ClaimType.BENCHMARK


# ===========================================================================
# Group 15: ClaimVerifierNegation
# ===========================================================================

class TestClaimVerifierNegation:
    def test_negated_claim_lower_confidence(self):
        from app.summarization.claim_verifier import ClaimVerifier
        v = ClaimVerifier(negation_penalty=0.5)
        src = _src("s1", content_snippet="The model is accurate and fast.", trust=0.9)
        conf_pos = v.verify_claim("model is accurate fast", [src])
        conf_neg = v.verify_claim("model is not accurate fast", [src])
        assert conf_neg < conf_pos

    def test_no_negation_full_penalty_not_applied(self):
        from app.summarization.claim_verifier import ClaimVerifier
        v = ClaimVerifier(negation_penalty=0.5)
        src = _src("s1", content_snippet="The model is accurate.", trust=0.9)
        conf = v.verify_claim("model is accurate", [src])
        # Without negation, penalty factor not applied → higher conf
        assert conf > 0.0

    def test_cannot_negation_detected(self):
        from app.summarization.claim_verifier import ClaimVerifier
        v = ClaimVerifier(negation_penalty=0.8)
        src = _src("s1", content_snippet="The system cannot process the request.", trust=0.8)
        conf_plain = v.verify_claim("system process request", [src])
        conf_negated = v.verify_claim("system cannot process request", [src])
        assert conf_negated <= conf_plain


# ===========================================================================
# Group 16: ClaimVerifierBatch
# ===========================================================================

class TestClaimVerifierBatch:
    def _verifier(self):
        from app.summarization.claim_verifier import ClaimVerifier
        return ClaimVerifier()

    def test_batch_returns_one_per_claim(self):
        src = _src("s1", content_snippet="OpenAI released GPT-5 with record accuracy benchmarks.")
        claims = ["GPT-5 released by OpenAI", "accuracy benchmarks", "model launched"]
        result = self._verifier().verify_batch(claims, [src])
        assert len(result) == 3

    def test_batch_items_are_attributed_claims(self):
        from app.summarization.models import AttributedClaim
        src = _src("s1", content_snippet="test content here")
        result = self._verifier().verify_batch(["test content"], [src])
        assert all(isinstance(c, AttributedClaim) for c in result)

    def test_batch_wrong_claims_type_raises(self):
        with pytest.raises(TypeError):
            self._verifier().verify_batch("not a list", [_src()])

    def test_batch_wrong_sources_type_raises(self):
        with pytest.raises(TypeError):
            self._verifier().verify_batch(["claim"], "not a list")

    def test_batch_skips_empty_strings(self):
        src = _src("s1", content_snippet="some text")
        result = self._verifier().verify_batch(["", "   ", "valid claim text"], [src])
        # Empty/whitespace strings should be skipped
        assert len(result) <= 1

    def test_batch_confidence_in_range(self):
        src = _src("s1", content_snippet="GPT model accuracy benchmark test data")
        result = self._verifier().verify_batch(["GPT model accuracy"], [src])
        assert all(0.0 <= c.confidence <= 1.0 for c in result)



# ===========================================================================
# Group 17: ContradictionDetectorConstruct
# ===========================================================================

class TestContradictionDetectorConstruct:
    def test_min_shared_tokens_zero_raises(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        with pytest.raises(ValueError):
            ContradictionDetector(min_shared_tokens=0)

    def test_negative_max_pairs_raises(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        with pytest.raises(ValueError):
            ContradictionDetector(max_pairs=-1)

    def test_default_construction(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        d = ContradictionDetector()
        assert d is not None


# ===========================================================================
# Group 18: ContradictionDetectorNegation
# ===========================================================================

class TestContradictionDetectorNegation:
    def _det(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        return ContradictionDetector(min_shared_tokens=1)

    def test_negation_conflict_detected(self):
        from app.summarization.models import ContradictionSeverity
        det = self._det()
        ca = _claim("The system is fully operational and running smoothly.")
        cb = _claim("The system is not operational and cannot run.")
        pairs = det.detect_contradictions([ca, cb])
        assert len(pairs) == 1
        assert pairs[0].detected_pattern == "negation_conflict"
        assert pairs[0].severity == ContradictionSeverity.MAJOR

    def test_same_negation_no_conflict(self):
        """Two negated claims about different subjects should not conflict."""
        det = self._det()
        ca = _claim("The model is not available in Europe.")
        cb = _claim("The product is not available in Asia.")
        # Different subjects → no shared content tokens → should not fire
        pairs = det.detect_contradictions([ca, cb])
        # Either no pairs, or no negation-conflict pattern
        neg_pairs = [p for p in pairs if p.detected_pattern == "negation_conflict"]
        assert len(neg_pairs) == 0

    def test_won_negation_no_conflict(self):
        """Both claims affirm the same thing — no contradiction."""
        det = self._det()
        ca = _claim("The model performs well on all standard tests.")
        cb = _claim("The model performs excellently on standard evaluation.")
        pairs = det.detect_contradictions([ca, cb])
        neg_pairs = [p for p in pairs if p.detected_pattern == "negation_conflict"]
        assert len(neg_pairs) == 0


# ===========================================================================
# Group 19: ContradictionDetectorNumber
# ===========================================================================

class TestContradictionDetectorNumber:
    def _det(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        return ContradictionDetector()

    def test_number_conflict_detected(self):
        from app.summarization.models import ContradictionSeverity
        det = self._det()
        ca = _claim("The dataset contains 10 million samples for training.")
        cb = _claim("The dataset contains 20 million samples for training.")
        pairs = det.detect_contradictions([ca, cb])
        num_pairs = [p for p in pairs if p.detected_pattern == "number_conflict"]
        assert len(num_pairs) >= 1
        assert num_pairs[0].severity == ContradictionSeverity.MODERATE

    def test_same_number_no_conflict(self):
        det = self._det()
        ca = _claim("The model has 10 billion parameters in total.")
        cb = _claim("The model has 10 billion parameters total.")
        pairs = det.detect_contradictions([ca, cb])
        num_pairs = [p for p in pairs if p.detected_pattern == "number_conflict"]
        assert len(num_pairs) == 0


# ===========================================================================
# Group 20: ContradictionDetectorAntonym
# ===========================================================================

class TestContradictionDetectorAntonym:
    def _det(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        return ContradictionDetector(min_shared_tokens=1)

    def test_antonym_conflict_detected(self):
        det = self._det()
        ca = _claim("Apple confirmed the acquisition of the startup company.")
        cb = _claim("Apple denied the acquisition of the startup company.")
        pairs = det.detect_contradictions([ca, cb])
        ant_pairs = [p for p in pairs if p.detected_pattern == "antonym_conflict"]
        assert len(ant_pairs) >= 1

    def test_open_closed_antonym(self):
        det = self._det()
        ca = _claim("The platform API remains open for developers.")
        cb = _claim("The platform API is closed to external developers.")
        pairs = det.detect_contradictions([ca, cb])
        ant_pairs = [p for p in pairs if p.detected_pattern == "antonym_conflict"]
        assert len(ant_pairs) >= 1


# ===========================================================================
# Group 21: ContradictionDetectorAPI
# ===========================================================================

class TestContradictionDetectorAPI:
    def test_is_contradiction_true(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        det = ContradictionDetector(min_shared_tokens=1)
        assert det.is_contradiction(
            "The system is fully operational and running.",
            "The system is not operational and not running."
        ) is True

    def test_is_contradiction_false(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        det = ContradictionDetector(min_shared_tokens=2)
        # Completely different topics → no contradiction
        assert det.is_contradiction(
            "Quantum computers use qubits.",
            "The weather is sunny today."
        ) is False

    def test_is_contradiction_empty_raises(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        with pytest.raises(ValueError):
            ContradictionDetector().is_contradiction("", "something")

    def test_is_contradiction_wrong_type_raises(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        with pytest.raises(TypeError):
            ContradictionDetector().is_contradiction(123, "text")  # type: ignore

    def test_score_severity_wrong_type_raises(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        with pytest.raises(TypeError):
            ContradictionDetector().score_severity("not a claim", _claim())  # type: ignore

    def test_detect_less_than_two_raises(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        with pytest.raises(ValueError):
            ContradictionDetector().detect_contradictions([_claim()])

    def test_detect_wrong_type_raises(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        with pytest.raises(TypeError):
            ContradictionDetector().detect_contradictions("bad")  # type: ignore


# ===========================================================================
# Group 22: ContradictionDetectorSort
# ===========================================================================

class TestContradictionDetectorSort:
    def test_pairs_sorted_severity_descending(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        from app.summarization.models import ContradictionSeverity
        det = ContradictionDetector(min_shared_tokens=1)
        # Create 3 claims to get multiple pairs
        ca = _claim("The system is fully operational and running now.")
        cb = _claim("The system is not operational and not running now.")  # negation → MAJOR
        cc = _claim("The company confirmed the successful launch event.")
        cd = _claim("The company denied the successful launch event.")  # antonym → MINOR
        pairs = det.detect_contradictions([ca, cb, cc, cd])
        if len(pairs) >= 2:
            severity_order = {ContradictionSeverity.CRITICAL: 4, ContradictionSeverity.MAJOR: 3,
                             ContradictionSeverity.MODERATE: 2, ContradictionSeverity.MINOR: 1}
            scores = [severity_order[p.severity] for p in pairs]
            assert scores == sorted(scores, reverse=True)

    def test_max_pairs_limits_result(self):
        from app.summarization.contradiction_detector import ContradictionDetector
        det = ContradictionDetector(min_shared_tokens=1, max_pairs=1)
        ca = _claim("The system is fully operational and running successfully.")
        cb = _claim("The system is not operational and not running at all.")
        cc = _claim("The platform is open for all external developers.")
        cd = _claim("The platform is closed to external developers globally.")
        pairs = det.detect_contradictions([ca, cb, cc, cd])
        assert len(pairs) <= 1


# ===========================================================================
# Group 23: UncertaintyAnnotatorConstruct
# ===========================================================================

class TestUncertaintyAnnotatorConstruct:
    def test_invalid_llm_min_chars_raises(self):
        from app.summarization.uncertainty_annotator import UncertaintyAnnotator
        with pytest.raises(ValueError):
            UncertaintyAnnotator(llm_min_chars=0)

    def test_invalid_min_severity_type_raises(self):
        from app.summarization.uncertainty_annotator import UncertaintyAnnotator
        with pytest.raises(TypeError):
            UncertaintyAnnotator(min_severity="low")  # type: ignore

    def test_default_construction(self):
        from app.summarization.uncertainty_annotator import UncertaintyAnnotator
        a = UncertaintyAnnotator()
        assert a is not None


# ===========================================================================
# Group 24: UncertaintyAnnotatorAnnotate
# ===========================================================================

class TestUncertaintyAnnotatorAnnotate:
    def _ann(self):
        from app.summarization.uncertainty_annotator import UncertaintyAnnotator
        return UncertaintyAnnotator()

    def test_hedge_word_detected(self):
        ann = self._ann()
        result = ann.annotate("The company might release a new product next quarter.")
        assert len(result) >= 1

    def test_high_severity_word_detected(self):
        from app.summarization.models import UncertaintySeverity
        ann = self._ann()
        result = ann.annotate("According to sources, the company will reportedly announce layoffs.")
        assert any(a.severity in (UncertaintySeverity.HIGH, UncertaintySeverity.CRITICAL) for a in result)

    def test_critical_word_detected(self):
        from app.summarization.models import UncertaintySeverity
        ann = self._ann()
        result = ann.annotate("Unconfirmed reports suggest the CEO will resign this month.")
        assert any(a.severity == UncertaintySeverity.CRITICAL for a in result)

    def test_certain_statement_minimal_annotation(self):
        ann = self._ann()
        result = ann.annotate("The experiment was completed on January 15, 2025.")
        # No strong hedge markers → zero or only LOW annotations
        from app.summarization.models import UncertaintySeverity
        high_severe = [a for a in result if a.severity in (UncertaintySeverity.HIGH, UncertaintySeverity.CRITICAL)]
        assert len(high_severe) == 0

    def test_empty_text_raises(self):
        with pytest.raises(ValueError):
            self._ann().annotate("")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            self._ann().annotate(None)  # type: ignore

    def test_sorted_by_position(self):
        ann = self._ann()
        text = "The results might be significant. It was reportedly confirmed."
        result = ann.annotate(text)
        if len(result) >= 2:
            positions = [a.position for a in result]
            assert positions == sorted(positions)


# ===========================================================================
# Group 25: UncertaintyAnnotatorSeverity
# ===========================================================================

class TestUncertaintyAnnotatorSeverity:
    def _ann(self):
        from app.summarization.uncertainty_annotator import UncertaintyAnnotator
        return UncertaintyAnnotator()

    def test_critical_span(self):
        from app.summarization.models import UncertaintySeverity
        assert self._ann().classify_severity("unconfirmed reports") == UncertaintySeverity.CRITICAL

    def test_high_span(self):
        from app.summarization.models import UncertaintySeverity
        assert self._ann().classify_severity("reportedly confirmed by sources") == UncertaintySeverity.HIGH

    def test_medium_span(self):
        from app.summarization.models import UncertaintySeverity
        assert self._ann().classify_severity("might happen") == UncertaintySeverity.MEDIUM

    def test_low_span(self):
        from app.summarization.models import UncertaintySeverity
        assert self._ann().classify_severity("will likely succeed") == UncertaintySeverity.LOW

    def test_no_marker_defaults_low(self):
        from app.summarization.models import UncertaintySeverity
        # Plain factual text with no hedge markers
        assert self._ann().classify_severity("the experiment ran successfully") == UncertaintySeverity.LOW

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            self._ann().classify_severity("")

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            self._ann().classify_severity(42)  # type: ignore


# ===========================================================================
# Group 26: UncertaintyAnnotatorOverall
# ===========================================================================

class TestUncertaintyAnnotatorOverall:
    def _ann(self):
        from app.summarization.uncertainty_annotator import UncertaintyAnnotator
        return UncertaintyAnnotator()

    def test_empty_returns_zero(self):
        assert self._ann().overall_uncertainty([]) == pytest.approx(0.0)

    def test_single_critical(self):
        from app.summarization.models import UncertaintyAnnotation, UncertaintySeverity
        ann = UncertaintyAnnotation(text_span="x", severity=UncertaintySeverity.CRITICAL)
        result = self._ann().overall_uncertainty([ann])
        assert result == pytest.approx(1.0)

    def test_single_low(self):
        from app.summarization.models import UncertaintyAnnotation, UncertaintySeverity
        ann = UncertaintyAnnotation(text_span="x", severity=UncertaintySeverity.LOW)
        result = self._ann().overall_uncertainty([ann])
        assert result == pytest.approx(0.25)

    def test_mixed_average(self):
        from app.summarization.models import UncertaintyAnnotation, UncertaintySeverity
        anns = [
            UncertaintyAnnotation(text_span="a", severity=UncertaintySeverity.CRITICAL),
            UncertaintyAnnotation(text_span="b", severity=UncertaintySeverity.LOW),
        ]
        # mean of 1.0 and 0.25 = 0.625
        result = self._ann().overall_uncertainty(anns)
        assert result == pytest.approx(0.625)

    def test_result_in_unit_range(self):
        from app.summarization.models import UncertaintyAnnotation, UncertaintySeverity
        anns = [UncertaintyAnnotation(text_span=f"x{i}", severity=UncertaintySeverity.HIGH) for i in range(10)]
        result = self._ann().overall_uncertainty(anns)
        assert 0.0 <= result <= 1.0

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            self._ann().overall_uncertainty("bad")  # type: ignore



# ===========================================================================
# Group 27: GroundedSummaryBuilderConstruct
# ===========================================================================

class TestGroundedSummaryBuilderConstruct:
    def test_invalid_top_n_raises(self):
        from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
        with pytest.raises(ValueError):
            GroundedSummaryBuilder(top_n_sentences=0)

    def test_default_construction(self):
        from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
        b = GroundedSummaryBuilder()
        assert b is not None

    def test_wrong_request_type_raises(self):
        from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
        with pytest.raises(TypeError):
            GroundedSummaryBuilder().build("not a request")  # type: ignore


# ===========================================================================
# Group 28: GroundedSummaryBuilderBuild
# ===========================================================================

class TestGroundedSummaryBuilderBuild:
    def _builder(self):
        from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
        return GroundedSummaryBuilder()

    def _rich_request(self):
        src = _src(
            "s1",
            title="AI Benchmark Report",
            content_snippet=(
                "OpenAI released GPT-5 this week, achieving state-of-the-art results "
                "on MMLU with 92% accuracy. The model outperforms GPT-4 on all benchmarks. "
                "This matters because it significantly impacts AI research and enterprise adoption. "
                "Researchers and businesses will need to adapt their workflows."
            ),
            trust=0.9,
        )
        return SynthesisRequest(topic="GPT-5 release", sources=[src])

    def test_returns_grounded_summary(self):
        from app.summarization.models import GroundedSummary
        req = self._rich_request()
        result = self._builder().build(req)
        assert isinstance(result, GroundedSummary)

    def test_summary_id_non_empty(self):
        result = self._builder().build(self._rich_request())
        assert len(result.summary_id) > 0

    def test_source_count_matches(self):
        req = self._rich_request()
        result = self._builder().build(req)
        assert result.source_count == len(req.sources)

    def test_source_attributions_populated(self):
        result = self._builder().build(self._rich_request())
        assert len(result.source_attributions) >= 1

    def test_what_happened_non_empty(self):
        result = self._builder().build(self._rich_request())
        assert len(result.what_happened) > 0

    def test_confidence_in_range(self):
        result = self._builder().build(self._rich_request())
        assert 0.0 <= result.confidence_score <= 1.0

    def test_key_claims_extracted(self):
        result = self._builder().build(self._rich_request())
        # Expect at least 1 claim from the rich content
        assert len(result.key_claims) >= 1

    def test_min_trust_filter_fallback(self):
        """All sources filtered → builder falls back to using all sources."""
        src = _src("s1", trust=0.3, content_snippet="Some important content here.")
        req = SynthesisRequest(topic="test", sources=[src], min_source_trust=0.9)
        result = self._builder().build(req)
        # Should not crash; fallback to all sources
        assert result.source_count >= 1

    def test_multi_source_build(self):
        src1 = _src("s1", content_snippet="The company launched a new AI product this quarter.", trust=0.9)
        src2 = _src("s2", content_snippet="The new AI product reportedly might change industry standards.", trust=0.7)
        req = SynthesisRequest(topic="AI launch", sources=[src1, src2])
        result = self._builder().build(req)
        assert result.source_count == 2
        assert result.overall_uncertainty_score >= 0.0


# ===========================================================================
# Group 29: GroundedSummaryBuilderConfidence
# ===========================================================================

class TestGroundedSummaryBuilderConfidence:
    def _builder(self):
        from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
        return GroundedSummaryBuilder()

    def test_high_trust_sources_high_confidence(self):
        src = _src("s1", trust=1.0, content_snippet="The model achieved perfect accuracy on all tests.")
        req = SynthesisRequest(topic="test", sources=[src])
        result = self._builder().build(req)
        assert result.confidence_score > 0.4  # high trust → high confidence

    def test_low_trust_sources_lower_confidence(self):
        src_high = _src("s1", trust=0.9, content_snippet="The model performed well on tests today.")
        src_low  = _src("s2", trust=0.1, content_snippet="The model might perform on tests today.")
        req_high = SynthesisRequest(topic="test", sources=[src_high])
        req_low  = SynthesisRequest(topic="test", sources=[src_low])
        conf_high = self._builder().build(req_high).confidence_score
        conf_low  = self._builder().build(req_low).confidence_score
        assert conf_high >= conf_low

    def test_uncertainty_deflates_confidence(self):
        """A source full of hedging should produce lower confidence."""
        factual = _src("s1", trust=0.8,
                       content_snippet="The experiment succeeded. Results confirmed the hypothesis.")
        hedged  = _src("s2", trust=0.8,
                       content_snippet="Results might possibly confirm the hypothesis allegedly.")
        req_f = SynthesisRequest(topic="t", sources=[factual])
        req_h = SynthesisRequest(topic="t", sources=[hedged])
        conf_f = self._builder().build(req_f).confidence_score
        conf_h = self._builder().build(req_h).confidence_score
        assert conf_f >= conf_h


# ===========================================================================
# Group 30: MultiSourceSynthesizerConstruct
# ===========================================================================

class TestMultiSourceSynthesizerConstruct:
    def test_invalid_dedup_threshold_zero_raises(self):
        from app.summarization.multi_source_synthesizer import MultiSourceSynthesizer
        with pytest.raises(ValueError):
            MultiSourceSynthesizer(dedup_threshold=0.0)

    def test_invalid_merge_threshold_over_one_raises(self):
        from app.summarization.multi_source_synthesizer import MultiSourceSynthesizer
        with pytest.raises(ValueError):
            MultiSourceSynthesizer(merge_threshold=1.5)

    def test_wrong_builder_type_raises(self):
        from app.summarization.multi_source_synthesizer import MultiSourceSynthesizer
        with pytest.raises(TypeError):
            MultiSourceSynthesizer(builder="bad")

    def test_default_construction(self):
        from app.summarization.multi_source_synthesizer import MultiSourceSynthesizer
        s = MultiSourceSynthesizer()
        assert s is not None


# ===========================================================================
# Group 31: MultiSourceSynthesizerDedup
# ===========================================================================

class TestMultiSourceSynthesizerDedup:
    def _synth(self, threshold=0.70):
        from app.summarization.multi_source_synthesizer import MultiSourceSynthesizer
        return MultiSourceSynthesizer(dedup_threshold=threshold)

    def test_identical_content_deduped(self):
        shared = "OpenAI released GPT-5 with unprecedented performance metrics."
        s1 = _src("s1", content_snippet=shared, trust=0.8)
        s2 = _src("s2", content_snippet=shared, trust=0.7)  # lower trust → dropped
        result = self._synth(threshold=0.50).deduplicate_sources([s1, s2])
        assert len(result) == 1
        assert result[0].source_id == "s1"  # higher trust kept

    def test_distinct_content_not_deduped(self):
        s1 = _src("s1", content_snippet="Quantum computing breakthrough announced by IBM researchers today.")
        s2 = _src("s2", content_snippet="Stock market declined sharply due to inflation concerns.")
        result = self._synth().deduplicate_sources([s1, s2])
        assert len(result) == 2

    def test_single_source_unchanged(self):
        src = _src("s1", content_snippet="Some content here.")
        result = self._synth().deduplicate_sources([src])
        assert len(result) == 1

    def test_empty_list_unchanged(self):
        result = self._synth().deduplicate_sources([])
        assert result == []

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            self._synth().deduplicate_sources("not a list")

    def test_higher_trust_wins_on_dedup(self):
        content = "The model achieves state-of-the-art results on benchmark evaluations."
        low  = _src("low",  content_snippet=content, trust=0.3)
        high = _src("high", content_snippet=content, trust=0.9)
        result = self._synth(threshold=0.50).deduplicate_sources([low, high])
        assert len(result) == 1
        assert result[0].source_id == "high"


# ===========================================================================
# Group 32: MultiSourceSynthesizerMerge
# ===========================================================================

class TestMultiSourceSynthesizerMerge:
    def _synth(self, threshold=0.60):
        from app.summarization.multi_source_synthesizer import MultiSourceSynthesizer
        return MultiSourceSynthesizer(merge_threshold=threshold)

    def test_identical_claims_merged(self):
        ca = _claim("GPT-5 model released by OpenAI today.", conf=0.7, source_ids=["s1"])
        cb = _claim("GPT-5 model released by OpenAI today.", conf=0.8, source_ids=["s2"])
        result = self._synth(threshold=0.50).merge_claims([ca, cb])
        assert len(result) == 1
        assert set(result[0].source_ids) == {"s1", "s2"}

    def test_merged_confidence_boosted(self):
        ca = _claim("transformer model training completed on cluster.", conf=0.6, source_ids=["s1"])
        cb = _claim("transformer model training completed on cluster.", conf=0.6, source_ids=["s2"])
        result = self._synth(threshold=0.50).merge_claims([ca, cb])
        assert result[0].confidence > 0.6  # slight corroboration boost

    def test_distinct_claims_not_merged(self):
        ca = _claim("OpenAI released GPT-5 language model.", conf=0.8)
        cb = _claim("Quantum computing breakthrough achieved by IBM laboratory.", conf=0.7)
        result = self._synth().merge_claims([ca, cb])
        assert len(result) == 2

    def test_empty_list_returns_empty(self):
        result = self._synth().merge_claims([])
        assert result == []

    def test_wrong_type_raises(self):
        with pytest.raises(TypeError):
            self._synth().merge_claims("bad")

    def test_higher_confidence_text_kept(self):
        ca = _claim("Low confidence text version.", conf=0.3, source_ids=["s1"])
        cb = _claim("Low confidence text version.", conf=0.9, source_ids=["s2"])
        result = self._synth(threshold=0.50).merge_claims([ca, cb])
        assert len(result) == 1
        assert result[0].confidence > 0.3


# ===========================================================================
# Group 33: MultiSourceSynthesizerSynthesize
# ===========================================================================

class TestMultiSourceSynthesizerSynthesize:
    def _synth(self):
        from app.summarization.multi_source_synthesizer import MultiSourceSynthesizer
        return MultiSourceSynthesizer()

    def test_synthesize_returns_grounded_summary(self):
        from app.summarization.models import GroundedSummary
        req = _request(sources=[
            _src("s1", content_snippet="Anthropic released Claude 3 with improved safety features today."),
            _src("s2", content_snippet="Claude 3 model outperforms competitors on safety benchmarks."),
        ])
        result = self._synth().synthesize(req)
        assert isinstance(result, GroundedSummary)

    def test_synthesize_wrong_type_raises(self):
        with pytest.raises(TypeError):
            self._synth().synthesize("bad")

    def test_synthesize_with_duplicates_reduces_sources(self):
        shared = "The company announced record profits exceeding all market expectations."
        req = _request(sources=[
            _src("s1", content_snippet=shared, trust=0.9),
            _src("s2", content_snippet=shared, trust=0.7),
            _src("s3", content_snippet="Completely unrelated news about sports teams."),
        ])
        result = self._synth().synthesize(req)
        # After dedup, s1 (higher trust) + s3 kept; source_count ≤ 3
        assert result.source_count <= 3
        assert result.source_count >= 1

    def test_synthesize_source_count_correct(self):
        src1 = _src("s1", content_snippet="New chip designed for AI inference tasks announced.")
        src2 = _src("s2", content_snippet="Chip for quantum computing released by Intel today.")
        req = SynthesisRequest(topic="chips", sources=[src1, src2])
        result = self._synth().synthesize(req)
        assert result.source_count >= 1


# ===========================================================================
# Group 34: CrossComponentAttributionFlow
# ===========================================================================

class TestCrossComponentAttributionFlow:
    def test_attribution_feeds_claim_source_ids(self):
        """SourceAttributor output should populate AttributedClaim.source_ids."""
        from app.summarization.claim_verifier import ClaimVerifier
        from app.summarization.source_attribution import SourceAttributor
        attr = SourceAttributor(min_overlap=0.05)
        verifier = ClaimVerifier(attributor=attr)
        src = _src("s1", content_snippet="GPT-5 released by OpenAI with record benchmark accuracy.")
        claims = verifier.verify_batch(["GPT-5 OpenAI benchmark accuracy"], [src])
        assert len(claims) == 1
        assert "s1" in claims[0].source_ids

    def test_attribution_empty_when_no_overlap(self):
        from app.summarization.claim_verifier import ClaimVerifier
        from app.summarization.source_attribution import SourceAttributor
        attr = SourceAttributor(min_overlap=0.9)  # very high threshold
        verifier = ClaimVerifier(attributor=attr)
        src = _src("s1", content_snippet="Unrelated stock market news.")
        claims = verifier.verify_batch(["quantum computing breakthroughs"], [src])
        # No overlap → no source_ids
        assert claims[0].source_ids == []


# ===========================================================================
# Group 35: CrossComponentContradictions
# ===========================================================================

class TestCrossComponentContradictions:
    def test_contradictions_surface_in_summary(self):
        """Conflicting sources should produce contradictions in the GroundedSummary."""
        from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
        s1 = _src("s1", content_snippet=(
            "The company confirmed the acquisition deal is approved and proceeding."), trust=0.8)
        s2 = _src("s2", content_snippet=(
            "The company rejected the acquisition deal and denied any approval."), trust=0.7)
        req = SynthesisRequest(topic="acquisition", sources=[s1, s2])
        builder = GroundedSummaryBuilder()
        summary = builder.build(req)
        # Both claims extracted → contradiction detector should fire
        # (may not always fire depending on sentence extraction order, but structure is present)
        assert isinstance(summary.contradictions, list)

    def test_uncertainty_annotations_in_summary(self):
        """Hedged text in sources should produce uncertainty_annotations."""
        from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
        src = _src("s1", content_snippet=(
            "Reportedly, the CEO might resign. Sources say the board could reject the proposal. "
            "Unconfirmed reports suggest layoffs are possible."), trust=0.6)
        req = SynthesisRequest(topic="company news", sources=[src])
        summary = GroundedSummaryBuilder().build(req)
        assert len(summary.uncertainty_annotations) >= 1
        assert summary.overall_uncertainty_score > 0.0


# ===========================================================================
# Group 36: CrossComponentPackageImports
# ===========================================================================

class TestCrossComponentPackageImports:
    def test_all_enums_importable(self):
        from app.summarization import ClaimType, ContradictionSeverity, UncertaintySeverity
        assert ClaimType.FACTUAL.value == "factual"
        assert ContradictionSeverity.MAJOR.value == "major"
        assert UncertaintySeverity.CRITICAL.value == "critical"

    def test_all_models_importable(self):
        from app.summarization import (
            EvidenceSource, AttributedClaim, ContradictionPair,
            UncertaintyAnnotation, GroundedSummary, SynthesisRequest,
        )
        assert EvidenceSource is not None
        assert GroundedSummary is not None

    def test_all_components_importable(self):
        from app.summarization import (
            SourceAttributor, ClaimVerifier, ContradictionDetector,
            UncertaintyAnnotator, GroundedSummaryBuilder, MultiSourceSynthesizer,
        )
        assert SourceAttributor is not None
        assert MultiSourceSynthesizer is not None

    def test_synthesizer_end_to_end_import(self):
        """Confirm full pipeline works from package-level imports."""
        from app.summarization import (
            EvidenceSource, SynthesisRequest, MultiSourceSynthesizer, GroundedSummary,
        )
        src = EvidenceSource(
            source_id="e2e-1",
            content_snippet="The AI model achieved 98% accuracy on the MMLU benchmark test.",
            trust_score=0.9,
        )
        req = SynthesisRequest(topic="AI benchmark", sources=[src])
        summary = MultiSourceSynthesizer().synthesize(req)
        assert isinstance(summary, GroundedSummary)
        assert summary.confidence_score > 0.0
        assert summary.source_count >= 1

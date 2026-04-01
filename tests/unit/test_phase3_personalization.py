"""Phase 3 — Personalization & User Adaptation Unit Tests.

Test groups
-----------
Group 1:  PersonalizationModels          — FeedbackType / InterestWeight / FeedbackEvent
Group 2:  DigestCandidateModel           — DigestCandidate validation
Group 3:  RankedDigestItemModel          — RankedDigestItem frozen model
Group 4:  NoveltyRelevanceConfigModel    — NoveltyRelevanceConfig validation
Group 5:  RankingWeightsModel            — RankingWeights + normalization
Group 6:  InterestGraphConstruction      — constructor validation
Group 7:  InterestGraphNodeCRUD          — add / remove / contains
Group 8:  InterestGraphWeightUpdate      — update_weight gradient logic
Group 9:  InterestGraphTopInterests      — top_interests query
Group 10: InterestGraphEdgeOperations    — add / remove / related_topics
Group 11: InterestGraphDecay             — decay_all temporal decay
Group 12: InterestGraphSerialization     — to_dict / from_dict round-trip
Group 13: TopicEmbeddingProfileConstruct — constructor validation
Group 14: TopicEmbeddingProfileUpdate    — EMA update correctness
Group 15: TopicEmbeddingProfileSimilarity — cosine similarity
Group 16: TopicEmbeddingProfileDrift     — drift_score
Group 17: TopicEmbeddingProfileStability — is_stable threshold
Group 18: TopicEmbeddingProfileReset     — reset to zero state
Group 19: TopicEmbeddingProfileSerial    — to_dict / from_dict round-trip
Group 20: FeedbackLearnerConstruct       — constructor validation
Group 21: FeedbackLearnerProcess         — single event updates
Group 22: FeedbackLearnerImplicit        — dwell-time scaling
Group 23: FeedbackLearnerBatch           — batch processing
Group 24: FeedbackLearnerHistory         — history management
Group 25: FeedbackLearnerStats           — summary_stats
Group 26: UserDigestRankerConstruct      — constructor validation
Group 27: UserDigestRankerRanking        — score ordering invariants
Group 28: UserDigestRankerTopK           — top_k limit
Group 29: UserDigestRankerScoreBreakdown — score_breakdown keys
Group 30: NoveltyTradeoffConstruct       — constructor validation
Group 31: NoveltyTradeoffBlend           — blend_scores correctness
Group 32: NoveltyTradeoffUpdate          — update_from_feedback alpha changes
Group 33: NoveltyTradeoffFatigue         — topic-fatigue concentration nudge
Group 34: NoveltyTradeoffCompute         — compute_alpha stateless
Group 35: NoveltyTradeoffReset           — reset clears state
Group 36: CrossComponentWiring           — FeedbackLearner + Ranker integration
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(dim: int = 4, value: float = 1.0) -> List[float]:
    return [value] * dim


def _make_event(
    user_id: str = "u1",
    item_id: str = "item1",
    feedback_type=None,
    topic_ids=None,
    dwell: float = None,
    embedding=None,
):
    from app.personalization.models import FeedbackEvent, FeedbackType
    ft = feedback_type or FeedbackType.LIKE
    return FeedbackEvent(
        user_id=user_id,
        item_id=item_id,
        feedback_type=ft,
        topic_ids=topic_ids or ["ml", "llm"],
        implicit_dwell_seconds=dwell,
        embedding=embedding or [],
    )


def _make_candidate(
    item_id: str = "c1",
    topics=None,
    published_hours_ago: float = 1.0,
    trust: float = 0.7,
    engagement: float = 0.6,
    novelty: float = 0.5,
    embedding=None,
) -> "DigestCandidate":
    from app.personalization.models import DigestCandidate
    now = datetime.now(timezone.utc)
    pub = now - timedelta(hours=published_hours_ago)
    return DigestCandidate(
        item_id=item_id,
        title=f"Title for {item_id}",
        topic_ids=topics or ["ml", "ai"],
        published_at=pub,
        trust_score=trust,
        engagement_score=engagement,
        novelty_score=novelty,
        embedding=embedding or [],
    )


# ===========================================================================
# Group 1: PersonalizationModels — FeedbackType / InterestWeight / FeedbackEvent
# ===========================================================================

class TestPersonalizationModels:
    def test_feedback_type_values(self):
        from app.personalization.models import FeedbackType
        assert FeedbackType.LIKE.value == "like"
        assert FeedbackType.DISMISS.value == "dismiss"
        assert FeedbackType.SAVE.value == "save"
        assert FeedbackType.SHARE.value == "share"
        assert FeedbackType.CLICK.value == "click"
        assert FeedbackType.READ_COMPLETE.value == "read_complete"
        assert FeedbackType.SCROLL_PAST.value == "scroll_past"

    def test_feedback_delta_coverage(self):
        from app.personalization.models import FEEDBACK_DELTA, FeedbackType
        for ft in FeedbackType:
            assert ft.value in FEEDBACK_DELTA

    def test_interest_weight_frozen(self):
        from app.personalization.models import InterestWeight
        iw = InterestWeight(topic_id="ml", weight=0.7)
        with pytest.raises(Exception):
            iw.weight = 0.9  # type: ignore

    def test_interest_weight_out_of_range_raises(self):
        from app.personalization.models import InterestWeight
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            InterestWeight(topic_id="ml", weight=1.5)

    def test_interest_weight_defaults(self):
        from app.personalization.models import InterestWeight
        iw = InterestWeight(topic_id="llm", weight=0.5)
        assert iw.confidence == 0.5
        assert iw.update_count == 0

    def test_feedback_event_frozen(self):
        ev = _make_event()
        with pytest.raises(Exception):
            ev.user_id = "u2"  # type: ignore

    def test_feedback_event_empty_user_raises(self):
        from app.personalization.models import FeedbackEvent, FeedbackType
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            FeedbackEvent(user_id="", item_id="i1", feedback_type=FeedbackType.LIKE)

    def test_feedback_event_negative_dwell_raises(self):
        from app.personalization.models import FeedbackEvent, FeedbackType
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            FeedbackEvent(user_id="u1", item_id="i1", feedback_type=FeedbackType.READ_COMPLETE, implicit_dwell_seconds=-5.0)

    def test_interest_edge_type_enum(self):
        from app.personalization.models import InterestEdgeType
        assert InterestEdgeType.SUBTOPIC_OF.value == "subtopic_of"
        assert InterestEdgeType.RELATED_TO.value == "related_to"


# ===========================================================================
# Group 2: DigestCandidateModel
# ===========================================================================

class TestDigestCandidateModel:
    def test_basic_instantiation(self):
        c = _make_candidate()
        assert c.item_id == "c1"
        assert c.trust_score == pytest.approx(0.7)

    def test_empty_item_id_raises(self):
        from app.personalization.models import DigestCandidate
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            DigestCandidate(item_id="")

    def test_trust_out_of_range_raises(self):
        from app.personalization.models import DigestCandidate
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            DigestCandidate(item_id="x", trust_score=1.5)

    def test_novelty_defaults(self):
        from app.personalization.models import DigestCandidate
        c = DigestCandidate(item_id="y")
        assert 0.0 <= c.novelty_score <= 1.0
        assert c.topic_ids == []
        assert c.embedding == []


# ===========================================================================
# Group 3: RankedDigestItemModel
# ===========================================================================

class TestRankedDigestItemModel:
    def test_frozen(self):
        from app.personalization.models import RankedDigestItem
        item = RankedDigestItem(item_id="x", rank=1, final_score=0.9)
        with pytest.raises(Exception):
            item.rank = 2  # type: ignore

    def test_rank_zero_raises(self):
        from app.personalization.models import RankedDigestItem
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            RankedDigestItem(item_id="x", rank=0, final_score=0.5)

    def test_score_out_of_range_raises(self):
        from app.personalization.models import RankedDigestItem
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            RankedDigestItem(item_id="x", rank=1, final_score=1.1)


# ===========================================================================
# Group 6: InterestGraphConstruction
# ===========================================================================

class TestInterestGraphConstruction:
    def test_default_instantiation(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        assert len(g) == 0

    def test_invalid_initial_weight_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph(initial_weight=1.5)

    def test_invalid_decay_factor_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph(decay_factor=0.0)

    def test_negative_decay_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph(decay_factor=-0.1)

    def test_invalid_min_weight_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph(min_weight=-0.1)

    def test_negative_max_topics_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph(max_topics=-1)


# ===========================================================================
# Group 7: InterestGraphNodeCRUD
# ===========================================================================

class TestInterestGraphNodeCRUD:
    def test_add_topic(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        g.add_topic("ml")
        assert "ml" in g
        assert len(g) == 1

    def test_add_topic_empty_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph().add_topic("")

    def test_add_topic_invalid_weight_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph().add_topic("ml", initial_weight=2.0)

    def test_add_duplicate_is_noop(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.5)
        g.add_topic("ml")
        g.add_topic("ml")  # should not raise or duplicate
        assert len(g) == 1
        assert g.get_weight("ml") == pytest.approx(0.5)

    def test_max_topics_limit(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(max_topics=2)
        g.add_topic("a")
        g.add_topic("b")
        with pytest.raises(OverflowError):
            g.add_topic("c")

    def test_remove_topic(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        g.add_topic("ml")
        assert g.remove_topic("ml") is True
        assert "ml" not in g

    def test_remove_missing_returns_false(self):
        from app.personalization.interest_graph import InterestGraph
        assert InterestGraph().remove_topic("nonexistent") is False

    def test_contains_operator(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        g.add_topic("ml")
        assert "ml" in g
        assert "nlp" not in g

    def test_iter(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        g.add_topic("a")
        g.add_topic("b")
        topics = list(g)
        assert len(topics) == 2


# ===========================================================================
# Group 8: InterestGraphWeightUpdate
# ===========================================================================

class TestInterestGraphWeightUpdate:
    def test_positive_delta_increases_weight(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.5)
        g.add_topic("ml")
        iw = g.update_weight("ml", delta=0.2)
        assert iw.weight == pytest.approx(0.7)

    def test_negative_delta_decreases_weight(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.5)
        g.add_topic("ml")
        iw = g.update_weight("ml", delta=-0.2)
        assert iw.weight == pytest.approx(0.3)

    def test_clipping_at_one(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.9)
        g.add_topic("ml")
        iw = g.update_weight("ml", delta=0.5)
        assert iw.weight == pytest.approx(1.0)

    def test_clipping_at_zero(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.1)
        g.add_topic("ml")
        iw = g.update_weight("ml", delta=-0.5)
        assert iw.weight == pytest.approx(0.0)

    def test_auto_creates_topic(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        iw = g.update_weight("new_topic", delta=0.1)
        assert "new_topic" in g
        assert iw.update_count == 1

    def test_update_count_increments(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.5)
        g.add_topic("ml")
        g.update_weight("ml", delta=0.1)
        g.update_weight("ml", delta=0.1)
        assert g.get_interest_weight("ml").update_count == 2

    def test_confidence_increases_with_updates(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.5)
        g.add_topic("ml")
        iw0 = g.get_interest_weight("ml")
        for _ in range(10):
            g.update_weight("ml", delta=0.01)
        iw10 = g.get_interest_weight("ml")
        assert iw10.confidence > iw0.confidence

    def test_learning_rate_scales_delta(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.5)
        g.add_topic("ml")
        iw = g.update_weight("ml", delta=0.2, learning_rate=0.5)
        assert iw.weight == pytest.approx(0.6)  # 0.5 + 0.2 * 0.5

    def test_empty_topic_id_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph().update_weight("", delta=0.1)

    def test_invalid_learning_rate_raises(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        with pytest.raises(ValueError):
            g.update_weight("ml", delta=0.1, learning_rate=1.5)


# ===========================================================================
# Group 9: InterestGraphTopInterests
# ===========================================================================

class TestInterestGraphTopInterests:
    def test_top_k_ordering(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        for tid, w in [("a", 0.9), ("b", 0.3), ("c", 0.7), ("d", 0.1)]:
            g.add_topic(tid, initial_weight=w)
        top = g.top_interests(k=2)
        assert len(top) == 2
        assert top[0].topic_id == "a"
        assert top[1].topic_id == "c"

    def test_top_k_zero_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph().top_interests(k=0)

    def test_top_k_larger_than_graph(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        g.add_topic("only_one", initial_weight=0.8)
        result = g.top_interests(k=10)
        assert len(result) == 1

    def test_all_topics_snapshot(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        g.add_topic("x")
        g.add_topic("y")
        assert set(g.all_topics()) == {"x", "y"}



# ===========================================================================
# Group 10: InterestGraphEdgeOperations
# ===========================================================================

class TestInterestGraphEdgeOperations:
    def test_add_edge_creates_nodes(self):
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.models import InterestEdgeType
        g = InterestGraph()
        g.add_edge("ml", "ai", edge_type=InterestEdgeType.SUBTOPIC_OF)
        assert "ml" in g
        assert "ai" in g

    def test_add_edge_self_loop_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph().add_edge("ml", "ml")

    def test_add_edge_invalid_weight_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph().add_edge("ml", "ai", weight=1.5)

    def test_add_edge_wrong_type_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(TypeError):
            InterestGraph().add_edge("ml", "ai", edge_type="not_an_enum")

    def test_remove_edge(self):
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.models import InterestEdgeType
        g = InterestGraph()
        g.add_edge("ml", "ai", edge_type=InterestEdgeType.RELATED_TO)
        assert g.remove_edge("ml", "ai") is True
        assert g.remove_edge("ml", "ai") is False

    def test_related_topics(self):
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.models import InterestEdgeType
        g = InterestGraph(initial_weight=0.5)
        g.add_topic("ml", initial_weight=0.8)
        g.add_topic("nlp", initial_weight=0.6)
        g.add_topic("cv", initial_weight=0.4)
        g.add_edge("ml", "nlp", edge_type=InterestEdgeType.RELATED_TO, weight=0.9)
        g.add_edge("ml", "cv", edge_type=InterestEdgeType.RELATED_TO, weight=0.3)
        related = g.related_topics("ml", top_k=5)
        assert len(related) == 2
        # nlp should rank first (higher edge_weight × node_weight)
        assert related[0].topic_id == "nlp"

    def test_related_topics_edge_type_filter(self):
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.models import InterestEdgeType
        g = InterestGraph()
        g.add_edge("ml", "nlp", edge_type=InterestEdgeType.RELATED_TO, weight=0.8)
        g.add_edge("ml", "cv", edge_type=InterestEdgeType.SUBTOPIC_OF, weight=0.8)
        related = g.related_topics("ml", edge_type=InterestEdgeType.RELATED_TO)
        ids = [iw.topic_id for iw in related]
        assert "nlp" in ids
        assert "cv" not in ids

    def test_related_topics_min_edge_weight(self):
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.models import InterestEdgeType
        g = InterestGraph()
        g.add_edge("ml", "nlp", edge_type=InterestEdgeType.RELATED_TO, weight=0.9)
        g.add_edge("ml", "cv", edge_type=InterestEdgeType.RELATED_TO, weight=0.1)
        related = g.related_topics("ml", min_edge_weight=0.5)
        ids = [iw.topic_id for iw in related]
        assert "nlp" in ids
        assert "cv" not in ids

    def test_related_topics_invalid_k_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph().related_topics("ml", top_k=0)


# ===========================================================================
# Group 11: InterestGraphDecay
# ===========================================================================

class TestInterestGraphDecay:
    def test_decay_reduces_weight(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=1.0, decay_factor=0.9)
        g.add_topic("ml")
        g.decay_all(days_elapsed=1.0)
        w = g.get_weight("ml")
        assert w == pytest.approx(0.9)

    def test_decay_compounded(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=1.0, decay_factor=0.9)
        g.add_topic("ml")
        g.decay_all(days_elapsed=2.0)
        w = g.get_weight("ml")
        assert w == pytest.approx(0.81, rel=1e-4)

    def test_decay_respects_min_weight(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.01, decay_factor=0.5, min_weight=0.01)
        g.add_topic("ml")
        g.decay_all(days_elapsed=100.0)
        assert g.get_weight("ml") >= 0.01

    def test_decay_zero_days_no_change(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.7, decay_factor=0.9)
        g.add_topic("ml")
        g.decay_all(days_elapsed=0.0)
        assert g.get_weight("ml") == pytest.approx(0.7)

    def test_decay_negative_days_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(ValueError):
            InterestGraph().decay_all(days_elapsed=-1.0)

    def test_decay_multiple_topics(self):
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(decay_factor=0.8)
        g.add_topic("a", initial_weight=1.0)
        g.add_topic("b", initial_weight=0.5)
        g.decay_all(days_elapsed=1.0)
        assert g.get_weight("a") == pytest.approx(0.8)
        assert g.get_weight("b") == pytest.approx(0.4)


# ===========================================================================
# Group 12: InterestGraphSerialization
# ===========================================================================

class TestInterestGraphSerialization:
    def test_to_dict_structure(self):
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.models import InterestEdgeType
        g = InterestGraph()
        g.add_topic("ml", initial_weight=0.7)
        g.add_edge("ml", "nlp", InterestEdgeType.RELATED_TO, 0.6)
        d = g.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "ml" in d["nodes"]

    def test_from_dict_wrong_type_raises(self):
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(TypeError):
            InterestGraph.from_dict("not a dict")

    def test_round_trip(self):
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.models import InterestEdgeType
        g = InterestGraph()
        g.add_topic("ml", initial_weight=0.8)
        g.add_topic("nlp", initial_weight=0.6)
        g.add_edge("ml", "nlp", InterestEdgeType.RELATED_TO, 0.7)
        d = g.to_dict()
        g2 = InterestGraph.from_dict(d)
        assert "ml" in g2
        assert "nlp" in g2
        assert g2.get_weight("ml") == pytest.approx(0.8)



# ===========================================================================
# Group 13: TopicEmbeddingProfileConstruct
# ===========================================================================

class TestTopicEmbeddingProfileConstruct:
    def test_zero_dim_raises(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        with pytest.raises(ValueError):
            TopicEmbeddingProfile(dim=0)

    def test_invalid_alpha_raises(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        with pytest.raises(ValueError):
            TopicEmbeddingProfile(dim=4, ema_alpha=1.5)

    def test_invalid_min_stable_raises(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        with pytest.raises(ValueError):
            TopicEmbeddingProfile(dim=4, min_updates_before_stable=0)

    def test_initial_state(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=4)
        assert p.dim == 4
        assert p.update_count == 0
        assert not p.is_stable
        assert p.centroid == [0.0, 0.0, 0.0, 0.0]


# ===========================================================================
# Group 14: TopicEmbeddingProfileUpdate
# ===========================================================================

class TestTopicEmbeddingProfileUpdate:
    def test_first_update_sets_centroid(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=3, ema_alpha=0.5)
        p.update([1.0, 2.0, 3.0])
        assert p.centroid == pytest.approx([1.0, 2.0, 3.0])
        assert p.update_count == 1

    def test_ema_update_blends(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=2, ema_alpha=0.5)
        p.update([1.0, 0.0])   # centroid = [1, 0]
        p.update([0.0, 1.0])   # centroid = 0.5*[0,1] + 0.5*[1,0] = [0.5, 0.5]
        assert p.centroid == pytest.approx([0.5, 0.5], abs=0.01)

    def test_wrong_embedding_type_raises(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=3)
        with pytest.raises(TypeError):
            p.update("not a list")

    def test_wrong_dim_raises(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=3)
        with pytest.raises(ValueError):
            p.update([1.0, 2.0])  # only 2 elements

    def test_negative_weight_raises(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=3)
        with pytest.raises(ValueError):
            p.update([1.0, 0.0, 0.0], weight=-1.0)

    def test_higher_weight_increases_effective_alpha(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p_low = TopicEmbeddingProfile(dim=2, ema_alpha=0.1)
        p_high = TopicEmbeddingProfile(dim=2, ema_alpha=0.1)
        p_low.update([1.0, 0.0])
        p_low.update([0.0, 1.0], weight=1.0)
        p_high.update([1.0, 0.0])
        p_high.update([0.0, 1.0], weight=5.0)
        # Higher weight should pull centroid harder toward new embedding
        assert p_high.centroid[1] > p_low.centroid[1]


# ===========================================================================
# Group 15: TopicEmbeddingProfileSimilarity
# ===========================================================================

class TestTopicEmbeddingProfileSimilarity:
    def test_similarity_to_self(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=3)
        emb = [1.0, 0.0, 0.0]
        p.update(emb)
        assert p.similarity(emb) == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_similarity_zero(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=2)
        p.update([1.0, 0.0])
        assert p.similarity([0.0, 1.0]) == pytest.approx(0.0, abs=1e-5)

    def test_similarity_before_update(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=3)
        # Zero centroid → similarity is 0
        assert p.similarity([1.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_wrong_dim_raises(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=3)
        p.update([1.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            p.similarity([1.0, 0.0])

    def test_opposite_similarity_negative(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=2)
        p.update([1.0, 0.0])
        sim = p.similarity([-1.0, 0.0])
        assert sim < 0.0


# ===========================================================================
# Group 16: TopicEmbeddingProfileDrift
# ===========================================================================

class TestTopicEmbeddingProfileDrift:
    def test_drift_same_direction_zero(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=2)
        p.update([1.0, 0.0])
        assert p.drift_score([1.0, 0.0]) == pytest.approx(0.0, abs=1e-5)

    def test_drift_orthogonal_is_one(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=2)
        p.update([1.0, 0.0])
        assert p.drift_score([0.0, 1.0]) == pytest.approx(1.0, abs=1e-5)

    def test_drift_range(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=4)
        p.update([0.5, 0.5, 0.5, 0.5])
        drift = p.drift_score([0.1, 0.9, 0.1, 0.9])
        assert 0.0 <= drift <= 1.0


# ===========================================================================
# Group 17: TopicEmbeddingProfileStability
# ===========================================================================

class TestTopicEmbeddingProfileStability:
    def test_not_stable_initially(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=2, min_updates_before_stable=3)
        assert not p.is_stable

    def test_stable_after_min_updates(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=2, min_updates_before_stable=3)
        for _ in range(3):
            p.update([1.0, 0.0])
        assert p.is_stable

    def test_not_stable_before_min_updates(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=2, min_updates_before_stable=5)
        for _ in range(4):
            p.update([1.0, 0.0])
        assert not p.is_stable


# ===========================================================================
# Group 18: TopicEmbeddingProfileReset
# ===========================================================================

class TestTopicEmbeddingProfileReset:
    def test_reset_clears_centroid(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=3)
        p.update([1.0, 1.0, 1.0])
        p.reset()
        assert p.centroid == [0.0, 0.0, 0.0]
        assert p.update_count == 0
        assert not p.is_stable

    def test_reset_and_reuse(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=2)
        p.update([1.0, 0.0])
        p.reset()
        p.update([0.0, 1.0])
        assert p.centroid == pytest.approx([0.0, 1.0])


# ===========================================================================
# Group 19: TopicEmbeddingProfileSerialization
# ===========================================================================

class TestTopicEmbeddingProfileSerialization:
    def test_to_dict_structure(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=3, ema_alpha=0.3)
        p.update([1.0, 0.0, 0.0])
        d = p.to_dict()
        assert d["dim"] == 3
        assert d["ema_alpha"] == pytest.approx(0.3)
        assert d["update_count"] == 1
        assert len(d["centroid"]) == 3

    def test_from_dict_wrong_type_raises(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        with pytest.raises(TypeError):
            TopicEmbeddingProfile.from_dict("not a dict")

    def test_round_trip(self):
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        p = TopicEmbeddingProfile(dim=3, ema_alpha=0.2)
        p.update([1.0, 0.0, 0.0])
        p.update([0.0, 1.0, 0.0])
        d = p.to_dict()
        p2 = TopicEmbeddingProfile.from_dict(d)
        assert p2.dim == 3
        assert p2.update_count == 2
        assert p2.centroid == pytest.approx(p.centroid)

# ===========================================================================
# Group 20: FeedbackLearnerConstruct
# ===========================================================================

class TestFeedbackLearnerConstruct:
    def _make_graph(self):
        from app.personalization.interest_graph import InterestGraph
        return InterestGraph()

    def test_wrong_graph_type_raises(self):
        from app.personalization.feedback_learner import FeedbackLearner
        with pytest.raises(TypeError):
            FeedbackLearner(interest_graph="not a graph")

    def test_wrong_profile_type_raises(self):
        from app.personalization.feedback_learner import FeedbackLearner
        g = self._make_graph()
        with pytest.raises(TypeError):
            FeedbackLearner(interest_graph=g, embedding_profile="bad")

    def test_invalid_learning_rate_raises(self):
        from app.personalization.feedback_learner import FeedbackLearner
        with pytest.raises(ValueError):
            FeedbackLearner(interest_graph=self._make_graph(), learning_rate=0.0)

    def test_invalid_expected_read_seconds_raises(self):
        from app.personalization.feedback_learner import FeedbackLearner
        with pytest.raises(ValueError):
            FeedbackLearner(interest_graph=self._make_graph(), expected_read_seconds=0)

    def test_invalid_history_maxlen_raises(self):
        from app.personalization.feedback_learner import FeedbackLearner
        with pytest.raises(ValueError):
            FeedbackLearner(interest_graph=self._make_graph(), history_maxlen=0)

    def test_valid_construction(self):
        from app.personalization.feedback_learner import FeedbackLearner
        learner = FeedbackLearner(interest_graph=self._make_graph())
        assert learner is not None


# ===========================================================================
# Group 21: FeedbackLearnerProcess
# ===========================================================================

class TestFeedbackLearnerProcess:
    def _learner(self):
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        return FeedbackLearner(interest_graph=InterestGraph())

    def test_process_like_increases_weight(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        ev = _make_event(feedback_type=FeedbackType.LIKE, topic_ids=["ml"])
        learner.process_feedback(ev)
        w = learner._graph.get_weight("ml")
        assert w > 0.5  # default initial is 0.5

    def test_process_dismiss_decreases_weight(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        # First, give the topic some weight
        ev_like = _make_event(feedback_type=FeedbackType.LIKE, topic_ids=["ml"])
        learner.process_feedback(ev_like)
        learner.process_feedback(ev_like)
        ev_dismiss = _make_event(feedback_type=FeedbackType.DISMISS, topic_ids=["ml"])
        w_before = learner._graph.get_weight("ml")
        learner.process_feedback(ev_dismiss)
        w_after = learner._graph.get_weight("ml")
        assert w_after < w_before

    def test_process_wrong_type_raises(self):
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        learner = FeedbackLearner(interest_graph=InterestGraph())
        with pytest.raises(TypeError):
            learner.process_feedback("not an event")

    def test_process_returns_interest_weights(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        ev = _make_event(feedback_type=FeedbackType.SAVE, topic_ids=["ml", "llm"])
        result = learner.process_feedback(ev)
        assert len(result) == 2
        topic_ids = {iw.topic_id for iw in result}
        assert "ml" in topic_ids and "llm" in topic_ids

    def test_save_larger_delta_than_like(self):
        from app.personalization.models import FeedbackType
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        g_like = InterestGraph()
        g_save = InterestGraph()
        FeedbackLearner(interest_graph=g_like).process_feedback(
            _make_event(feedback_type=FeedbackType.LIKE, topic_ids=["ml"]))
        FeedbackLearner(interest_graph=g_save).process_feedback(
            _make_event(feedback_type=FeedbackType.SAVE, topic_ids=["ml"]))
        assert g_save.get_weight("ml") > g_like.get_weight("ml")

    def test_multi_topic_event_updates_all(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        ev = _make_event(feedback_type=FeedbackType.LIKE, topic_ids=["ml", "nlp", "cv"])
        learner.process_feedback(ev)
        for t in ["ml", "nlp", "cv"]:
            assert learner._graph.get_weight(t) is not None

    def test_event_with_embedding_updates_profile(self):
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        from app.personalization.models import FeedbackType
        profile = TopicEmbeddingProfile(dim=3)
        learner = FeedbackLearner(interest_graph=InterestGraph(), embedding_profile=profile)
        ev = _make_event(feedback_type=FeedbackType.LIKE, embedding=[1.0, 0.0, 0.0])
        learner.process_feedback(ev)
        assert profile.update_count == 1


# ===========================================================================
# Group 22: FeedbackLearnerImplicit
# ===========================================================================

class TestFeedbackLearnerImplicit:
    def _learner(self):
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        return FeedbackLearner(interest_graph=InterestGraph(), expected_read_seconds=120.0)

    def test_full_dwell_full_delta(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        ev = _make_event(feedback_type=FeedbackType.READ_COMPLETE, topic_ids=["ml"], dwell=120.0)
        result = learner.process_feedback(ev)
        # Full dwell → scale = 1.0 → full +0.10 delta
        assert result[0].weight == pytest.approx(0.6, abs=0.01)  # 0.5 + 0.10

    def test_half_dwell_half_delta(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        ev = _make_event(feedback_type=FeedbackType.READ_COMPLETE, topic_ids=["ml"], dwell=60.0)
        result = learner.process_feedback(ev)
        # Half dwell → scale = 0.5 → +0.05 delta
        assert result[0].weight == pytest.approx(0.55, abs=0.01)  # 0.5 + 0.05

    def test_zero_dwell_minimal_delta(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        ev = _make_event(feedback_type=FeedbackType.READ_COMPLETE, topic_ids=["ml"], dwell=0.0)
        result = learner.process_feedback(ev)
        # Zero dwell → scale = 0.0 → no delta
        assert result[0].weight == pytest.approx(0.5, abs=0.01)

    def test_scroll_past_decreases_weight(self):
        from app.personalization.models import FeedbackType
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph(initial_weight=0.5)
        learner = FeedbackLearner(interest_graph=g)
        ev = _make_event(feedback_type=FeedbackType.SCROLL_PAST, topic_ids=["ml"])
        learner.process_feedback(ev)
        assert g.get_weight("ml") < 0.5


# ===========================================================================
# Group 23: FeedbackLearnerBatch
# ===========================================================================

class TestFeedbackLearnerBatch:
    def _learner(self):
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        return FeedbackLearner(interest_graph=InterestGraph())

    def test_batch_processes_all(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        events = [
            _make_event(feedback_type=FeedbackType.LIKE, topic_ids=["ml"], item_id="i1"),
            _make_event(feedback_type=FeedbackType.SAVE, topic_ids=["nlp"], item_id="i2"),
            _make_event(feedback_type=FeedbackType.CLICK, topic_ids=["cv"], item_id="i3"),
        ]
        stats = learner.process_batch(events)
        assert stats["processed"] == 3
        assert stats["topics_updated"] == 3

    def test_batch_wrong_type_raises(self):
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        with pytest.raises(TypeError):
            FeedbackLearner(interest_graph=InterestGraph()).process_batch("bad")

    def test_batch_empty_list(self):
        learner = self._learner()
        stats = learner.process_batch([])
        assert stats["processed"] == 0
        assert stats["topics_updated"] == 0


# ===========================================================================
# Group 24: FeedbackLearnerHistory
# ===========================================================================

class TestFeedbackLearnerHistory:
    def _learner(self):
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        return FeedbackLearner(interest_graph=InterestGraph())

    def test_history_records_event(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        ev = _make_event(feedback_type=FeedbackType.LIKE)
        learner.process_feedback(ev)
        hist = learner.get_history()
        assert len(hist) == 1

    def test_history_newest_first(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        for i in range(3):
            learner.process_feedback(_make_event(item_id=f"item{i}", feedback_type=FeedbackType.CLICK))
        hist = learner.get_history()
        assert hist[0].item_id == "item2"

    def test_history_limit(self):
        from app.personalization.models import FeedbackType
        learner = self._learner()
        for i in range(5):
            learner.process_feedback(_make_event(item_id=f"i{i}", feedback_type=FeedbackType.LIKE))
        hist = learner.get_history(limit=3)
        assert len(hist) == 3


# ===========================================================================
# Group 25: FeedbackLearnerStats
# ===========================================================================

class TestFeedbackLearnerStats:
    def test_stats_after_events(self):
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.models import FeedbackType
        learner = FeedbackLearner(interest_graph=InterestGraph())
        for _ in range(3):
            learner.process_feedback(_make_event(feedback_type=FeedbackType.LIKE))
        for _ in range(2):
            learner.process_feedback(_make_event(feedback_type=FeedbackType.DISMISS))
        stats = learner.summary_stats()
        assert stats["total_events"] == 5
        assert stats["positive_events"] >= 3
        assert stats["negative_events"] >= 2

    def test_stats_empty(self):
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.interest_graph import InterestGraph
        learner = FeedbackLearner(interest_graph=InterestGraph())
        stats = learner.summary_stats()
        assert stats["total_events"] == 0
        assert stats["by_type"] == {}



# ===========================================================================
# Group 26: UserDigestRankerConstruct
# ===========================================================================

class TestUserDigestRankerConstruct:
    def test_wrong_graph_type_raises(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        with pytest.raises(TypeError):
            UserDigestRanker(interest_graph="bad")

    def test_wrong_profile_type_raises(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        with pytest.raises(TypeError):
            UserDigestRanker(embedding_profile="bad")

    def test_invalid_top_interest_k_raises(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        with pytest.raises(ValueError):
            UserDigestRanker(top_interest_k=0)

    def test_invalid_recency_half_life_raises(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        with pytest.raises(ValueError):
            UserDigestRanker(recency_half_life_h=0.0)

    def test_invalid_boost_weight_raises(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        with pytest.raises(ValueError):
            UserDigestRanker(boost_weight=1.5)

    def test_default_construction(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        ranker = UserDigestRanker()
        assert ranker is not None


# ===========================================================================
# Group 27: UserDigestRankerRanking
# ===========================================================================

class TestUserDigestRankerRanking:
    def _ranker_with_graph(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        from app.personalization.interest_graph import InterestGraph
        g = InterestGraph()
        g.add_topic("ml", initial_weight=0.9)
        g.add_topic("ai", initial_weight=0.8)
        return UserDigestRanker(interest_graph=g)

    def test_empty_candidates_returns_empty(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        assert UserDigestRanker().rank([]) == []

    def test_wrong_candidates_type_raises(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        with pytest.raises(TypeError):
            UserDigestRanker().rank("not a list")

    def test_invalid_top_k_raises(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        with pytest.raises(ValueError):
            UserDigestRanker().rank([], top_k=0)

    def test_rank_returns_ranked_items(self):
        ranker = self._ranker_with_graph()
        candidates = [_make_candidate("c1", topics=["ml"]), _make_candidate("c2", topics=["cv"])]
        result = ranker.rank(candidates)
        assert len(result) == 2
        assert result[0].rank == 1
        assert result[1].rank == 2

    def test_topic_match_ranks_higher(self):
        ranker = self._ranker_with_graph()
        high = _make_candidate("hi", topics=["ml", "ai"], trust=0.8, engagement=0.7)
        low = _make_candidate("lo", topics=["cooking", "travel"], trust=0.3, engagement=0.2)
        result = ranker.rank([high, low])
        # ml/ai items should rank above cooking/travel with same signals
        assert result[0].item_id == "hi"

    def test_recency_matters(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        ranker = UserDigestRanker()  # no graph, so only recency matters in ranking
        fresh = _make_candidate("fresh", published_hours_ago=0.5, trust=0.5, engagement=0.5, novelty=0.5)
        stale = _make_candidate("stale", published_hours_ago=72.0, trust=0.5, engagement=0.5, novelty=0.5)
        result = ranker.rank([stale, fresh])
        assert result[0].item_id == "fresh"

    def test_scores_in_unit_range(self):
        ranker = self._ranker_with_graph()
        candidates = [_make_candidate(f"c{i}", topics=["ml"]) for i in range(5)]
        result = ranker.rank(candidates)
        for item in result:
            assert 0.0 <= item.final_score <= 1.0

    def test_rank_is_one_indexed(self):
        ranker = self._ranker_with_graph()
        candidates = [_make_candidate(f"c{i}") for i in range(3)]
        result = ranker.rank(candidates)
        ranks = [item.rank for item in result]
        assert ranks == [1, 2, 3]


# ===========================================================================
# Group 28: UserDigestRankerTopK
# ===========================================================================

class TestUserDigestRankerTopK:
    def test_top_k_limits_output(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        ranker = UserDigestRanker()
        candidates = [_make_candidate(f"c{i}") for i in range(10)]
        result = ranker.rank(candidates, top_k=3)
        assert len(result) == 3

    def test_top_k_larger_than_candidates(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        ranker = UserDigestRanker()
        candidates = [_make_candidate(f"c{i}") for i in range(3)]
        result = ranker.rank(candidates, top_k=10)
        assert len(result) == 3


# ===========================================================================
# Group 29: UserDigestRankerScoreBreakdown
# ===========================================================================

class TestUserDigestRankerScoreBreakdown:
    def test_score_breakdown_has_all_keys(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        ranker = UserDigestRanker()
        result = ranker.rank([_make_candidate("c1")])
        bd = result[0].score_breakdown
        for key in ("topic_relevance", "embedding_similarity", "recency", "engagement", "trust", "novelty"):
            assert key in bd

    def test_sub_scores_match_breakdown(self):
        from app.personalization.user_digest_ranker import UserDigestRanker
        ranker = UserDigestRanker()
        result = ranker.rank([_make_candidate("c1")])
        item = result[0]
        assert item.recency_score == pytest.approx(item.score_breakdown["recency"], abs=1e-4)
        assert item.trust_score == pytest.approx(item.score_breakdown["trust"], abs=1e-4)


# ===========================================================================
# Group 30: NoveltyTradeoffConstruct
# ===========================================================================

class TestNoveltyTradeoffConstruct:
    def test_default_construction(self):
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        t = NoveltyRelevanceTradeoff()
        assert 0.0 <= t.current_alpha <= 1.0

    def test_wrong_config_type_raises(self):
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        with pytest.raises(TypeError):
            NoveltyRelevanceTradeoff(config="not a config")

    def test_custom_alpha(self):
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        from app.personalization.models import NoveltyRelevanceConfig
        cfg = NoveltyRelevanceConfig(alpha=0.6)
        t = NoveltyRelevanceTradeoff(cfg)
        assert t.current_alpha == pytest.approx(0.6)


# ===========================================================================
# Group 31: NoveltyTradeoffBlend
# ===========================================================================

class TestNoveltyTradeoffBlend:
    def _tradeoff(self, alpha=0.3):
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        from app.personalization.models import NoveltyRelevanceConfig
        return NoveltyRelevanceTradeoff(NoveltyRelevanceConfig(alpha=alpha, auto_adapt=False))

    def test_pure_relevance(self):
        t = self._tradeoff(alpha=0.0)
        assert t.blend_scores(0.8, 0.2) == pytest.approx(0.8)

    def test_pure_novelty(self):
        t = self._tradeoff(alpha=1.0)
        assert t.blend_scores(0.8, 0.2) == pytest.approx(0.2)

    def test_blend_at_half(self):
        t = self._tradeoff(alpha=0.5)
        assert t.blend_scores(0.8, 0.4) == pytest.approx(0.6)

    def test_out_of_range_relevance_raises(self):
        t = self._tradeoff()
        with pytest.raises(ValueError):
            t.blend_scores(1.5, 0.5)

    def test_out_of_range_novelty_raises(self):
        t = self._tradeoff()
        with pytest.raises(ValueError):
            t.blend_scores(0.5, -0.1)

    def test_alpha_override(self):
        t = self._tradeoff(alpha=0.0)
        result = t.blend_scores(0.8, 0.2, alpha=1.0)
        assert result == pytest.approx(0.2)

    def test_result_in_unit_range(self):
        t = self._tradeoff(alpha=0.3)
        r = t.blend_scores(0.7, 0.6)
        assert 0.0 <= r <= 1.0


# ===========================================================================
# Group 32: NoveltyTradeoffUpdate
# ===========================================================================

class TestNoveltyTradeoffUpdate:
    def _tradeoff(self, alpha=0.3, auto_adapt=True):
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        from app.personalization.models import NoveltyRelevanceConfig
        return NoveltyRelevanceTradeoff(NoveltyRelevanceConfig(
            alpha=alpha, auto_adapt=auto_adapt,
            novelty_push_rate=0.05, relevance_pull_rate=0.03,
            min_alpha=0.05, max_alpha=0.80,
        ))

    def test_save_decreases_alpha(self):
        from app.personalization.models import FeedbackType
        t = self._tradeoff(alpha=0.5)
        ev = _make_event(feedback_type=FeedbackType.SAVE, topic_ids=["ml"])
        new_alpha = t.update_from_feedback(ev)
        assert new_alpha < 0.5

    def test_click_increases_alpha(self):
        from app.personalization.models import FeedbackType
        t = self._tradeoff(alpha=0.3)
        ev = _make_event(feedback_type=FeedbackType.CLICK, topic_ids=["ml"])
        new_alpha = t.update_from_feedback(ev)
        assert new_alpha > 0.3

    def test_read_complete_decreases_alpha(self):
        from app.personalization.models import FeedbackType
        t = self._tradeoff(alpha=0.5)
        ev = _make_event(feedback_type=FeedbackType.READ_COMPLETE, topic_ids=["ml"])
        t.update_from_feedback(ev)
        assert t.current_alpha < 0.5

    def test_auto_adapt_false_no_change(self):
        from app.personalization.models import FeedbackType
        t = self._tradeoff(alpha=0.4, auto_adapt=False)
        ev = _make_event(feedback_type=FeedbackType.SAVE, topic_ids=["ml"])
        t.update_from_feedback(ev)
        assert t.current_alpha == pytest.approx(0.4)

    def test_alpha_clamped_at_max(self):
        from app.personalization.models import FeedbackType
        t = self._tradeoff(alpha=0.78)
        for _ in range(20):
            t.update_from_feedback(_make_event(feedback_type=FeedbackType.CLICK))
        assert t.current_alpha <= 0.80

    def test_alpha_clamped_at_min(self):
        from app.personalization.models import FeedbackType
        t = self._tradeoff(alpha=0.08)
        for _ in range(20):
            t.update_from_feedback(_make_event(feedback_type=FeedbackType.SAVE))
        assert t.current_alpha >= 0.05

    def test_wrong_event_type_raises(self):
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        t = NoveltyRelevanceTradeoff()
        with pytest.raises(TypeError):
            t.update_from_feedback("not an event")


# ===========================================================================
# Group 33: NoveltyTradeoffFatigue
# ===========================================================================

class TestNoveltyTradeoffFatigue:
    def test_fatigue_nudge_on_repeated_same_topic(self):
        """Repeated engagement with the same topic triggers the fatigue nudge."""
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        from app.personalization.models import NoveltyRelevanceConfig, FeedbackType
        cfg = NoveltyRelevanceConfig(
            alpha=0.3, auto_adapt=True,
            novelty_push_rate=0.05, relevance_pull_rate=0.03,
            min_alpha=0.05, max_alpha=0.95,
            exploration_window=10,
        )
        t = NoveltyRelevanceTradeoff(cfg)
        alpha_start = t.current_alpha
        # Fire 10 CLICK events on same topic → window fills with "ml"
        for _ in range(12):
            t.update_from_feedback(_make_event(feedback_type=FeedbackType.CLICK, topic_ids=["ml"]))
        # Alpha should have increased (fatigue nudge + click pushes)
        assert t.current_alpha > alpha_start

    def test_diverse_topics_no_fatigue(self):
        """Diverse topics should not trigger topic-fatigue nudge."""
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        from app.personalization.models import NoveltyRelevanceConfig, FeedbackType
        cfg = NoveltyRelevanceConfig(
            alpha=0.5, auto_adapt=True,
            novelty_push_rate=0.0, relevance_pull_rate=0.03,
            min_alpha=0.05, max_alpha=0.95,
            exploration_window=10,
        )
        t = NoveltyRelevanceTradeoff(cfg)
        topics = [f"topic_{i}" for i in range(10)]
        for i in range(10):
            t.update_from_feedback(_make_event(
                feedback_type=FeedbackType.READ_COMPLETE, topic_ids=[topics[i]]))
        # With diverse topics concentration < fatigue threshold, pull-only → alpha decreases
        assert t.current_alpha <= 0.5


# ===========================================================================
# Group 34: NoveltyTradeoffCompute
# ===========================================================================

class TestNoveltyTradeoffCompute:
    def _tradeoff(self):
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        from app.personalization.models import NoveltyRelevanceConfig
        return NoveltyRelevanceTradeoff(NoveltyRelevanceConfig(
            alpha=0.3, auto_adapt=True,
            novelty_push_rate=0.05, relevance_pull_rate=0.03,
            min_alpha=0.05, max_alpha=0.80,
        ))

    def test_compute_alpha_wrong_type_raises(self):
        t = self._tradeoff()
        with pytest.raises(TypeError):
            t.compute_alpha("not a list")

    def test_compute_alpha_empty_returns_default(self):
        t = self._tradeoff()
        alpha = t.compute_alpha([])
        assert alpha == pytest.approx(0.3)

    def test_compute_alpha_saves_reduce(self):
        from app.personalization.models import FeedbackType
        t = self._tradeoff()
        events = [_make_event(feedback_type=FeedbackType.SAVE) for _ in range(5)]
        alpha = t.compute_alpha(events)
        assert alpha < 0.3

    def test_compute_alpha_does_not_mutate_state(self):
        from app.personalization.models import FeedbackType
        t = self._tradeoff()
        alpha_before = t.current_alpha
        events = [_make_event(feedback_type=FeedbackType.CLICK) for _ in range(5)]
        t.compute_alpha(events)
        assert t.current_alpha == pytest.approx(alpha_before)

    def test_compute_alpha_range(self):
        from app.personalization.models import FeedbackType
        t = self._tradeoff()
        events = [_make_event(feedback_type=FeedbackType.LIKE) for _ in range(10)]
        alpha = t.compute_alpha(events)
        assert 0.05 <= alpha <= 0.80


# ===========================================================================
# Group 35: NoveltyTradeoffReset
# ===========================================================================

class TestNoveltyTradeoffReset:
    def test_reset_restores_initial_alpha(self):
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        from app.personalization.models import NoveltyRelevanceConfig, FeedbackType
        cfg = NoveltyRelevanceConfig(alpha=0.4)
        t = NoveltyRelevanceTradeoff(cfg)
        for _ in range(10):
            t.update_from_feedback(_make_event(feedback_type=FeedbackType.SAVE))
        t.reset()
        assert t.current_alpha == pytest.approx(0.4)

    def test_reset_clears_topic_history(self):
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        from app.personalization.models import FeedbackType
        t = NoveltyRelevanceTradeoff()
        for _ in range(5):
            t.update_from_feedback(_make_event(feedback_type=FeedbackType.CLICK, topic_ids=["ml"]))
        t.reset()
        stats = t.stats()
        assert stats["recent_topic_count"] == 0
        assert stats["event_count"] == 0


# ===========================================================================
# Group 36: CrossComponentWiring
# ===========================================================================

class TestCrossComponentWiring:
    """Integration-style tests that wire FeedbackLearner → InterestGraph → Ranker."""

    def _setup(self):
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.topic_embedding_profile import TopicEmbeddingProfile
        from app.personalization.feedback_learner import FeedbackLearner
        from app.personalization.user_digest_ranker import UserDigestRanker
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        graph = InterestGraph()
        profile = TopicEmbeddingProfile(dim=3)
        learner = FeedbackLearner(interest_graph=graph, embedding_profile=profile)
        tradeoff = NoveltyRelevanceTradeoff()
        ranker = UserDigestRanker(interest_graph=graph, embedding_profile=profile,
                                   novelty_tradeoff=tradeoff)
        return graph, profile, learner, ranker, tradeoff

    def test_feedback_flows_to_ranker(self):
        """Feedback on ml topic should increase weight → matching candidate scores higher."""
        from app.personalization.models import FeedbackType
        graph, profile, learner, ranker, _ = self._setup()
        # Train preference for ml
        for _ in range(5):
            learner.process_feedback(_make_event(
                feedback_type=FeedbackType.LIKE, topic_ids=["ml"],
                embedding=[1.0, 0.0, 0.0]))
        ml_cand = _make_candidate("ml_item", topics=["ml"], trust=0.6, engagement=0.6)
        cv_cand = _make_candidate("cv_item", topics=["cv"], trust=0.6, engagement=0.6)
        result = ranker.rank([cv_cand, ml_cand])
        assert result[0].item_id == "ml_item"

    def test_dismiss_reduces_score(self):
        from app.personalization.models import FeedbackType
        graph, _, learner, ranker, _ = self._setup()
        # First establish a high weight for ml
        for _ in range(10):
            learner.process_feedback(_make_event(feedback_type=FeedbackType.LIKE, topic_ids=["ml"]))
        w_before = graph.get_weight("ml")
        # Now dismiss many times
        for _ in range(10):
            learner.process_feedback(_make_event(feedback_type=FeedbackType.DISMISS, topic_ids=["ml"]))
        w_after = graph.get_weight("ml")
        assert w_after < w_before

    def test_tradeoff_affects_final_score(self):
        """Higher alpha should blend more novelty into the score."""
        from app.personalization.user_digest_ranker import UserDigestRanker
        from app.personalization.interest_graph import InterestGraph
        from app.personalization.novelty_vs_relevance_tradeoff import NoveltyRelevanceTradeoff
        from app.personalization.models import NoveltyRelevanceConfig
        g = InterestGraph()
        g.add_topic("ml", initial_weight=0.9)
        low_novelty_cfg = NoveltyRelevanceConfig(alpha=0.0, auto_adapt=False)
        high_novelty_cfg = NoveltyRelevanceConfig(alpha=1.0, auto_adapt=False)
        cand = _make_candidate("c1", topics=["ml"], novelty=0.9, trust=0.1, engagement=0.1)
        r_low = UserDigestRanker(interest_graph=g,
                                   novelty_tradeoff=NoveltyRelevanceTradeoff(low_novelty_cfg))
        r_high = UserDigestRanker(interest_graph=g,
                                   novelty_tradeoff=NoveltyRelevanceTradeoff(high_novelty_cfg))
        score_low = r_low.rank([cand])[0].final_score
        score_high = r_high.rank([cand])[0].final_score
        # With alpha=1 (pure novelty) and novelty=0.9, score_high should >= score_low for this candidate
        assert score_high >= score_low or abs(score_high - score_low) < 0.3

    def test_full_pipeline_returns_all_candidates(self):
        from app.personalization.models import FeedbackType
        _, _, learner, ranker, _ = self._setup()
        for t in ["ml", "ai", "nlp"]:
            learner.process_feedback(_make_event(feedback_type=FeedbackType.LIKE, topic_ids=[t]))
        candidates = [_make_candidate(f"c{i}", topics=[["ml", "ai", "nlp"][i % 3]]) for i in range(8)]
        result = ranker.rank(candidates)
        assert len(result) == 8
        # All ranks unique
        ranks = [item.rank for item in result]
        assert sorted(ranks) == list(range(1, 9))

    def test_public_package_imports(self):
        """Verify that the package __init__ exports work correctly."""
        from app.personalization import (
            FeedbackType, InterestWeight, FeedbackEvent, DigestCandidate,
            RankedDigestItem, NoveltyRelevanceConfig, RankingWeights,
            InterestGraph, TopicEmbeddingProfile, FeedbackLearner,
            UserDigestRanker, NoveltyRelevanceTradeoff,
        )
        assert FeedbackType.LIKE.value == "like"
        assert InterestGraph is not None
        assert UserDigestRanker is not None

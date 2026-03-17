"""Tests for core data models."""

from datetime import datetime
from uuid import uuid4

import pytest

from app.core.models import (
    ContentItem,
    MediaType,
    SourcePlatform,
    UserInterestProfile,
    Cluster,
)


def test_content_item_creation():
    """Test creating a ContentItem."""
    user_id = uuid4()
    item = ContentItem(
        user_id=user_id,
        source_platform=SourcePlatform.REDDIT,
        source_id="test123",
        source_url="https://reddit.com/r/test/comments/test123",
        title="Test Post",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )

    assert item.user_id == user_id
    assert item.source_platform == SourcePlatform.REDDIT
    assert item.title == "Test Post"
    assert item.media_type == MediaType.TEXT


def test_content_item_with_topics():
    """Test ContentItem with topics."""
    item = ContentItem(
        user_id=uuid4(),
        source_platform=SourcePlatform.YOUTUBE,
        source_id="video123",
        source_url="https://youtube.com/watch?v=video123",
        title="AI Tutorial",
        media_type=MediaType.VIDEO,
        published_at=datetime.utcnow(),
        topics=["AI", "machine learning", "tutorial"],
    )

    assert len(item.topics) == 3
    assert "AI" in item.topics


def test_user_interest_profile():
    """Test UserInterestProfile creation."""
    user_id = uuid4()
    profile = UserInterestProfile(
        user_id=user_id,
        interest_topics=["AI", "technology", "science"],
        negative_filters=["sports", "celebrity"],
    )

    assert profile.user_id == user_id
    assert len(profile.interest_topics) == 3
    assert len(profile.negative_filters) == 2


def test_cluster_creation():
    """Test Cluster creation."""
    user_id = uuid4()
    items = [
        ContentItem(
            user_id=user_id,
            source_platform=SourcePlatform.REDDIT,
            source_id=f"post{i}",
            source_url=f"https://reddit.com/post{i}",
            title=f"Post {i}",
            media_type=MediaType.TEXT,
            published_at=datetime.utcnow(),
        )
        for i in range(3)
    ]

    cluster = Cluster(
        user_id=user_id,
        topic="AI News",
        summary="Latest developments in AI",
        items=items,
        item_ids=[item.id for item in items],
        relevance_score=0.85,
        platforms_represented=[SourcePlatform.REDDIT],
    )

    assert cluster.topic == "AI News"
    assert len(cluster.items) == 3
    assert cluster.relevance_score == 0.85
    assert SourcePlatform.REDDIT in cluster.platforms_represented



# ---------------------------------------------------------------------------
# DataResidencyGuard unit tests (Step 1 — competitive_analysis.md §5.1)
# ---------------------------------------------------------------------------

from app.core.data_residency import DataResidencyGuard
from app.core.errors import DataResidencyViolationError


def _make_item(**kwargs) -> ContentItem:
    """Helper: build a minimal ContentItem, overriding any field via kwargs."""
    defaults = dict(
        user_id=uuid4(),
        source_platform=SourcePlatform.REDDIT,
        source_id="test123",
        source_url="https://reddit.com/r/test/comments/test123",
        title="Test Post",
        media_type=MediaType.TEXT,
        published_at=datetime.utcnow(),
    )
    defaults.update(kwargs)
    return ContentItem(**defaults)


class TestDataResidencyGuard:
    """Unit tests for DataResidencyGuard — implements competitive_analysis.md §5.1."""

    def setup_method(self):
        self.guard = DataResidencyGuard()

    # (a) PII is stripped from author fields
    def test_author_is_pseudonymised(self):
        """Author real name is replaced with a stable anon_ prefix pseudonym."""
        item = _make_item(author="john.doe")
        clean = self.guard.redact(item)

        assert clean.author is not None
        assert clean.author.startswith("anon_"), f"Expected pseudonym, got: {clean.author}"
        assert "john" not in clean.author.lower()
        assert "doe" not in clean.author.lower()

    def test_author_pseudonym_is_stable(self):
        """Same author name always produces the same pseudonym (deterministic)."""
        item1 = _make_item(author="alice")
        item2 = _make_item(author="alice")
        g = DataResidencyGuard()
        assert g.redact(item1).author == g.redact(item2).author

    # (b) URLs are pseudonymised (PII query params stripped)
    def test_url_pii_query_params_redacted(self):
        """Profile PII query parameters in source_url are replaced (jane.smith removed)."""
        pii_url = "https://example.com/post?user=jane.smith&page=1"
        item = _make_item(source_url=pii_url)
        clean = self.guard.redact(item)

        assert "jane.smith" not in clean.source_url
        # After URL encoding, <redacted> may appear as %3Credacted%3E — either form is valid
        assert "redacted" in clean.source_url.lower()
        # Non-PII param should survive
        assert "page" in clean.source_url

    def test_url_without_pii_params_unchanged(self):
        """URLs without PII query parameters are not modified."""
        safe_url = "https://reddit.com/r/python/comments/abc123"
        item = _make_item(source_url=safe_url)
        clean = self.guard.redact(item)
        assert clean.source_url == safe_url

    # (c) DataResidencyViolationError is raised when bypass is detected
    def test_verify_clean_raises_on_email_in_text(self):
        """verify_clean raises DataResidencyViolationError if email found in raw_text."""
        item = _make_item(raw_text="Contact me at user@example.com for details")
        with pytest.raises(DataResidencyViolationError) as exc_info:
            self.guard.verify_clean(item)
        assert exc_info.value.field == "raw_text"
        assert "email" in exc_info.value.pattern

    def test_verify_clean_raises_on_phone_in_text(self):
        """verify_clean raises DataResidencyViolationError if phone number found in raw_text."""
        item = _make_item(raw_text="Call me at 555-867-5309 anytime")
        with pytest.raises(DataResidencyViolationError) as exc_info:
            self.guard.verify_clean(item)
        assert exc_info.value.field == "raw_text"

    def test_verify_clean_raises_on_pii_url(self):
        """verify_clean raises DataResidencyViolationError if source_url has PII params."""
        item = _make_item(source_url="https://example.com/?username=secret_user")
        with pytest.raises(DataResidencyViolationError) as exc_info:
            self.guard.verify_clean(item)
        assert exc_info.value.field == "source_url"

    # (d) Clean content passes through unchanged
    def test_clean_content_passes_unchanged(self):
        """Content with no PII passes through redact() with minimal changes."""
        item = _make_item(
            author=None,
            raw_text="This product is great, highly recommend it!",
            source_url="https://reddit.com/r/python/comments/clean123",
        )
        clean = self.guard.redact(item)

        assert clean.raw_text == item.raw_text
        assert clean.source_url == item.source_url
        assert clean.author is None

    def test_email_scrubbed_from_raw_text(self):
        """Email addresses in raw_text are replaced with <email_redacted>."""
        item = _make_item(raw_text="Reach out at support@company.com for help.")
        clean = self.guard.redact(item)

        assert "<email_redacted>" in clean.raw_text
        assert "support@company.com" not in clean.raw_text

    def test_redact_is_idempotent(self):
        """Calling redact() twice on the same item produces the same result."""
        item = _make_item(author="jane.doe", raw_text="Email: test@test.com")
        once = self.guard.redact(item)
        twice = self.guard.redact(once)
        assert once.author == twice.author
        assert once.raw_text == twice.raw_text

    def test_verify_clean_passes_on_redacted_item(self):
        """verify_clean does not raise after redact() has processed the item."""
        item = _make_item(
            author="john.doe",
            raw_text="Email me: user@example.com",
            source_url="https://example.com/?user=john",
        )
        clean = self.guard.redact(item)
        # Should NOT raise after full redaction
        self.guard.verify_clean(clean)



# ---------------------------------------------------------------------------
# TeamRole unit tests (Step 4 — competitive_analysis.md §5.5)
# ---------------------------------------------------------------------------

from app.core.models import TeamRole
from app.core.signal_models import TeamDigest


class TestTeamRole:
    """Unit tests for TeamRole privilege model — competitive_analysis.md §5.5."""

    # (a) VIEWER cannot assign (privilege too low)
    def test_viewer_does_not_have_manager_privilege(self):
        """VIEWER role has insufficient privilege for MANAGER-only actions."""
        assert not TeamRole.has_role_at_least(TeamRole.VIEWER, TeamRole.MANAGER)

    def test_analyst_does_not_have_manager_privilege(self):
        """ANALYST role has insufficient privilege for MANAGER-only actions."""
        assert not TeamRole.has_role_at_least(TeamRole.ANALYST, TeamRole.MANAGER)

    # (b) MANAGER can assign
    def test_manager_has_manager_privilege(self):
        """MANAGER role satisfies MANAGER requirement."""
        assert TeamRole.has_role_at_least(TeamRole.MANAGER, TeamRole.MANAGER)

    def test_manager_also_has_viewer_and_analyst_privileges(self):
        """MANAGER satisfies both VIEWER and ANALYST requirements."""
        assert TeamRole.has_role_at_least(TeamRole.MANAGER, TeamRole.VIEWER)
        assert TeamRole.has_role_at_least(TeamRole.MANAGER, TeamRole.ANALYST)

    def test_viewer_has_viewer_privilege_only(self):
        """VIEWER satisfies VIEWER but not ANALYST or MANAGER."""
        assert TeamRole.has_role_at_least(TeamRole.VIEWER, TeamRole.VIEWER)
        assert not TeamRole.has_role_at_least(TeamRole.VIEWER, TeamRole.ANALYST)

    def test_analyst_has_viewer_and_analyst_privileges(self):
        """ANALYST satisfies VIEWER and ANALYST but not MANAGER."""
        assert TeamRole.has_role_at_least(TeamRole.ANALYST, TeamRole.VIEWER)
        assert TeamRole.has_role_at_least(TeamRole.ANALYST, TeamRole.ANALYST)
        assert not TeamRole.has_role_at_least(TeamRole.ANALYST, TeamRole.MANAGER)

    def test_privilege_levels_are_ordered(self):
        """Privilege levels are strictly ordered: VIEWER < ANALYST < MANAGER."""
        assert TeamRole.privilege_level(TeamRole.VIEWER) < TeamRole.privilege_level(TeamRole.ANALYST)
        assert TeamRole.privilege_level(TeamRole.ANALYST) < TeamRole.privilege_level(TeamRole.MANAGER)

    # (c) TeamDigest returns correct counts
    def test_team_digest_instantiation(self):
        """TeamDigest instantiates correctly with default zero counts."""
        from uuid import uuid4
        from datetime import datetime
        now = datetime.utcnow()
        digest = TeamDigest(
            team_id=uuid4(),
            period_start=now,
            period_end=now,
        )
        assert digest.total_signals == 0
        assert digest.by_status == {}
        assert digest.by_type == {}
        assert digest.unassigned_count == 0
        assert digest.high_urgency_count == 0

    def test_team_digest_with_counts(self):
        """TeamDigest correctly stores by_status and by_type mappings."""
        from uuid import uuid4
        from datetime import datetime
        now = datetime.utcnow()
        digest = TeamDigest(
            team_id=uuid4(),
            period_start=now,
            period_end=now,
            total_signals=10,
            by_status={"new": 7, "acted": 3},
            by_type={"churn_risk": 4, "feature_request": 6},
            unassigned_count=2,
            high_urgency_count=4,
        )
        assert digest.total_signals == 10
        assert digest.by_status["new"] == 7
        assert digest.by_type["churn_risk"] == 4
        assert digest.unassigned_count == 2
        assert digest.high_urgency_count == 4


# ===========================================================================
# ConfidenceCalibrator tests (Enhancement 2)
# ===========================================================================

import json as _json
import math
from pathlib import Path
from uuid import uuid4 as _uuid4

import pytest

from app.domain.inference_models import SignalType
from app.intelligence.calibration import ConfidenceCalibrator


class TestConfidenceCalibrator:
    """Unit tests for ConfidenceCalibrator (per-SignalType temperature scaling)."""

    def _make_calibrator(self, tmp_path: Path, scalars: dict = None) -> ConfidenceCalibrator:
        state_file = tmp_path / "calibration_state.json"
        if scalars is not None:
            state_file.write_text(
                _json.dumps({"version": "1.0", "scalars": scalars}), encoding="utf-8"
            )
        return ConfidenceCalibrator(state_path=state_file)

    def test_calibrate_identity_at_T1(self, tmp_path: Path) -> None:
        """T=1.0 is the sigmoid identity: calibrate(logit, T=1) == sigmoid(logit)."""
        cal = self._make_calibrator(tmp_path, {"churn_risk": 1.0})
        raw_prob = 0.7
        logit = math.log(raw_prob / (1 - raw_prob))
        result = cal.calibrate(logit, SignalType.CHURN_RISK)
        assert abs(result - raw_prob) < 1e-6

    def test_calibrate_sharpens_at_low_T(self, tmp_path: Path) -> None:
        """T<1.0 sharpens probabilities away from 0.5."""
        cal = self._make_calibrator(tmp_path, {"churn_risk": 0.5})
        raw_prob = 0.7
        logit = math.log(raw_prob / (1 - raw_prob))
        sharpened = cal.calibrate(logit, SignalType.CHURN_RISK)
        assert sharpened > raw_prob

    def test_calibrate_flattens_at_high_T(self, tmp_path: Path) -> None:
        """T>1.0 flattens probabilities toward 0.5."""
        cal = self._make_calibrator(tmp_path, {"churn_risk": 3.0})
        raw_prob = 0.9
        logit = math.log(raw_prob / (1 - raw_prob))
        flattened = cal.calibrate(logit, SignalType.CHURN_RISK)
        assert flattened < raw_prob

    def test_calibrate_missing_type_defaults_T1(self, tmp_path: Path) -> None:
        """An unknown SignalType falls back to T=1.0."""
        cal = self._make_calibrator(tmp_path, {})
        raw_prob = 0.6
        logit = math.log(raw_prob / (1 - raw_prob))
        result = cal.calibrate(logit, SignalType.COMPLAINT)
        assert abs(result - raw_prob) < 1e-6

    def test_update_adjusts_T_on_wrong_prediction(self, tmp_path: Path) -> None:
        """update() modifies T when predicted_prob contradicts true_label."""
        cal = self._make_calibrator(tmp_path, {"complaint": 1.0})
        initial_t = cal._scalars.get("complaint", 1.0)
        cal.update(SignalType.COMPLAINT, predicted_prob=0.9, true_label=False)
        new_t = cal._scalars.get("complaint", 1.0)
        assert new_t != initial_t

    def test_state_roundtrip(self, tmp_path: Path) -> None:
        """Scalars persisted by _save() are reloaded identically by _load()."""
        cal = self._make_calibrator(tmp_path, {"praise": 1.2})
        cal.update(SignalType.PRAISE, predicted_prob=0.8, true_label=True)
        saved_t = cal._scalars["praise"]
        cal2 = ConfidenceCalibrator(state_path=tmp_path / "calibration_state.json")
        assert abs(cal2._scalars.get("praise", 0.0) - saved_t) < 1e-9

    def test_update_never_below_t_min(self, tmp_path: Path) -> None:
        """Temperature must never drop below the minimum value (0.1)."""
        cal = self._make_calibrator(tmp_path, {"bug_report": 0.11})
        for _ in range(500):
            cal.update(SignalType.BUG_REPORT, predicted_prob=0.99, true_label=False)
        assert cal._scalars["bug_report"] >= 0.1

    def test_missing_state_file_gives_default_T1(self, tmp_path: Path) -> None:
        """When state file is absent the calibrator starts at T=1.0."""
        cal = ConfidenceCalibrator(state_path=tmp_path / "nonexistent.json")
        raw_prob = 0.65
        logit = math.log(raw_prob / (1 - raw_prob))
        result = cal.calibrate(logit, SignalType.FEATURE_REQUEST)
        assert abs(result - raw_prob) < 1e-6


# ===========================================================================
# FeedbackStore tests (Enhancement 4)
# ===========================================================================

import pytest
from uuid import UUID, uuid4 as _uuid4
from datetime import timezone

from app.intelligence.feedback_store import FeedbackRecord, FeedbackStore


class TestFeedbackRecord:
    """Tests for the FeedbackRecord dataclass."""

    def test_fields_populated(self) -> None:
        sid = _uuid4()
        uid = _uuid4()
        rec = FeedbackRecord(
            signal_id=sid,
            predicted_type="complaint",
            true_type="churn_risk",
            predicted_confidence=0.75,
            user_id=uid,
        )
        assert rec.signal_id == sid
        assert rec.predicted_type == "complaint"
        assert rec.true_type == "churn_risk"
        assert abs(rec.predicted_confidence - 0.75) < 1e-9
        assert rec.user_id == uid

    def test_auto_id_is_uuid(self) -> None:
        rec = FeedbackRecord(
            signal_id=_uuid4(),
            predicted_type="praise",
            true_type="praise",
            predicted_confidence=0.9,
            user_id=_uuid4(),
        )
        assert isinstance(rec.id, UUID)

    def test_created_at_is_timezone_aware(self) -> None:
        rec = FeedbackRecord(
            signal_id=_uuid4(),
            predicted_type="feature_request",
            true_type="feature_request",
            predicted_confidence=0.8,
            user_id=_uuid4(),
        )
        assert rec.created_at.tzinfo is not None


class TestFeedbackStore:
    """Tests for FeedbackStore (in-memory backend)."""

    @pytest.mark.asyncio
    async def test_record_appends_to_memory(self) -> None:
        store = FeedbackStore()
        rec = await store.record(
            signal_id=_uuid4(),
            predicted_type="complaint",
            true_type="churn_risk",
            predicted_confidence=0.7,
            user_id=_uuid4(),
        )
        assert isinstance(rec, FeedbackRecord)
        recent = await store.get_recent(limit=10)
        assert len(recent) == 1
        assert recent[0].predicted_type == "complaint"

    @pytest.mark.asyncio
    async def test_get_recent_respects_limit(self) -> None:
        store = FeedbackStore()
        for _ in range(5):
            await store.record(
                signal_id=_uuid4(),
                predicted_type="praise",
                true_type="praise",
                predicted_confidence=0.9,
                user_id=_uuid4(),
            )
        recent = await store.get_recent(limit=2)
        assert len(recent) == 2

    @pytest.mark.asyncio
    async def test_get_recent_newest_first(self) -> None:
        store = FeedbackStore()
        for label in ["a", "b", "c"]:
            await store.record(
                signal_id=_uuid4(),
                predicted_type=label,
                true_type=label,
                predicted_confidence=0.5,
                user_id=_uuid4(),
            )
        recent = await store.get_recent()
        assert recent[0].predicted_type == "c"

    @pytest.mark.asyncio
    async def test_calibrator_update_called(self, tmp_path: Path) -> None:
        """record() triggers calibrator.update() when calibrator is injected."""
        state_file = tmp_path / "calibration_state.json"
        cal = ConfidenceCalibrator(state_path=state_file)
        store = FeedbackStore(confidence_calibrator=cal)
        before_t = cal._scalars.get("complaint", 1.0)
        await store.record(
            signal_id=_uuid4(),
            predicted_type="complaint",
            true_type="churn_risk",
            predicted_confidence=0.8,
            user_id=_uuid4(),
        )
        after_t = cal._scalars.get("complaint", 1.0)
        assert after_t != before_t


# ============================================================================
# ConfidenceCalibrator tests (Enhancement 2)
# ============================================================================

import math
import tempfile
import json as _json
from pathlib import Path

from app.domain.inference_models import SignalType
from app.intelligence.calibration import ConfidenceCalibrator


class TestConfidenceCalibrator:
    """Unit tests for ``ConfidenceCalibrator`` (per-SignalType temperature scaling)."""

    def _make_calibrator(self, tmp_path: Path, scalars: dict = None) -> ConfidenceCalibrator:
        """Build a calibrator with a temp state file."""
        state_file = tmp_path / "calibration_state.json"
        if scalars is not None:
            state_file.write_text(
                _json.dumps({"version": "1.0", "scalars": scalars}), encoding="utf-8"
            )
        return ConfidenceCalibrator(state_path=state_file)

    def test_calibrate_identity_at_T1(self, tmp_path: Path) -> None:
        """T=1.0 should leave sigmoid(logit) unchanged."""
        cal = self._make_calibrator(tmp_path, {"churn_risk": 1.0})
        raw_prob = 0.7
        logit = math.log(raw_prob / (1 - raw_prob))
        result = cal.calibrate(logit, SignalType.CHURN_RISK)
        assert abs(result - raw_prob) < 1e-6

    def test_calibrate_sharpens_at_low_T(self, tmp_path: Path) -> None:
        """T<1.0 sharpens — a 0.7 probability should become higher."""
        cal = self._make_calibrator(tmp_path, {"churn_risk": 0.5})
        raw_prob = 0.7
        logit = math.log(raw_prob / (1 - raw_prob))
        sharpened = cal.calibrate(logit, SignalType.CHURN_RISK)
        assert sharpened > raw_prob

    def test_calibrate_flattens_at_high_T(self, tmp_path: Path) -> None:
        """T>1.0 flattens — a 0.9 probability should come closer to 0.5."""
        cal = self._make_calibrator(tmp_path, {"churn_risk": 3.0})
        raw_prob = 0.9
        logit = math.log(raw_prob / (1 - raw_prob))
        flattened = cal.calibrate(logit, SignalType.CHURN_RISK)
        assert flattened < raw_prob

    def test_calibrate_missing_type_defaults_T1(self, tmp_path: Path) -> None:
        """An unknown SignalType should fall back to T=1.0 (identity)."""
        cal = self._make_calibrator(tmp_path, {})
        raw_prob = 0.6
        logit = math.log(raw_prob / (1 - raw_prob))
        result = cal.calibrate(logit, SignalType.COMPLAINT)
        assert abs(result - raw_prob) < 1e-6

    def test_update_adjusts_T_on_wrong_prediction(self, tmp_path: Path) -> None:
        """update() should move T when predicted label differs from true label."""
        cal = self._make_calibrator(tmp_path, {"complaint": 1.0})
        initial_t = cal._scalars.get("complaint", 1.0)
        # Predicted 0.9 confidence but true_label=False
        cal.update(SignalType.COMPLAINT, predicted_prob=0.9, true_label=False)
        new_t = cal._scalars.get("complaint", 1.0)
        assert new_t != initial_t

    def test_state_roundtrip(self, tmp_path: Path) -> None:
        """Scalars written by _save() are identical when reloaded by _load()."""
        cal = self._make_calibrator(tmp_path, {"praise": 1.2})
        cal.update(SignalType.PRAISE, predicted_prob=0.8, true_label=True)
        saved_t = cal._scalars["praise"]

        # Reload fresh instance from same file
        cal2 = ConfidenceCalibrator(state_path=tmp_path / "calibration_state.json")
        assert abs(cal2._scalars.get("praise", 0.0) - saved_t) < 1e-9

    def test_update_does_not_go_below_t_min(self, tmp_path: Path) -> None:
        """Temperature must never drop below the minimum (0.1)."""
        cal = self._make_calibrator(tmp_path, {"bug_report": 0.11})
        # Many aggressive updates should hit the floor
        for _ in range(1000):
            cal.update(SignalType.BUG_REPORT, predicted_prob=0.9, true_label=False)
        assert cal._scalars["bug_report"] >= 0.1


# ============================================================================
# FeedbackStore tests (Enhancement 4)
# ============================================================================

import pytest as _pytest
from uuid import uuid4 as _uuid4
from app.intelligence.feedback_store import FeedbackRecord, FeedbackStore


class TestFeedbackRecord:
    """Tests for the ``FeedbackRecord`` dataclass."""

    def test_fields_populated(self) -> None:
        """All required fields are stored correctly."""
        sid = _uuid4()
        uid = _uuid4()
        rec = FeedbackRecord(
            signal_id=sid,
            predicted_type="complaint",
            true_type="churn_risk",
            predicted_confidence=0.75,
            user_id=uid,
        )
        assert rec.signal_id == sid
        assert rec.predicted_type == "complaint"
        assert rec.true_type == "churn_risk"
        assert abs(rec.predicted_confidence - 0.75) < 1e-9
        assert rec.user_id == uid

    def test_id_is_uuid(self) -> None:
        """Auto-generated id is a UUID."""
        rec = FeedbackRecord(
            signal_id=_uuid4(),
            predicted_type="praise",
            true_type="praise",
            predicted_confidence=0.9,
            user_id=_uuid4(),
        )
        from uuid import UUID
        assert isinstance(rec.id, UUID)

    def test_created_at_is_utc(self) -> None:
        """created_at is timezone-aware."""
        from datetime import timezone
        rec = FeedbackRecord(
            signal_id=_uuid4(),
            predicted_type="feature_request",
            true_type="feature_request",
            predicted_confidence=0.8,
            user_id=_uuid4(),
        )
        assert rec.created_at.tzinfo is not None


class TestFeedbackStore:
    """Tests for ``FeedbackStore`` (in-memory backend)."""

    @_pytest.mark.asyncio
    async def test_record_appends_to_memory(self) -> None:
        """record() persists a FeedbackRecord in the in-memory list."""
        store = FeedbackStore()
        rec = await store.record(
            signal_id=_uuid4(),
            predicted_type="complaint",
            true_type="churn_risk",
            predicted_confidence=0.7,
            user_id=_uuid4(),
        )
        assert isinstance(rec, FeedbackRecord)
        recent = await store.get_recent(limit=10)
        assert len(recent) == 1
        assert recent[0].predicted_type == "complaint"

    @_pytest.mark.asyncio
    async def test_get_recent_respects_limit(self) -> None:
        """get_recent(limit=N) returns at most N records."""
        store = FeedbackStore()
        for _ in range(5):
            await store.record(
                signal_id=_uuid4(),
                predicted_type="praise",
                true_type="praise",
                predicted_confidence=0.9,
                user_id=_uuid4(),
            )
        recent = await store.get_recent(limit=2)
        assert len(recent) == 2

    @_pytest.mark.asyncio
    async def test_get_recent_newest_first(self) -> None:
        """get_recent() returns records ordered newest first."""
        store = FeedbackStore()
        for label in ["a", "b", "c"]:
            await store.record(
                signal_id=_uuid4(),
                predicted_type=label,
                true_type=label,
                predicted_confidence=0.5,
                user_id=_uuid4(),
            )
        recent = await store.get_recent()
        # Newest is last inserted → first in result
        assert recent[0].predicted_type == "c"

    @_pytest.mark.asyncio
    async def test_calibrator_update_called(self, tmp_path: Path) -> None:
        """record() calls calibrator.update() when a calibrator is injected."""
        state_file = tmp_path / "calibration_state.json"
        cal = ConfidenceCalibrator(state_path=state_file)
        store = FeedbackStore(confidence_calibrator=cal)
        before_t = cal._scalars.get("complaint", 1.0)
        await store.record(
            signal_id=_uuid4(),
            predicted_type="complaint",
            true_type="churn_risk",  # wrong → true_label=False
            predicted_confidence=0.8,
            user_id=_uuid4(),
        )
        after_t = cal._scalars.get("complaint", 1.0)
        assert after_t != before_t

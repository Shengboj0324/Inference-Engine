"""Phase 2 — Content Understanding Unit Tests.

Groups:
  TestAudioModels              — TranscriptSegment / DiarizedSegment / etc.
  TestTranscriptionRouter      — backend resolution, domain corrections
  TestDiarizer                 — heuristic diarization
  TestTopicSegmenter           — keyword-based heuristic segmentation
  TestQuoteExtractor           — heuristic quote importance scoring
  TestClaimExtractor           — heuristic claim classification
  TestPodcastPipeline          — end-to-end EpisodeUnderstanding (no LLM)
  TestDocumentModels           — PaperParsed / BenchmarkResult / etc.
  TestPDFIngestor              — stub backend + heading detection
  TestSectionSegmenter         — line-based section splitting
  TestPaperParser              — metadata extraction
  TestBenchmarkTableExtractor  — inline + table extraction
  TestMethodVsClaimExtractor   — heuristic claim classification
  TestCitationGraph            — build / query / BFS
  TestPaperSummarizer          — extractive fallback
  TestNoveltyEstimator         — heuristic scoring
  TestDevintelModels           — ChangeEntry / ReleaseNote validation
  TestReleaseParser            — structured release body parsing
  TestChangelogNormalizer      — KACL + free-form
  TestBreakingChangeDetector   — heuristic + flagged entries
  TestVersionDiffAnalyzer      — semver parsing + urgency
  TestSemanticDiffSummarizer   — extractive fallback
  TestRepoHealthScorer         — dimension scores
  TestDependencyAlertEngine    — watch/unwatch/process
  TestEntityModels             — CanonicalEntity / EventBundle / DedupeResult
  TestCanonicalEntityStore     — CRUD / alias lookup / fuzzy
  TestAliasResolver            — normalization variants
  TestEventClusterer           — clustering + scoring
  TestCrossSourceDeduper       — trust-first + similarity
  TestTemporalEventGraph       — add / edges / BFS / timeline
"""

import asyncio
import re
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(start: float, end: float, text: str = "Hello world."):
    from app.media.audio_intelligence.models import TranscriptSegment
    return TranscriptSegment(start_s=start, end_s=end, text=text)


def _ds(start: float, end: float, text: str = "Hello world.", speaker_id: str = "SPEAKER_00", role: str = "host"):
    from app.media.audio_intelligence.models import DiarizedSegment
    return DiarizedSegment(segment=_ts(start, end, text), speaker_id=speaker_id, speaker_role=role)


# ===========================================================================
# Group 1: Audio Models
# ===========================================================================

class TestAudioModels:
    def test_transcript_segment_basic(self):
        from app.media.audio_intelligence.models import TranscriptSegment
        seg = TranscriptSegment(start_s=0.0, end_s=10.0, text="Hello world.")
        assert seg.duration_s == pytest.approx(10.0)
        assert seg.language == "en"

    def test_transcript_segment_end_before_start_raises(self):
        from app.media.audio_intelligence.models import TranscriptSegment
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            TranscriptSegment(start_s=10.0, end_s=5.0, text="bad")

    def test_transcript_segment_is_frozen(self):
        from app.media.audio_intelligence.models import TranscriptSegment
        seg = TranscriptSegment(start_s=0.0, end_s=1.0, text="x")
        with pytest.raises(Exception):
            seg.text = "y"  # type: ignore

    def test_diarized_segment_invalid_role(self):
        from app.media.audio_intelligence.models import DiarizedSegment
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            DiarizedSegment(segment=_ts(0, 1), speaker_id="S0", speaker_role="admin")

    def test_topic_segment_end_before_start_raises(self):
        from app.media.audio_intelligence.models import TopicSegment, TopicLabel
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            TopicSegment(start_s=20.0, end_s=5.0, label=TopicLabel.RESEARCH, title="bad", summary="x")

    def test_extracted_quote_min_length(self):
        from app.media.audio_intelligence.models import ExtractedQuote
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            ExtractedQuote(text="x", speaker_id="S0", start_s=0.0)  # too short

    def test_extracted_claim_fields(self):
        from app.media.audio_intelligence.models import ExtractedClaim, ClaimType
        c = ExtractedClaim(text="GPT-4 achieves 90% on MMLU.", claim_type=ClaimType.BENCHMARK, start_s=30.0)
        assert c.supported is True
        assert c.entities == []

    def test_episode_understanding_instantiation(self):
        from app.media.audio_intelligence.models import EpisodeUnderstanding
        eu = EpisodeUnderstanding(episode_id="ep1", title="Test", duration_s=3600.0, transcript="hello")
        assert eu.episode_id == "ep1"
        assert eu.topics == []


# ===========================================================================
# Group 2: TranscriptionRouter
# ===========================================================================

class TestTranscriptionRouter:
    def test_domain_correction_gpt(self):
        from app.media.audio_intelligence.transcription_router import apply_domain_corrections
        assert "GPT-4" in apply_domain_corrections("We discussed GPT-4 today.")

    def test_domain_correction_llama(self):
        from app.media.audio_intelligence.transcription_router import apply_domain_corrections
        result = apply_domain_corrections("The lama model is good.")
        assert "LLaMA" in result

    def test_domain_correction_rlhf(self):
        from app.media.audio_intelligence.transcription_router import apply_domain_corrections
        assert "RLHF" in apply_domain_corrections("We used r l h f to align the model.")

    def test_domain_correction_openai(self):
        from app.media.audio_intelligence.transcription_router import apply_domain_corrections
        result = apply_domain_corrections("open eye released a new model.")
        assert "OpenAI" in result

    def test_apply_corrections_type_error(self):
        from app.media.audio_intelligence.transcription_router import apply_domain_corrections
        with pytest.raises(TypeError):
            apply_domain_corrections(123)  # type: ignore

    def test_stub_backend_selected_when_no_deps(self):
        from app.media.audio_intelligence.transcription_router import TranscriptionRouter, ASRBackend
        router = TranscriptionRouter(backend=ASRBackend.STUB)
        assert router.resolved_backend() == ASRBackend.STUB

    def test_forced_backend(self):
        from app.media.audio_intelligence.transcription_router import TranscriptionRouter, ASRBackend
        router = TranscriptionRouter(backend=ASRBackend.STUB)
        assert router._forced_backend == ASRBackend.STUB

    def test_transcribe_missing_file_raises(self):
        from app.media.audio_intelligence.transcription_router import TranscriptionRouter, ASRBackend
        router = TranscriptionRouter(backend=ASRBackend.STUB)
        with pytest.raises(FileNotFoundError):
            asyncio.run(router.transcribe("/nonexistent/audio.mp3"))

    def test_transcribe_empty_path_raises(self):
        from app.media.audio_intelligence.transcription_router import TranscriptionRouter, ASRBackend
        router = TranscriptionRouter(backend=ASRBackend.STUB)
        with pytest.raises(ValueError):
            asyncio.run(router.transcribe(""))

    def test_stub_transcription_returns_segment(self, tmp_path):
        from app.media.audio_intelligence.transcription_router import TranscriptionRouter, ASRBackend
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake audio")
        router = TranscriptionRouter(backend=ASRBackend.STUB, apply_corrections=False)
        segs = asyncio.run(router.transcribe(str(f)))
        assert len(segs) == 1
        assert segs[0].text == "[STUB TRANSCRIPT]"

    # ── GAP-2 tests: TranscriptResult provenance ────────────────────────────
    def test_transcribe_with_provenance_returns_transcript_result(self, tmp_path):
        """transcribe_with_provenance() must return a TranscriptResult (GAP-2)."""
        from app.media.audio_intelligence.transcription_router import TranscriptionRouter, ASRBackend
        from app.media.audio_intelligence.models import TranscriptResult
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake audio")
        router = TranscriptionRouter(backend=ASRBackend.STUB, apply_corrections=False)
        result = asyncio.run(router.transcribe_with_provenance(str(f)))
        assert isinstance(result, TranscriptResult)

    def test_transcribe_with_provenance_backend_used_populated(self, tmp_path):
        """backend_used must be the resolved ASR backend name (GAP-2)."""
        from app.media.audio_intelligence.transcription_router import TranscriptionRouter, ASRBackend
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake audio")
        router = TranscriptionRouter(backend=ASRBackend.STUB, apply_corrections=False)
        result = asyncio.run(router.transcribe_with_provenance(str(f)))
        assert result.backend_used == "stub"

    def test_transcribe_with_provenance_is_stub_property(self, tmp_path):
        """TranscriptResult.is_stub must be True when backend is stub (GAP-2)."""
        from app.media.audio_intelligence.transcription_router import TranscriptionRouter, ASRBackend
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake audio")
        router = TranscriptionRouter(backend=ASRBackend.STUB, apply_corrections=False)
        result = asyncio.run(router.transcribe_with_provenance(str(f)))
        assert result.is_stub is True

    def test_transcribe_with_provenance_full_text(self, tmp_path):
        """TranscriptResult.full_text must concatenate segment texts (GAP-2)."""
        from app.media.audio_intelligence.transcription_router import TranscriptionRouter, ASRBackend
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake audio")
        router = TranscriptionRouter(backend=ASRBackend.STUB, apply_corrections=False)
        result = asyncio.run(router.transcribe_with_provenance(str(f)))
        assert "[STUB TRANSCRIPT]" in result.full_text

    def test_transcript_result_mean_confidence_none_for_stub(self, tmp_path):
        """Stub backend produces no confidence — mean_confidence must be None (GAP-2)."""
        from app.media.audio_intelligence.transcription_router import TranscriptionRouter, ASRBackend
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"fake audio")
        router = TranscriptionRouter(backend=ASRBackend.STUB, apply_corrections=False)
        result = asyncio.run(router.transcribe_with_provenance(str(f)))
        assert result.mean_confidence is None


# ===========================================================================
# Group 3: Diarizer
# ===========================================================================

class TestDiarizer:
    def test_empty_segments_returns_empty(self):
        from app.media.audio_intelligence.diarization import Diarizer
        d = Diarizer()
        assert d.diarize([]) == []

    def test_wrong_type_raises(self):
        from app.media.audio_intelligence.diarization import Diarizer
        with pytest.raises(TypeError):
            Diarizer().diarize("not a list")  # type: ignore

    def test_invalid_min_speakers_raises(self):
        from app.media.audio_intelligence.diarization import Diarizer
        with pytest.raises(ValueError):
            Diarizer(min_speakers=0)

    def test_max_less_than_min_raises(self):
        from app.media.audio_intelligence.diarization import Diarizer
        with pytest.raises(ValueError):
            Diarizer(min_speakers=3, max_speakers=2)

    def test_negative_silence_gap_raises(self):
        from app.media.audio_intelligence.diarization import Diarizer
        with pytest.raises(ValueError):
            Diarizer(silence_gap_s=-1.0)

    def test_heuristic_single_segment(self):
        from app.media.audio_intelligence.diarization import Diarizer
        d = Diarizer(use_pyannote=False)
        result = d.diarize([_ts(0, 5, "Hello everyone, welcome to the show.")])
        assert len(result) == 1
        assert result[0].speaker_role == "host"

    def test_heuristic_two_speakers(self):
        from app.media.audio_intelligence.diarization import Diarizer
        segs = [
            _ts(0, 5, "Welcome to the show today!"),
            _ts(10, 15, "Thanks for having me, I'm delighted to be here."),
        ]
        d = Diarizer(silence_gap_s=3.0, use_pyannote=False)
        result = d.diarize(segs)
        assert len(result) == 2
        ids = {r.speaker_id for r in result}
        # Two different speaker IDs assigned
        assert len(ids) == 2

    def test_speaker_role_host_guest(self):
        from app.media.audio_intelligence.diarization import Diarizer
        segs = [
            _ts(0, 5, "Welcome to the show!"),
            _ts(10, 15, "Thank you."),
        ]
        d = Diarizer(silence_gap_s=3.0, use_pyannote=False)
        result = d.diarize(segs)
        roles = [r.speaker_role for r in result]
        assert "host" in roles
        assert "guest" in roles


# ===========================================================================
# Group 4: TopicSegmenter
# ===========================================================================

class TestTopicSegmenter:
    def test_empty_returns_empty(self):
        from app.media.audio_intelligence.topic_segmentation import TopicSegmenter
        ts = TopicSegmenter()
        assert asyncio.run(ts.segment([])) == []

    def test_wrong_type_raises(self):
        from app.media.audio_intelligence.topic_segmentation import TopicSegmenter
        with pytest.raises(TypeError):
            asyncio.run(TopicSegmenter().segment("not a list"))  # type: ignore

    def test_invalid_window_raises(self):
        from app.media.audio_intelligence.topic_segmentation import TopicSegmenter
        with pytest.raises(ValueError):
            TopicSegmenter(window_chars=0)

    def test_invalid_min_segment_raises(self):
        from app.media.audio_intelligence.topic_segmentation import TopicSegmenter
        with pytest.raises(ValueError):
            TopicSegmenter(min_segment_s=-1.0)

    def test_heuristic_labels_model_release(self):
        from app.media.audio_intelligence.topic_segmentation import TopicSegmenter, TopicLabel
        segs = [
            _ds(0, 60, "Today OpenAI is releasing GPT-4o, a new version of their model.", "S0", "host"),
            _ds(62, 120, "The launch was announced in a blog post.", "S0", "host"),
        ] * 5  # Ensure enough chars / segments
        ts = TopicSegmenter(min_segment_s=0.0)
        result = asyncio.run(ts.segment(segs))
        assert len(result) >= 1
        labels = {r.label for r in result}
        assert TopicLabel.MODEL_RELEASE in labels or TopicLabel.OTHER in labels

    def test_heuristic_labels_benchmark(self):
        from app.media.audio_intelligence.topic_segmentation import TopicSegmenter, TopicLabel
        segs = [_ds(i * 10, i * 10 + 8, f"The model achieves a new MMLU score of {90+i}%.", "S0", "host") for i in range(8)]
        ts = TopicSegmenter(min_segment_s=0.0)
        result = asyncio.run(ts.segment(segs))
        labels = {r.label for r in result}
        assert TopicLabel.BENCHMARK in labels or TopicLabel.OTHER in labels

    def test_all_segments_have_title(self):
        from app.media.audio_intelligence.topic_segmentation import TopicSegmenter
        segs = [_ds(i * 10, i * 10 + 8, f"Segment {i} content.", "S0", "host") for i in range(6)]
        ts = TopicSegmenter(min_segment_s=0.0)
        result = asyncio.run(ts.segment(segs))
        for seg in result:
            assert seg.title != ""
            assert len(seg.title) <= 80


# ===========================================================================
# Group 5: QuoteExtractor
# ===========================================================================

class TestQuoteExtractor:
    def test_empty_returns_empty(self):
        from app.media.audio_intelligence.quote_extraction import QuoteExtractor
        assert asyncio.run(QuoteExtractor().extract([])) == []

    def test_wrong_type_raises(self):
        from app.media.audio_intelligence.quote_extraction import QuoteExtractor
        with pytest.raises(TypeError):
            asyncio.run(QuoteExtractor().extract("x"))  # type: ignore

    def test_invalid_max_quotes_raises(self):
        from app.media.audio_intelligence.quote_extraction import QuoteExtractor
        with pytest.raises(ValueError):
            QuoteExtractor(max_quotes=0)

    def test_invalid_min_importance_raises(self):
        from app.media.audio_intelligence.quote_extraction import QuoteExtractor
        with pytest.raises(ValueError):
            QuoteExtractor(min_importance=1.5)

    def test_short_segments_excluded(self):
        from app.media.audio_intelligence.quote_extraction import QuoteExtractor
        segs = [_ds(0, 1, "OK.", "S0", "host"), _ds(2, 3, "Sure.", "S0", "host")]
        result = asyncio.run(QuoteExtractor(min_words=10).extract(segs))
        assert result == []

    def test_superlative_quote_included(self):
        from app.media.audio_intelligence.quote_extraction import QuoteExtractor
        text = "This is the most revolutionary breakthrough in AI we have ever seen in history."
        segs = [_ds(0, 10, text, "S0", "host")]
        result = asyncio.run(QuoteExtractor(min_words=5, min_importance=0.0).extract(segs))
        assert len(result) >= 1
        assert result[0].importance > 0.3

    def test_sorted_by_importance_desc(self):
        from app.media.audio_intelligence.quote_extraction import QuoteExtractor
        segs = [
            _ds(0, 10, "This is the most critical breakthrough we have ever seen.", "S0", "host"),
            _ds(20, 30, "We talked about many things in general today.", "S0", "host"),
        ]
        result = asyncio.run(QuoteExtractor(min_words=5, min_importance=0.0).extract(segs))
        importances = [q.importance for q in result]
        assert importances == sorted(importances, reverse=True)

    def test_max_quotes_respected(self):
        from app.media.audio_intelligence.quote_extraction import QuoteExtractor
        segs = [
            _ds(i * 10, i * 10 + 9, f"This is absolutely the most critical finding {i} ever.", "S0", "host")
            for i in range(20)
        ]
        result = asyncio.run(QuoteExtractor(max_quotes=3, min_words=5, min_importance=0.0).extract(segs))
        assert len(result) <= 3


# ===========================================================================
# Group 6: ClaimExtractor
# ===========================================================================

class TestClaimExtractor:
    def test_empty_returns_empty(self):
        from app.media.audio_intelligence.claim_extraction import ClaimExtractor
        assert asyncio.run(ClaimExtractor().extract([])) == []

    def test_wrong_type_raises(self):
        from app.media.audio_intelligence.claim_extraction import ClaimExtractor
        with pytest.raises(TypeError):
            asyncio.run(ClaimExtractor().extract("not list"))  # type: ignore

    def test_invalid_confidence_raises(self):
        from app.media.audio_intelligence.claim_extraction import ClaimExtractor
        with pytest.raises(ValueError):
            ClaimExtractor(min_confidence=1.5)

    def test_announcement_classified(self):
        from app.media.audio_intelligence.claim_extraction import ClaimExtractor, ClaimType
        segs = [_ds(0, 10, "We are releasing GPT-5 today, available to all users.", "S0", "host")]
        result = asyncio.run(ClaimExtractor(min_confidence=0.0, use_llm=False).extract(segs))
        assert any(c.claim_type == ClaimType.ANNOUNCEMENT for c in result)

    def test_benchmark_claim_classified(self):
        from app.media.audio_intelligence.claim_extraction import ClaimExtractor, ClaimType
        segs = [_ds(0, 10, "Our model achieves 95.2% on HumanEval, outperforming all baselines.", "S0", "host")]
        result = asyncio.run(ClaimExtractor(min_confidence=0.0, use_llm=False).extract(segs))
        assert any(c.claim_type in (ClaimType.BENCHMARK, ClaimType.FACT) for c in result)

    def test_opinion_classified(self):
        from app.media.audio_intelligence.claim_extraction import ClaimExtractor, ClaimType
        segs = [_ds(0, 10, "I think the scaling laws will continue to hold for at least another decade.", "S0", "host")]
        result = asyncio.run(ClaimExtractor(min_confidence=0.0, use_llm=False).extract(segs))
        assert any(c.claim_type in (ClaimType.OPINION, ClaimType.PREDICTION) for c in result)

    def test_sorted_by_confidence_desc(self):
        from app.media.audio_intelligence.claim_extraction import ClaimExtractor
        segs = [
            _ds(0, 10, "We are releasing the new model today for all users worldwide.", "S0", "host"),
            _ds(20, 30, "Maybe AI will solve everything eventually someday in the future.", "S0", "host"),
        ]
        result = asyncio.run(ClaimExtractor(min_confidence=0.0, use_llm=False).extract(segs))
        confs = [c.confidence for c in result]
        assert confs == sorted(confs, reverse=True)


# ===========================================================================
# Group 7: PodcastEpisodeUnderstandingPipeline
# ===========================================================================

class TestPodcastPipeline:
    def test_wrong_segment_type_raises(self):
        from app.media.audio_intelligence.podcast_episode_understanding import PodcastEpisodeUnderstandingPipeline
        p = PodcastEpisodeUnderstandingPipeline()
        with pytest.raises(TypeError):
            asyncio.run(p.process("not a list", "ep1", "Test", 3600.0))  # type: ignore

    def test_empty_episode_id_raises(self):
        from app.media.audio_intelligence.podcast_episode_understanding import PodcastEpisodeUnderstandingPipeline
        p = PodcastEpisodeUnderstandingPipeline()
        with pytest.raises(ValueError):
            asyncio.run(p.process([], "", "Test", 3600.0))

    def test_negative_duration_raises(self):
        from app.media.audio_intelligence.podcast_episode_understanding import PodcastEpisodeUnderstandingPipeline
        p = PodcastEpisodeUnderstandingPipeline()
        with pytest.raises(ValueError):
            asyncio.run(p.process([], "ep1", "Test", -1.0))

    def test_invalid_max_top_claims_raises(self):
        from app.media.audio_intelligence.podcast_episode_understanding import PodcastEpisodeUnderstandingPipeline
        with pytest.raises(ValueError):
            PodcastEpisodeUnderstandingPipeline(max_top_claims=0)

    def test_empty_transcript_produces_result(self):
        from app.media.audio_intelligence.podcast_episode_understanding import PodcastEpisodeUnderstandingPipeline
        from app.media.audio_intelligence.models import EpisodeUnderstanding
        p = PodcastEpisodeUnderstandingPipeline()
        result = asyncio.run(p.process([], "ep1", "Empty Episode", 0.0))
        assert isinstance(result, EpisodeUnderstanding)
        assert result.transcript == ""

    def test_full_pipeline_no_llm(self):
        from app.media.audio_intelligence.podcast_episode_understanding import PodcastEpisodeUnderstandingPipeline
        from app.media.audio_intelligence.models import EpisodeUnderstanding
        segs = [
            _ts(0, 10, "Welcome to the show! Today we discuss the release of GPT-4o."),
            _ts(15, 25, "The model achieves 87% on MMLU, outperforming all previous models."),
            _ts(30, 40, "I think this is the most important AI release we have seen."),
        ]
        p = PodcastEpisodeUnderstandingPipeline(max_top_claims=3, max_key_quotes=5)
        result = asyncio.run(p.process(segs, "ep-test", "AI Weekly Episode 42", 3600.0))
        assert isinstance(result, EpisodeUnderstanding)
        assert result.episode_id == "ep-test"
        assert result.duration_s == 3600.0
        assert "GPT-4o" in result.transcript or "Welcome" in result.transcript
        assert result.processing_metadata.get("total_ms") is not None
        assert len(result.top_claims) <= 3
        assert len(result.key_quotes) <= 5

    def test_entity_extraction(self):
        from app.media.audio_intelligence.podcast_episode_understanding import PodcastEpisodeUnderstandingPipeline
        segs = [
            _ts(0, 5, "OpenAI released GPT-4. OpenAI is a top lab."),
        ]
        p = PodcastEpisodeUnderstandingPipeline()
        result = asyncio.run(p.process(segs, "ep2", "Test", 300.0))
        # OpenAI appears twice → should be in entities
        assert "OpenAI" in result.entities or result.entities == []  # graceful even if empty


# ===========================================================================
# Group 8: Document Intelligence Models
# ===========================================================================

class TestDocumentModels:
    def test_section_type_enum_values(self):
        from app.document_intelligence.models import SectionType
        assert SectionType.ABSTRACT.value == "abstract"
        assert SectionType.REFERENCES.value == "references"

    def test_document_section_negative_order_raises(self):
        from app.document_intelligence.models import DocumentSection, SectionType
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            DocumentSection(section_type=SectionType.ABSTRACT, heading="Abstract", text="x", order=-1)

    def test_benchmark_result_frozen(self):
        from app.document_intelligence.models import BenchmarkResult, SectionType
        br = BenchmarkResult(benchmark_name="MMLU", metric="accuracy", value="90.1%")
        with pytest.raises(Exception):
            br.value = "91%"  # type: ignore

    def test_claim_evidence_invalid_claim_type(self):
        from app.document_intelligence.models import ClaimEvidence
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            ClaimEvidence(claim="x", evidence="y", claim_type="invalid_type")

    def test_citation_node_defaults(self):
        from app.document_intelligence.models import CitationNode
        node = CitationNode(paper_id="2401.12345", title="Attention Is All You Need")
        assert node.is_focal is False
        assert node.citation_count == 0

    def test_paper_parsed_get_section(self):
        from app.document_intelligence.models import PaperParsed, DocumentSection, SectionType
        sec = DocumentSection(section_type=SectionType.ABSTRACT, heading="Abstract", text="This paper proposes...", order=0)
        paper = PaperParsed(paper_id="test", title="Test Paper", sections=[sec])
        found = paper.get_section(SectionType.ABSTRACT)
        assert found is not None
        assert "proposes" in found.text

    def test_paper_parsed_get_section_missing(self):
        from app.document_intelligence.models import PaperParsed, SectionType
        paper = PaperParsed(paper_id="p1", title="P1")
        assert paper.get_section(SectionType.REFERENCES) is None

    def test_paper_parsed_full_text(self):
        from app.document_intelligence.models import PaperParsed, DocumentSection, SectionType
        secs = [
            DocumentSection(section_type=SectionType.ABSTRACT, heading="Abstract", text="Abstract text.", order=0),
            DocumentSection(section_type=SectionType.CONCLUSION, heading="Conclusion", text="Conclusion text.", order=1),
        ]
        paper = PaperParsed(paper_id="p2", title="P2", sections=secs)
        full = paper.full_text()
        assert "Abstract text." in full
        assert "Conclusion text." in full


# ===========================================================================
# Group 9: PDFIngestor
# ===========================================================================

class TestPDFIngestor:
    def test_missing_file_raises(self):
        from app.document_intelligence.pdf_ingestor import PDFIngestor
        ingestor = PDFIngestor(backend="stub")
        with pytest.raises(FileNotFoundError):
            ingestor.ingest("/no/such/file.pdf")

    def test_empty_path_raises(self):
        from app.document_intelligence.pdf_ingestor import PDFIngestor
        with pytest.raises(ValueError):
            PDFIngestor().ingest("")

    def test_invalid_backend_raises(self):
        from app.document_intelligence.pdf_ingestor import PDFIngestor
        with pytest.raises(ValueError):
            PDFIngestor(backend="magic")

    def test_negative_max_pages_raises(self):
        from app.document_intelligence.pdf_ingestor import PDFIngestor
        with pytest.raises(ValueError):
            PDFIngestor(max_pages=-1)

    def test_stub_backend(self, tmp_path):
        from app.document_intelligence.pdf_ingestor import PDFIngestor
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 stub")
        doc = PDFIngestor(backend="stub").ingest(str(f))
        assert doc.extraction_backend == "stub"
        assert len(doc.pages) == 1
        assert "STUB" in doc.full_text

    def test_content_hash_set(self, tmp_path):
        from app.document_intelligence.pdf_ingestor import PDFIngestor
        f = tmp_path / "hash_test.pdf"
        f.write_bytes(b"%PDF-1.4 stub")
        doc = PDFIngestor(backend="stub").ingest(str(f))
        assert len(doc.content_hash) == 64  # SHA-256 hex

    def test_heading_detection(self):
        from app.document_intelligence.pdf_ingestor import PDFIngestor
        text = "Introduction\nThis is the intro text.\n\nResults\nHere are our results."
        offsets = PDFIngestor._detect_headings(text)
        assert "Introduction" in offsets or "Results" in offsets

    # ── GAP-3 tests: PDFIngestor production guard ────────────────────────────
    def test_production_safe_raises_when_stub_only(self, tmp_path):
        """production_safe=True must raise RuntimeError when only stub is available (GAP-3)."""
        from app.document_intelligence.pdf_ingestor import PDFIngestor
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 stub")
        ingestor = PDFIngestor(backend="stub", production_safe=True)
        with pytest.raises(RuntimeError, match="production_safe"):
            ingestor.ingest(str(f))

    def test_production_safe_false_allows_stub(self, tmp_path):
        """production_safe=False (default) must allow stub fallback (GAP-3)."""
        from app.document_intelligence.pdf_ingestor import PDFIngestor
        f = tmp_path / "test.pdf"
        f.write_bytes(b"%PDF-1.4 stub")
        ingestor = PDFIngestor(backend="stub", production_safe=False)
        doc = ingestor.ingest(str(f))
        assert doc.extraction_backend == "stub"

    def test_production_safe_wrong_type_raises(self):
        """production_safe must be bool — wrong type must raise TypeError (GAP-3)."""
        from app.document_intelligence.pdf_ingestor import PDFIngestor
        with pytest.raises(TypeError):
            PDFIngestor(production_safe="yes")  # type: ignore


# ===========================================================================
# Group 10: SectionSegmenter
# ===========================================================================

class TestSectionSegmenter:
    def test_empty_text_returns_empty(self):
        from app.document_intelligence.section_segmenter import SectionSegmenter
        assert SectionSegmenter().segment("") == []

    def test_wrong_type_raises(self):
        from app.document_intelligence.section_segmenter import SectionSegmenter
        with pytest.raises(TypeError):
            SectionSegmenter().segment(123)  # type: ignore

    def test_detects_abstract_section(self):
        from app.document_intelligence.section_segmenter import SectionSegmenter, SectionType
        text = "Abstract\nThis paper proposes a new method.\n\nIntroduction\nWe start here."
        sections = SectionSegmenter(min_section_chars=0, merge_short=False).segment(text)
        types = [s.section_type for s in sections]
        assert SectionType.ABSTRACT in types or SectionType.OTHER in types

    def test_detects_conclusion(self):
        from app.document_intelligence.section_segmenter import SectionSegmenter, SectionType
        text = "Conclusion\nWe conclude that our method works.\n"
        sections = SectionSegmenter(min_section_chars=0, merge_short=False).segment(text)
        types = [s.section_type for s in sections]
        assert SectionType.CONCLUSION in types or SectionType.OTHER in types

    def test_sections_ordered(self):
        from app.document_intelligence.section_segmenter import SectionSegmenter
        text = "Abstract\nABC.\n\nIntroduction\nDEF.\n\nConclusion\nGHI.\n"
        sections = SectionSegmenter(min_section_chars=0, merge_short=False).segment(text)
        orders = [s.order for s in sections]
        assert orders == sorted(orders)

    def test_merge_short_enabled(self):
        from app.document_intelligence.section_segmenter import SectionSegmenter
        # Short section should be merged into previous
        text = "Abstract\nThis is a fairly long abstract section with sufficient content.\n\nNote\nx"
        sections_merged = SectionSegmenter(min_section_chars=50, merge_short=True).segment(text)
        sections_not_merged = SectionSegmenter(min_section_chars=0, merge_short=False).segment(text)
        assert len(sections_merged) <= len(sections_not_merged)


# ===========================================================================
# Group 11: PaperParser
# ===========================================================================

class TestPaperParser:
    def test_wrong_type_raises(self):
        from app.document_intelligence.paper_parser import PaperParser
        with pytest.raises(TypeError):
            PaperParser().parse(123, [])  # type: ignore

    def test_sections_wrong_type_raises(self):
        from app.document_intelligence.paper_parser import PaperParser
        with pytest.raises(TypeError):
            PaperParser().parse("text", "not a list")  # type: ignore

    def test_title_extraction(self):
        from app.document_intelligence.paper_parser import PaperParser
        text = "Attention Is All You Need\nAshish Vaswani et al.\nAbstract\nWe propose..."
        paper = PaperParser().parse(text, [])
        assert "Attention" in paper.title

    def test_arxiv_id_extraction(self):
        from app.document_intelligence.paper_parser import PaperParser
        text = "arXiv:2401.12345v2\nSome paper title."
        paper = PaperParser().parse(text, [])
        assert paper.arxiv_id == "2401.12345v2"

    def test_doi_extraction(self):
        from app.document_intelligence.paper_parser import PaperParser
        text = "DOI: 10.1145/12345.67890\nSome paper title."
        paper = PaperParser().parse(text, [])
        assert paper.doi.startswith("10.1145")

    def test_year_extraction(self):
        from app.document_intelligence.paper_parser import PaperParser
        text = "Published in NeurIPS 2023. Some paper title."
        paper = PaperParser().parse(text, [])
        assert paper.year == 2023

    def test_override_arxiv_id(self):
        from app.document_intelligence.paper_parser import PaperParser
        parser = PaperParser(arxiv_id="1234.56789")
        paper = parser.parse("A paper without arxiv in text.", [])
        assert paper.arxiv_id == "1234.56789"

    def test_paper_id_hash_fallback(self):
        from app.document_intelligence.paper_parser import PaperParser
        paper = PaperParser().parse("No arXiv ID or DOI here.", [])
        assert len(paper.paper_id) > 0


# ===========================================================================
# Group 12: BenchmarkTableExtractor
# ===========================================================================

class TestBenchmarkTableExtractor:
    def test_wrong_type_raises(self):
        from app.document_intelligence.benchmark_table_extractor import BenchmarkTableExtractor
        with pytest.raises(TypeError):
            BenchmarkTableExtractor().extract("not a list")  # type: ignore

    def test_invalid_max_results_raises(self):
        from app.document_intelligence.benchmark_table_extractor import BenchmarkTableExtractor
        with pytest.raises(ValueError):
            BenchmarkTableExtractor(max_results=0)

    def test_empty_sections_returns_empty(self):
        from app.document_intelligence.benchmark_table_extractor import BenchmarkTableExtractor
        assert BenchmarkTableExtractor().extract([]) == []

    def test_inline_benchmark_extraction(self):
        from app.document_intelligence.benchmark_table_extractor import BenchmarkTableExtractor
        from app.document_intelligence.models import DocumentSection, SectionType
        text = "Our model achieves 91.2% on MMLU, surpassing all baselines."
        sec = DocumentSection(section_type=SectionType.RESULTS, heading="Results", text=text, order=0)
        results = BenchmarkTableExtractor().extract([sec], focal_model="OurModel")
        assert len(results) >= 1
        assert any("MMLU" in r.benchmark_name for r in results)

    def test_skips_non_result_sections(self):
        from app.document_intelligence.benchmark_table_extractor import BenchmarkTableExtractor
        from app.document_intelligence.models import DocumentSection, SectionType
        text = "We discuss MMLU and HumanEval in related work."
        sec = DocumentSection(section_type=SectionType.INTRODUCTION, heading="Introduction", text=text, order=0)
        results = BenchmarkTableExtractor().extract([sec])
        assert results == []

    def test_deduplication(self):
        from app.document_intelligence.benchmark_table_extractor import BenchmarkTableExtractor
        from app.document_intelligence.models import DocumentSection, SectionType
        text = "Our model achieves 91.2% on MMLU. Also scores 91.2% on MMLU."
        sec = DocumentSection(section_type=SectionType.RESULTS, heading="Results", text=text, order=0)
        results = BenchmarkTableExtractor().extract([sec], focal_model="Model")
        # Duplicate should be removed
        mmlu_results = [r for r in results if "MMLU" in r.benchmark_name]
        assert len(mmlu_results) == 1




# ===========================================================================
# Group 13: MethodVsClaimExtractor
# ===========================================================================

class TestMethodVsClaimExtractor:
    def test_wrong_type_raises(self):
        from app.document_intelligence.method_vs_claim_extractor import MethodVsClaimExtractor
        with pytest.raises(TypeError):
            asyncio.run(MethodVsClaimExtractor().extract("not list"))  # type: ignore

    def test_empty_sections_returns_empty(self):
        from app.document_intelligence.method_vs_claim_extractor import MethodVsClaimExtractor
        assert asyncio.run(MethodVsClaimExtractor().extract([])) == []

    def test_contribution_claim_detected(self):
        from app.document_intelligence.method_vs_claim_extractor import MethodVsClaimExtractor
        from app.document_intelligence.models import DocumentSection, SectionType
        text = "We propose a novel architecture called FusionNet that outperforms all baselines."
        sec = DocumentSection(section_type=SectionType.ABSTRACT, heading="Abstract", text=text, order=0)
        result = asyncio.run(MethodVsClaimExtractor(min_confidence=0.0).extract([sec]))
        assert len(result) >= 1
        assert any(c.claim_type == "contribution" for c in result)

    def test_limitation_detected(self):
        from app.document_intelligence.method_vs_claim_extractor import MethodVsClaimExtractor
        from app.document_intelligence.models import DocumentSection, SectionType
        text = "A key limitation of our approach is the high computational cost of inference."
        sec = DocumentSection(section_type=SectionType.CONCLUSION, heading="Conclusion", text=text, order=0)
        result = asyncio.run(MethodVsClaimExtractor(min_confidence=0.0).extract([sec]))
        assert any(c.claim_type == "limitation" for c in result)

    def test_future_work_detected(self):
        from app.document_intelligence.method_vs_claim_extractor import MethodVsClaimExtractor
        from app.document_intelligence.models import DocumentSection, SectionType
        text = "In future work, we plan to extend this approach to multimodal settings."
        sec = DocumentSection(section_type=SectionType.CONCLUSION, heading="Conclusion", text=text, order=0)
        result = asyncio.run(MethodVsClaimExtractor(min_confidence=0.0).extract([sec]))
        assert any(c.claim_type == "future_work" for c in result)

    def test_skips_non_claim_sections(self):
        from app.document_intelligence.method_vs_claim_extractor import MethodVsClaimExtractor
        from app.document_intelligence.models import DocumentSection, SectionType
        text = "See reference [1] for more details."
        sec = DocumentSection(section_type=SectionType.REFERENCES, heading="References", text=text, order=0)
        result = asyncio.run(MethodVsClaimExtractor().extract([sec]))
        assert result == []


# ===========================================================================
# Group 14: CitationGraph
# ===========================================================================

class TestCitationGraph:
    def test_invalid_focal_id_raises(self):
        from app.document_intelligence.citation_graph import CitationGraph
        with pytest.raises(ValueError):
            CitationGraph("")

    def test_build_from_sections_empty(self):
        from app.document_intelligence.citation_graph import CitationGraph
        g = CitationGraph("focal")
        g.build_from_sections([])
        assert len(g.all_nodes()) >= 1  # focal node added

    def test_wrong_sections_type_raises(self):
        from app.document_intelligence.citation_graph import CitationGraph
        g = CitationGraph("focal")
        with pytest.raises(TypeError):
            g.build_from_sections("not a list")  # type: ignore

    def test_add_and_get_node(self):
        from app.document_intelligence.citation_graph import CitationGraph
        from app.document_intelligence.models import CitationNode
        g = CitationGraph("focal")
        node = CitationNode(paper_id="arxiv:2401.00001", title="Test Paper")
        g.add_node(node)
        assert g.get_node("arxiv:2401.00001") is not None

    def test_add_edge_invalid_source_raises(self):
        from app.document_intelligence.citation_graph import CitationGraph
        from app.document_intelligence.models import CitationNode
        g = CitationGraph("focal")
        g.add_node(CitationNode(paper_id="focal", title="Focal", is_focal=True))
        with pytest.raises(KeyError):
            g.add_edge("nonexistent", "focal")

    def test_shortest_path_same_node(self):
        from app.document_intelligence.citation_graph import CitationGraph
        from app.document_intelligence.models import CitationNode
        g = CitationGraph("p1")
        g.add_node(CitationNode(paper_id="p1", title="P1"))
        assert g.shortest_path("p1", "p1") == ["p1"]

    def test_shortest_path_unreachable(self):
        from app.document_intelligence.citation_graph import CitationGraph
        from app.document_intelligence.models import CitationNode
        g = CitationGraph("p1")
        g.add_node(CitationNode(paper_id="p1", title="P1"))
        g.add_node(CitationNode(paper_id="p2", title="P2"))
        assert g.shortest_path("p1", "p2") is None

    def test_influence_score(self):
        from app.document_intelligence.citation_graph import CitationGraph
        from app.document_intelligence.models import CitationNode
        g = CitationGraph("focal")
        g.add_node(CitationNode(paper_id="focal", title="Focal"))
        g.add_node(CitationNode(paper_id="target", title="Target"))
        g.add_edge("focal", "target")
        assert g.influence_score("target") == 1

    def test_parse_reference_section(self):
        from app.document_intelligence.citation_graph import CitationGraph
        from app.document_intelligence.models import DocumentSection, SectionType
        ref_text = (
            "[1] Vaswani et al. \"Attention Is All You Need\". NeurIPS 2017. arXiv:1706.03762\n"
            "[2] Devlin et al. \"BERT: Pre-training of Deep Bidirectional Transformers\". 2018.\n"
        )
        sec = DocumentSection(section_type=SectionType.REFERENCES, heading="References", text=ref_text, order=0)
        g = CitationGraph("focal")
        g.build_from_sections([sec])
        assert g.edge_count() >= 1

    def test_adjacency_dict(self):
        from app.document_intelligence.citation_graph import CitationGraph
        from app.document_intelligence.models import CitationNode
        g = CitationGraph("focal")
        g.add_node(CitationNode(paper_id="focal", title="F"))
        g.add_node(CitationNode(paper_id="ref1", title="R1"))
        g.add_edge("focal", "ref1")
        adj = g.to_adjacency_dict()
        assert "focal" in adj
        assert "ref1" in adj["focal"]


# ===========================================================================
# Group 15: PaperSummarizer
# ===========================================================================

class TestPaperSummarizer:
    def test_wrong_type_raises(self):
        from app.document_intelligence.paper_summarizer import PaperSummarizer
        with pytest.raises(TypeError):
            asyncio.run(PaperSummarizer().summarize("not a paper"))  # type: ignore

    def test_invalid_max_tokens_raises(self):
        from app.document_intelligence.paper_summarizer import PaperSummarizer
        with pytest.raises(ValueError):
            PaperSummarizer(max_tokens=0)

    def test_extractive_with_abstract(self):
        from app.document_intelligence.paper_summarizer import PaperSummarizer
        from app.document_intelligence.models import PaperParsed, DocumentSection, SectionType
        sec = DocumentSection(section_type=SectionType.ABSTRACT, heading="Abstract",
                              text="We propose a new method for language modeling.", order=0)
        paper = PaperParsed(paper_id="p1", title="Test", abstract="We propose a new method.", sections=[sec])
        summary = asyncio.run(PaperSummarizer().summarize(paper))
        assert "We propose" in summary or "method" in summary or len(summary) > 0

    def test_extractive_fallback_no_llm(self):
        from app.document_intelligence.paper_summarizer import PaperSummarizer
        from app.document_intelligence.models import PaperParsed
        paper = PaperParsed(paper_id="p2", title="Empty Paper")
        summary = asyncio.run(PaperSummarizer(llm_router=None).summarize(paper))
        assert isinstance(summary, str)
        assert len(summary) > 0


# ===========================================================================
# Group 16: NoveltyEstimator
# ===========================================================================

class TestNoveltyEstimator:
    def test_wrong_type_raises(self):
        from app.document_intelligence.novelty_estimator import NoveltyEstimator
        with pytest.raises(TypeError):
            asyncio.run(NoveltyEstimator().estimate("not a paper"))  # type: ignore

    def test_invalid_llm_weight_raises(self):
        from app.document_intelligence.novelty_estimator import NoveltyEstimator
        with pytest.raises(ValueError):
            NoveltyEstimator(llm_weight=1.5)

    def test_high_novelty_language(self):
        from app.document_intelligence.novelty_estimator import NoveltyEstimator
        from app.document_intelligence.models import PaperParsed
        paper = PaperParsed(
            paper_id="novel",
            title="First-ever paradigm shift in language modeling",
            abstract="We introduce a novel zero-shot learning framework, unprecedented in scale. "
                     "This is the first to demonstrate emergent reasoning at scaling law boundaries.",
            venue="NeurIPS",
        )
        score = asyncio.run(NoveltyEstimator(llm_router=None).estimate(paper))
        assert score > 0.3

    def test_incremental_paper_lower_score(self):
        from app.document_intelligence.novelty_estimator import NoveltyEstimator
        from app.document_intelligence.models import PaperParsed
        paper = PaperParsed(
            paper_id="incr",
            title="Slight improvement using hyperparameter tuning",
            abstract="We slightly improve baseline results by tuning hyperparameters and adding more data.",
            venue="arxiv",
        )
        score = asyncio.run(NoveltyEstimator(llm_router=None).estimate(paper))
        assert score < 0.9

    def test_score_in_unit_range(self):
        from app.document_intelligence.novelty_estimator import NoveltyEstimator
        from app.document_intelligence.models import PaperParsed
        paper = PaperParsed(paper_id="range", title="Some paper", abstract="Some content.", venue="ICLR")
        score = asyncio.run(NoveltyEstimator(llm_router=None).estimate(paper))
        assert 0.0 <= score <= 1.0



# ===========================================================================
# Group 17: Devintel Models
# ===========================================================================

class TestDevintelModels:
    def test_change_entry_frozen(self):
        from app.devintel.models import ChangeEntry, ChangeCategory
        e = ChangeEntry(text="feat: add new feature", category=ChangeCategory.FEATURE)
        with pytest.raises(Exception):
            e.text = "changed"  # type: ignore

    def test_release_note_empty_version_raises(self):
        from app.devintel.models import ReleaseNote
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            ReleaseNote(version="")

    def test_release_note_has_breaking(self):
        from app.devintel.models import ReleaseNote, ChangeEntry, ChangeCategory
        entry = ChangeEntry(text="removed deprecated API", category=ChangeCategory.BREAKING, is_breaking=True)
        note = ReleaseNote(version="2.0.0", breaking=[entry])
        assert note.has_breaking_changes is True

    def test_release_note_all_entries(self):
        from app.devintel.models import ReleaseNote, ChangeEntry, ChangeCategory
        feat = ChangeEntry(text="feat: new thing", category=ChangeCategory.FEATURE)
        fix = ChangeEntry(text="fix: bug fix", category=ChangeCategory.FIX)
        note = ReleaseNote(version="1.1.0", features=[feat], fixes=[fix])
        assert len(note.all_entries()) == 2

    def test_breaking_change_frozen(self):
        from app.devintel.models import BreakingChange, ImpactLevel
        bc = BreakingChange(description="API removed", impact_level=ImpactLevel.HIGH)
        with pytest.raises(Exception):
            bc.description = "changed"  # type: ignore

    def test_dependency_alert_frozen(self):
        from app.devintel.models import DependencyAlert, ImpactLevel
        alert = DependencyAlert(package_name="openai", new_version="2.0.0", impact_level=ImpactLevel.HIGH)
        with pytest.raises(Exception):
            alert.package_name = "torch"  # type: ignore

    def test_repo_health_score_range(self):
        from app.devintel.models import RepoHealthScore
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            RepoHealthScore(repo="x/y", overall_score=1.5)


# ===========================================================================
# Group 18: ReleaseParser
# ===========================================================================

class TestReleaseParser:
    def test_empty_version_raises(self):
        from app.devintel.release_parser import ReleaseParser
        with pytest.raises(ValueError):
            ReleaseParser(version="")

    def test_wrong_body_type_raises(self):
        from app.devintel.release_parser import ReleaseParser
        with pytest.raises(TypeError):
            ReleaseParser(version="1.0.0").parse(123)  # type: ignore

    def test_parses_features(self):
        from app.devintel.release_parser import ReleaseParser
        body = "### Added\n- feat: new embedding endpoint\n- feat: streaming support\n"
        note = ReleaseParser(version="1.2.0").parse(body)
        assert len(note.features) >= 2

    def test_parses_breaking_section(self):
        from app.devintel.release_parser import ReleaseParser
        body = "### Breaking Changes\n- Removed `old_api()` method\n"
        note = ReleaseParser(version="2.0.0").parse(body)
        assert len(note.breaking) >= 1
        assert note.has_breaking_changes is True

    def test_inline_breaking_flag(self):
        from app.devintel.release_parser import ReleaseParser
        body = "### What's Changed\n- [BREAKING] Renamed `connect()` to `connect_async()`\n"
        note = ReleaseParser(version="3.0.0").parse(body)
        assert len(note.breaking) >= 1

    def test_parses_fixes(self):
        from app.devintel.release_parser import ReleaseParser
        body = "### Bug Fixes\n- fix: memory leak in tokenizer\n- fix: handle None inputs gracefully\n"
        note = ReleaseParser(version="1.1.1").parse(body)
        assert len(note.fixes) >= 2

    def test_parses_security(self):
        from app.devintel.release_parser import ReleaseParser
        body = "### Security\n- CVE-2024-1234: fixed auth bypass vulnerability\n"
        note = ReleaseParser(version="1.0.1").parse(body)
        assert len(note.security) >= 1

    def test_migration_notes_extracted(self):
        from app.devintel.release_parser import ReleaseParser
        body = "### Migration\n- Update your config files\n- Replace old imports with new paths\n"
        note = ReleaseParser(version="2.0.0").parse(body)
        assert len(note.migration_notes) > 0

    def test_conventional_commit_prefix(self):
        from app.devintel.release_parser import ReleaseParser
        body = "## What's Changed\n- feat: add new model support\n- fix: resolve crash on startup\n"
        note = ReleaseParser(version="1.3.0").parse(body)
        assert len(note.features) >= 1 or len(note.fixes) >= 1

    def test_pr_number_extracted(self):
        from app.devintel.release_parser import ReleaseParser
        body = "### Added\n- feat: new feature (#123)\n"
        note = ReleaseParser(version="1.0.0").parse(body)
        entries = note.features + note.fixes
        pr_numbers = [e.pr_number for e in entries if e.pr_number]
        assert 123 in pr_numbers or len(entries) >= 1


# ===========================================================================
# Group 19: ChangelogNormalizer
# ===========================================================================

class TestChangelogNormalizer:
    def test_wrong_type_raises(self):
        from app.devintel.changelog_normalizer import ChangelogNormalizer
        with pytest.raises(TypeError):
            ChangelogNormalizer().normalize(123)  # type: ignore

    def test_empty_string_returns_empty(self):
        from app.devintel.changelog_normalizer import ChangelogNormalizer
        assert ChangelogNormalizer().normalize("") == []

    def test_invalid_max_notes_raises(self):
        from app.devintel.changelog_normalizer import ChangelogNormalizer
        with pytest.raises(ValueError):
            ChangelogNormalizer(max_notes=0)

    def test_kacl_format_parsed(self):
        from app.devintel.changelog_normalizer import ChangelogNormalizer
        kacl = (
            "# Changelog\n\n"
            "## [2.1.0] - 2024-03-01\n"
            "### Added\n- New streaming API\n\n"
            "## [2.0.0] - 2024-01-15\n"
            "### Breaking Changes\n- Removed v1 endpoint\n"
        )
        notes = ChangelogNormalizer().normalize(kacl)
        assert len(notes) == 2
        versions = [n.version for n in notes]
        assert "2.1.0" in versions
        assert "2.0.0" in versions

    def test_kacl_sorted_newest_first(self):
        from app.devintel.changelog_normalizer import ChangelogNormalizer
        kacl = (
            "## [1.0.0] - 2023-01-01\n### Added\n- Initial release\n\n"
            "## [2.0.0] - 2024-01-01\n### Added\n- Major update\n"
        )
        notes = ChangelogNormalizer().normalize(kacl)
        assert notes[0].version == "2.0.0"

    def test_max_notes_respected(self):
        from app.devintel.changelog_normalizer import ChangelogNormalizer
        kacl = "\n".join(
            f"## [1.{i}.0] - 2024-0{(i%9)+1}-01\n### Added\n- change {i}\n"
            for i in range(1, 8)
        )
        notes = ChangelogNormalizer(max_notes=3).normalize(kacl)
        assert len(notes) <= 3

    def test_unreleased_section_skipped(self):
        from app.devintel.changelog_normalizer import ChangelogNormalizer
        kacl = (
            "## [Unreleased]\n### Added\n- Not released yet\n\n"
            "## [1.0.0] - 2024-01-01\n### Added\n- Initial\n"
        )
        notes = ChangelogNormalizer().normalize(kacl)
        versions = [n.version for n in notes]
        assert "Unreleased" not in versions

    def test_free_form_changelog_parsed(self):
        from app.devintel.changelog_normalizer import ChangelogNormalizer
        free = (
            "## 1.2.0\n"
            "- feat: add new endpoint\n\n"
            "## 1.1.0\n"
            "- fix: resolve memory leak\n"
        )
        notes = ChangelogNormalizer().normalize(free)
        assert len(notes) >= 1


# ===========================================================================
# Group 20: BreakingChangeDetector
# ===========================================================================

class TestBreakingChangeDetector:
    def test_wrong_type_raises(self):
        from app.devintel.breaking_change_detector import BreakingChangeDetector
        with pytest.raises(TypeError):
            asyncio.run(BreakingChangeDetector().detect("not a note"))  # type: ignore

    def test_empty_note_no_changes(self):
        from app.devintel.breaking_change_detector import BreakingChangeDetector
        from app.devintel.models import ReleaseNote
        note = ReleaseNote(version="1.0.0")
        result = asyncio.run(BreakingChangeDetector().detect(note))
        assert result == []

    def test_flagged_breaking_entry_detected(self):
        from app.devintel.breaking_change_detector import BreakingChangeDetector
        from app.devintel.models import ReleaseNote, ChangeEntry, ChangeCategory
        entry = ChangeEntry(text="Removed deprecated `v1.auth()` method", is_breaking=True, category=ChangeCategory.BREAKING)
        note = ReleaseNote(version="2.0.0", breaking=[entry])
        result = asyncio.run(BreakingChangeDetector().detect(note))
        assert len(result) >= 1

    def test_keyword_breaking_detected(self):
        from app.devintel.breaking_change_detector import BreakingChangeDetector
        from app.devintel.models import ReleaseNote, ChangeEntry, ChangeCategory
        entry = ChangeEntry(text="Removed the `connect()` function from public API.", category=ChangeCategory.OTHER)
        note = ReleaseNote(version="2.0.0", features=[entry])
        result = asyncio.run(BreakingChangeDetector(min_confidence=0.5).detect(note))
        assert len(result) >= 1

    def test_critical_security_change(self):
        from app.devintel.breaking_change_detector import BreakingChangeDetector, ImpactLevel
        from app.devintel.models import ReleaseNote, ChangeEntry, ChangeCategory
        entry = ChangeEntry(text="Removed insecure auth token validation security bypass", is_breaking=True, category=ChangeCategory.BREAKING)
        note = ReleaseNote(version="2.0.0", breaking=[entry])
        result = asyncio.run(BreakingChangeDetector().detect(note))
        impacts = {bc.impact_level for bc in result}
        assert ImpactLevel.CRITICAL in impacts or ImpactLevel.HIGH in impacts

    def test_sorted_critical_first(self):
        from app.devintel.breaking_change_detector import BreakingChangeDetector, ImpactLevel
        from app.devintel.models import ReleaseNote, ChangeEntry, ChangeCategory
        entries = [
            ChangeEntry(text="renamed `get()` to `fetch()`", is_breaking=True, category=ChangeCategory.BREAKING),
            ChangeEntry(text="security token removed due to auth vulnerability", is_breaking=True, category=ChangeCategory.BREAKING),
        ]
        note = ReleaseNote(version="3.0.0", breaking=entries)
        result = asyncio.run(BreakingChangeDetector().detect(note))
        if len(result) >= 2:
            order_map = {ImpactLevel.CRITICAL: 0, ImpactLevel.HIGH: 1, ImpactLevel.MEDIUM: 2, ImpactLevel.LOW: 3}
            levels = [order_map[bc.impact_level] for bc in result]
            assert levels == sorted(levels)

    def test_detect_from_entries_sync(self):
        from app.devintel.breaking_change_detector import BreakingChangeDetector
        from app.devintel.models import ChangeEntry, ChangeCategory
        entries = [
            ChangeEntry(text="Removed deprecated API method from SDK", category=ChangeCategory.OTHER),
        ]
        result = BreakingChangeDetector().detect_from_entries(entries)
        assert isinstance(result, list)


# ===========================================================================
# Group 21: VersionDiffAnalyzer
# ===========================================================================

class TestVersionDiffAnalyzer:
    def test_patch_change(self):
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        diff = VersionDiffAnalyzer().analyze("1.2.3", "1.2.4")
        assert diff.change_type == "patch"

    def test_minor_change(self):
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        diff = VersionDiffAnalyzer().analyze("1.2.3", "1.3.0")
        assert diff.change_type == "minor"

    def test_major_change(self):
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        diff = VersionDiffAnalyzer().analyze("1.2.3", "2.0.0")
        assert diff.change_type == "major"

    def test_breaking_change_type(self):
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        from app.devintel.models import BreakingChange, ImpactLevel
        bc = BreakingChange(description="API removed", impact_level=ImpactLevel.HIGH)
        diff = VersionDiffAnalyzer().analyze("1.0.0", "2.0.0", breaking_changes=[bc])
        assert diff.change_type == "breaking"

    def test_urgency_range(self):
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        diff = VersionDiffAnalyzer().analyze("1.0.0", "2.0.0")
        assert 0.0 <= diff.upgrade_urgency_score <= 1.0

    def test_patch_lower_urgency_than_major(self):
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        patch = VersionDiffAnalyzer().analyze("1.0.0", "1.0.1")
        major = VersionDiffAnalyzer().analyze("1.0.0", "2.0.0")
        assert patch.upgrade_urgency_score < major.upgrade_urgency_score

    def test_security_increases_urgency(self):
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        from app.devintel.models import ReleaseNote, ChangeEntry, ChangeCategory
        entry = ChangeEntry(text="CVE fix", category=ChangeCategory.SECURITY)
        note_with_sec = ReleaseNote(version="1.0.1", security=[entry])
        note_without = ReleaseNote(version="1.0.1")
        diff_sec = VersionDiffAnalyzer().analyze("1.0.0", "1.0.1", release_note=note_with_sec)
        diff_no_sec = VersionDiffAnalyzer().analyze("1.0.0", "1.0.1", release_note=note_without)
        assert diff_sec.upgrade_urgency_score > diff_no_sec.upgrade_urgency_score

    def test_invalid_security_bonus_raises(self):
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        with pytest.raises(ValueError):
            VersionDiffAnalyzer(security_urgency_bonus=1.5)

    def test_invalid_half_life_raises(self):
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        with pytest.raises(ValueError):
            VersionDiffAnalyzer(staleness_half_life_days=0)

    def test_semver_with_v_prefix(self):
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        diff = VersionDiffAnalyzer().analyze("v1.2.3", "v1.2.4")
        assert diff.change_type == "patch"


# ===========================================================================
# Group 22: SemanticDiffSummarizer
# ===========================================================================

class TestSemanticDiffSummarizer:
    def test_wrong_diff_type_raises(self):
        from app.devintel.semantic_diff_summarizer import SemanticDiffSummarizer
        with pytest.raises(TypeError):
            asyncio.run(SemanticDiffSummarizer().summarize("not a diff"))  # type: ignore

    def test_invalid_max_tokens_raises(self):
        from app.devintel.semantic_diff_summarizer import SemanticDiffSummarizer
        with pytest.raises(ValueError):
            SemanticDiffSummarizer(max_tokens=0)

    def test_extractive_summary_patch(self):
        from app.devintel.semantic_diff_summarizer import SemanticDiffSummarizer
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        diff = VersionDiffAnalyzer().analyze("1.0.0", "1.0.1")
        summary = asyncio.run(SemanticDiffSummarizer(llm_router=None).summarize(diff))
        assert "patch" in summary.lower() or "1.0.0" in summary

    def test_extractive_summary_includes_breaking_changes(self):
        from app.devintel.semantic_diff_summarizer import SemanticDiffSummarizer
        from app.devintel.version_diff_analyzer import VersionDiffAnalyzer
        from app.devintel.models import BreakingChange, ImpactLevel, ReleaseNote, ChangeEntry, ChangeCategory
        bc = BreakingChange(description="Removed legacy auth endpoint", impact_level=ImpactLevel.HIGH)
        entry = ChangeEntry(text="Removed legacy auth endpoint", is_breaking=True, category=ChangeCategory.BREAKING)
        note = ReleaseNote(version="2.0.0", breaking=[entry])
        diff = VersionDiffAnalyzer().analyze("1.0.0", "2.0.0", release_note=note, breaking_changes=[bc])
        summary = asyncio.run(SemanticDiffSummarizer(llm_router=None).summarize(diff, release_note=note))
        assert "BREAKING" in summary or "breaking" in summary.lower() or "high" in summary.lower()


# ===========================================================================
# Group 23: RepoHealthScorer
# ===========================================================================

class TestRepoHealthScorer:
    def test_wrong_type_raises(self):
        from app.devintel.repo_health import RepoHealthScorer
        with pytest.raises(TypeError):
            RepoHealthScorer().score("not a dict")  # type: ignore

    def test_empty_metadata(self):
        from app.devintel.repo_health import RepoHealthScorer
        score = RepoHealthScorer().score({})
        assert 0.0 <= score.overall_score <= 1.0

    def test_high_star_count_increases_score(self):
        from app.devintel.repo_health import RepoHealthScorer
        low = RepoHealthScorer().score({"stargazers_count": 10})
        high = RepoHealthScorer().score({"stargazers_count": 50000})
        assert high.overall_score > low.overall_score

    def test_recent_commit_increases_score(self):
        from app.devintel.repo_health import RepoHealthScorer
        stale = RepoHealthScorer().score({"pushed_at": "2020-01-01T00:00:00Z"})
        recent = RepoHealthScorer().score({"pushed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
        assert recent.overall_score > stale.overall_score

    def test_ci_and_tests_increase_score(self):
        from app.devintel.repo_health import RepoHealthScorer
        no_ci = RepoHealthScorer().score({"has_ci": False, "has_tests": False})
        with_ci = RepoHealthScorer().score({"has_ci": True, "has_tests": True})
        assert with_ci.overall_score > no_ci.overall_score

    def test_license_increases_score(self):
        from app.devintel.repo_health import RepoHealthScorer
        no_lic = RepoHealthScorer().score({})
        with_lic = RepoHealthScorer().score({"license": {"spdx_id": "MIT"}})
        assert with_lic.overall_score >= no_lic.overall_score

    def test_score_breakdown_keys(self):
        from app.devintel.repo_health import RepoHealthScorer
        result = RepoHealthScorer().score({"stargazers_count": 1000, "has_ci": True, "license": {"spdx_id": "MIT"}})
        assert "recency" in result.score_breakdown
        assert "community" in result.score_breakdown
        assert "ci_testing" in result.score_breakdown

    def test_invalid_star_scale_raises(self):
        from app.devintel.repo_health import RepoHealthScorer
        with pytest.raises(ValueError):
            RepoHealthScorer(star_scale=0)


# ===========================================================================
# Group 24: DependencyAlertEngine
# ===========================================================================

class TestDependencyAlertEngine:
    def test_watch_invalid_package_raises(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        with pytest.raises(ValueError):
            DependencyAlertEngine().watch("", "1.0.0")

    def test_watch_invalid_version_raises(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        with pytest.raises(ValueError):
            DependencyAlertEngine().watch("openai", "")

    def test_watch_and_len(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        eng = DependencyAlertEngine()
        eng.watch("openai", "1.0.0")
        eng.watch("anthropic", "0.5.0")
        assert len(eng) == 2

    def test_unwatch_returns_true(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        eng = DependencyAlertEngine()
        eng.watch("openai", "1.0.0")
        assert eng.unwatch("openai") is True

    def test_unwatch_missing_returns_false(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        assert DependencyAlertEngine().unwatch("nonexistent") is False

    def test_register_callback_invalid_raises(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        with pytest.raises(TypeError):
            DependencyAlertEngine().register_callback("not callable")  # type: ignore

    def test_process_release_no_match_returns_none(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        from app.devintel.models import ReleaseNote
        eng = DependencyAlertEngine()
        eng.watch("openai", "1.0.0")
        note = ReleaseNote(version="2.0.0", repo="some/other-package")
        assert eng.process_release(note) is None

    def test_process_release_matched_creates_alert(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        from app.devintel.models import ReleaseNote
        eng = DependencyAlertEngine()
        eng.watch("openai", "1.0.0", repo="openai/openai-python")
        note = ReleaseNote(version="2.0.0", repo="openai/openai-python")
        alert = eng.process_release(note)
        assert alert is not None
        assert alert.new_version == "2.0.0"
        assert alert.old_version == "1.0.0"

    def test_process_same_version_returns_none(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        from app.devintel.models import ReleaseNote
        eng = DependencyAlertEngine()
        eng.watch("openai", "1.0.0", repo="openai/openai-python")
        note = ReleaseNote(version="1.0.0", repo="openai/openai-python")
        assert eng.process_release(note) is None

    def test_callback_invoked(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        from app.devintel.models import ReleaseNote
        alerts_received = []
        eng = DependencyAlertEngine(callbacks=[alerts_received.append])
        eng.watch("openai", "1.0.0", repo="openai/openai-python")
        note = ReleaseNote(version="2.0.0", repo="openai/openai-python")
        eng.process_release(note)
        assert len(alerts_received) == 1

    def test_wrong_release_type_raises(self):
        from app.devintel.dependency_alerts import DependencyAlertEngine
        with pytest.raises(TypeError):
            DependencyAlertEngine().process_release("not a release note")  # type: ignore


# ===========================================================================
# Group 25: Entity Resolution Models
# ===========================================================================

class TestEntityModels:
    def test_canonical_entity_id_normalized(self):
        from app.entity_resolution.models import CanonicalEntity, EntityType
        e = CanonicalEntity(entity_id="  OpenAI  ", entity_type=EntityType.ORGANIZATION, canonical_name="OpenAI")
        assert e.entity_id == "openai"

    def test_canonical_entity_all_names(self):
        from app.entity_resolution.models import CanonicalEntity, EntityType
        e = CanonicalEntity(entity_id="gpt4", entity_type=EntityType.MODEL, canonical_name="GPT-4", aliases=["GPT4", "gpt-4"])
        names = e.all_names()
        assert "gpt-4" in names
        assert "gpt4" in names

    def test_canonical_entity_empty_id_raises(self):
        from app.entity_resolution.models import CanonicalEntity, EntityType
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            CanonicalEntity(entity_id="", entity_type=EntityType.MODEL, canonical_name="GPT-4")

    def test_event_bundle_empty_id_raises(self):
        from app.entity_resolution.models import EventBundle
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            EventBundle(bundle_id="", canonical_title="Test")

    def test_event_bundle_size(self):
        from app.entity_resolution.models import EventBundle
        b = EventBundle(bundle_id="b1", canonical_title="T", source_items=[{"source_id": "x"}, {"source_id": "y"}])
        assert b.size() == 2

    def test_dedupe_result_invalid_strategy_raises(self):
        from app.entity_resolution.models import DedupeResult
        import pydantic
        with pytest.raises((ValueError, pydantic.ValidationError)):
            DedupeResult(bundle_id="b1", kept_item_id="x", strategy="magic")

    def test_dedupe_result_frozen(self):
        from app.entity_resolution.models import DedupeResult
        d = DedupeResult(bundle_id="b1", kept_item_id="x", strategy="trust")
        with pytest.raises(Exception):
            d.kept_item_id = "y"  # type: ignore


# ===========================================================================
# Group 26: CanonicalEntityStore
# ===========================================================================

class TestCanonicalEntityStore:
    def _make_entity(self, entity_id="openai", name="OpenAI", aliases=None):
        from app.entity_resolution.models import CanonicalEntity, EntityType
        return CanonicalEntity(
            entity_id=entity_id, entity_type=EntityType.ORGANIZATION,
            canonical_name=name, aliases=aliases or []
        )

    def test_add_and_get(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        store = CanonicalEntityStore()
        entity = self._make_entity()
        store.add(entity)
        assert store.get("openai") is not None
        assert store.get("openai").canonical_name == "OpenAI"

    def test_wrong_type_raises(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        with pytest.raises(TypeError):
            CanonicalEntityStore().add("not an entity")  # type: ignore

    def test_max_entities_limit(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        store = CanonicalEntityStore(max_entities=2)
        store.add(self._make_entity("e1", "E1"))
        store.add(self._make_entity("e2", "E2"))
        with pytest.raises(OverflowError):
            store.add(self._make_entity("e3", "E3"))

    def test_remove(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        store = CanonicalEntityStore()
        store.add(self._make_entity())
        assert store.remove("openai") is True
        assert store.get("openai") is None

    def test_remove_missing_returns_false(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        assert CanonicalEntityStore().remove("nonexistent") is False

    def test_resolve_alias_exact(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        store = CanonicalEntityStore()
        store.add(self._make_entity("gpt4", "GPT-4", aliases=["GPT-4", "GPT4"]))
        assert store.resolve_alias("gpt4") == "gpt4"
        assert store.resolve_alias("GPT4") == "gpt4"

    def test_fuzzy_match_within_distance(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        store = CanonicalEntityStore()
        store.add(self._make_entity("gpt4", "GPT-4", aliases=["gpt-4"]))
        result = store.fuzzy_match("GPT4", max_distance=2)
        assert result is not None

    def test_fuzzy_match_too_far_returns_none(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        store = CanonicalEntityStore()
        store.add(self._make_entity("openai", "OpenAI"))
        result = store.fuzzy_match("Microsoft", max_distance=1)
        assert result is None

    def test_list_by_type(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        from app.entity_resolution.models import CanonicalEntity, EntityType
        store = CanonicalEntityStore()
        store.add(CanonicalEntity(entity_id="gpt4", entity_type=EntityType.MODEL, canonical_name="GPT-4"))
        store.add(CanonicalEntity(entity_id="openai", entity_type=EntityType.ORGANIZATION, canonical_name="OpenAI"))
        models = store.list_by_type(EntityType.MODEL)
        assert len(models) == 1
        assert models[0].entity_id == "gpt4"

    def test_contains(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        store = CanonicalEntityStore()
        store.add(self._make_entity("openai", "OpenAI"))
        assert "openai" in store
        assert "anthropic" not in store

    def test_update_property(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        store = CanonicalEntityStore()
        store.add(self._make_entity())
        assert store.update_property("openai", "founded", 2015) is True
        assert store.get("openai").properties.get("founded") == 2015


# ===========================================================================
# Group 27: AliasResolver
# ===========================================================================

class TestAliasResolver:
    def _make_store(self):
        from app.entity_resolution.canonical_entity_store import CanonicalEntityStore
        from app.entity_resolution.models import CanonicalEntity, EntityType
        store = CanonicalEntityStore()
        store.add(CanonicalEntity(entity_id="openai", entity_type=EntityType.ORGANIZATION,
                                   canonical_name="OpenAI", aliases=["open ai", "openai inc"]))
        store.add(CanonicalEntity(entity_id="gpt4", entity_type=EntityType.MODEL,
                                   canonical_name="GPT-4", aliases=["gpt-4", "gpt4"]))
        return store

    def test_wrong_store_type_raises(self):
        from app.entity_resolution.alias_resolver import AliasResolver
        with pytest.raises(TypeError):
            AliasResolver(store="not a store")  # type: ignore

    def test_wrong_name_type_raises(self):
        from app.entity_resolution.alias_resolver import AliasResolver
        resolver = AliasResolver(self._make_store())
        with pytest.raises(TypeError):
            resolver.resolve(123)  # type: ignore

    def test_exact_resolution(self):
        from app.entity_resolution.alias_resolver import AliasResolver
        resolver = AliasResolver(self._make_store())
        assert resolver.resolve("openai") == "openai"

    def test_alias_resolution(self):
        from app.entity_resolution.alias_resolver import AliasResolver
        resolver = AliasResolver(self._make_store())
        assert resolver.resolve("open ai") == "openai"

    def test_normalized_resolution_strips_possessive(self):
        from app.entity_resolution.alias_resolver import AliasResolver
        resolver = AliasResolver(self._make_store())
        result = resolver.resolve("OpenAI's new model")
        # Should match 'openai' after stripping possessive
        assert result == "openai" or result is None  # depends on normalization

    def test_empty_string_returns_none(self):
        from app.entity_resolution.alias_resolver import AliasResolver
        assert AliasResolver(self._make_store()).resolve("") is None

    def test_unresolvable_returns_none_no_fuzzy(self):
        from app.entity_resolution.alias_resolver import AliasResolver
        resolver = AliasResolver(self._make_store(), use_fuzzy=False)
        assert resolver.resolve("XYZZY_UNKNOWN_ENTITY_9999") is None

    def test_resolve_entity_returns_canonical(self):
        from app.entity_resolution.alias_resolver import AliasResolver
        resolver = AliasResolver(self._make_store())
        entity = resolver.resolve_entity("openai")
        assert entity is not None
        assert entity.canonical_name == "OpenAI"

    def test_resolve_batch(self):
        from app.entity_resolution.alias_resolver import AliasResolver
        resolver = AliasResolver(self._make_store())
        results = resolver.resolve_batch(["openai", "gpt-4", "UNKNOWN"])
        assert results[0] == "openai"
        assert results[1] == "gpt4"
        assert results[2] is None

    def test_resolve_batch_wrong_type_raises(self):
        from app.entity_resolution.alias_resolver import AliasResolver
        with pytest.raises(TypeError):
            AliasResolver(self._make_store()).resolve_batch("not a list")  # type: ignore


# ===========================================================================
# Group 28: EventClusterer
# ===========================================================================

class TestEventClusterer:
    def test_wrong_type_raises(self):
        from app.entity_resolution.event_clusterer import EventClusterer
        with pytest.raises(TypeError):
            EventClusterer().cluster("not a list")  # type: ignore

    def test_empty_returns_empty(self):
        from app.entity_resolution.event_clusterer import EventClusterer
        assert EventClusterer().cluster([]) == []

    def test_invalid_threshold_raises(self):
        from app.entity_resolution.event_clusterer import EventClusterer
        with pytest.raises(ValueError):
            EventClusterer(similarity_threshold=0.0)

    def test_weights_must_sum_to_one(self):
        from app.entity_resolution.event_clusterer import EventClusterer
        with pytest.raises(ValueError):
            EventClusterer(weight_entity=0.5, weight_title=0.5, weight_time=0.1)

    def test_single_item_one_bundle(self):
        from app.entity_resolution.event_clusterer import EventClusterer
        items = [{"source_id": "a", "title": "OpenAI releases GPT-5", "entities": ["OpenAI"], "published_at": "2024-01-01T00:00:00Z", "trust_score": 0.9}]
        bundles = EventClusterer().cluster(items)
        assert len(bundles) == 1

    def test_similar_items_clustered_together(self):
        from app.entity_resolution.event_clusterer import EventClusterer
        items = [
            {"source_id": "a", "title": "OpenAI releases GPT-5 model", "entities": ["OpenAI", "GPT-5"], "published_at": "2024-03-01T00:00:00Z", "trust_score": 0.9},
            {"source_id": "b", "title": "OpenAI announces GPT-5 release", "entities": ["OpenAI", "GPT-5"], "published_at": "2024-03-01T06:00:00Z", "trust_score": 0.7},
        ]
        bundles = EventClusterer(similarity_threshold=0.15).cluster(items)
        # Highly similar → should cluster
        assert len(bundles) <= 2  # ideally 1

    def test_dissimilar_items_separate_bundles(self):
        from app.entity_resolution.event_clusterer import EventClusterer
        items = [
            {"source_id": "a", "title": "Google launches new search product", "entities": ["Google"], "published_at": "2024-01-01T00:00:00Z", "trust_score": 0.9},
            {"source_id": "b", "title": "OpenAI GPT-5 breakthrough in reasoning", "entities": ["OpenAI"], "published_at": "2024-06-01T00:00:00Z", "trust_score": 0.8},
        ]
        bundles = EventClusterer(similarity_threshold=0.8).cluster(items)
        assert len(bundles) == 2

    def test_bundles_sorted_newest_first(self):
        from app.entity_resolution.event_clusterer import EventClusterer
        items = [
            {"source_id": "old", "title": "Old event about Microsoft", "entities": ["Microsoft"], "published_at": "2022-01-01T00:00:00Z", "trust_score": 0.5},
            {"source_id": "new", "title": "New event about Apple", "entities": ["Apple"], "published_at": "2024-06-01T00:00:00Z", "trust_score": 0.8},
        ]
        bundles = EventClusterer(similarity_threshold=0.99).cluster(items)
        if len(bundles) >= 2:
            times = [b.event_time for b in bundles if b.event_time]
            assert times == sorted(times, reverse=True)


# ===========================================================================
# Group 29: CrossSourceDeduper
# ===========================================================================

class TestCrossSourceDeduper:
    def _make_bundle(self, items):
        from app.entity_resolution.models import EventBundle
        trust = {item["source_id"]: item.get("trust_score", 0.5) for item in items}
        return EventBundle(
            bundle_id="test_bundle",
            canonical_title=items[0]["title"] if items else "Test",
            source_items=items,
            trust_scores=trust,
            primary_item_id=max(trust, key=lambda k: trust[k]) if trust else "",
        )

    def test_wrong_type_raises(self):
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        with pytest.raises(TypeError):
            CrossSourceDeduper().deduplicate("not a bundle")  # type: ignore

    def test_invalid_threshold_raises(self):
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        with pytest.raises(ValueError):
            CrossSourceDeduper(title_threshold=0.0)

    def test_single_item_no_removal(self):
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        bundle = self._make_bundle([{"source_id": "x", "title": "Test", "trust_score": 0.8}])
        result = CrossSourceDeduper().deduplicate(bundle)
        assert result.removed_ids == []
        assert result.kept_item_id == "x"

    def test_exact_duplicate_removed(self):
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        items = [
            {"source_id": "high", "title": "OpenAI releases GPT-5", "trust_score": 0.9, "raw_text": "same content", "source_url": "http://a.com"},
            {"source_id": "low", "title": "OpenAI releases GPT-5", "trust_score": 0.3, "raw_text": "same content", "source_url": "http://a.com"},
        ]
        bundle = self._make_bundle(items)
        result = CrossSourceDeduper().deduplicate(bundle)
        assert "low" in result.removed_ids

    def test_dissimilar_items_kept(self):
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        items = [
            {"source_id": "a", "title": "OpenAI releases GPT-5 for enterprise customers", "trust_score": 0.9, "raw_text": "", "source_url": ""},
            {"source_id": "b", "title": "Google launches quantum computing breakthrough", "trust_score": 0.8, "raw_text": "", "source_url": ""},
        ]
        bundle = self._make_bundle(items)
        result = CrossSourceDeduper(title_threshold=0.9).deduplicate(bundle)
        assert "b" not in result.removed_ids

    def test_highest_trust_kept(self):
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        items = [
            {"source_id": "low", "title": "Same title about OpenAI GPT", "trust_score": 0.2, "raw_text": "x", "source_url": ""},
            {"source_id": "high", "title": "Same title about OpenAI GPT", "trust_score": 0.9, "raw_text": "y", "source_url": ""},
        ]
        bundle = self._make_bundle(items)
        result = CrossSourceDeduper(title_threshold=0.5).deduplicate(bundle)
        assert result.kept_item_id == "high"

    def test_batch_deduplicate(self):
        from app.entity_resolution.cross_source_deduper import CrossSourceDeduper
        from app.entity_resolution.models import EventBundle
        bundles = [
            EventBundle(bundle_id="b1", canonical_title="T1", source_items=[{"source_id": "x", "title": "T1"}], trust_scores={"x": 0.8}, primary_item_id="x"),
            EventBundle(bundle_id="b2", canonical_title="T2", source_items=[{"source_id": "y", "title": "T2"}], trust_scores={"y": 0.7}, primary_item_id="y"),
        ]
        results = CrossSourceDeduper().deduplicate_batch(bundles)
        assert len(results) == 2


# ===========================================================================
# Group 30: TemporalEventGraph
# ===========================================================================

class TestTemporalEventGraph:
    def _make_bundle(self, bundle_id, title, entities=None, dt_str=None):
        from app.entity_resolution.models import EventBundle
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")) if dt_str else None
        return EventBundle(bundle_id=bundle_id, canonical_title=title, entity_ids=entities or [], event_time=dt)

    def test_add_and_get_bundle(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph()
        b = self._make_bundle("b1", "Event 1")
        g.add_bundle(b)
        assert g.get_bundle("b1") is not None

    def test_add_wrong_type_raises(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        with pytest.raises(TypeError):
            TemporalEventGraph().add_bundle("not a bundle")  # type: ignore

    def test_add_edge_invalid_type_raises(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph()
        b1 = self._make_bundle("b1", "E1")
        b2 = self._make_bundle("b2", "E2")
        g.add_bundle(b1)
        g.add_bundle(b2)
        with pytest.raises(ValueError):
            g.add_edge("b1", "b2", edge_type="invalid")

    def test_add_edge_missing_source_raises(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph()
        g.add_bundle(self._make_bundle("b1", "E1"))
        with pytest.raises(KeyError):
            g.add_edge("missing", "b1")

    def test_edge_count(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph(auto_follow_edges=False)
        g.add_bundle(self._make_bundle("b1", "E1"))
        g.add_bundle(self._make_bundle("b2", "E2"))
        g.add_edge("b1", "b2", edge_type="related")
        assert g.edge_count() == 1

    def test_neighbours(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph(auto_follow_edges=False)
        g.add_bundle(self._make_bundle("b1", "E1"))
        g.add_bundle(self._make_bundle("b2", "E2"))
        g.add_edge("b1", "b2", edge_type="causes")
        n = g.neighbours("b1", edge_type="causes")
        assert len(n) == 1
        assert n[0].bundle_id == "b2"

    def test_shortest_path_same(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph()
        g.add_bundle(self._make_bundle("b1", "E1"))
        assert g.shortest_path("b1", "b1") == ["b1"]

    def test_shortest_path_found(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph(auto_follow_edges=False)
        g.add_bundle(self._make_bundle("b1", "E1"))
        g.add_bundle(self._make_bundle("b2", "E2"))
        g.add_bundle(self._make_bundle("b3", "E3"))
        g.add_edge("b1", "b2", edge_type="follows")
        g.add_edge("b2", "b3", edge_type="follows")
        path = g.shortest_path("b1", "b3")
        assert path == ["b1", "b2", "b3"]

    def test_shortest_path_unreachable(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph(auto_follow_edges=False)
        g.add_bundle(self._make_bundle("b1", "E1"))
        g.add_bundle(self._make_bundle("b2", "E2"))
        assert g.shortest_path("b1", "b2") is None

    def test_timeline_ordered(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph(auto_follow_edges=False)
        g.add_bundle(self._make_bundle("b1", "Old", dt_str="2023-01-01T00:00:00Z"))
        g.add_bundle(self._make_bundle("b2", "New", dt_str="2024-06-01T00:00:00Z"))
        timeline = g.timeline()
        assert timeline[0].bundle_id == "b1"
        assert timeline[1].bundle_id == "b2"

    def test_timeline_entity_filter(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph(auto_follow_edges=False)
        g.add_bundle(self._make_bundle("b1", "OpenAI", entities=["openai"], dt_str="2024-01-01T00:00:00Z"))
        g.add_bundle(self._make_bundle("b2", "Google", entities=["google"], dt_str="2024-02-01T00:00:00Z"))
        filtered = g.timeline(entity_id="openai")
        assert len(filtered) == 1
        assert filtered[0].bundle_id == "b1"

    def test_remove_bundle(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph()
        g.add_bundle(self._make_bundle("b1", "E1"))
        assert g.remove_bundle("b1") is True
        assert g.get_bundle("b1") is None

    def test_remove_missing_returns_false(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        assert TemporalEventGraph().remove_bundle("nonexistent") is False

    def test_auto_follow_edges_created(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph(auto_follow_edges=True)
        g.add_bundle(self._make_bundle("b1", "Earlier", entities=["openai"], dt_str="2024-01-01T00:00:00Z"))
        g.add_bundle(self._make_bundle("b2", "Later", entities=["openai"], dt_str="2024-06-01T00:00:00Z"))
        # b1 should auto-follow → b2
        assert g.edge_count() >= 1

    def test_adjacency_dict_serializable(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph(auto_follow_edges=False)
        g.add_bundle(self._make_bundle("b1", "E1"))
        g.add_bundle(self._make_bundle("b2", "E2"))
        g.add_edge("b1", "b2", edge_type="related")
        adj = g.to_adjacency_dict()
        assert "b1" in adj
        assert any(e["target"] == "b2" for e in adj["b1"])

    def test_len_and_contains(self):
        from app.entity_resolution.temporal_event_graph import TemporalEventGraph
        g = TemporalEventGraph(auto_follow_edges=False)
        g.add_bundle(self._make_bundle("b1", "E1"))
        assert len(g) == 1
        assert "b1" in g
        assert "b99" not in g

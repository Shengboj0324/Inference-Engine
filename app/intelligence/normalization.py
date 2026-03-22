"""Normalization engine for converting RawObservation to NormalizedObservation.

This module implements the first stage of the inference pipeline:
- Merges title/body/quoted text
- Detects language and translates non-English content
- Extracts entities and competitor mentions
- Attaches thread context
- Computes engagement/freshness features
- Generates embeddings

Follows the strict contract defined in app/domain/normalized_models.py
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional, List
from uuid import UUID

logger = logging.getLogger(__name__)

from app.domain.raw_models import RawObservation
from app.domain.normalized_models import (
    NormalizedObservation,
    EntityMention,
    ThreadContext,
    ContentQuality,
    SentimentPolarity,
)

# Optional imports
try:
    from app.intelligence.entity_extractor import EntityExtractor
except ImportError:
    EntityExtractor = None  # type: ignore[assignment,misc]
    logger.warning("EntityExtractor not available - entity extraction will be disabled")

try:
    from app.llm.providers.openai_provider import OpenAIEmbeddingClient
except ImportError:
    OpenAIEmbeddingClient = None  # type: ignore[assignment,misc]
    logger.warning("OpenAIEmbeddingClient not available - embedding generation will be disabled")

from app.llm.router import get_router


class NormalizationEngine:
    """Converts RawObservation to NormalizedObservation with enrichment.
    
    This is Stage A of the inference pipeline as defined in the blueprint.
    """
    
    def __init__(
        self,
        enable_translation: bool = True,
        enable_entity_extraction: bool = True,
        enable_embedding_generation: bool = True,
    ):
        """Initialize normalization engine.

        Args:
            enable_translation: Enable translation for non-English content
            enable_entity_extraction: Enable entity extraction
            enable_embedding_generation: Enable embedding generation
        """
        self.enable_translation = enable_translation
        self.enable_entity_extraction = enable_entity_extraction
        self.enable_embedding_generation = enable_embedding_generation

        # Initialize entity extractor
        if enable_entity_extraction and EntityExtractor is not None:
            self.entity_extractor = EntityExtractor()
        else:
            self.entity_extractor = None
            if enable_entity_extraction and EntityExtractor is None:
                logger.warning("Entity extraction requested but EntityExtractor not available")

        # Initialize LLM router for translation
        if enable_translation:
            self.llm_router = get_router()
        else:
            self.llm_router = None

        # Initialize dedicated embedding client (separate from LLM router)
        if enable_embedding_generation and OpenAIEmbeddingClient is not None:
            self._embedding_client = OpenAIEmbeddingClient()
        else:
            self._embedding_client = None
            if enable_embedding_generation and OpenAIEmbeddingClient is None:
                logger.warning("OpenAIEmbeddingClient not available - embeddings disabled")
        
        logger.info(
            f"NormalizationEngine initialized: "
            f"translation={enable_translation}, "
            f"entities={enable_entity_extraction}, "
            f"embeddings={enable_embedding_generation}"
        )
    
    async def normalize(self, raw: RawObservation) -> NormalizedObservation:
        """Convert RawObservation to NormalizedObservation.
        
        Args:
            raw: Raw observation from connector
            
        Returns:
            Normalized observation with enrichment
        """
        # Merge and normalize text content
        normalized_text = self._merge_and_normalize_text(raw)

        # Detect language (sync operation)
        original_language = self._detect_language(normalized_text)

        # Translate if needed
        translated_text = None
        if original_language and original_language != "en" and self.enable_translation:
            translated_text = await self._translate_text(normalized_text, original_language)

        # Extract entities
        entities: List[EntityMention] = []
        if self.enable_entity_extraction and self.entity_extractor:
            entities = await self._extract_entities(normalized_text)

        # Generate embedding
        embedding: Optional[List[float]] = None
        if self.enable_embedding_generation and self._embedding_client:
            embedding = await self._generate_embedding(normalized_text)

        # Compute quality scores
        quality, quality_score, completeness_score = self._compute_quality(raw, normalized_text)

        # Detect sentiment
        sentiment = self._detect_sentiment(normalized_text)

        # Extract topics and keywords
        topics, keywords = self._extract_topics_keywords(normalized_text, entities)

        # Compute engagement features from platform metadata
        engagement_velocity, virality_score = self._compute_engagement_features(raw)

        # Create normalized observation
        normalized = NormalizedObservation(
            raw_observation_id=raw.id,
            user_id=raw.user_id,
            source_platform=raw.source_platform,
            source_id=raw.source_id,
            source_url=raw.source_url,
            author=raw.author,
            channel=raw.channel,
            title=raw.title,
            normalized_text=normalized_text,
            original_language=original_language,
            translated_text=translated_text,
            media_type=raw.media_type,
            media_urls=raw.media_urls or [],
            published_at=raw.published_at,
            fetched_at=raw.fetched_at,
            normalized_at=datetime.now(timezone.utc),
            entities=entities,
            topics=topics,
            keywords=keywords,
            sentiment=sentiment if sentiment is not None else SentimentPolarity.UNKNOWN,
            quality=quality,
            quality_score=quality_score,
            completeness_score=completeness_score,
            engagement_velocity=engagement_velocity,
            virality_score=virality_score,
            embedding=embedding,
        )
        
        logger.debug(f"Normalized observation {raw.id} -> {normalized.id}")
        return normalized

    def _merge_and_normalize_text(self, raw: RawObservation) -> str:
        """Merge and normalize text content.

        Args:
            raw: Raw observation

        Returns:
            Merged and normalized text
        """
        parts = []

        if raw.title:
            parts.append(raw.title)

        if raw.raw_text:
            parts.append(raw.raw_text)

        merged = "\n\n".join(parts)

        # Basic normalization: strip whitespace, normalize newlines
        normalized = " ".join(merged.split())

        return normalized

    def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            ISO 639-1 language code (e.g., 'en', 'es', 'fr')
        """
        if not text:
            return None

        try:
            import langdetect
            lang = langdetect.detect(text)
            logger.debug(f"Detected language: {lang}")
            return lang
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return None

    async def _translate_text(self, text: str, source_lang: str) -> Optional[str]:
        """Translate text to English.

        Args:
            text: Text to translate
            source_lang: Source language code

        Returns:
            Translated text or None if translation fails
        """
        if not text or not self.llm_router:
            return None

        try:
            # Use LLM router's simple interface for translation
            prompt = f"Translate the following {source_lang} text to English. Output only the translated text, nothing else:\n\n{text}"
            translated = await self.llm_router.generate_simple(
                prompt=prompt,
                max_tokens=len(text) * 2,  # Rough estimate
                temperature=0.3,
            )
            return translated.strip() if translated else None
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return None

    async def _extract_entities(self, text: str) -> List[EntityMention]:
        """Extract entities from text.

        Args:
            text: Text to analyze

        Returns:
            List of entity mentions
        """
        if not text or not self.entity_extractor:
            return []

        try:
            # Extract entities using entity extractor (async method)
            result = await self.entity_extractor.extract_entities(text)

            # Convert to EntityMention objects - map Entity fields to EntityMention fields
            entities = []
            for entity in result.entities:
                entities.append(
                    EntityMention(
                        entity_name=entity.text,
                        entity_type=entity.type.value,
                        span_start=entity.start_char,
                        span_end=entity.end_char,
                        confidence=entity.confidence,
                    )
                )

            return entities
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if generation fails
        """
        if not text or not self._embedding_client:
            return None

        try:
            response = await self._embedding_client.embed_text(text)
            return response.embedding
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def _compute_quality(
        self, raw: RawObservation, normalized_text: str
    ) -> tuple[ContentQuality, float, float]:
        """Compute quality and completeness scores.

        Args:
            raw: Raw observation
            normalized_text: Normalized text content

        Returns:
            Tuple of (quality enum, quality_score, completeness_score)
        """
        # Compute completeness score
        completeness_score = 0.0
        if raw.title:
            completeness_score += 0.3
        if raw.raw_text:
            completeness_score += 0.4
        if raw.author:
            completeness_score += 0.1
        if raw.platform_metadata:
            completeness_score += 0.2

        # Compute quality score based on text length and structure
        quality_score = 0.5  # Base score

        if len(normalized_text) > 100:
            quality_score += 0.2
        if len(normalized_text) > 500:
            quality_score += 0.2
        if raw.media_urls:
            quality_score += 0.1

        # Determine quality enum
        if quality_score >= 0.8:
            quality = ContentQuality.HIGH
        elif quality_score >= 0.5:
            quality = ContentQuality.MEDIUM
        else:
            quality = ContentQuality.LOW

        return quality, min(quality_score, 1.0), min(completeness_score, 1.0)

    def _compute_engagement_features(
        self, raw: RawObservation
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute engagement velocity and virality score from platform metadata.

        Args:
            raw: Raw observation

        Returns:
            Tuple of (engagement_velocity, virality_score)
        """
        if not raw.platform_metadata:
            return None, None

        metadata = raw.platform_metadata

        # Try to extract engagement metrics from platform metadata
        # Different platforms use different field names
        likes = metadata.get('likes', metadata.get('upvotes', metadata.get('score', 0)))
        shares = metadata.get('shares', metadata.get('retweets', metadata.get('crossposts', 0)))
        comments = metadata.get('comments', metadata.get('num_comments', metadata.get('replies', 0)))

        if not any([likes, shares, comments]):
            return None, None

        # Compute engagement velocity (engagement per hour)
        engagement_velocity = None
        if raw.published_at:
            hours_since_publish = (
                datetime.now(timezone.utc) - raw.published_at
            ).total_seconds() / 3600

            if hours_since_publish > 0:
                total_engagement = likes + shares + comments
                engagement_velocity = total_engagement / hours_since_publish

        # Compute virality score (shares / (likes + comments + 1))
        virality_score = None
        if shares > 0:
            virality_score = shares / (likes + comments + 1)
            virality_score = min(virality_score, 1.0)  # Clamp to [0, 1]

        return engagement_velocity, virality_score

    # ------------------------------------------------------------------
    # AFINN-style sentiment lexicon with negation handling
    # ------------------------------------------------------------------

    # Valence scores follow AFINN conventions: positive = good, negative = bad.
    # Range: −3 (very negative) to +3 (very positive).
    # Only lemmatised base forms are stored; the scorer looks up individual
    # whitespace-split tokens so morphological variants (e.g. "loving",
    # "excellent") must be added explicitly if needed.
    _SENTIMENT_LEXICON: dict = {
        # ── Strong positive (+3) ─────────────────────────────────────────
        "excellent": 3, "amazing": 3, "outstanding": 3, "fantastic": 3,
        "perfect": 3, "brilliant": 3, "superb": 3, "exceptional": 3,
        "love": 3, "wonderful": 3, "awesome": 3,
        # ── Moderate positive (+2) ───────────────────────────────────────
        "good": 2, "great": 2, "nice": 2, "helpful": 2, "useful": 2,
        "easy": 2, "better": 2, "improved": 2, "recommend": 2, "happy": 2,
        "pleased": 2, "impressed": 2, "smooth": 2, "reliable": 2,
        # ── Weak positive (+1) ───────────────────────────────────────────
        "ok": 1, "fine": 1, "decent": 1, "acceptable": 1, "working": 1,
        "okay": 1, "solid": 1,
        # ── Weak negative (−1) ───────────────────────────────────────────
        "issue": -1, "problem": -1, "slow": -1, "confusing": -1, "hard": -1,
        "annoying": -1, "missing": -1, "complicated": -1,
        # ── Moderate negative (−2) ───────────────────────────────────────
        "bad": -2, "poor": -2, "terrible": -2, "broken": -2, "wrong": -2,
        "hate": -2, "frustrating": -2, "useless": -2, "fail": -2,
        "failed": -2, "failing": -2, "error": -2, "disappointed": -2,
        "unreliable": -2, "buggy": -2,
        # ── Strong negative (−3) ─────────────────────────────────────────
        "awful": -3, "horrible": -3, "worst": -3, "disaster": -3,
        "garbage": -3, "unacceptable": -3, "appalling": -3, "dreadful": -3,
    }

    # Words that flip the valence of the immediately following sentiment word.
    # Stored as a frozenset for O(1) lookup.
    _NEGATION_WORDS: frozenset = frozenset({
        "not", "no", "never", "neither", "without", "cant", "cannot",
        "dont", "doesnt", "didnt", "wont", "isnt", "arent", "wasnt",
        "werent", "hardly", "barely", "scarcely",
    })

    # Tokenisation regex: extract lower-cased alphanumeric tokens.
    _TOKEN_RE = re.compile(r"\b[a-z']+\b")

    def _detect_sentiment(self, text: str) -> Optional[SentimentPolarity]:
        """Detect sentiment polarity using a lexicon-weighted scorer with negation.

        Algorithm
        ---------
        1. Tokenise *text* into lower-cased word tokens.
        2. For each token in ``_SENTIMENT_LEXICON``, retrieve its base valence.
        3. If any of the three tokens immediately preceding the current token
           is in ``_NEGATION_WORDS``, flip the valence (e.g. "not great" → −2).
        4. Sum all valence contributions and normalise by ``sqrt(n_tokens)``
           to reduce length bias (longer posts accumulate more signal).
        5. Classify using fixed thresholds:
           - normalised score > +0.20  → POSITIVE
           - normalised score < −0.20  → NEGATIVE
           - otherwise                 → NEUTRAL
           Threshold rationale: ±0.20 corresponds to roughly 1 unambiguous
           sentiment word per 25 tokens, which is a conservative floor that
           avoids mislabelling factual/technical content as neutral.

        Args:
            text: Normalised content text.

        Returns:
            ``SentimentPolarity`` or ``None`` for empty input.
        """
        if not text:
            return None

        tokens = self._TOKEN_RE.findall(text.lower())
        if not tokens:
            return SentimentPolarity.NEUTRAL

        score = 0.0
        for i, token in enumerate(tokens):
            valence = self._SENTIMENT_LEXICON.get(token)
            if valence is None:
                continue
            # Check the three preceding tokens for negation words
            preceding = tokens[max(0, i - 3) : i]
            if any(neg in preceding for neg in self._NEGATION_WORDS):
                valence = -valence
            score += valence

        # Normalise by sqrt(n_tokens) to reduce length bias
        normalised = score / (len(tokens) ** 0.5)

        # Thresholds: ±0.20 gives a conservative neutral band; see docstring.
        if normalised > 0.20:
            return SentimentPolarity.POSITIVE
        if normalised < -0.20:
            return SentimentPolarity.NEGATIVE
        return SentimentPolarity.NEUTRAL

    # ------------------------------------------------------------------
    # TF-style keyword extractor
    # ------------------------------------------------------------------

    # English stop-words to exclude from keyword extraction.  This set is
    # intentionally broader than the old 10-word list so that function words,
    # auxiliary verbs, and common prepositions do not pollute the output.
    _STOP_WORDS: frozenset = frozenset({
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or",
        "but", "with", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "this", "that", "these",
        "those", "it", "its", "as", "by", "from", "up", "about", "into",
        "through", "during", "if", "then", "than", "so", "yet", "both",
        "each", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "too", "very", "just", "because", "while",
        "although", "though", "when", "where", "who", "which", "what", "how",
        "all", "any", "can", "our", "your", "their", "my", "we", "you",
        "they", "he", "she", "i", "me", "him", "her", "us", "them",
    })

    # Regex to extract candidate keyword tokens: lower-case letters, digits,
    # and hyphens (for compound terms like "sign-up").  Minimum length 3.
    _KEYWORD_RE = re.compile(r"\b[a-z][a-z0-9\-]{2,}\b")

    def _extract_topics_keywords(
        self, text: str, entities: List[EntityMention]
    ) -> tuple[List[str], List[str]]:
        """Extract topics and keywords from *text*.

        Topics are derived from named entities of type PRODUCT, ORG, or EVENT;
        they represent proper-noun anchors for the content.

        Keywords use term-frequency (TF) ranking over the token vocabulary
        after stop-word filtering.  TF ranking is preferable to raw word
        frequency because it is not biased by token length; the previous
        implementation filtered only on ``len(word) > 4`` which excluded
        semantically important short words like "bug", "api", "sdk".

        Threshold rationale
        -------------------
        * Minimum token length: 3 characters — filters "ok", "id", "vs".
        * Stop-word list: ~60 high-frequency English function words.
        * Top-K: 10 keywords — balances retrieval recall vs noise.

        Args:
            text: Normalised content text to analyse.
            entities: List of ``EntityMention`` objects from the NER stage.

        Returns:
            ``(topics, keywords)`` where *topics* is a ``List[str]`` of up to
            10 entity names and *keywords* is a ``List[str]`` of up to 10
            frequency-ranked content words.
        """
        # Topics from named entities
        topics = [
            e.entity_name
            for e in entities
            if e.entity_type in ("PRODUCT", "ORG", "EVENT")
        ][:10]

        # TF-ranked keyword extraction with stop-word filtering
        tokens = self._KEYWORD_RE.findall(text.lower())
        freq: dict = {}
        for tok in tokens:
            if tok not in self._STOP_WORDS:
                freq[tok] = freq.get(tok, 0) + 1

        keywords = [
            word
            for word, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:10]
        ]
        return topics, keywords


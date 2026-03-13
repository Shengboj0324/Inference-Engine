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
    EntityExtractor = None  # type: ignore
    logger.warning("EntityExtractor not available - entity extraction will be disabled")

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
        
        # Initialize LLM router for embeddings
        if enable_embedding_generation:
            self.llm_router = get_router()
        else:
            self.llm_router = None
        
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

        # Detect language
        original_language = await self._detect_language(normalized_text)

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
        if self.enable_embedding_generation and self.llm_router:
            embedding = await self._generate_embedding(normalized_text)

        # Compute quality scores
        quality, quality_score, completeness_score = self._compute_quality(raw, normalized_text)

        # Detect sentiment
        sentiment = self._detect_sentiment(normalized_text)

        # Extract topics and keywords
        topics, keywords = self._extract_topics_keywords(normalized_text, entities)

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
            sentiment=sentiment,
            quality=quality,
            quality_score=quality_score,
            completeness_score=completeness_score,
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

    async def _detect_language(self, text: str) -> Optional[str]:
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
            # Use LLM for translation
            prompt = f"Translate the following {source_lang} text to English:\n\n{text}"
            response = await self.llm_router.generate(
                prompt=prompt,
                max_tokens=len(text) * 2,  # Rough estimate
                temperature=0.3,
            )
            return response.content
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
            # Extract entities using entity extractor
            result = await self.entity_extractor.extract_entities_async(text)

            # Convert to EntityMention objects
            entities = []
            for entity in result.entities:
                entities.append(
                    EntityMention(
                        text=entity.text,
                        entity_type=entity.entity_type,
                        start_char=entity.start_char,
                        end_char=entity.end_char,
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
        if not text or not self.llm_router:
            return None

        try:
            # Generate embedding using LLM router
            response = await self.llm_router.embed(text)
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

    def _detect_sentiment(self, text: str) -> Optional[SentimentPolarity]:
        """Detect sentiment polarity of text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment polarity or None
        """
        if not text:
            return None

        # Simple keyword-based sentiment detection
        # In production, use a proper sentiment model
        positive_keywords = ["great", "excellent", "love", "amazing", "best"]
        negative_keywords = ["bad", "terrible", "hate", "worst", "awful"]

        text_lower = text.lower()
        positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in text_lower)

        if positive_count > negative_count:
            return SentimentPolarity.POSITIVE
        elif negative_count > positive_count:
            return SentimentPolarity.NEGATIVE
        else:
            return SentimentPolarity.NEUTRAL

    def _extract_topics_keywords(
        self, text: str, entities: List[EntityMention]
    ) -> tuple[List[str], List[str]]:
        """Extract topics and keywords from text.

        Args:
            text: Text to analyze
            entities: Extracted entities

        Returns:
            Tuple of (topics, keywords)
        """
        # Extract topics from entities
        topics = []
        for entity in entities:
            if entity.entity_type in ["PRODUCT", "ORG", "EVENT"]:
                topics.append(entity.text)

        # Extract keywords (simple word frequency)
        # In production, use TF-IDF or other keyword extraction
        words = text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only consider words longer than 4 chars
                word_freq[word] = word_freq.get(word, 0) + 1

        # Get top 10 keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        keywords = [word for word, _ in keywords]

        return topics[:10], keywords  # Limit to 10 each


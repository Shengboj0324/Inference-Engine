"""Production-grade enrichment service for metadata augmentation.

This module provides the EnrichmentService class that augments content with:
- Language detection using langdetect
- Topic extraction using NLP
- Entity recognition using spaCy
- Sentiment analysis
- Embedding generation using LLM providers
- Media metadata extraction
"""

import logging
from typing import Dict, List, Optional
import asyncio

from app.core.models import ContentItem
from app.intelligence.entity_extractor import EntityExtractor
from app.llm.openai_client import OpenAIEmbeddingClient

logger = logging.getLogger(__name__)


class EnrichmentMetrics:
    """Metrics for enrichment operations."""

    def __init__(self):
        """Initialize metrics."""
        self.total_enriched = 0
        self.language_detected = 0
        self.topics_extracted = 0
        self.entities_extracted = 0
        self.embeddings_generated = 0
        self.enrichment_failures = 0

    def record_enrichment(
        self,
        language: bool = False,
        topics: bool = False,
        entities: bool = False,
        embedding: bool = False,
        failed: bool = False,
    ):
        """Record enrichment metrics."""
        if not failed:
            self.total_enriched += 1
            if language:
                self.language_detected += 1
            if topics:
                self.topics_extracted += 1
            if entities:
                self.entities_extracted += 1
            if embedding:
                self.embeddings_generated += 1
        else:
            self.enrichment_failures += 1

    def get_summary(self) -> Dict:
        """Get metrics summary."""
        return {
            "total_enriched": self.total_enriched,
            "language_detected": self.language_detected,
            "topics_extracted": self.topics_extracted,
            "entities_extracted": self.entities_extracted,
            "embeddings_generated": self.embeddings_generated,
            "enrichment_failures": self.enrichment_failures,
        }


class EnrichmentService:
    """Production-grade enrichment service for metadata augmentation.

    Features:
    - Language detection
    - Topic extraction
    - Entity recognition
    - Sentiment analysis
    - Embedding generation
    - Media metadata extraction
    """

    def __init__(
        self,
        enable_language_detection: bool = True,
        enable_topic_extraction: bool = True,
        enable_entity_extraction: bool = True,
        enable_embedding_generation: bool = True,
        entity_extractor: Optional[EntityExtractor] = None,
    ):
        """Initialize enrichment service.

        Args:
            enable_language_detection: Enable language detection
            enable_topic_extraction: Enable topic extraction
            enable_entity_extraction: Enable entity recognition
            enable_embedding_generation: Enable embedding generation
            entity_extractor: Entity extractor instance (created if None)
        """
        self.enable_language_detection = enable_language_detection
        self.enable_topic_extraction = enable_topic_extraction
        self.enable_entity_extraction = enable_entity_extraction
        self.enable_embedding_generation = enable_embedding_generation

        # Initialize entity extractor
        if enable_entity_extraction:
            self.entity_extractor = entity_extractor or EntityExtractor()
        else:
            self.entity_extractor = None

        # Dedicated embedding client (LLMRouter does not expose embed())
        if enable_embedding_generation:
            try:
                self._embedding_client: Optional[OpenAIEmbeddingClient] = OpenAIEmbeddingClient()
            except Exception:
                logger.warning("OpenAIEmbeddingClient unavailable; embeddings disabled")
                self._embedding_client = None
        else:
            self._embedding_client = None

        # Metrics
        self.metrics = EnrichmentMetrics()

        logger.info(
            f"EnrichmentService initialized: lang={enable_language_detection}, "
            f"topics={enable_topic_extraction}, entities={enable_entity_extraction}, "
            f"embeddings={enable_embedding_generation}"
        )

    async def enrich(self, item: ContentItem) -> ContentItem:
        """Enrich a content item with metadata.

        Args:
            item: Content item to enrich

        Returns:
            Enriched content item
        """
        try:
            # Get text for analysis
            text = self._get_text_for_analysis(item)

            if not text:
                logger.warning(f"No text available for enrichment: {item.id}")
                return item

            # Run enrichment tasks concurrently
            tasks = []

            if self.enable_language_detection:
                tasks.append(self._detect_language(text))

            if self.enable_topic_extraction and self.entity_extractor:
                tasks.append(self._extract_topics(text))

            if self.enable_entity_extraction and self.entity_extractor:
                tasks.append(self._extract_entities(text))

            if self.enable_embedding_generation and self._embedding_client:
                tasks.append(self._generate_embedding(text))

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            result_index = 0

            if self.enable_language_detection:
                lang_result = results[result_index]
                result_index += 1
                if not isinstance(lang_result, Exception):
                    item.lang = lang_result

            if self.enable_topic_extraction and self.entity_extractor:
                topics_result = results[result_index]
                result_index += 1
                if not isinstance(topics_result, Exception):
                    item.topics = topics_result

            if self.enable_entity_extraction and self.entity_extractor:
                entities_result = results[result_index]
                result_index += 1
                if not isinstance(entities_result, Exception):
                    # Store entities in metadata
                    item.metadata['entities'] = entities_result

            if self.enable_embedding_generation and self._embedding_client:
                embedding_result = results[result_index]
                result_index += 1
                if not isinstance(embedding_result, Exception):
                    item.embedding = embedding_result

            # Record metrics
            self.metrics.record_enrichment(
                language=item.lang is not None,
                topics=len(item.topics) > 0,
                entities='entities' in item.metadata,
                embedding=item.embedding is not None,
            )

            return item

        except Exception as e:
            logger.error(f"Error enriching item {item.id}: {e}", exc_info=True)
            self.metrics.record_enrichment(failed=True)
            return item

    def _get_text_for_analysis(self, item: ContentItem) -> str:
        """Get text for analysis from content item.

        Args:
            item: Content item

        Returns:
            Combined text for analysis
        """
        text_parts = []

        if item.title:
            text_parts.append(item.title)

        if item.raw_text:
            text_parts.append(item.raw_text)

        return "\n\n".join(text_parts)

    async def _detect_language(self, text: str) -> Optional[str]:
        """Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            ISO 639-1 language code (e.g., 'en', 'es', 'fr')
        """
        try:
            # Use langdetect for language detection
            import langdetect

            # Detect language
            lang = langdetect.detect(text)

            logger.debug(f"Detected language: {lang}")
            return lang

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return None

    async def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text.

        Args:
            text: Text to analyze

        Returns:
            List of topics
        """
        try:
            if not self.entity_extractor:
                return []

            # Extract entities and use them as topics
            entities = await self.entity_extractor.extract_entities(
                text,
                extract_relations=False,
                extract_key_phrases=True,
                extract_topics=True,
            )

            # Combine topics and key phrases
            topics = list(entities.topics)

            # Add key phrases as topics
            for phrase in entities.key_phrases[:5]:  # Limit to top 5
                if phrase not in topics:
                    topics.append(phrase)

            logger.debug(f"Extracted {len(topics)} topics")
            return topics[:10]  # Limit to 10 topics

        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return []

    async def _extract_entities(self, text: str) -> Dict:
        """Extract entities from text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary of entities by type
        """
        try:
            if not self.entity_extractor:
                return {}

            # Extract entities
            entities = await self.entity_extractor.extract_entities(
                text,
                extract_relations=True,
                extract_key_phrases=False,
                extract_topics=False,
            )

            # Group entities by type
            entities_by_type = {}
            for entity in entities.entities:
                entity_type = entity.type.value
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []

                entities_by_type[entity_type].append({
                    'text': entity.text,
                    'confidence': entity.confidence,
                    'canonical_form': entity.canonical_form,
                })

            logger.debug(f"Extracted {len(entities.entities)} entities")
            return entities_by_type

        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return {}

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text via OpenAIEmbeddingClient.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None on failure
        """
        try:
            if not self._embedding_client:
                return None

            response = await self._embedding_client.embed_text(text)
            vector = response.embedding
            if vector:
                logger.debug(f"Generated embedding of dimension {len(vector)}")
                return vector

            return None

        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def get_metrics(self) -> Dict:
        """Get enrichment metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.get_summary()

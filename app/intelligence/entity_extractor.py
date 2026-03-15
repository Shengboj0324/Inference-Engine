"""Industrial-grade entity extraction and NER for content intelligence."""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

try:
    import spacy
    from spacy.tokens import Doc
    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None  # type: ignore[assignment]
    Doc = None  # type: ignore[assignment,misc]
    _SPACY_AVAILABLE = False

from pydantic import BaseModel

from app.core.cache import cached, get_cache_manager

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Entity types for classification."""

    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "GPE"  # Geopolitical entity
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    WORK_OF_ART = "WORK_OF_ART"


class Entity(BaseModel):
    """Extracted entity with metadata."""

    text: str
    type: EntityType
    start_char: int
    end_char: int
    confidence: float = 1.0
    context: Optional[str] = None  # Surrounding text
    canonical_form: Optional[str] = None  # Normalized form


class EntityRelation(BaseModel):
    """Relationship between two entities."""

    subject: Entity
    predicate: str  # Relationship type (e.g., "works_for", "located_in")
    object: Entity
    confidence: float = 1.0
    evidence: Optional[str] = None  # Text supporting the relation


class ExtractedEntities(BaseModel):
    """Complete entity extraction result."""

    entities: List[Entity]
    relations: List[EntityRelation]
    key_phrases: List[str]
    topics: List[str]
    sentiment: Optional[Dict[str, float]] = None


class EntityExtractor:
    """Industrial-grade entity extraction with NER, relations, and topics.

    Features:
    - Named Entity Recognition (NER) using spaCy
    - Entity relationship extraction
    - Key phrase extraction
    - Topic modeling
    - Sentiment analysis
    - Entity disambiguation and normalization
    - Caching for performance
    """

    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        enable_caching: bool = True,
    ):
        """Initialize entity extractor.

        Args:
            model_name: spaCy model to use (en_core_web_lg recommended)
            enable_caching: Enable result caching
        """
        self.model_name = model_name
        self.enable_caching = enable_caching
        self._nlp: Optional[spacy.Language] = None
        self._cache = get_cache_manager() if enable_caching else None

    def _ensure_model(self):
        """Ensure spaCy model is loaded."""
        if not _SPACY_AVAILABLE:
            raise ImportError(
                "spacy is not installed. Run: pip install spacy && "
                "python -m spacy download en_core_web_sm"
            )
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_name)
                logger.info(f"Loaded spaCy model: {self.model_name}")
            except OSError:
                logger.warning(
                    f"Model {self.model_name} not found. "
                    f"Downloading... (this may take a few minutes)"
                )
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", self.model_name])
                self._nlp = spacy.load(self.model_name)

    async def extract_entities(
        self,
        text: str,
        extract_relations: bool = True,
        extract_key_phrases: bool = True,
        extract_topics: bool = True,
        context_window: int = 50,
    ) -> ExtractedEntities:
        """Extract entities and relationships from text.

        Args:
            text: Text to analyze
            extract_relations: Extract entity relationships
            extract_key_phrases: Extract key phrases
            extract_topics: Extract topics
            context_window: Characters of context around each entity

        Returns:
            Extracted entities with metadata
        """
        # Check cache
        if self.enable_caching and self._cache:
            cache_key = self._cache._hash_key(f"{text}:{extract_relations}:{extract_key_phrases}")
            cached_result = await self._cache.get("entities", cache_key)
            if cached_result:
                logger.debug("Entity extraction cache hit")
                return ExtractedEntities(**cached_result)

        # Ensure model is loaded
        self._ensure_model()

        # Process text
        doc = self._nlp(text)

        # Extract entities
        entities = self._extract_named_entities(doc, text, context_window)

        # Extract relations
        relations = []
        if extract_relations:
            relations = self._extract_relations(doc, entities)

        # Extract key phrases
        key_phrases = []
        if extract_key_phrases:
            key_phrases = self._extract_key_phrases(doc)

        # Extract topics
        topics = []
        if extract_topics:
            topics = self._extract_topics(doc)

        # Build result
        result = ExtractedEntities(
            entities=entities,
            relations=relations,
            key_phrases=key_phrases,
            topics=topics,
        )

        # Cache result
        if self.enable_caching and self._cache:
            await self._cache.set("entities", cache_key, result.model_dump(), ttl=3600)

        return result

    def _extract_named_entities(
        self,
        doc: Doc,
        text: str,
        context_window: int,
    ) -> List[Entity]:
        """Extract named entities from spaCy doc.

        Args:
            doc: spaCy processed document
            text: Original text
            context_window: Characters of context

        Returns:
            List of extracted entities
        """
        entities = []

        for ent in doc.ents:
            # Map spaCy entity type to our enum
            try:
                entity_type = EntityType(ent.label_)
            except ValueError:
                # Skip unknown entity types
                continue

            # Extract context
            start = max(0, ent.start_char - context_window)
            end = min(len(text), ent.end_char + context_window)
            context = text[start:end]

            # Create entity
            entity = Entity(
                text=ent.text,
                type=entity_type,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=1.0,  # spaCy doesn't provide confidence
                context=context,
                canonical_form=self._normalize_entity(ent.text, entity_type),
            )

            entities.append(entity)

        return entities

    def _normalize_entity(self, text: str, entity_type: EntityType) -> str:
        """Normalize entity to canonical form.

        Args:
            text: Entity text
            entity_type: Entity type

        Returns:
            Normalized form
        """
        # Basic normalization
        normalized = text.strip()

        # Type-specific normalization
        if entity_type == EntityType.PERSON:
            # Remove titles (Mr., Mrs., Dr., etc.)
            for title in ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof."]:
                normalized = normalized.replace(title, "").strip()

        elif entity_type == EntityType.ORGANIZATION:
            # Remove common suffixes
            for suffix in [" Inc.", " Corp.", " LLC", " Ltd."]:
                if normalized.endswith(suffix):
                    normalized = normalized[:-len(suffix)]

        return normalized

    def _extract_relations(
        self,
        doc: Doc,
        entities: List[Entity],
    ) -> List[EntityRelation]:
        """Extract relationships between entities.

        Args:
            doc: spaCy processed document
            entities: Extracted entities

        Returns:
            List of entity relationships
        """
        relations = []

        # Simple rule-based relation extraction
        # For production, consider using a dedicated relation extraction model

        # Create entity lookup by position
        entity_by_start = {e.start_char: e for e in entities}

        # Look for common patterns
        for token in doc:
            # Pattern: PERSON works for/at ORGANIZATION
            if token.lemma_ in ["work", "join", "lead", "head"]:
                # Find subject (PERSON) and object (ORG)
                subject = None
                obj = None

                for child in token.children:
                    if child.dep_ == "nsubj":  # Subject
                        # Check if this is a PERSON entity
                        for ent in entities:
                            if (ent.start_char <= child.idx < ent.end_char and
                                ent.type == EntityType.PERSON):
                                subject = ent
                                break

                    elif child.dep_ in ["dobj", "pobj"]:  # Object
                        # Check if this is an ORG entity
                        for ent in entities:
                            if (ent.start_char <= child.idx < ent.end_char and
                                ent.type == EntityType.ORGANIZATION):
                                obj = ent
                                break

                if subject and obj:
                    relation = EntityRelation(
                        subject=subject,
                        predicate="works_for",
                        object=obj,
                        confidence=0.8,
                        evidence=token.sent.text,
                    )
                    relations.append(relation)

            # Pattern: ORGANIZATION located in/at LOCATION
            elif token.lemma_ in ["locate", "base", "headquarter"]:
                subject = None
                obj = None

                for child in token.children:
                    if child.dep_ == "nsubjpass":  # Passive subject
                        for ent in entities:
                            if (ent.start_char <= child.idx < ent.end_char and
                                ent.type == EntityType.ORGANIZATION):
                                subject = ent
                                break

                    elif child.dep_ == "pobj":  # Prepositional object
                        for ent in entities:
                            if (ent.start_char <= child.idx < ent.end_char and
                                ent.type == EntityType.LOCATION):
                                obj = ent
                                break

                if subject and obj:
                    relation = EntityRelation(
                        subject=subject,
                        predicate="located_in",
                        object=obj,
                        confidence=0.8,
                        evidence=token.sent.text,
                    )
                    relations.append(relation)

        return relations

    def _extract_key_phrases(self, doc: Doc) -> List[str]:
        """Extract key phrases using noun chunks and named entities.

        Args:
            doc: spaCy processed document

        Returns:
            List of key phrases
        """
        key_phrases = set()

        # Add noun chunks
        for chunk in doc.noun_chunks:
            # Filter out very short or very long chunks
            if 2 <= len(chunk.text.split()) <= 5:
                key_phrases.add(chunk.text.lower())

        # Add named entities as key phrases
        for ent in doc.ents:
            key_phrases.add(ent.text.lower())

        # Sort by frequency in text
        phrase_counts = {}
        text_lower = doc.text.lower()
        for phrase in key_phrases:
            phrase_counts[phrase] = text_lower.count(phrase)

        # Return top phrases by frequency
        sorted_phrases = sorted(
            phrase_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [phrase for phrase, count in sorted_phrases[:20]]

    def _extract_topics(self, doc: Doc) -> List[str]:
        """Extract topics using entity types and key verbs.

        Args:
            doc: spaCy processed document

        Returns:
            List of topics
        """
        topics = set()

        # Topic inference from entity types
        entity_types = [ent.label_ for ent in doc.ents]

        if "PERSON" in entity_types and "ORG" in entity_types:
            topics.add("business")
            topics.add("leadership")

        if "GPE" in entity_types and "EVENT" in entity_types:
            topics.add("news")
            topics.add("current_events")

        if "MONEY" in entity_types or "PERCENT" in entity_types:
            topics.add("finance")
            topics.add("economics")

        if "PRODUCT" in entity_types:
            topics.add("technology")
            topics.add("products")

        # Topic inference from key verbs
        key_verbs = set()
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                key_verbs.add(token.lemma_)

        # Business/finance verbs
        if key_verbs & {"acquire", "merge", "invest", "fund", "raise"}:
            topics.add("business")
            topics.add("finance")

        # Technology verbs
        if key_verbs & {"launch", "release", "develop", "innovate", "build"}:
            topics.add("technology")
            topics.add("innovation")

        # Politics verbs
        if key_verbs & {"elect", "vote", "legislate", "govern", "regulate"}:
            topics.add("politics")
            topics.add("government")

        return list(topics)

    async def batch_extract_entities(
        self,
        texts: List[str],
        max_concurrency: int = 10,
    ) -> List[ExtractedEntities]:
        """Extract entities from multiple texts in batch.

        Args:
            texts: List of texts to process
            max_concurrency: Maximum concurrent extractions

        Returns:
            List of extraction results
        """
        import asyncio

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrency)

        async def extract_with_semaphore(text: str) -> ExtractedEntities:
            async with semaphore:
                return await self.extract_entities(text)

        # Process all texts concurrently
        tasks = [extract_with_semaphore(text) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Entity extraction failed for text {i}: {result}")
                # Return empty result
                valid_results.append(ExtractedEntities(
                    entities=[],
                    relations=[],
                    key_phrases=[],
                    topics=[],
                ))
            else:
                valid_results.append(result)

        return valid_results


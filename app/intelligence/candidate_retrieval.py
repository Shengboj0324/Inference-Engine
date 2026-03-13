"""Candidate retrieval system for signal classification.

This module implements Stage B of the inference pipeline:
- Embedding similarity to canonical signal exemplars
- Lightweight classifier probabilities
- Entity-conditioned rules
- Platform-specific prior adjustments

Outputs top-k signal candidates with weak scores to guide LLM adjudication.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from uuid import UUID

import numpy as np
from pydantic import BaseModel

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalType
from app.intelligence.hnsw_search import HNSWIndex

logger = logging.getLogger(__name__)


class SignalCandidate(BaseModel):
    """A candidate signal type with weak prior score."""
    
    signal_type: SignalType
    score: float
    reasoning: str
    source: str  # 'embedding', 'entity', 'platform_prior', 'classifier'


@dataclass
class ExemplarSignal:
    """Canonical exemplar for a signal type."""
    
    signal_type: SignalType
    text: str
    embedding: List[float]
    entities: List[str]
    platform: str


class CandidateRetriever:
    """Retrieves candidate signal types for a normalized observation.
    
    This is Stage B of the inference pipeline as defined in the blueprint.
    Uses multiple weak signals to generate candidate hypotheses.
    """
    
    def __init__(
        self,
        exemplar_bank: Optional[List[ExemplarSignal]] = None,
        top_k: int = 5,
        embedding_weight: float = 0.4,
        entity_weight: float = 0.3,
        platform_weight: float = 0.3,
    ):
        """Initialize candidate retriever.
        
        Args:
            exemplar_bank: List of canonical signal exemplars
            top_k: Number of top candidates to return
            embedding_weight: Weight for embedding similarity
            entity_weight: Weight for entity matching
            platform_weight: Weight for platform priors
        """
        self.exemplar_bank = exemplar_bank or []
        self.top_k = top_k
        self.embedding_weight = embedding_weight
        self.entity_weight = entity_weight
        self.platform_weight = platform_weight
        
        # Build HNSW index for fast similarity search
        self.hnsw_index: Optional[HNSWIndex] = None
        if self.exemplar_bank:
            self._build_index()
        
        # Platform-specific priors (learned from data)
        self.platform_priors = self._initialize_platform_priors()
        
        logger.info(
            f"CandidateRetriever initialized with {len(self.exemplar_bank)} exemplars, "
            f"top_k={top_k}"
        )
    
    def _build_index(self) -> None:
        """Build HNSW index from exemplar embeddings."""
        if not self.exemplar_bank:
            return
        
        # Extract embeddings
        embeddings = np.array([ex.embedding for ex in self.exemplar_bank])
        
        # Build HNSW index
        self.hnsw_index = HNSWIndex(
            dimension=len(embeddings[0]),
            max_elements=len(embeddings),
        )
        
        # Add exemplars to index
        for i, exemplar in enumerate(self.exemplar_bank):
            self.hnsw_index.add_item(embeddings[i], i)
        
        logger.info(f"Built HNSW index with {len(self.exemplar_bank)} exemplars")
    
    def _initialize_platform_priors(self) -> Dict[str, Dict[SignalType, float]]:
        """Initialize platform-specific signal type priors.
        
        Returns:
            Dict mapping platform to signal type probabilities
        """
        # These would be learned from historical data
        # For now, use reasonable defaults
        return {
            "reddit": {
                SignalType.SUPPORT_REQUEST: 0.3,
                SignalType.FEATURE_REQUEST: 0.2,
                SignalType.COMPETITOR_MENTION: 0.15,
                SignalType.ALTERNATIVE_SEEKING: 0.15,
            },
            "twitter": {
                SignalType.COMPETITOR_MENTION: 0.25,
                SignalType.COMPLAINT: 0.2,
                SignalType.CHURN_RISK: 0.15,
            },
            "linkedin": {
                SignalType.PARTNERSHIP_OPPORTUNITY: 0.3,
                SignalType.EXPANSION_OPPORTUNITY: 0.2,
            },
        }
    
    async def retrieve_candidates(
        self, observation: NormalizedObservation
    ) -> List[SignalCandidate]:
        """Retrieve candidate signal types for an observation.
        
        Args:
            observation: Normalized observation
            
        Returns:
            List of signal candidates with scores
        """
        candidates: Dict[SignalType, float] = {}
        reasoning: Dict[SignalType, List[str]] = {}
        
        # 1. Embedding similarity
        if observation.embedding and self.hnsw_index:
            embedding_candidates = self._retrieve_by_embedding(observation)
            for signal_type, score, reason in embedding_candidates:
                candidates[signal_type] = candidates.get(signal_type, 0.0) + (
                    score * self.embedding_weight
                )
                reasoning.setdefault(signal_type, []).append(reason)
        
        # 2. Entity matching
        if observation.entities:
            entity_candidates = self._retrieve_by_entities(observation)
            for signal_type, score, reason in entity_candidates:
                candidates[signal_type] = candidates.get(signal_type, 0.0) + (
                    score * self.entity_weight
                )
                reasoning.setdefault(signal_type, []).append(reason)
        
        # 3. Platform priors
        platform_candidates = self._retrieve_by_platform(observation)
        for signal_type, score, reason in platform_candidates:
            candidates[signal_type] = candidates.get(signal_type, 0.0) + (
                score * self.platform_weight
            )
            reasoning.setdefault(signal_type, []).append(reason)
        
        # Convert to SignalCandidate objects
        result = []
        for signal_type, score in sorted(
            candidates.items(), key=lambda x: x[1], reverse=True
        )[:self.top_k]:
            result.append(
                SignalCandidate(
                    signal_type=signal_type,
                    score=min(score, 1.0),  # Normalize to [0, 1]
                    reasoning="; ".join(reasoning[signal_type]),
                    source="hybrid",
                )
            )
        
        logger.debug(
            f"Retrieved {len(result)} candidates for observation {observation.id}"
        )
        return result

    def _retrieve_by_embedding(
        self, observation: NormalizedObservation
    ) -> List[Tuple[SignalType, float, str]]:
        """Retrieve candidates by embedding similarity.

        Args:
            observation: Normalized observation

        Returns:
            List of (signal_type, score, reasoning) tuples
        """
        if not observation.embedding or not self.hnsw_index:
            return []

        # Search for nearest neighbors
        neighbors, distances = self.hnsw_index.search(
            np.array(observation.embedding), k=self.top_k
        )

        # Convert to candidates
        candidates = []
        for idx, distance in zip(neighbors, distances):
            if idx < len(self.exemplar_bank):
                exemplar = self.exemplar_bank[idx]
                similarity = 1.0 - distance  # Convert distance to similarity
                candidates.append((
                    exemplar.signal_type,
                    similarity,
                    f"Similar to exemplar: '{exemplar.text[:50]}...' (sim={similarity:.2f})"
                ))

        return candidates

    def _retrieve_by_entities(
        self, observation: NormalizedObservation
    ) -> List[Tuple[SignalType, float, str]]:
        """Retrieve candidates by entity matching.

        Args:
            observation: Normalized observation

        Returns:
            List of (signal_type, score, reasoning) tuples
        """
        if not observation.entities:
            return []

        candidates = []
        entity_texts = [e.text.lower() for e in observation.entities]

        # Check for competitor mentions
        competitor_keywords = ["competitor", "alternative", "vs", "versus", "better than"]
        if any(kw in observation.merged_text.lower() for kw in competitor_keywords):
            candidates.append((
                SignalType.COMPETITOR_MENTION,
                0.8,
                "Detected competitor-related keywords"
            ))

        # Check for product mentions
        product_entities = [e for e in observation.entities if e.entity_type == "PRODUCT"]
        if product_entities:
            candidates.append((
                SignalType.FEATURE_REQUEST,
                0.6,
                f"Mentioned products: {', '.join([e.text for e in product_entities[:3]])}"
            ))

        # Check for organization mentions
        org_entities = [e for e in observation.entities if e.entity_type == "ORG"]
        if org_entities:
            candidates.append((
                SignalType.PARTNERSHIP_OPPORTUNITY,
                0.5,
                f"Mentioned organizations: {', '.join([e.text for e in org_entities[:3]])}"
            ))

        return candidates

    def _retrieve_by_platform(
        self, observation: NormalizedObservation
    ) -> List[Tuple[SignalType, float, str]]:
        """Retrieve candidates by platform priors.

        Args:
            observation: Normalized observation

        Returns:
            List of (signal_type, score, reasoning) tuples
        """
        platform = observation.source_platform.value.lower()
        priors = self.platform_priors.get(platform, {})

        candidates = []
        for signal_type, prior_prob in priors.items():
            candidates.append((
                signal_type,
                prior_prob,
                f"Platform prior for {platform}"
            ))

        return candidates

    def add_exemplar(self, exemplar: ExemplarSignal) -> None:
        """Add a new exemplar to the bank.

        Args:
            exemplar: Exemplar signal to add
        """
        self.exemplar_bank.append(exemplar)

        # Rebuild index
        self._build_index()

        logger.info(f"Added exemplar for {exemplar.signal_type}, total={len(self.exemplar_bank)}")

    def get_stats(self) -> Dict:
        """Get retriever statistics.

        Returns:
            Dict with statistics
        """
        signal_type_counts = {}
        for exemplar in self.exemplar_bank:
            signal_type_counts[exemplar.signal_type.value] = (
                signal_type_counts.get(exemplar.signal_type.value, 0) + 1
            )

        return {
            "total_exemplars": len(self.exemplar_bank),
            "signal_type_distribution": signal_type_counts,
            "top_k": self.top_k,
            "weights": {
                "embedding": self.embedding_weight,
                "entity": self.entity_weight,
                "platform": self.platform_weight,
            },
        }


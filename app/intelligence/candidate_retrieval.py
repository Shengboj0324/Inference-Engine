"""Candidate retrieval system for signal classification.

This module implements Stage B of the inference pipeline:
- Embedding similarity to canonical signal exemplars
- Lightweight classifier probabilities
- Entity-conditioned rules
- Platform-specific prior adjustments

Outputs top-k signal candidates with weak scores to guide LLM adjudication.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from uuid import UUID

import numpy as np
from pydantic import BaseModel

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalType
from app.intelligence.hnsw_search import HNSWIndex, HNSWConfig, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level compiled regex patterns for entity-based candidate detection.
# Using compiled patterns (not string membership checks) gives O(n) scanning
# with explicit coverage and is far easier to extend than keyword lists.
# ---------------------------------------------------------------------------

#: Matches explicit competitor/alternative language.
_RE_COMPETITOR = re.compile(
    r"\b(vs\.?|versus|alternative|better than|compared? to|switching? (to|from)|"
    r"instead of|replace[sd]?|competitor)\b",
    re.IGNORECASE,
)

#: Matches price/cost sensitivity language.
_RE_PRICE = re.compile(
    r"\b(price|pricing|cost|expensive|cheap|afford|discount|subscription fee|"
    r"overpriced|budget|too pricey|pay less)\b",
    re.IGNORECASE,
)

#: Matches bug/error language.
_RE_BUG = re.compile(
    r"\b(bug|broken|error|crash|fail(ed|ing|ure)?|not working|doesn'?t work|"
    r"glitch|issue|problem|exception|stack trace|traceback)\b",
    re.IGNORECASE,
)

#: Matches feature-request language.
_RE_FEATURE = re.compile(
    r"\b(would (love|like)|please add|feature request|any plans|roadmap|"
    r"wish(list)?|suggest(ion)?|could you add|looking for)\b",
    re.IGNORECASE,
)

#: Matches churn/cancellation language.
_RE_CHURN = re.compile(
    r"\b(cancel(l(ing|ed|ation))?|switching? away|leaving|churning?|"
    r"terminating? (my |the )?subscription|not renewing)\b",
    re.IGNORECASE,
)

#: Default path for the persisted exemplar bank JSON file.
_DEFAULT_EXEMPLAR_BANK_PATH: Path = Path("training/exemplar_bank.json")

#: Default path for the platform-prior configuration file.
_DEFAULT_PRIORS_CONFIG_PATH: Path = Path("training/retrieval_config.json")


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
        exemplar_bank_path: Optional[Path] = None,
        priors_config_path: Optional[Path] = None,
    ):
        """Initialize candidate retriever.

        On startup the retriever tries to load the exemplar bank from
        ``exemplar_bank_path`` (defaults to ``training/exemplar_bank.json``)
        and the platform priors from ``priors_config_path`` (defaults to
        ``training/retrieval_config.json``).  If either file does not exist
        the retriever falls back to the ``exemplar_bank`` argument and the
        built-in default priors respectively.

        Args:
            exemplar_bank: In-memory list of canonical signal exemplars.
                Merged with any exemplars loaded from ``exemplar_bank_path``.
            top_k: Number of top candidates to return.
            embedding_weight: Weight for embedding similarity scores.
            entity_weight: Weight for regex entity-matching scores.
            platform_weight: Weight for platform-prior scores.
            exemplar_bank_path: Path to a JSON file that persists the exemplar
                bank across restarts.  Defaults to
                ``training/exemplar_bank.json``.
            priors_config_path: Path to a JSON file containing platform-prior
                probabilities.  Defaults to ``training/retrieval_config.json``.
                Falls back to built-in defaults when the file does not exist.
        """
        self._exemplar_bank_path: Path = exemplar_bank_path or _DEFAULT_EXEMPLAR_BANK_PATH
        self._priors_config_path: Path = priors_config_path or _DEFAULT_PRIORS_CONFIG_PATH

        # Merge in-memory exemplars with any persisted ones
        loaded_exemplars = self._load_exemplar_bank()
        self.exemplar_bank: List[ExemplarSignal] = list(exemplar_bank or []) + loaded_exemplars
        self.top_k = top_k
        self.embedding_weight = embedding_weight
        self.entity_weight = entity_weight
        self.platform_weight = platform_weight

        # Build HNSW index for fast similarity search
        self.hnsw_index: Optional[HNSWIndex] = None
        if self.exemplar_bank:
            self._build_index()

        # Platform-specific priors — prefer config file, fall back to built-ins
        self.platform_priors = self._load_platform_priors()

        logger.info(
            "CandidateRetriever initialized with %d exemplars, top_k=%d",
            len(self.exemplar_bank), top_k,
        )
    
    def _build_index(self) -> None:
        """Build HNSW index from exemplar embeddings."""
        if not self.exemplar_bank:
            return

        # Extract embeddings
        embeddings = np.array([ex.embedding for ex in self.exemplar_bank])

        # Build HNSW index with proper config
        config = HNSWConfig(
            dimension=len(embeddings[0]),
            max_elements=len(embeddings),
        )
        self.hnsw_index = HNSWIndex(config=config)

        # Add exemplars to index
        for i, exemplar in enumerate(self.exemplar_bank):
            # Use add_vector with string ID
            self.hnsw_index.add_vector(
                id=str(i),
                vector=embeddings[i].tolist(),
            )

        logger.info(f"Built HNSW index with {len(self.exemplar_bank)} exemplars")
    
    # ------------------------------------------------------------------
    # Persistence: exemplar bank
    # ------------------------------------------------------------------

    def _load_exemplar_bank(self) -> List[ExemplarSignal]:
        """Load exemplars from the JSON file at ``self._exemplar_bank_path``.

        Returns:
            List of ``ExemplarSignal`` instances, or empty list if the file
            does not exist or cannot be parsed.
        """
        if not self._exemplar_bank_path.exists():
            logger.debug(
                "CandidateRetriever: no exemplar bank file at %s",
                self._exemplar_bank_path,
            )
            return []
        try:
            with self._exemplar_bank_path.open("r", encoding="utf-8") as fh:
                records = json.load(fh)
            exemplars = [
                ExemplarSignal(
                    signal_type=SignalType(r["signal_type"]),
                    text=r["text"],
                    embedding=r["embedding"],
                    entities=r.get("entities", []),
                    platform=r.get("platform", ""),
                )
                for r in records
            ]
            logger.info(
                "CandidateRetriever: loaded %d exemplars from %s",
                len(exemplars), self._exemplar_bank_path,
            )
            return exemplars
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            logger.warning(
                "CandidateRetriever: failed to load exemplar bank from %s: %s",
                self._exemplar_bank_path, exc,
            )
            return []

    def _persist_exemplar_bank(self) -> None:
        """Persist the current exemplar bank to ``self._exemplar_bank_path``.

        Raises:
            OSError: If the file cannot be written (logged, not re-raised).
        """
        try:
            self._exemplar_bank_path.parent.mkdir(parents=True, exist_ok=True)
            records = [
                {
                    "signal_type": ex.signal_type.value,
                    "text": ex.text,
                    "embedding": ex.embedding,
                    "entities": ex.entities,
                    "platform": ex.platform,
                }
                for ex in self.exemplar_bank
            ]
            with self._exemplar_bank_path.open("w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)
            logger.debug(
                "CandidateRetriever: persisted %d exemplars to %s",
                len(records), self._exemplar_bank_path,
            )
        except OSError as exc:
            logger.error(
                "CandidateRetriever: failed to persist exemplar bank: %s", exc
            )

    # ------------------------------------------------------------------
    # Persistence: platform priors
    # ------------------------------------------------------------------

    def _load_platform_priors(self) -> Dict[str, Dict[SignalType, float]]:
        """Load platform priors from ``self._priors_config_path``.

        The JSON file must have the shape::

            {
              "reddit":   { "support_request": 0.30, "feature_request": 0.20 },
              "twitter":  { "competitor_mention": 0.25, "complaint": 0.20 },
              "linkedin": { "partnership_opportunity": 0.30 }
            }

        Falls back to built-in defaults when the file is absent or malformed.

        Returns:
            Dict mapping platform name to ``{SignalType: float}`` prior map.
        """
        _BUILTIN_DEFAULTS: Dict[str, Dict[SignalType, float]] = {
            # Probabilities reflect observed signal-type distributions per platform.
            # reddit: community Q&A → heavy support + feature traffic
            "reddit": {
                SignalType.SUPPORT_REQUEST: 0.30,
                SignalType.FEATURE_REQUEST: 0.20,
                SignalType.COMPETITOR_MENTION: 0.15,
                SignalType.ALTERNATIVE_SEEKING: 0.15,
            },
            # twitter/x: real-time reactions → complaints + competitor comparisons
            "twitter": {
                SignalType.COMPETITOR_MENTION: 0.25,
                SignalType.COMPLAINT: 0.20,
                SignalType.CHURN_RISK: 0.15,
            },
            # linkedin: professional network → partnership + expansion signals
            "linkedin": {
                SignalType.PARTNERSHIP_OPPORTUNITY: 0.30,
                SignalType.EXPANSION_OPPORTUNITY: 0.20,
            },
        }

        if not self._priors_config_path.exists():
            logger.debug(
                "CandidateRetriever: priors config not found at %s; using built-ins",
                self._priors_config_path,
            )
            return _BUILTIN_DEFAULTS

        try:
            with self._priors_config_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            priors: Dict[str, Dict[SignalType, float]] = {}
            for platform, type_map in raw.items():
                priors[platform.lower()] = {
                    SignalType(signal_type_str): float(prob)
                    for signal_type_str, prob in type_map.items()
                }
            logger.info(
                "CandidateRetriever: loaded platform priors for %d platforms from %s",
                len(priors), self._priors_config_path,
            )
            return priors
        except (json.JSONDecodeError, KeyError, ValueError, OSError) as exc:
            logger.warning(
                "CandidateRetriever: failed to load priors config from %s: %s; "
                "using built-in defaults",
                self._priors_config_path, exc,
            )
            return _BUILTIN_DEFAULTS
    
    def retrieve_candidates(
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
        
        # 1. Embedding similarity — HNSW required; warn loudly on fallback
        if observation.embedding and not self.hnsw_index:
            logger.warning(
                "HNSW index is not initialised (no exemplars loaded). "
                "Candidate retrieval is falling back to entity matching and platform priors only. "
                "Populate the exemplar bank via CandidateRetriever(exemplar_bank=[...]) or "
                "add_exemplar() to restore HNSW-based retrieval."
            )

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

        # Search for nearest neighbors - returns List[SearchResult]
        results = self.hnsw_index.search(
            query_vector=observation.embedding,
            k=self.top_k
        )

        # Convert to candidates
        candidates = []
        for result in results:
            # Parse ID back to index
            try:
                idx = int(result.id)
                if idx < len(self.exemplar_bank):
                    exemplar = self.exemplar_bank[idx]
                    # Distance is already in [0, 1] for cosine, convert to similarity
                    similarity = 1.0 - result.distance
                    candidates.append((
                        exemplar.signal_type,
                        max(0.0, min(1.0, similarity)),  # Clamp to [0, 1]
                        f"Similar to exemplar: '{exemplar.text[:50]}...' (sim={similarity:.2f})"
                    ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse search result: {e}")
                continue

        return candidates

    def _retrieve_by_entities(
        self, observation: NormalizedObservation
    ) -> List[Tuple[SignalType, float, str]]:
        """Retrieve candidates via compiled regex patterns applied to the merged text.

        Replaces the old keyword-membership approach with module-level compiled
        regex patterns (``_RE_COMPETITOR``, ``_RE_PRICE``, ``_RE_BUG``,
        ``_RE_FEATURE``, ``_RE_CHURN``) that have explicit coverage, handle
        morphological variants (e.g. "failing", "cancellation"), and avoid
        O(k) string-search-per-keyword overhead.

        Named-entity types extracted by the NER stage are used as secondary
        boosting signals on top of the regex matches.

        Args:
            observation: Normalized observation whose ``merged_text`` is scanned.

        Returns:
            List of ``(signal_type, score, reasoning)`` tuples; empty when no
            pattern matches.
        """
        text = observation.merged_text or observation.normalized_text or ""
        if not text:
            return []

        candidates: List[Tuple[SignalType, float, str]] = []

        # ── Regex-based signal detection ────────────────────────────────────
        if _RE_COMPETITOR.search(text):
            candidates.append((
                SignalType.COMPETITOR_MENTION,
                0.75,
                "Regex pattern: competitor/alternative language detected",
            ))

        if _RE_CHURN.search(text):
            candidates.append((
                SignalType.CHURN_RISK,
                0.80,
                "Regex pattern: cancellation/churn language detected",
            ))

        if _RE_PRICE.search(text):
            candidates.append((
                SignalType.PRICE_SENSITIVITY,
                0.65,
                "Regex pattern: price/cost sensitivity language detected",
            ))

        if _RE_BUG.search(text):
            candidates.append((
                SignalType.BUG_REPORT,
                0.70,
                "Regex pattern: bug/error language detected",
            ))

        if _RE_FEATURE.search(text):
            candidates.append((
                SignalType.FEATURE_REQUEST,
                0.65,
                "Regex pattern: feature-request language detected",
            ))

        # ── NER-entity boosts ────────────────────────────────────────────────
        if observation.entities:
            product_entities = [
                e for e in observation.entities if e.entity_type == "PRODUCT"
            ]
            if product_entities:
                names = ", ".join(e.entity_name for e in product_entities[:3])
                # A product mention alongside churn/competitor language strengthens
                # those signals; standalone it weakly suggests a feature request.
                candidates.append((
                    SignalType.FEATURE_REQUEST,
                    0.45,
                    f"NER: product entities mentioned — {names}",
                ))

            org_entities = [
                e for e in observation.entities if e.entity_type == "ORG"
            ]
            if org_entities:
                names = ", ".join(e.entity_name for e in org_entities[:3])
                candidates.append((
                    SignalType.PARTNERSHIP_OPPORTUNITY,
                    0.40,
                    f"NER: organisation entities mentioned — {names}",
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
        """Add a new exemplar to the in-memory bank, rebuild the HNSW index,
        and immediately persist the updated bank to disk.

        Persisting on every add ensures the bank survives process restarts even
        when ``add_exemplar`` is called during live serving (e.g. via an admin
        endpoint).  Callers that add many exemplars in a tight loop should call
        ``add_exemplar_batch`` instead to avoid repeated index rebuilds and
        file writes.

        Args:
            exemplar: Exemplar signal to add.
        """
        self.exemplar_bank.append(exemplar)
        self._build_index()
        self._persist_exemplar_bank()
        logger.info(
            "CandidateRetriever: added exemplar for %s, total=%d",
            exemplar.signal_type.value, len(self.exemplar_bank),
        )

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


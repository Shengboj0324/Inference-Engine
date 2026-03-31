"""Candidate retrieval system for signal classification.

This module implements Stage B of the inference pipeline:
- Embedding similarity to canonical signal exemplars
- Lightweight classifier probabilities
- Entity-conditioned rules
- Platform-specific prior adjustments

Outputs top-k signal candidates with weak scores to guide LLM adjudication.
"""

import heapq
import json
import logging
import math
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from uuid import UUID

import numpy as np
from pydantic import BaseModel

from app.domain.normalized_models import NormalizedObservation
from app.domain.inference_models import SignalType
from app.intelligence.hnsw_search import HNSWIndex, HNSWConfig, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hybrid retrieval utilities — Reciprocal Rank Fusion + Query Expansion
# ---------------------------------------------------------------------------


def _rrf_merge(
    ranked_lists: List[List[int]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion over multiple ranked document lists.

    Merges any number of ranked lists (e.g. dense HNSW + sparse TF-IDF) into
    a single unified ranking.  The RRF constant ``k=60`` is the value
    recommended in Cormack et al. (2009) and empirically validated on a wide
    range of retrieval benchmarks.

    Formula::

        score(d) = Σ_lists  1 / (rank(d, list) + k)

    where ``rank`` is 1-indexed.  Documents appearing in more lists receive
    additive boosts; documents appearing in only one list are never excluded.

    Args:
        ranked_lists: Each inner list contains document indices in rank order
                      (most relevant first).  Lists may have different lengths.
        k: RRF constant.  Higher values reduce the influence of top-ranked
           documents; lower values amplify rank-1 boosts.

    Returns:
        List of ``(document_index, rrf_score)`` tuples sorted by score
        descending.  Only documents that appear in at least one list are
        included.
    """
    scores: Dict[int, float] = {}
    for ranked in ranked_lists:
        if ranked is None:  # guard: callers must not pass None, but be defensive
            continue
        for rank, idx in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _expand_query_with_kb(
    text: str,
    entity_kb: Dict[str, Tuple[str, str]],
    max_terms: int = 3,
) -> str:
    """Expand a query text with canonical names from the entity knowledge base.

    Finds entity KB entries whose surface forms appear in *text* and appends
    up to *max_terms* canonical names.  The expansion improves sparse-retrieval
    recall for abbreviated or alias-form entity mentions (e.g. "MSFT" → expands
    to include "Microsoft").

    Privacy note: this function logs only at ``DEBUG`` level — never at
    ``INFO`` or above — to avoid leaking PII contained in the query text.

    Args:
        text: The query text to expand (merged_text of a NormalizedObservation).
        entity_kb: Mapping from lower-cased surface form to
                   ``(canonical_id, canonical_name)`` pair.
        max_terms: Maximum number of expansion terms to append.

    Returns:
        Original *text* with up to *max_terms* canonical names appended,
        space-separated.  Returns *text* unchanged when no KB match is found.
    """
    lower_text = text.lower()
    expansion: List[str] = []
    for surface, value in entity_kb.items():
        # Guard against malformed KB entries (e.g. loaded from an external JSON
        # file where a value might be a plain string, None, or a tuple of wrong
        # arity).  Skip silently so one bad entry never blocks the whole batch.
        try:
            _, canonical_name = value
        except (TypeError, ValueError):
            continue
        if surface in lower_text and canonical_name not in expansion:
            expansion.append(canonical_name)
            if len(expansion) >= max_terms:
                break

    if expansion:
        logger.debug(
            "RAG query expansion: appended %d KB term(s): %s",
            len(expansion),
            expansion,
        )
        return text + " " + " ".join(expansion)
    return text

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
        """Build HNSW (dense) and TF-IDF (sparse) indices from exemplar data."""
        if not self.exemplar_bank:
            return

        # ── Dense: HNSW ──────────────────────────────────────────────────────
        embeddings = np.array([ex.embedding for ex in self.exemplar_bank])
        config = HNSWConfig(
            dimension=len(embeddings[0]),
            max_elements=len(embeddings),
        )
        self.hnsw_index = HNSWIndex(config=config)
        for i, exemplar in enumerate(self.exemplar_bank):
            self.hnsw_index.add_vector(id=str(i), vector=embeddings[i].tolist())

        # ── Sparse: TF-IDF ────────────────────────────────────────────────────
        self._build_sparse_index()

        logger.info(
            "Built HNSW + TF-IDF hybrid index with %d exemplars",
            len(self.exemplar_bank),
        )

    def _build_sparse_index(self) -> None:
        """Build an in-memory TF-IDF index over exemplar texts for hybrid retrieval.

        Stores:
            self._sparse_vocab  — term → column index mapping.
            self._sparse_matrix — list of per-document {col: tfidf} dicts.
            self._doc_freq      — term → document frequency across all exemplars.
        """
        all_tokens: List[List[str]] = []
        for ex in self.exemplar_bank:
            toks = re.findall(r"[a-z0-9]+", ex.text.lower())
            all_tokens.append(toks)

        # Vocabulary
        vocab: set = set()
        for toks in all_tokens:
            vocab.update(toks)
        self._sparse_vocab: Dict[str, int] = {
            term: idx for idx, term in enumerate(sorted(vocab))
        }

        # Document frequencies
        self._doc_freq: Dict[str, int] = Counter()
        for toks in all_tokens:
            self._doc_freq.update(set(toks))

        # TF-IDF per document (stored as sparse dict)
        n_docs = len(all_tokens)
        self._sparse_matrix: List[Dict[int, float]] = []
        for toks in all_tokens:
            tf = Counter(toks)
            doc_len = len(toks)
            row: Dict[int, float] = {}
            for term, col in self._sparse_vocab.items():
                if term in tf:
                    term_tf = tf[term] / doc_len if doc_len else 0.0
                    idf = math.log((1 + n_docs) / (1 + self._doc_freq[term]))
                    val = term_tf * idf
                    if val > 0:
                        row[col] = val
            self._sparse_matrix.append(row)

        logger.debug(
            "TF-IDF sparse index: %d terms, %d documents",
            len(self._sparse_vocab),
            n_docs,
        )

    def _sparse_search(self, query_text: str, k: int) -> List[int]:
        """Return the top-k exemplar indices by TF-IDF cosine similarity.

        Args:
            query_text: Expanded query text (output of _expand_query_with_kb).
            k: Maximum number of indices to return.

        Returns:
            List of exemplar indices sorted by descending cosine similarity.
            Returns [] when the sparse index is not built or query is empty.
        """
        if not getattr(self, "_sparse_matrix", None):
            return []

        q_tokens = re.findall(r"[a-z0-9]+", query_text.lower())
        if not q_tokens:
            return []

        n_docs = len(self._sparse_matrix)
        q_tf = Counter(q_tokens)
        q_len = len(q_tokens)

        # Build query TF-IDF vector (sparse dict)
        q_vec: Dict[int, float] = {}
        for term, cnt in q_tf.items():
            if term in self._sparse_vocab:
                col = self._sparse_vocab[term]
                idf = math.log(
                    (1 + n_docs) / (1 + self._doc_freq.get(term, 0))
                )
                val = (cnt / q_len) * idf
                if val > 0:
                    q_vec[col] = val

        if not q_vec:
            return []

        q_norm = math.sqrt(sum(v * v for v in q_vec.values()))
        scores: List[Tuple[int, float]] = []
        for doc_idx, doc_vec in enumerate(self._sparse_matrix):
            dot = sum(q_vec.get(col, 0.0) * val for col, val in doc_vec.items())
            d_norm = math.sqrt(sum(v * v for v in doc_vec.values()))
            sim = dot / (q_norm * d_norm) if q_norm > 0 and d_norm > 0 else 0.0
            scores.append((doc_idx, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in scores[:k]]
    
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
        """Hybrid retrieval: dense HNSW + sparse TF-IDF merged via RRF.

        Combines:
        1. Dense cosine similarity search via the HNSW index (observation embedding).
        2. Sparse TF-IDF cosine similarity over exemplar texts, with the query
           expanded by up to 3 entity KB terms for improved recall.
        3. Reciprocal Rank Fusion (k=60) to merge both ranked lists into a
           single ordering that captures complementary signals.

        Args:
            observation: Normalized observation with an embedding vector.

        Returns:
            List of ``(signal_type, score, reasoning)`` tuples, most relevant
            first, capped at ``self.top_k``.
        """
        if not observation.embedding or not self.hnsw_index:
            return []

        # ── 1. Dense HNSW retrieval ───────────────────────────────────────────
        hnsw_results = self.hnsw_index.search(
            query_vector=observation.embedding,
            k=self.top_k,
        )
        dense_indices: List[int] = []
        dense_scores: Dict[int, float] = {}
        for result in hnsw_results:
            try:
                idx = int(result.id)
                if idx < len(self.exemplar_bank):
                    dense_indices.append(idx)
                    dense_scores[idx] = max(0.0, min(1.0, 1.0 - result.distance))
            except (ValueError, IndexError):
                pass

        # ── 2. Sparse TF-IDF retrieval (query-expanded) ───────────────────────
        query_text = observation.normalized_text or observation.title or ""
        try:
            from app.intelligence.normalization import _ENTITY_KB  # lazy import
            entity_kb = _ENTITY_KB
        except Exception:
            # Normalization module unavailable (e.g. circular import in tests).
            # Use an empty KB — query expansion is purely additive, so skipping
            # it never degrades correctness, only slightly reduces recall.
            entity_kb = {}

        expanded_text = _expand_query_with_kb(query_text, entity_kb)
        sparse_indices = self._sparse_search(expanded_text, k=self.top_k)

        # ── 3. RRF merge ──────────────────────────────────────────────────────
        merged = _rrf_merge([dense_indices, sparse_indices])

        # ── 4. Convert to (SignalType, score, reason) tuples ─────────────────
        candidates = []
        for idx, rrf_score in merged[: self.top_k]:
            if idx >= len(self.exemplar_bank):
                continue
            exemplar = self.exemplar_bank[idx]
            dense_sim = dense_scores.get(idx, 0.0)
            in_sparse = idx in sparse_indices
            reason = (
                f"Hybrid RRF (dense_sim={dense_sim:.2f}, sparse={'yes' if in_sparse else 'no'}): "
                f"'{exemplar.text[:50]}...'"
            )
            candidates.append((
                exemplar.signal_type,
                max(0.0, min(1.0, rrf_score * 10)),  # scale RRF score to [0,1]
                reason,
            ))
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


# ---------------------------------------------------------------------------
# ExemplarBank — global cross-user high-confidence exemplar store (Step 3a)
# ---------------------------------------------------------------------------


class ExemplarBank:
    """Thread-safe, per-``SignalType``-capped store of high-confidence exemplars.

    Exemplars are sourced from non-abstained inferences with calibrated
    probability ≥ 0.85 across **all** users.  ``CandidateRetriever`` can query
    the bank as an additional retrieval source alongside the per-user RAG pool
    via :meth:`search_similar`.

    Design constraints
    ------------------
    * Capped at ``max_per_signal_type`` exemplars per ``SignalType`` (default
      10 000).  When the cap is reached, the exemplar with the **lowest**
      confidence is evicted using a min-heap — **O(log n)** per insert rather
      than O(n) sort.
    * All public methods are thread-safe via a single ``threading.Lock``.
    * ``add_nonblocking()`` dispatches ``add()`` to
      ``asyncio.get_running_loop().run_in_executor`` so async callers are not
      blocked.
    * State can be serialised / restored via :meth:`persist` / :meth:`load`.

    Args:
        max_per_signal_type: Per-``SignalType`` exemplar cap.
    """

    _MAX_PER_SIGNAL_TYPE: int = 10_000

    def __init__(self, max_per_signal_type: int = _MAX_PER_SIGNAL_TYPE) -> None:
        """Initialise an empty exemplar bank.

        Args:
            max_per_signal_type: Cap per ``SignalType`` before lowest-confidence
                eviction.
        """
        self._max: int = max_per_signal_type
        # signal_type.value → min-heap of (confidence, seq_id, ExemplarSignal)
        # The seq_id breaks ties deterministically (monotonically increasing
        # counter) so that the heap comparison never falls through to the
        # non-comparable ExemplarSignal object.
        self._bank: Dict[str, List[Tuple[float, int, ExemplarSignal]]] = {}
        self._seq: int = 0  # global monotonic insertion counter
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, exemplar: ExemplarSignal, confidence: float) -> None:
        """Add ``exemplar`` to the bank; evict lowest-confidence on overflow.

        Uses a min-heap so the eviction is **O(log n)** rather than O(n).

        This method is safe to call from multiple threads concurrently.

        Args:
            exemplar: Exemplar signal to store.
            confidence: Calibrated probability of the inference that produced
                this exemplar.  Used for eviction ordering (lowest evicted
                first when the cap is reached).
        """
        key = exemplar.signal_type.value
        with self._lock:
            if key not in self._bank:
                self._bank[key] = []
            heap = self._bank[key]
            seq = self._seq
            self._seq += 1
            heapq.heappush(heap, (confidence, seq, exemplar))
            if len(heap) > self._max:
                heapq.heappop(heap)  # O(log n) — removes minimum-confidence entry

    async def add_nonblocking(
        self, exemplar: ExemplarSignal, confidence: float
    ) -> None:
        """Async wrapper around ``add()`` — does not block the event loop.

        Dispatches ``add()`` to the default thread-pool executor so the await
        returns quickly.  Uses ``asyncio.get_running_loop()`` (Python 3.7+).

        Args:
            exemplar: Exemplar signal to store.
            confidence: Calibrated probability.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.add, exemplar, confidence)

    def get_for_signal_type(
        self, signal_type: SignalType, top_k: int = 50
    ) -> List[ExemplarSignal]:
        """Return the top-``top_k`` highest-confidence exemplars for ``signal_type``.

        Args:
            signal_type: Signal type to retrieve exemplars for.
            top_k: Maximum number of exemplars to return.

        Returns:
            List of ``ExemplarSignal`` objects sorted by confidence descending.
            Empty list if the signal type has no exemplars.
        """
        key = signal_type.value
        with self._lock:
            heap_copy = list(self._bank.get(key, []))
        # Sort copy descending by confidence (index 0 in the tuple)
        heap_copy.sort(key=lambda t: t[0], reverse=True)
        return [ex for _, _seq, ex in heap_copy[:top_k]]

    def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        signal_type: Optional[SignalType] = None,
    ) -> List[Tuple[float, "ExemplarSignal"]]:
        """Return the ``top_k`` exemplars most similar to ``query_embedding``.

        Uses cosine similarity over the stored ``ExemplarSignal.embedding``
        vectors.  If ``signal_type`` is specified, only exemplars of that type
        are searched.

        This allows ``CandidateRetriever`` to use the bank as an additional
        retrieval source alongside the per-user RAG pool.

        Args:
            query_embedding: Query vector to compare against stored embeddings.
                Must be non-empty and have at least one non-zero component.
            top_k: Maximum number of exemplars to return.
            signal_type: Optional filter; when provided only exemplars of this
                type are searched.

        Returns:
            List of ``(cosine_similarity, ExemplarSignal)`` pairs, sorted by
            similarity descending.  Empty list if the bank is empty or all
            embeddings have zero norm.
        """
        _t0 = time.perf_counter()
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-9:
            return []

        with self._lock:
            # Gather all relevant (confidence, seq, exemplar) triples
            if signal_type is not None:
                items: List[Tuple[float, int, ExemplarSignal]] = list(
                    self._bank.get(signal_type.value, [])
                )
            else:
                items = [
                    entry
                    for bucket in self._bank.values()
                    for entry in bucket
                ]

        scored: List[Tuple[float, ExemplarSignal]] = []
        for _conf, _seq, exemplar in items:
            emb = exemplar.embedding
            if emb is None or len(emb) == 0:
                continue
            v = np.array(emb, dtype=np.float32)
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-9:
                continue
            sim = float(np.dot(q, v) / (q_norm * v_norm))
            scored.append((sim, exemplar))

        scored.sort(key=lambda t: t[0], reverse=True)
        results = scored[:top_k]
        logger.debug(
            "ExemplarBank.search_similar: top_k=%d results=%d latency_ms=%.1f",
            top_k, len(results), (time.perf_counter() - _t0) * 1000,
        )
        return results

    def total_size(self) -> int:
        """Return the total number of exemplars across all signal types."""
        with self._lock:
            return sum(len(v) for v in self._bank.values())

    def size_per_type(self) -> Dict[str, int]:
        """Return a dict mapping each ``SignalType.value`` to its bucket size."""
        with self._lock:
            return {k: len(v) for k, v in self._bank.items()}

    def clear(self) -> None:
        """Remove all exemplars from the bank (primarily used in tests)."""
        with self._lock:
            self._bank.clear()
            self._seq = 0

    def get_signal_types_represented(self) -> Set[str]:
        """Return the set of ``SignalType.value`` strings present in the bank."""
        with self._lock:
            return set(self._bank.keys())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, path: Path) -> None:
        """Serialise the bank to ``path`` as a JSON file (atomic write).

        Each exemplar is stored with its confidence, signal type, and embedding
        so the bank can be fully restored via :meth:`load`.

        The write is atomic: payload is written to a sibling ``.tmp`` file and
        then renamed over ``path``.

        Args:
            path: Destination file path (created / overwritten).
        """
        path = Path(path)
        _t0 = time.perf_counter()
        with self._lock:
            serialised: Dict[str, List[Dict]] = {}
            for key, heap in self._bank.items():
                serialised[key] = [
                    {
                        "confidence": conf,
                        "exemplar": {
                            "signal_type": ex.signal_type.value,
                            "text": ex.text,
                            "embedding": ex.embedding or [],
                            "entities": ex.entities or [],
                            "platform": ex.platform or "",
                        },
                    }
                    for conf, _seq, ex in heap
                ]
            payload = {"version": "1.0", "max_per_signal_type": self._max, "bank": serialised}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        logger.info(
            "ExemplarBank.persist: wrote %d exemplars to %s in %.1f ms",
            sum(len(v) for v in serialised.values()),
            path, (time.perf_counter() - _t0) * 1000,
        )

    def load(self, path: Path) -> None:
        """Restore bank state from a file written by :meth:`persist`.

        Existing bank contents are **replaced** (not merged) with the loaded
        data.  The min-heap invariant is re-established via ``heapq.heapify``.

        Args:
            path: Source file path.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file is not valid JSON or has an unsupported
                version.
        """
        from app.domain.inference_models import SignalType as _ST

        path = Path(path)
        _t0 = time.perf_counter()
        raw = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ExemplarBank.load: invalid JSON in {path}: {exc}") from exc

        new_bank: Dict[str, List[Tuple[float, int, ExemplarSignal]]] = {}
        seq = 0
        for key, entries in payload.get("bank", {}).items():
            heap: List[Tuple[float, int, ExemplarSignal]] = []
            for item in entries:
                conf = float(item["confidence"])
                ed = item["exemplar"]
                try:
                    st = _ST(ed["signal_type"])
                except ValueError:
                    continue  # skip unknown signal types gracefully
                ex = ExemplarSignal(
                    signal_type=st,
                    text=ed.get("text", ""),
                    embedding=ed.get("embedding") or [],
                    entities=ed.get("entities") or [],
                    platform=ed.get("platform", ""),
                )
                heap.append((conf, seq, ex))
                seq += 1
            heapq.heapify(heap)
            new_bank[key] = heap

        with self._lock:
            self._bank = new_bank
            self._seq = seq

        total = sum(len(v) for v in new_bank.values())
        logger.info(
            "ExemplarBank.load: loaded %d exemplars from %s in %.1f ms",
            total, path, (time.perf_counter() - _t0) * 1000,
        )


#: Module-level singleton — shared across all users and all pipeline instances.
#: Populated asynchronously after every qualifying inference by
#: ``InferencePipeline.run()``.
_GLOBAL_EXEMPLAR_BANK: ExemplarBank = ExemplarBank()


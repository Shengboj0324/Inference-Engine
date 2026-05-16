"""Multi-source synthesizer.

Provides two capabilities on top of ``GroundedSummaryBuilder``:

1. **Source deduplication** — removes near-duplicate ``EvidenceSource``
   objects before synthesis so the downstream pipeline is not skewed by
   repeated identical content from different platforms.

2. **Claim merging** — consolidates ``AttributedClaim`` objects that assert
   the same fact (high token-overlap) into a single, higher-confidence claim
   that carries the union of supporting source_ids.

3. **Full synthesis** — calls ``GroundedSummaryBuilder.build()`` on the
   de-duplicated, claim-merged request and returns a ``GroundedSummary``.

Deduplication algorithm
------------------------
Two sources are considered near-duplicates when their content_snippet
Jaccard-similarity exceeds ``dedup_threshold``.  The source with the higher
``trust_score`` is kept; in case of a tie, the earlier one (lower index) is
retained.  Deduplication is O(n²) and adequate for typical digest sizes
(≤ 100 sources).

Claim merging algorithm
------------------------
Claims are merged greedily: a new claim is started whenever its token
overlap with all existing merged claims is below ``merge_threshold``.
When two claims are merged:
- The text of the higher-confidence claim is kept.
- ``confidence`` = min(1.0, mean of both confidences × 1.05) (small boost
  because corroboration increases confidence).
- ``source_ids`` = union of both source_id lists.
- ``negation_detected`` = True if either input is negated (conservative).

Optional LLM path
-----------------
Passed through to ``GroundedSummaryBuilder``.  No direct LLM use in the
dedup / merge steps (pure heuristic).

Thread safety
-------------
``MultiSourceSynthesizer`` is stateless; all public methods are re-entrant.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from app.summarization.models import (
    AttributedClaim,
    EvidenceSource,
    GroundedSummary,
    SynthesisRequest,
)
from app.summarization.grounded_summary_builder import GroundedSummaryBuilder
from app.summarization.source_attribution import _tokenise

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Source-relationship graph (Chain Analysis)
# ---------------------------------------------------------------------------

class SourceRelationshipType(str, Enum):
    """Kinds of cross-source links the synthesizer detects."""

    CITES = "cites"                          # snippet of A references URL of B
    SHARES_GITHUB_REPO = "shares_github_repo"
    SHARES_YOUTUBE_VIDEO = "shares_youtube_video"
    SHARES_ENTITY = "shares_entity"          # high lexical / entity overlap
    TEMPORAL_FOLLOWS = "temporal_follows"    # B published after A (same chain)


@dataclass(frozen=True)
class SourceEdge:
    """Directed edge between two sources in a :class:`SourceRelationshipGraph`.

    Attributes:
        from_source_id: ``EvidenceSource.source_id`` of the originator.
        to_source_id:   ``EvidenceSource.source_id`` of the target.
        relation:       Detected ``SourceRelationshipType``.
        evidence:       Short string describing why the edge exists (URL,
                        repo slug, video id, overlap fraction, …).
    """

    from_source_id: str
    to_source_id: str
    relation: SourceRelationshipType
    evidence: str = ""


@dataclass
class SourceRelationshipGraph:
    """Directed multigraph of :class:`EvidenceSource` links.

    Two helper accessors (``edges_for`` and ``connected_components``) make it
    easy for downstream code to render chain narratives or attach
    per-component trust / temporal annotations.
    """

    nodes: Dict[str, EvidenceSource] = field(default_factory=dict)
    edges: List[SourceEdge] = field(default_factory=list)

    def add_node(self, source: EvidenceSource) -> None:
        self.nodes[source.source_id] = source

    def add_edge(self, edge: SourceEdge) -> None:
        if edge.from_source_id == edge.to_source_id:
            return
        if edge.from_source_id not in self.nodes or edge.to_source_id not in self.nodes:
            return
        self.edges.append(edge)

    def edges_for(self, source_id: str) -> List[SourceEdge]:
        return [e for e in self.edges
                if e.from_source_id == source_id or e.to_source_id == source_id]

    def connected_components(self) -> List[List[str]]:
        """Return source-id sets that form weakly-connected components."""
        adj: Dict[str, Set[str]] = {sid: set() for sid in self.nodes}
        for e in self.edges:
            adj[e.from_source_id].add(e.to_source_id)
            adj[e.to_source_id].add(e.from_source_id)
        seen: Set[str] = set()
        out: List[List[str]] = []
        for sid in self.nodes:
            if sid in seen:
                continue
            comp: List[str] = []
            stack = [sid]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.append(cur)
                stack.extend(adj[cur] - seen)
            out.append(comp)
        return out


# Regex helpers for chain detection.  Anchored to public URL shapes only so
# private or self-hosted hostnames are never mis-classified as a GitHub repo
# or YouTube video.
_GITHUB_REPO_RE = re.compile(
    r"github\.com/([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+?)(?:/|\.git|$|[?#])"
)
_YOUTUBE_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)"
    r"([A-Za-z0-9_\-]{11})"
)


def _github_repo_of(source: EvidenceSource) -> Optional[str]:
    """Return ``"org/repo"`` slug if the source URL points to a GitHub repo."""
    if not source.url:
        return None
    m = _GITHUB_REPO_RE.search(source.url)
    return m.group(1).lower() if m else None


def _youtube_id_of(source: EvidenceSource) -> Optional[str]:
    """Return the 11-character YouTube video id if the URL is a YouTube video."""
    if not source.url:
        return None
    m = _YOUTUBE_ID_RE.search(source.url)
    return m.group(1) if m else None

_DEFAULT_DEDUP_THRESHOLD: float = 0.70  # Jaccard similarity above which sources are dupes
_DEFAULT_MERGE_THRESHOLD: float = 0.60  # token-overlap above which claims are merged


def _source_jaccard(a: EvidenceSource, b: EvidenceSource) -> float:
    """Title + snippet Jaccard similarity between two sources."""
    tokens_a = _tokenise(a.content_snippet + " " + a.title)
    tokens_b = _tokenise(b.content_snippet + " " + b.title)
    if not tokens_a or not tokens_b:
        return 0.0
    inter = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(inter) / len(union)


def _claim_token_overlap(a: AttributedClaim, b: AttributedClaim) -> float:
    """Recall-based token overlap: |tokens_a ∩ tokens_b| / |tokens_a|."""
    ta = _tokenise(a.text)
    tb = _tokenise(b.text)
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


class MultiSourceSynthesizer:
    """Deduplicates sources, merges claims, then builds a ``GroundedSummary``.

    Args:
        dedup_threshold: Jaccard similarity above which sources are near-dupes.
        merge_threshold: Token-overlap above which claims are consolidated.
        builder:         ``GroundedSummaryBuilder`` to delegate the final build.
        llm_router:      Optional LLM router passed to the builder.
    """

    def __init__(
        self,
        dedup_threshold: float = _DEFAULT_DEDUP_THRESHOLD,
        merge_threshold: float = _DEFAULT_MERGE_THRESHOLD,
        builder: Optional[GroundedSummaryBuilder] = None,
        llm_router: Optional[Any] = None,
    ) -> None:
        if not (0.0 < dedup_threshold <= 1.0):
            raise ValueError(f"'dedup_threshold' must be in (0, 1], got {dedup_threshold!r}")
        if not (0.0 < merge_threshold <= 1.0):
            raise ValueError(f"'merge_threshold' must be in (0, 1], got {merge_threshold!r}")
        if builder is not None and not isinstance(builder, GroundedSummaryBuilder):
            raise TypeError(f"'builder' must be GroundedSummaryBuilder or None, got {type(builder)!r}")

        self._dedup_t = dedup_threshold
        self._merge_t = merge_threshold
        self._builder = builder or GroundedSummaryBuilder(llm_router=llm_router)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, request: SynthesisRequest) -> GroundedSummary:
        """Deduplicate sources and synthesize a ``GroundedSummary``.

        Args:
            request: Validated ``SynthesisRequest`` (possibly with duplicate sources).

        Returns:
            ``GroundedSummary`` built from the deduplicated source set.

        Raises:
            TypeError: *request* is not a ``SynthesisRequest``.
        """
        if not isinstance(request, SynthesisRequest):
            raise TypeError(f"'request' must be SynthesisRequest, got {type(request)!r}")

        deduped = self.deduplicate_sources(request.sources)
        logger.info(
            "MultiSourceSynthesizer.synthesize: %d → %d sources after dedup",
            len(request.sources), len(deduped),
        )
        clean_request = SynthesisRequest(
            topic=request.topic,
            sources=deduped,
            context=request.context,
            max_claims=request.max_claims,
            min_source_trust=request.min_source_trust,
            who_it_affects=request.who_it_affects,
        )
        return self._builder.build(clean_request)

    def deduplicate_sources(self, sources: List[EvidenceSource]) -> List[EvidenceSource]:
        """Remove near-duplicate sources, keeping the highest-trust copy.

        Args:
            sources: List of ``EvidenceSource`` objects.

        Returns:
            Deduplicated list, preserving original ordering of kept items.

        Raises:
            TypeError: *sources* is not a list.
        """
        if not isinstance(sources, list):
            raise TypeError(f"'sources' must be a list, got {type(sources)!r}")
        if len(sources) <= 1:
            return list(sources)

        dropped: Set[int] = set()
        for i in range(len(sources)):
            if i in dropped:
                continue
            for j in range(i + 1, len(sources)):
                if j in dropped:
                    continue
                sim = _source_jaccard(sources[i], sources[j])
                if sim >= self._dedup_t:
                    # Drop the lower-trust one; ties → drop the later one (j)
                    if sources[j].trust_score > sources[i].trust_score:
                        dropped.add(i)
                        break
                    else:
                        dropped.add(j)

        result = [s for idx, s in enumerate(sources) if idx not in dropped]
        logger.debug(
            "deduplicate_sources: %d → %d (threshold=%.2f)",
            len(sources), len(result), self._dedup_t,
        )
        return result

    def merge_claims(self, claims: List[AttributedClaim]) -> List[AttributedClaim]:
        """Consolidate near-duplicate claims into corroborated single claims.

        Args:
            claims: ``AttributedClaim`` list (from one or more sources).

        Returns:
            Reduced list of merged ``AttributedClaim`` objects.

        Raises:
            TypeError: *claims* is not a list.
        """
        if not isinstance(claims, list):
            raise TypeError(f"'claims' must be a list, got {type(claims)!r}")
        if not claims:
            return []

        merged: List[AttributedClaim] = []
        for claim in claims:
            absorbed = False
            for idx, existing in enumerate(merged):
                overlap = _claim_token_overlap(claim, existing)
                if overlap >= self._merge_t:
                    # Keep the higher-confidence text, boost confidence slightly
                    if claim.confidence >= existing.confidence:
                        primary_text = claim.text
                        primary_type = claim.claim_type
                    else:
                        primary_text = existing.text
                        primary_type = existing.claim_type
                    merged_confidence = min(
                        1.0, (claim.confidence + existing.confidence) / 2.0 * 1.05
                    )
                    combined_source_ids = list(
                        dict.fromkeys(existing.source_ids + claim.source_ids)
                    )
                    merged[idx] = AttributedClaim(
                        claim_id=existing.claim_id,
                        text=primary_text,
                        claim_type=primary_type,
                        confidence=round(merged_confidence, 5),
                        source_ids=combined_source_ids,
                        negation_detected=existing.negation_detected or claim.negation_detected,
                        extracted_at=existing.extracted_at,
                    )
                    absorbed = True
                    break
            if not absorbed:
                merged.append(claim)

        logger.debug("merge_claims: %d → %d claims (threshold=%.2f)", len(claims), len(merged), self._merge_t)
        return merged

    # ------------------------------------------------------------------
    # Chain analysis: cross-source relationship graph + narrative
    # ------------------------------------------------------------------

    def build_relationship_graph(
        self,
        sources: List[EvidenceSource],
        entity_overlap_threshold: float = 0.35,
    ) -> SourceRelationshipGraph:
        """Build a directed link graph across *sources*.

        Detects five categories of edges:

        * ``CITES``                — URL of source *B* appears verbatim in
          source *A*'s ``content_snippet``.
        * ``SHARES_GITHUB_REPO``   — both sources reference the same
          ``github.com/<org>/<repo>`` slug (case-insensitive).
        * ``SHARES_YOUTUBE_VIDEO`` — both sources reference the same
          11-character YouTube video id.
        * ``SHARES_ENTITY``        — Jaccard overlap of the ``title + snippet``
          token sets is ≥ *entity_overlap_threshold* but the sources are
          not strict duplicates (handled by :meth:`deduplicate_sources`).
        * ``TEMPORAL_FOLLOWS``     — for every other edge between *A* and *B*
          with both ``published_at`` set, an additional directed
          ``TEMPORAL_FOLLOWS`` edge from the earlier to the later source is
          recorded so callers can trivially read the chain in publication
          order.

        The graph is O(n²) over *sources* and intended for digest-sized
        inputs (≤ 100 sources).

        Args:
            sources: Evidence sources from the same digest / query window.
            entity_overlap_threshold: Jaccard floor for ``SHARES_ENTITY``;
                must be in ``(0.0, 1.0]``.

        Returns:
            Populated :class:`SourceRelationshipGraph`.

        Raises:
            TypeError: *sources* is not a list.
            ValueError: *entity_overlap_threshold* is out of range.
        """
        if not isinstance(sources, list):
            raise TypeError(f"'sources' must be a list, got {type(sources)!r}")
        if not (0.0 < entity_overlap_threshold <= 1.0):
            raise ValueError(
                f"'entity_overlap_threshold' must be in (0, 1], got {entity_overlap_threshold!r}"
            )

        graph = SourceRelationshipGraph()
        for src in sources:
            graph.add_node(src)

        gh_index: Dict[str, List[str]] = {}
        yt_index: Dict[str, List[str]] = {}
        for src in sources:
            slug = _github_repo_of(src)
            if slug:
                gh_index.setdefault(slug, []).append(src.source_id)
            vid = _youtube_id_of(src)
            if vid:
                yt_index.setdefault(vid, []).append(src.source_id)

        for slug, ids in gh_index.items():
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    graph.add_edge(SourceEdge(
                        from_source_id=ids[i], to_source_id=ids[j],
                        relation=SourceRelationshipType.SHARES_GITHUB_REPO,
                        evidence=f"github.com/{slug}",
                    ))
        for vid, ids in yt_index.items():
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    graph.add_edge(SourceEdge(
                        from_source_id=ids[i], to_source_id=ids[j],
                        relation=SourceRelationshipType.SHARES_YOUTUBE_VIDEO,
                        evidence=f"youtube:{vid}",
                    ))

        for i in range(len(sources)):
            a = sources[i]
            for j in range(len(sources)):
                if i == j:
                    continue
                b = sources[j]
                if b.url and b.url in a.content_snippet:
                    graph.add_edge(SourceEdge(
                        from_source_id=a.source_id, to_source_id=b.source_id,
                        relation=SourceRelationshipType.CITES,
                        evidence=b.url[:120],
                    ))

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                sim = _source_jaccard(sources[i], sources[j])
                if entity_overlap_threshold <= sim < self._dedup_t:
                    graph.add_edge(SourceEdge(
                        from_source_id=sources[i].source_id,
                        to_source_id=sources[j].source_id,
                        relation=SourceRelationshipType.SHARES_ENTITY,
                        evidence=f"jaccard={sim:.2f}",
                    ))

        existing_pairs: Set[Tuple[str, str]] = {
            tuple(sorted((e.from_source_id, e.to_source_id)))
            for e in graph.edges
        }
        by_id = {s.source_id: s for s in sources}
        for a_id, b_id in existing_pairs:
            a = by_id[a_id]
            b = by_id[b_id]
            if a.published_at is None or b.published_at is None:
                continue
            if a.published_at == b.published_at:
                continue
            earlier, later = (a, b) if a.published_at < b.published_at else (b, a)
            graph.add_edge(SourceEdge(
                from_source_id=earlier.source_id, to_source_id=later.source_id,
                relation=SourceRelationshipType.TEMPORAL_FOLLOWS,
                evidence=(
                    f"{earlier.published_at.isoformat()} -> "
                    f"{later.published_at.isoformat()}"
                ),
            ))

        logger.info(
            "build_relationship_graph: nodes=%d edges=%d",
            len(graph.nodes), len(graph.edges),
        )
        return graph

    def chain_analysis(
        self,
        sources: List[EvidenceSource],
        entity_overlap_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """Return a structured chain-analysis report for *sources*.

        Builds the relationship graph and then, for every weakly-connected
        component of size ≥ 2, emits:

        * ``component_id``     — zero-based index in the returned list.
        * ``source_ids``       — member source ids in publication order
                                 (sources without ``published_at`` go last).
        * ``narrative``        — short human-readable arc describing the
                                 chain (e.g. *"YouTube video discusses
                                 github.com/foo/bar then The Verge cites
                                 the video"*).
        * ``relation_summary`` — dict mapping relation-type → count for the
                                 component, useful for downstream UI badges.

        Isolated nodes (components of size 1) are returned separately under
        ``isolated_source_ids`` so callers can decide whether to surface them
        as standalone items.

        Args:
            sources: Evidence sources from the same digest / query window.
            entity_overlap_threshold: Forwarded to
                :meth:`build_relationship_graph`.

        Returns:
            ``{"graph": SourceRelationshipGraph, "chains": [...],
              "isolated_source_ids": [...]}``.
        """
        graph = self.build_relationship_graph(
            sources, entity_overlap_threshold=entity_overlap_threshold
        )
        chains: List[Dict[str, Any]] = []
        isolated: List[str] = []
        components = graph.connected_components()

        for cid, comp in enumerate(components):
            if len(comp) < 2:
                isolated.extend(comp)
                continue
            members = [graph.nodes[sid] for sid in comp]
            members.sort(
                key=lambda s: (s.published_at is None, s.published_at or 0)
            )
            ordered_ids = [s.source_id for s in members]
            comp_edges = [
                e for e in graph.edges
                if e.from_source_id in comp and e.to_source_id in comp
            ]
            relation_summary: Dict[str, int] = {}
            for e in comp_edges:
                relation_summary[e.relation.value] = (
                    relation_summary.get(e.relation.value, 0) + 1
                )

            parts: List[str] = []
            for idx, src in enumerate(members):
                label = src.title or src.platform or src.source_id[:8]
                ts = src.published_at.isoformat() if src.published_at else "undated"
                parts.append(f"[{idx + 1}] {label} ({src.platform or 'unknown'}, {ts})")
            arc = " -> ".join(parts)
            relations_human = ", ".join(
                f"{k}×{v}" for k, v in sorted(relation_summary.items())
            )
            narrative = (
                f"Chain across {len(members)} sources linked via "
                f"{relations_human}: {arc}"
            )
            chains.append({
                "component_id": cid,
                "source_ids": ordered_ids,
                "narrative": narrative,
                "relation_summary": relation_summary,
            })

        logger.info(
            "chain_analysis: components=%d chains=%d isolated=%d",
            len(components), len(chains), len(isolated),
        )
        return {
            "graph": graph,
            "chains": chains,
            "isolated_source_ids": isolated,
        }


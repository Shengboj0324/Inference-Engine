"""Citation graph.

Builds an in-memory directed citation graph from the references section
of an academic paper.  Supports:
- Shortest-path queries (BFS)
- Influence scoring (in-degree of each node)
- Neighbour lookup
- Serialisation to adjacency dict

The graph is stored as a dict-of-sets (adjacency list) for O(1) edge lookup.
Nodes are ``CitationNode`` objects keyed by ``paper_id``.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from app.document_intelligence.models import CitationEdge, CitationNode, DocumentSection, SectionType

logger = logging.getLogger(__name__)

_REF_LINE = re.compile(
    r"^\s*\[?\d+\]?\s*(.{10,300})$",
    re.MULTILINE,
)
_ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")
_YEAR = re.compile(r"\b(20\d{2}|19[89]\d)\b")
_DOI = re.compile(r"\b(10\.\d{4,9}/\S+)\b")


class CitationGraph:
    """In-memory directed citation graph for a focal paper.

    Edges go from the focal paper → cited paper (``cites`` direction).

    Args:
        focal_id: ``paper_id`` of the paper being analyzed.
    """

    def __init__(self, focal_id: str) -> None:
        if not focal_id or not isinstance(focal_id, str):
            raise ValueError("'focal_id' must be a non-empty string")
        self._focal_id = focal_id
        self._nodes: Dict[str, CitationNode] = {}
        self._edges: Dict[str, Set[str]] = {}  # source_id → set of target_ids
        self._edge_contexts: Dict[Tuple[str, str], str] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build_from_sections(self, sections: List[DocumentSection], focal_node: Optional[CitationNode] = None) -> None:
        """Populate graph from parsed paper sections.

        Processes the ``REFERENCES`` section to extract cited papers, and
        other sections to extract in-text citation contexts.

        Args:
            sections:    Parsed document sections.
            focal_node:  ``CitationNode`` for the focal paper itself.
        """
        if not isinstance(sections, list):
            raise TypeError(f"'sections' must be a list, got {type(sections)!r}")

        # Register focal node
        focal = focal_node or CitationNode(paper_id=self._focal_id, title="Focal Paper", is_focal=True)
        self.add_node(focal)

        # Extract references
        for sec in sections:
            if sec.section_type == SectionType.REFERENCES:
                self._parse_references(sec.text)
                break

        logger.debug("CitationGraph: %d nodes, %d edges", len(self._nodes), self.edge_count())

    def add_node(self, node: CitationNode) -> None:
        """Add or update a ``CitationNode``."""
        if not isinstance(node, CitationNode):
            raise TypeError(f"Expected CitationNode, got {type(node)!r}")
        self._nodes[node.paper_id] = node

    def add_edge(self, source_id: str, target_id: str, context: str = "") -> None:
        """Add a directed edge (citation) from *source_id* → *target_id*.

        Both nodes must be added separately via ``add_node()``.
        """
        if not source_id or not target_id:
            raise ValueError("'source_id' and 'target_id' must be non-empty strings")
        if source_id not in self._nodes:
            raise KeyError(f"Source node not in graph: {source_id!r}")
        if target_id not in self._nodes:
            raise KeyError(f"Target node not in graph: {target_id!r}")
        self._edges.setdefault(source_id, set()).add(target_id)
        self._edge_contexts[(source_id, target_id)] = context

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_node(self, paper_id: str) -> Optional[CitationNode]:
        return self._nodes.get(paper_id)

    def neighbours(self, paper_id: str) -> List[CitationNode]:
        """Return nodes cited by *paper_id* (outgoing edges)."""
        targets = self._edges.get(paper_id, set())
        return [self._nodes[t] for t in targets if t in self._nodes]

    def cited_by(self, paper_id: str) -> List[CitationNode]:
        """Return nodes that cite *paper_id* (incoming edges)."""
        result: List[CitationNode] = []
        for src, targets in self._edges.items():
            if paper_id in targets and src in self._nodes:
                result.append(self._nodes[src])
        return result

    def influence_score(self, paper_id: str) -> int:
        """In-degree of *paper_id* (number of papers that cite it)."""
        return sum(1 for targets in self._edges.values() if paper_id in targets)

    def shortest_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """BFS shortest path from *source_id* to *target_id*.

        Returns:
            List of paper_ids on the path (inclusive), or None if unreachable.
        """
        if source_id == target_id:
            return [source_id]
        visited: Set[str] = {source_id}
        queue: deque[List[str]] = deque([[source_id]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            for neighbour in self._edges.get(current, set()):
                if neighbour == target_id:
                    return path + [neighbour]
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(path + [neighbour])
        return None

    def all_nodes(self) -> List[CitationNode]:
        return list(self._nodes.values())

    def all_edges(self) -> List[CitationEdge]:
        edges: List[CitationEdge] = []
        for source, targets in self._edges.items():
            for target in targets:
                ctx = self._edge_contexts.get((source, target), "")
                edges.append(CitationEdge(source_id=source, target_id=target, context=ctx))
        return edges

    def edge_count(self) -> int:
        return sum(len(v) for v in self._edges.values())

    def to_adjacency_dict(self) -> Dict[str, List[str]]:
        """Serialise graph to JSON-friendly adjacency dict."""
        return {src: sorted(targets) for src, targets in self._edges.items()}

    # ------------------------------------------------------------------
    # Reference parsing
    # ------------------------------------------------------------------

    def _parse_references(self, ref_text: str) -> None:
        for match in _REF_LINE.finditer(ref_text):
            raw = match.group(1).strip()
            if not raw or len(raw) < 15:
                continue
            paper_id = self._extract_id(raw)
            year_m = _YEAR.search(raw)
            year = int(year_m.group(1)) if year_m else None
            # Title heuristic: first quoted string or first clause
            title = self._extract_title(raw)
            node = CitationNode(paper_id=paper_id, title=title, year=year)
            self.add_node(node)
            self.add_edge(self._focal_id, paper_id, context=raw[:200])

    @staticmethod
    def _extract_id(text: str) -> str:
        m = _ARXIV_ID.search(text)
        if m:
            return f"arxiv:{m.group(1)}"
        m = _DOI.search(text)
        if m:
            return f"doi:{m.group(1)}"
        import hashlib
        return "ref:" + hashlib.sha256(text.encode()).hexdigest()[:12]

    @staticmethod
    def _extract_title(text: str) -> str:
        m = re.search(r'"([^"]{5,120})"', text)
        if m:
            return m.group(1)
        # First clause up to first comma or period after 20 chars
        m = re.match(r".{20,}?[,.]", text)
        if m:
            return m.group(0).strip("., ")
        return text[:80]


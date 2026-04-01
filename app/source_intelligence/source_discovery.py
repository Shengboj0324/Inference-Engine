"""Source discovery engine.

Discovers new relevant sources from entity names, topic keywords, and
existing source metadata.  The engine does not perform live HTTP calls —
instead it produces ``DiscoveredSource`` candidates that the caller can
inspect and optionally register in ``SourceRegistryStore``.

Discovery strategies (all stateless heuristics):
1. **GitHub repo inference** — entity name → ``{entity}/sdk``,
   ``{entity}/python``, ``{entity}/api`` patterns.
2. **arXiv category mapping** — topic keywords → arXiv category codes.
3. **Podcast keyword matching** — entity name → known AI podcast RSS URLs.
4. **Known source catalogue** — a hardcoded catalogue of authoritative
   AI-industry sources (maintained as a module-level constant).

All methods return ``List[DiscoveredSource]``.  Deduplication against the
existing ``SourceRegistryStore`` is the caller's responsibility.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.models import SourcePlatform
from app.source_intelligence.source_registry import SourceCapability, SourceFamily

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredSource:
    """Candidate source found by the discovery engine.

    Attributes:
        source_id:      Suggested ``SourceSpec.source_id``.
        platform:       Suggested ``SourcePlatform``.
        family:         Suggested ``SourceFamily``.
        confidence:     Heuristic confidence score [0, 1].
        display_name:   Human-readable label.
        homepage:       Canonical URL.
        capabilities:   Suggested ``SourceCapability`` flags.
        discovery_reason: Why this source was suggested.
        metadata:       Arbitrary extra data.
    """

    source_id: str
    platform: SourcePlatform
    family: SourceFamily
    confidence: float
    display_name: str = ""
    homepage: str = ""
    capabilities: frozenset = field(default_factory=frozenset)
    discovery_reason: str = ""
    metadata: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"'confidence' must be in [0, 1], got {self.confidence!r}")


# ---------------------------------------------------------------------------
# Known AI-industry source catalogue
# ---------------------------------------------------------------------------

_KNOWN_GITHUB_REPOS: List[Dict] = [
    {"repo": "openai/openai-python", "org": "OpenAI", "conf": 0.99},
    {"repo": "anthropics/anthropic-sdk-python", "org": "Anthropic", "conf": 0.99},
    {"repo": "google-deepmind/gemma", "org": "Google DeepMind", "conf": 0.95},
    {"repo": "meta-llama/llama", "org": "Meta", "conf": 0.95},
    {"repo": "mistralai/mistral-src", "org": "Mistral AI", "conf": 0.93},
    {"repo": "huggingface/transformers", "org": "Hugging Face", "conf": 0.98},
    {"repo": "pytorch/pytorch", "org": "PyTorch", "conf": 0.97},
    {"repo": "microsoft/autogen", "org": "Microsoft", "conf": 0.92},
    {"repo": "langchain-ai/langchain", "org": "LangChain", "conf": 0.91},
    {"repo": "BerriAI/litellm", "org": "LiteLLM", "conf": 0.88},
]

_KNOWN_ARXIV_QUERIES: List[Dict] = [
    {"query": "cat:cs.AI", "label": "arXiv cs.AI", "conf": 0.95},
    {"query": "cat:cs.LG", "label": "arXiv cs.LG (Machine Learning)", "conf": 0.95},
    {"query": "cat:cs.CL", "label": "arXiv cs.CL (Computation and Language)", "conf": 0.93},
    {"query": "ti:large language model AND cat:cs.CL", "label": "LLM papers (cs.CL)", "conf": 0.90},
    {"query": "ti:transformer AND cat:cs.LG", "label": "Transformer papers (cs.LG)", "conf": 0.88},
]

_KNOWN_PODCAST_FEEDS: List[Dict] = [
    {"feed_url": "https://lexfridman.com/feed/podcast/", "label": "Lex Fridman Podcast", "conf": 0.97},
    {"feed_url": "https://anchor.fm/s/98551cc4/podcast/rss", "label": "Dwarkesh Patel Podcast", "conf": 0.92},
    {"feed_url": "https://feeds.simplecast.com/54nAGcIl", "label": "TWIML AI Podcast", "conf": 0.90},
    {"feed_url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCbmNph6atAoGfqLoCL_duAg", "label": "Andrej Karpathy", "conf": 0.93},
]

_ORG_TO_REPOS: Dict[str, List[str]] = {
    "openai": ["openai/openai-python", "openai/openai-node", "openai/whisper", "openai/tiktoken"],
    "anthropic": ["anthropics/anthropic-sdk-python", "anthropics/claude-3-haiku"],
    "google": ["google-deepmind/gemma", "google-deepmind/graphcast"],
    "meta": ["meta-llama/llama", "facebookresearch/llama-recipes"],
    "mistral": ["mistralai/mistral-src", "mistralai/mistral-inference"],
    "huggingface": ["huggingface/transformers", "huggingface/diffusers", "huggingface/trl"],
}

_TOPIC_TO_ARXIV: Dict[str, str] = {
    "llm": "ti:large language model",
    "transformer": "ti:transformer",
    "rag": "ti:retrieval augmented generation",
    "agent": "ti:autonomous agent AND cat:cs.AI",
    "alignment": "ti:alignment AND cat:cs.AI",
    "diffusion": "ti:diffusion model AND cat:cs.LG",
    "multimodal": "ti:multimodal AND cat:cs.CV",
    "embedding": "ti:embedding AND cat:cs.CL",
}


class SourceDiscoveryEngine:
    """Discovers candidate sources for a set of entities and keywords.

    All methods are stateless (no side effects) and safe to call concurrently.
    """

    def discover_for_entity(self, entity_name: str) -> List[DiscoveredSource]:
        """Return candidate sources relevant to *entity_name*.

        Performs:
        1. Exact match against known GitHub repos (org name).
        2. Fuzzy match against podcast labels.
        3. Returns known high-confidence repositories.

        Args:
            entity_name: E.g. ``"OpenAI"``, ``"Mistral"`` (case-insensitive).

        Returns:
            List of ``DiscoveredSource`` sorted by confidence descending.
        """
        if not entity_name or not isinstance(entity_name, str):
            raise ValueError("'entity_name' must be a non-empty string")

        results: List[DiscoveredSource] = []
        lower = entity_name.lower()

        # GitHub repos by org
        repos = _ORG_TO_REPOS.get(lower, [])
        for repo in repos:
            results.append(DiscoveredSource(
                source_id=repo,
                platform=SourcePlatform.GITHUB_RELEASES,
                family=SourceFamily.DEVELOPER_RELEASE,
                confidence=0.90,
                display_name=f"{repo} GitHub Releases",
                homepage=f"https://github.com/{repo}",
                capabilities=frozenset({SourceCapability.SUPPORTS_SINCE, SourceCapability.VERSIONED, SourceCapability.HAS_STRUCTURED_DATA}),
                discovery_reason=f"Known GitHub repos for entity '{entity_name}'",
            ))

        # Known catalogue match
        for entry in _KNOWN_GITHUB_REPOS:
            if lower in entry["repo"].lower() or lower in entry["org"].lower():
                results.append(DiscoveredSource(
                    source_id=entry["repo"],
                    platform=SourcePlatform.GITHUB_RELEASES,
                    family=SourceFamily.DEVELOPER_RELEASE,
                    confidence=entry["conf"],
                    display_name=f"{entry['org']} — {entry['repo']}",
                    homepage=f"https://github.com/{entry['repo']}",
                    capabilities=frozenset({SourceCapability.SUPPORTS_SINCE, SourceCapability.VERSIONED}),
                    discovery_reason=f"Known AI-industry catalogue match for '{entity_name}'",
                ))

        # Podcast label match
        for pod in _KNOWN_PODCAST_FEEDS:
            if lower in pod["label"].lower():
                results.append(DiscoveredSource(
                    source_id=pod["feed_url"],
                    platform=SourcePlatform.PODCAST_RSS,
                    family=SourceFamily.MEDIA_AUDIO,
                    confidence=pod["conf"],
                    display_name=pod["label"],
                    homepage=pod["feed_url"],
                    capabilities=frozenset({SourceCapability.PROVIDES_AUDIO, SourceCapability.SUPPORTS_SINCE}),
                    discovery_reason=f"Known AI podcast match for '{entity_name}'",
                ))

        # Deduplicate by source_id, keep highest confidence
        seen: Dict[str, DiscoveredSource] = {}
        for src in results:
            if src.source_id not in seen or src.confidence > seen[src.source_id].confidence:
                seen[src.source_id] = src
        logger.debug("SourceDiscoveryEngine.discover_for_entity: entity=%r found=%d", entity_name, len(seen))
        return sorted(seen.values(), key=lambda s: s.confidence, reverse=True)

    def discover_for_topic(self, topic: str) -> List[DiscoveredSource]:
        """Return candidate arXiv queries for a given *topic* keyword.

        Args:
            topic: E.g. ``"llm"``, ``"rag"``, ``"alignment"`` (case-insensitive).

        Returns:
            List of ``DiscoveredSource`` for arXiv.
        """
        if not topic or not isinstance(topic, str):
            raise ValueError("'topic' must be a non-empty string")

        lower = topic.lower()
        results: List[DiscoveredSource] = []

        # Exact keyword match against topic map
        if lower in _TOPIC_TO_ARXIV:
            query = _TOPIC_TO_ARXIV[lower]
            results.append(DiscoveredSource(
                source_id=f"arxiv:{query}",
                platform=SourcePlatform.ARXIV,
                family=SourceFamily.RESEARCH,
                confidence=0.88,
                display_name=f"arXiv: {topic}",
                capabilities=frozenset({SourceCapability.SUPPORTS_SEARCH, SourceCapability.PROVIDES_FULL_TEXT, SourceCapability.SUPPORTS_SINCE}),
                discovery_reason=f"Topic keyword '{topic}' matched arXiv query map",
                metadata={"arxiv_query": query},
            ))

        # Always surface the core AI/ML categories
        for cat in _KNOWN_ARXIV_QUERIES:
            if topic.lower() in cat["label"].lower() or topic.lower() in cat["query"].lower():
                results.append(DiscoveredSource(
                    source_id=f"arxiv:{cat['query']}",
                    platform=SourcePlatform.ARXIV,
                    family=SourceFamily.RESEARCH,
                    confidence=cat["conf"],
                    display_name=cat["label"],
                    capabilities=frozenset({SourceCapability.SUPPORTS_SINCE, SourceCapability.PROVIDES_FULL_TEXT}),
                    discovery_reason=f"arXiv category catalogue match for '{topic}'",
                    metadata={"arxiv_query": cat["query"]},
                ))

        logger.debug("SourceDiscoveryEngine.discover_for_topic: topic=%r found=%d", topic, len(results))
        return sorted(results, key=lambda s: s.confidence, reverse=True)

    def get_known_catalogue(self) -> List[DiscoveredSource]:
        """Return the full hardcoded catalogue of high-priority AI-industry sources."""
        results: List[DiscoveredSource] = []
        for entry in _KNOWN_GITHUB_REPOS:
            results.append(DiscoveredSource(
                source_id=entry["repo"],
                platform=SourcePlatform.GITHUB_RELEASES,
                family=SourceFamily.DEVELOPER_RELEASE,
                confidence=entry["conf"],
                display_name=f"{entry['org']} — {entry['repo']}",
                homepage=f"https://github.com/{entry['repo']}",
                capabilities=frozenset({SourceCapability.SUPPORTS_SINCE, SourceCapability.VERSIONED}),
                discovery_reason="Known AI-industry catalogue",
            ))
        for pod in _KNOWN_PODCAST_FEEDS:
            results.append(DiscoveredSource(
                source_id=pod["feed_url"],
                platform=SourcePlatform.PODCAST_RSS,
                family=SourceFamily.MEDIA_AUDIO,
                confidence=pod["conf"],
                display_name=pod["label"],
                homepage=pod["feed_url"],
                capabilities=frozenset({SourceCapability.PROVIDES_AUDIO, SourceCapability.SUPPORTS_SINCE}),
                discovery_reason="Known AI podcast catalogue",
            ))
        for cat in _KNOWN_ARXIV_QUERIES:
            results.append(DiscoveredSource(
                source_id=f"arxiv:{cat['query']}",
                platform=SourcePlatform.ARXIV,
                family=SourceFamily.RESEARCH,
                confidence=cat["conf"],
                display_name=cat["label"],
                capabilities=frozenset({SourceCapability.SUPPORTS_SINCE, SourceCapability.PROVIDES_FULL_TEXT}),
                discovery_reason="Known arXiv category catalogue",
            ))
        return sorted(results, key=lambda s: s.confidence, reverse=True)


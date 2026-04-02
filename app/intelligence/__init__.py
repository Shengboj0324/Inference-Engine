"""Intelligence layer for content analysis and digest generation.

Imports are **lazy** — heavyweight dependencies (transformers, torch, etc.) are
only pulled in when a specific symbol is first accessed.  This keeps
``import app.intelligence`` fast for unit tests and CLI tools that only need a
subset of the layer.

Usage
-----
    # Fine — no heavy modules loaded yet
    import app.intelligence

    # Triggers lazy import of digest_engine only
    from app.intelligence import DigestEngine
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from app.intelligence.digest_engine import DigestEngine
    from app.intelligence.cluster_summarizer import ClusterSummarizer
    from app.intelligence.inference_pipeline import InferencePipeline
    from app.intelligence.action_pipeline import ActionPipeline
    from app.intelligence.normalization import NormalizationEngine
    from app.intelligence.llm_adjudicator import LLMAdjudicator
    from app.intelligence.response_generator import ResponseGenerator

__all__ = [
    "DigestEngine",
    "ClusterSummarizer",
    "InferencePipeline",
    "ActionPipeline",
    "NormalizationEngine",
    "LLMAdjudicator",
    "ResponseGenerator",
    "MultimodalAnalyzer",
    "CapabilityMode",
]

_MODULE_MAP: dict[str, str] = {
    "DigestEngine":       "app.intelligence.digest_engine",
    "ClusterSummarizer":  "app.intelligence.cluster_summarizer",
    "InferencePipeline":  "app.intelligence.inference_pipeline",
    "ActionPipeline":     "app.intelligence.action_pipeline",
    "NormalizationEngine":"app.intelligence.normalization",
    "LLMAdjudicator":     "app.intelligence.llm_adjudicator",
    "ResponseGenerator":  "app.intelligence.response_generator",
    "MultimodalAnalyzer": "app.intelligence.multimodal",
    "CapabilityMode":     "app.intelligence.multimodal",
}


def __getattr__(name: str):  # noqa: ANN001, ANN201
    if name in _MODULE_MAP:
        module = importlib.import_module(_MODULE_MAP[name])
        obj = getattr(module, name)
        globals()[name] = obj   # cache — subsequent access is O(1)
        return obj
    raise AttributeError(f"module 'app.intelligence' has no attribute {name!r}")

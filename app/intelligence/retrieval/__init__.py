"""Retrieval asset lifecycle package.

Provides the chunk corpus (``ChunkStore``) and retrieval quality evaluation
(``RetrievalEvaluator``) that form the RAG foundation for the inference engine.

Public API
----------
- :class:`ChunkRecord` — single indexable chunk with optional embedding.
- :class:`ChunkStore` — thread-safe in-memory corpus with cosine-similarity
  and keyword search.
- :class:`SearchHit` — scored retrieval result.
- :class:`RetrievalQuery` — labelled query for evaluation.
- :class:`RetrievalMetrics` — aggregated Recall@k, MRR, nDCG@k.
- :class:`RetrievalEvaluator` — computes retrieval metrics for any retriever.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "ChunkRecord",
    "ChunkStore",
    "SearchHit",
    "RetrievalQuery",
    "RetrievalMetrics",
    "RetrievalEvaluator",
]


def __getattr__(name: str):
    if name in {"ChunkRecord", "ChunkStore", "SearchHit"}:
        import importlib
        mod = importlib.import_module("app.intelligence.retrieval.chunk_store")
        return getattr(mod, name)
    if name in {"RetrievalQuery", "RetrievalMetrics", "RetrievalEvaluator"}:
        import importlib
        mod = importlib.import_module("app.intelligence.retrieval.retrieval_evaluator")
        return getattr(mod, name)
    raise AttributeError(f"module 'app.intelligence.retrieval' has no attribute {name!r}")


if TYPE_CHECKING:
    from app.intelligence.retrieval.chunk_store import ChunkRecord, ChunkStore, SearchHit
    from app.intelligence.retrieval.retrieval_evaluator import (
        RetrievalEvaluator,
        RetrievalMetrics,
        RetrievalQuery,
    )


"""Local-RAG control endpoints for the desktop sidecar.

Two routes, both gated by :func:`require_desktop_mode` so they return
404 in server mode and are subject to the loopback-token middleware in
desktop mode:

* ``GET  /api/v1/rag/status`` — reports whether the retrieval layer can
  serve queries right now (key configured + permission granted) and how
  many local items currently carry embeddings.
* ``POST /api/v1/rag/search`` — embeds ``query`` and returns the top-k
  PII-scrubbed snippets from :class:`ContentStore`.

The chat-time path uses the same :class:`LocalRAGRetriever`; these
routes exist so the shell can render a stand-alone "Ask your local feed"
panel without going through the chat session machinery.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import require_desktop_mode
from app.local.rag_retriever import (
    LocalRAGRetriever,
    RetrievedSnippet,
    get_rag_retriever,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_desktop_mode)])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class RAGStatusResponse(BaseModel):
    available: bool = Field(
        ...,
        description="True when the embedding provider is configured AND the USE_LLM_KEY permission is granted.",
    )
    indexed_count: int = Field(
        ..., description="Number of locally stored items that carry an embedding."
    )


class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4_000)
    k: int = Field(6, ge=1, le=20)
    min_score: float = Field(0.0, ge=-1.0, le=1.0)
    platforms: Optional[List[str]] = Field(
        None,
        description="Optional list of SourcePlatform values to restrict the search to.",
    )


class RAGSnippet(BaseModel):
    content_id: str
    title: str
    source_url: str
    source_platform: str
    published_at: float
    score: float
    text: str


class RAGSearchResponse(BaseModel):
    query: str
    snippets: List[RAGSnippet]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _retriever() -> LocalRAGRetriever:
    return get_rag_retriever()


@router.get("/status", response_model=RAGStatusResponse)
async def get_status() -> RAGStatusResponse:
    r = _retriever()
    return RAGStatusResponse(
        available=r.is_available(),
        indexed_count=r.indexed_count(),
    )


@router.post("/search", response_model=RAGSearchResponse)
async def search(request: RAGSearchRequest) -> RAGSearchResponse:
    from app.core.models import SourcePlatform  # local import keeps this route cheap at import time

    platforms = None
    if request.platforms:
        try:
            platforms = [SourcePlatform(p) for p in request.platforms]
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unknown platform value: {exc}",
            )
    r = _retriever()
    snippets: List[RetrievedSnippet] = await r.retrieve(
        request.query, k=request.k, platforms=platforms, min_score=request.min_score,
    )
    return RAGSearchResponse(
        query=request.query,
        snippets=[RAGSnippet(**s.to_dict()) for s in snippets],
    )

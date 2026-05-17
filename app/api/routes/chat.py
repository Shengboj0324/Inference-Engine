"""Chat-session and SSE-streaming endpoints for the desktop sidecar.

All endpoints are gated behind :func:`require_desktop_mode`.  The
non-streaming reply lives at ``POST /sessions/{id}/messages`` and
returns the assembled assistant message; the streaming reply lives at
``POST /sessions/{id}/stream`` and returns ``text/event-stream`` with
``data:`` frames containing each redacted chunk, terminated by a
``data: [DONE]`` sentinel.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import require_desktop_mode
from app.local import chat_streamer
from app.local.chat_store import ChatMessage, ChatSession, get_chat_store

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_desktop_mode)])

# Default streaming model used when the user has not configured a tier.
_DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SessionCreateRequest(BaseModel):
    title: str = Field("New chat", min_length=1, max_length=200)
    model: Optional[str] = Field(None, max_length=200)


class SessionRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class SessionResponse(BaseModel):
    id: str
    title: str
    model: Optional[str]
    created_at: float
    updated_at: float

    @classmethod
    def from_orm_like(cls, s: ChatSession) -> "SessionResponse":
        return cls(id=s.id, title=s.title, model=s.model,
                   created_at=s.created_at, updated_at=s.updated_at)


class MessageResponse(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    tokens: Optional[int]
    created_at: float

    @classmethod
    def from_orm_like(cls, m: ChatMessage) -> "MessageResponse":
        return cls(id=m.id, session_id=m.session_id, role=m.role,
                   content=m.content, tokens=m.tokens, created_at=m.created_at)


class SessionDetail(BaseModel):
    session: SessionResponse
    messages: List[MessageResponse]


class MessageCreateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=100_000)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32_000)


class ReplyResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
    source: str


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions() -> List[SessionResponse]:
    return [SessionResponse.from_orm_like(s)
            for s in get_chat_store().list_sessions()]


@router.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session(request: SessionCreateRequest) -> SessionResponse:
    s = get_chat_store().create_session(title=request.title, model=request.model)
    return SessionResponse.from_orm_like(s)


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str) -> SessionDetail:
    store = get_chat_store()
    s = store.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="session not found")
    msgs = store.list_messages(session_id)
    return SessionDetail(
        session=SessionResponse.from_orm_like(s),
        messages=[MessageResponse.from_orm_like(m) for m in msgs],
    )


@router.patch("/sessions/{session_id}", response_model=SessionResponse)
async def rename_session(session_id: str, request: SessionRenameRequest) -> SessionResponse:
    store = get_chat_store()
    if not store.rename_session(session_id, request.title):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="session not found")
    return SessionResponse.from_orm_like(store.get_session(session_id))  # type: ignore[arg-type]


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    removed = get_chat_store().delete_session(session_id)
    if not removed:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="session not found")
    return {"session_id": session_id, "removed": True}


# ---------------------------------------------------------------------------
# Messages (non-streaming reply)
# ---------------------------------------------------------------------------


def _stream_messages_for(session_id: str) -> List[chat_streamer.StreamMessage]:
    history = get_chat_store().list_messages(session_id)
    return [chat_streamer.StreamMessage(role=m.role, content=m.content)
            for m in history]


@router.post("/sessions/{session_id}/messages", response_model=ReplyResponse,
             status_code=201)
async def post_message(session_id: str, request: MessageCreateRequest) -> ReplyResponse:
    store = get_chat_store()
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="session not found")
    user_msg = store.append_message(session_id, role="user", content=request.content)
    source = chat_streamer.pick_default_source()
    parts: List[str] = []
    async for chunk in source.stream(
        _stream_messages_for(session_id),
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    ):
        parts.append(chat_streamer.scrub_chunk(chunk))
    assistant_text = "".join(parts).strip() or "(no response)"
    assistant_msg = store.append_message(
        session_id, role="assistant", content=assistant_text
    )
    return ReplyResponse(
        user_message=MessageResponse.from_orm_like(user_msg),
        assistant_message=MessageResponse.from_orm_like(assistant_msg),
        source=source.name,
    )


# ---------------------------------------------------------------------------
# SSE streaming reply
# ---------------------------------------------------------------------------


def _sse(event: str, data: dict) -> str:
    """Format a single SSE frame.  Keep the wire format tight and stable."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/sessions/{session_id}/stream")
async def stream_message(session_id: str, request: MessageCreateRequest) -> StreamingResponse:
    store = get_chat_store()
    if store.get_session(session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="session not found")
    user_msg = store.append_message(session_id, role="user", content=request.content)
    source = chat_streamer.pick_default_source()

    async def event_stream():
        yield _sse("user_message", MessageResponse.from_orm_like(user_msg).model_dump())
        assembled: List[str] = []
        try:
            async for chunk in source.stream(
                _stream_messages_for(session_id),
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            ):
                clean = chat_streamer.scrub_chunk(chunk)
                if not clean:
                    continue
                assembled.append(clean)
                yield _sse("delta", {"text": clean})
                # Cooperative yield so backpressure works in tests.
                await asyncio.sleep(0)
        except Exception as exc:  # surface to the client; do not crash the response
            logger.exception("chat stream failed for session=%s", session_id)
            yield _sse("error", {"message": str(exc)})
            return
        final = "".join(assembled).strip() or "(no response)"
        assistant_msg = store.append_message(
            session_id, role="assistant", content=final
        )
        yield _sse("assistant_message",
                   MessageResponse.from_orm_like(assistant_msg).model_dump())
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

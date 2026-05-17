"""Permission grant/revoke endpoints for the desktop sidecar."""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import require_desktop_mode
from app.local.permissions import (
    Decision,
    Grant,
    Permission,
    get_permission_manager,
)

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_desktop_mode)])


class GrantResponse(BaseModel):
    permission: str
    decision: str
    granted_at: Optional[float]
    expires_at: Optional[float]
    note: Optional[str]

    @classmethod
    def from_grant(cls, g: Grant) -> "GrantResponse":
        return cls(
            permission=g.permission.value,
            decision=g.decision.value,
            granted_at=g.granted_at,
            expires_at=g.expires_at,
            note=g.note,
        )


class GrantRequest(BaseModel):
    ttl_seconds: Optional[float] = Field(
        None, gt=0, description="Optional auto-expiry in seconds."
    )
    note: Optional[str] = Field(None, max_length=500)


class RevokeRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=500)


def _parse_permission_or_400(name: str) -> Permission:
    try:
        return Permission(name.lower())
    except ValueError:
        allowed = ", ".join(p.value for p in Permission)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unknown permission {name!r}; expected one of: {allowed}",
        )


@router.get("", response_model=List[GrantResponse])
async def list_grants() -> List[GrantResponse]:
    return [GrantResponse.from_grant(g)
            for g in get_permission_manager().list_grants()]


@router.get("/{name}", response_model=GrantResponse)
async def get_grant(name: str) -> GrantResponse:
    perm = _parse_permission_or_400(name)
    return GrantResponse.from_grant(get_permission_manager().get(perm))


@router.post("/{name}/grant", response_model=GrantResponse)
async def grant(name: str, request: GrantRequest) -> GrantResponse:
    perm = _parse_permission_or_400(name)
    g = get_permission_manager().grant(
        perm, ttl_seconds=request.ttl_seconds, note=request.note
    )
    logger.info("Permission granted: %s (ttl=%s)", perm.value, request.ttl_seconds)
    return GrantResponse.from_grant(g)


@router.post("/{name}/revoke", response_model=GrantResponse)
async def revoke(name: str, request: RevokeRequest) -> GrantResponse:
    perm = _parse_permission_or_400(name)
    g = get_permission_manager().revoke(perm, note=request.note)
    logger.info("Permission revoked: %s", perm.value)
    return GrantResponse.from_grant(g)


@router.post("/{name}/deny", response_model=GrantResponse)
async def deny(name: str, request: RevokeRequest) -> GrantResponse:
    perm = _parse_permission_or_400(name)
    g = get_permission_manager().deny(perm, note=request.note)
    logger.info("Permission denied: %s", perm.value)
    return GrantResponse.from_grant(g)

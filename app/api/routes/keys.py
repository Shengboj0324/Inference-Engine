"""BYOK key-management endpoints for the desktop sidecar.

All endpoints require :func:`require_desktop_mode` so they only exist
when the API is running embedded in the desktop shell.  The vault
backend (OS keychain or Fernet-encrypted file) is chosen by
:func:`app.local.key_vault.get_key_vault` and never exposed in the
response payloads — only the provider id and a masked preview leave
the process.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.api.deps import require_desktop_mode
from app.local import key_policy
from app.local.key_vault import get_key_vault

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_desktop_mode)])


class ProviderDescriptor(BaseModel):
    id: str
    label: str
    has_key: bool
    masked: Optional[str] = None


class KeyUpsertRequest(BaseModel):
    api_key: str = Field(..., min_length=1, max_length=1024)


class KeyUpsertResponse(BaseModel):
    provider: str
    masked: str
    backend: str


class KeyTestResponse(BaseModel):
    provider: str
    ok: bool
    detail: str


@router.get("", response_model=List[ProviderDescriptor])
async def list_keys() -> List[ProviderDescriptor]:
    """List every allowed provider with whether a key is stored.

    The masked preview is included for providers that have a key so the
    UI can render which key is active without ever needing the raw
    secret.
    """
    vault = get_key_vault()
    stored = set(vault.list_providers())
    out: List[ProviderDescriptor] = []
    for spec in key_policy.list_provider_specs().values():
        pid = spec["id"]
        has_key = pid in stored
        masked = None
        if has_key:
            raw = vault.get(pid)
            masked = key_policy.mask(raw) if raw else None
        out.append(
            ProviderDescriptor(
                id=pid,
                label=spec["label"],
                has_key=has_key,
                masked=masked,
            )
        )
    return out


@router.put("/{provider}", response_model=KeyUpsertResponse)
async def upsert_key(provider: str, request: KeyUpsertRequest) -> KeyUpsertResponse:
    """Store or replace the API key for ``provider``.

    Validates the key shape against :mod:`app.local.key_policy` before
    persisting; any validation failure returns HTTP 400 without echoing
    the rejected key.
    """
    try:
        p = key_policy.normalise(provider)
        clean = key_policy.validate_key(p, request.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    vault = get_key_vault()
    vault.set(p, clean)
    logger.info("Stored API key for provider=%s via backend=%s", p, vault.backend_name)
    return KeyUpsertResponse(
        provider=p,
        masked=key_policy.mask(clean),
        backend=vault.backend_name,
    )


@router.delete("/{provider}")
async def delete_key(provider: str) -> Dict[str, object]:
    """Remove the stored API key for ``provider``."""
    try:
        p = key_policy.normalise(provider)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    vault = get_key_vault()
    removed = vault.delete(p)
    return {"provider": p, "removed": removed}


@router.post("/{provider}/test", response_model=KeyTestResponse)
async def test_key(provider: str) -> KeyTestResponse:
    """Confirm a key exists and is shaped correctly.

    Performs a local-only sanity check (presence + shape).  A real
    round-trip to the provider is intentionally deferred to a later
    phase to avoid leaking key liveness to network observers from a
    casual UI action.
    """
    try:
        p = key_policy.normalise(provider)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    vault = get_key_vault()
    raw = vault.get(p)
    if raw is None:
        return KeyTestResponse(provider=p, ok=False, detail="no key stored")
    try:
        key_policy.validate_key(p, raw)
    except ValueError as exc:
        return KeyTestResponse(provider=p, ok=False, detail=str(exc))
    return KeyTestResponse(provider=p, ok=True, detail="ok")

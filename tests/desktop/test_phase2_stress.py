"""Phase 2 concurrency stress tests.

Exercises the two highest-risk concurrent paths in the new local-first
surface:

* **Chat streaming** — many sessions streaming simultaneously must all
  receive their full delta sequence in order and have their assistant
  message committed exactly once.
* **Key rotation** — concurrent PUT/DELETE/GET on the same provider
  must never leave the vault in a torn state nor leak the raw key.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, List

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.core.config import settings
from app.local import chat_store, chat_streamer, key_vault


_RAW_KEY = "sk-stress-" + "b" * 40


@pytest.fixture
def desktop_client(
    tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "deployment_mode", "desktop")
    monkeypatch.setenv("SMR_KEY_VAULT_BACKEND", "file")
    key_vault.reset_key_vault()
    chat_store.reset_chat_store()
    with TestClient(app) as client:
        yield client
    key_vault.reset_key_vault()
    chat_store.reset_chat_store()


class TestConcurrentChatStreams:
    def test_20_parallel_streams_each_complete(
        self, desktop_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        chunks = [f"part-{i} " for i in range(8)]
        monkeypatch.setattr(
            chat_streamer,
            "pick_default_source",
            lambda: chat_streamer.StaticChunkSource(list(chunks)),
        )

        # Pre-create sessions.
        session_ids: List[str] = []
        for _ in range(20):
            sid = desktop_client.post(
                "/api/v1/chat/sessions", json={"title": "s"}
            ).json()["id"]
            session_ids.append(sid)

        def run_one(sid: str) -> int:
            with desktop_client.stream(
                "POST",
                f"/api/v1/chat/sessions/{sid}/stream",
                json={"content": "go"},
            ) as r:
                assert r.status_code == 200
                body = "".join(r.iter_text())
            assert body.count("event: delta") == len(chunks)
            assert body.rstrip().endswith("data: [DONE]")
            return len(body)

        with ThreadPoolExecutor(max_workers=10) as ex:
            results = [f.result() for f in
                       as_completed(ex.submit(run_one, sid) for sid in session_ids)]
        assert len(results) == 20

        # Each session ends with exactly one user + one assistant message,
        # and assistant content matches the concatenated chunks.
        expected = "".join(chunks).strip()
        for sid in session_ids:
            detail = desktop_client.get(f"/api/v1/chat/sessions/{sid}").json()
            roles = [m["role"] for m in detail["messages"]]
            assert roles == ["user", "assistant"]
            assert detail["messages"][1]["content"] == expected


class TestConcurrentKeyRotation:
    def test_repeated_put_delete_does_not_corrupt_vault(
        self, desktop_client: TestClient
    ) -> None:
        errors: List[str] = []
        barrier = threading.Barrier(8)

        def hammer(worker_id: int) -> None:
            barrier.wait()
            for _ in range(20):
                r = desktop_client.put(
                    "/api/v1/keys/openai", json={"api_key": _RAW_KEY}
                )
                if r.status_code != 200:
                    errors.append(f"put w{worker_id}: {r.status_code} {r.text}")
                    return
                r2 = desktop_client.get("/api/v1/keys")
                if r2.status_code != 200:
                    errors.append(f"get w{worker_id}: {r2.status_code}")
                    return
                if _RAW_KEY in r2.text:
                    errors.append(f"raw key leaked in list response w{worker_id}")
                    return
                desktop_client.delete("/api/v1/keys/openai")

        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(hammer, range(8)))
        assert not errors, errors

        # Final state: a fresh PUT then a single test must succeed.
        assert desktop_client.put(
            "/api/v1/keys/openai", json={"api_key": _RAW_KEY}
        ).status_code == 200
        r = desktop_client.post("/api/v1/keys/openai/test")
        assert r.status_code == 200
        assert r.json()["ok"] is True
        assert _RAW_KEY not in r.text

    def test_concurrent_grants_and_revokes(
        self, desktop_client: TestClient
    ) -> None:
        from app.local import permissions as perm_mod

        perm_mod.reset_permission_manager()

        def grant_revoke(worker_id: int) -> None:
            for _ in range(25):
                desktop_client.post(
                    "/api/v1/permissions/read_files/grant", json={}
                )
                desktop_client.post(
                    "/api/v1/permissions/read_files/revoke", json={}
                )

        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(grant_revoke, range(6)))

        # End state well-defined; file is intact and parseable.
        r = desktop_client.get("/api/v1/permissions/read_files")
        assert r.status_code == 200
        assert r.json()["decision"] in {"grant", "ask"}

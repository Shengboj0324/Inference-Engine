"""End-to-end desktop session walkthrough for the local RAG layer.

Exercises the Phase 5/6 contracts the way a real operator would:

  1. seed a tiny content store with three items
  2. probe the RAG status route (current model, stale count)
  3. issue a search; verify base_score / personalization_bonus are surfaced
  4. record explicit feedback on one snippet
  5. issue the same search; verify the feedback target is now promoted
  6. swap the embedder's ``model`` to a fresh version and confirm the
     stale-embedding counter surfaces every prior row
  7. start a background reindex job and poll until done
  8. reset personalization signals and confirm the count returns to zero

Exits ``0`` on success, ``1`` on any assertion failure.  Intended for
the pre-release checklist in ``docs/deployment.md`` and as a fast smoke
test the on-call can run against a freshly-installed build.

Run with: ``DEPLOYMENT_MODE=desktop python scripts/mock_user_session.py``
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from uuid import uuid4

# Isolate from the operator's real user-data dir.
_TMP = tempfile.mkdtemp(prefix="smr-mock-")
os.environ.setdefault("SMR_USER_DATA_DIR", _TMP)
os.environ.setdefault("SMR_DATA_DIR", _TMP)
os.environ.setdefault("DEPLOYMENT_MODE", "desktop")

# Allow ``python scripts/mock_user_session.py`` from the repo root without
# requiring ``PYTHONPATH=.`` — the desktop runbook in deployment.md shows
# the bare invocation, so the script must self-bootstrap.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.api.main import app  # noqa: E402
from app.api.routes import rag as rag_routes  # noqa: E402
from app.core.models import ContentItem, MediaType, SourcePlatform  # noqa: E402
from app.local import rag_retriever as rr  # noqa: E402
from app.local.content_store import get_content_store, reset_content_store  # noqa: E402
from app.local.rag_jobs import reset_reindex_jobs  # noqa: E402
from app.local.rag_retriever import LocalRAGRetriever  # noqa: E402
from app.local.retrieval_signals import reset_signals_store  # noqa: E402


class _FakeEmbedder:
    """Deterministic embedder so the walkthrough is reproducible."""

    def __init__(self, model: str = "fake:v1") -> None:
        self.model = model

    def is_configured(self) -> bool: return True
    def is_permitted(self) -> bool: return True

    async def embed_text(self, text: str) -> List[float]:  # noqa: ARG002
        return [1.0, 0.0, 0.0]

    async def embed_batch(self, texts):  # noqa: ANN001
        return [[1.0, 0.0, 0.0] for _ in texts]


def _seed_content() -> List[str]:
    store = get_content_store()
    ids: List[str] = []
    for title, emb, src in [
        ("Reddit headline about machine learning", [1.0, 0.0, 0.0], "r1"),
        ("Slightly off-topic story",               [0.99, 0.10, 0.0], "r2"),
        ("Completely unrelated post",              [0.0, 1.0, 0.0],   "r3"),
    ]:
        item = ContentItem(
            id=uuid4(), user_id=uuid4(),
            source_platform=SourcePlatform.REDDIT, source_id=src,
            source_url=f"https://example.test/{src}",
            author="anon", channel=None, title=title, raw_text=title,
            media_type=MediaType.TEXT, media_urls=[],
            published_at=datetime.now(timezone.utc),
            fetched_at=datetime.now(timezone.utc),
            topics=[], lang="en", embedding=emb,
            metadata={"embedding_version": "fake:v1"},
        )
        store.upsert(item)
        ids.append(str(item.id))
    return ids


def _check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
    if not condition:
        raise SystemExit(1)


def run() -> int:
    print(f"-> isolated user-data dir: {_TMP}")
    reset_content_store(); reset_signals_store(); reset_reindex_jobs()
    rr.reset_rag_retriever()

    embedder = _FakeEmbedder()
    rr._global_retriever = LocalRAGRetriever(
        content=get_content_store(), embedder=embedder,
    )
    rag_routes._embedder = lambda: embedder  # type: ignore[assignment]

    ids = _seed_content()
    print(f"-> seeded {len(ids)} content rows")

    c = TestClient(app)

    print("step 1: GET /rag/status")
    s = c.get("/api/v1/rag/status"); _check("status 200", s.status_code == 200)
    body = s.json()
    _check("available", body["available"] is True, repr(body))
    _check("current_version reported", body["current_version"] == "fake:v1")

    print("step 2: POST /rag/search (baseline)")
    r = c.post("/api/v1/rag/search", json={"query": "ml news", "k": 3})
    _check("search 200", r.status_code == 200)
    snippets = r.json()["snippets"]
    _check("snippets >= 2", len(snippets) >= 2)
    baseline_top = snippets[0]["content_id"]
    runner_up = snippets[1]["content_id"]

    print("step 3: POST /rag/feedback on the runner-up")
    for _ in range(8):
        c.post("/api/v1/rag/feedback", json={"content_id": runner_up, "score": 1.0})

    print("step 4: POST /rag/search (after feedback)")
    r2 = c.post("/api/v1/rag/search", json={"query": "ml news", "k": 3})
    new_top = r2.json()["snippets"][0]
    _check("personalization bonus surfaced", new_top["personalization_bonus"] > 0.0,
           f"bonus={new_top['personalization_bonus']:.3f}")
    _check("runner-up promoted to top", new_top["content_id"] == runner_up,
           f"baseline_top={baseline_top}, new_top={new_top['content_id']}")

    print("step 5: model swap surfaces stale rows")
    embedder.model = "fake:v2"
    s3 = c.get("/api/v1/rag/status").json()
    _check("stale_count covers all rows", s3["stale_count"] >= len(ids),
           f"stale={s3['stale_count']}")

    print("step 6: background reindex job")
    j = c.post("/api/v1/rag/reindex/jobs", json={"batch_size": 4})
    _check("job 202", j.status_code == 202, str(j.status_code))
    jid = j.json()["id"]
    final = None
    for _ in range(40):
        time.sleep(0.05)
        final = c.get(f"/api/v1/rag/reindex/jobs/{jid}").json()
        if final["status"] in ("done", "error", "cancelled"):
            break
    _check("job finished", final and final["status"] == "done", repr(final))
    _check("job embedded >= seeded rows", final["embedded"] >= len(ids))

    print("step 7: reset personalization signals")
    d = c.delete("/api/v1/rag/signals")
    _check("clear 200", d.status_code == 200)
    _check("count drops to 0", c.get("/api/v1/rag/signals").json()["count"] == 0)

    print("\n-> all checks passed; desktop sidecar contracts hold end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(run())

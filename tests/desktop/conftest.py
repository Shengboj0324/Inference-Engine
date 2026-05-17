"""Shared fixtures for the Phase 1 desktop backend test suite.

``tmp_data_dir`` redirects every ``get_user_data_dir`` / cache / log call
to a per-test ``tmp_path`` so concurrent test workers cannot collide and
no operator data is ever touched.

``reset_pubsub_hub`` ensures the module-level message broker starts each
test in a known-empty state — critical because pub/sub state is process-
global by design.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect every user-data path to ``tmp_path`` for the test."""
    monkeypatch.setenv("SMR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SMR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SMR_LOG_DIR", str(tmp_path / "logs"))
    return tmp_path


@pytest.fixture(autouse=True)
def reset_pubsub_hub() -> Iterator[None]:
    """Reset the shared in-process pub/sub hub between tests."""
    from app.local import local_pubsub

    local_pubsub.reset_hub()
    yield
    local_pubsub.reset_hub()

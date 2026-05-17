"""Tests for ``app.local.chat_store``."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.local.chat_store import ChatStore


@pytest.fixture
def store(tmp_path: Path) -> ChatStore:
    return ChatStore(db_path=tmp_path / "chat.sqlite3")


class TestSessions:
    def test_create_then_list(self, store: ChatStore) -> None:
        s = store.create_session(title="Hello", model="gpt-4o-mini")
        sessions = store.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].id == s.id
        assert sessions[0].title == "Hello"
        assert sessions[0].model == "gpt-4o-mini"

    def test_list_orders_most_recent_first(self, store: ChatStore) -> None:
        a = store.create_session(title="first")
        time.sleep(0.01)
        b = store.create_session(title="second")
        ids = [s.id for s in store.list_sessions()]
        assert ids == [b.id, a.id]

    def test_get_session_missing_returns_none(self, store: ChatStore) -> None:
        assert store.get_session("does-not-exist") is None

    def test_rename_session(self, store: ChatStore) -> None:
        s = store.create_session(title="old")
        assert store.rename_session(s.id, "new") is True
        assert store.get_session(s.id).title == "new"  # type: ignore[union-attr]

    def test_rename_missing(self, store: ChatStore) -> None:
        assert store.rename_session("nope", "new") is False

    def test_delete_session_cascades_messages(self, store: ChatStore) -> None:
        s = store.create_session(title="x")
        store.append_message(s.id, role="user", content="hi")
        store.append_message(s.id, role="assistant", content="hello")
        assert store.delete_session(s.id) is True
        assert store.list_messages(s.id) == []


class TestMessages:
    def test_append_then_list(self, store: ChatStore) -> None:
        s = store.create_session(title="x")
        store.append_message(s.id, role="user", content="hello")
        store.append_message(s.id, role="assistant", content="world", tokens=2)
        msgs = store.list_messages(s.id)
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert [m.content for m in msgs] == ["hello", "world"]
        assert msgs[1].tokens == 2

    def test_append_unknown_session_raises(self, store: ChatStore) -> None:
        with pytest.raises(KeyError):
            store.append_message("nope", role="user", content="x")

    def test_append_invalid_role(self, store: ChatStore) -> None:
        s = store.create_session(title="x")
        with pytest.raises(ValueError):
            store.append_message(s.id, role="root", content="x")

    def test_append_empty_content(self, store: ChatStore) -> None:
        s = store.create_session(title="x")
        with pytest.raises(ValueError):
            store.append_message(s.id, role="user", content="")

    def test_append_updates_session_timestamp(self, store: ChatStore) -> None:
        s = store.create_session(title="x")
        original = store.get_session(s.id).updated_at  # type: ignore[union-attr]
        time.sleep(0.01)
        store.append_message(s.id, role="user", content="hi")
        new_ts = store.get_session(s.id).updated_at  # type: ignore[union-attr]
        assert new_ts > original

    def test_messages_ordered_by_creation(self, store: ChatStore) -> None:
        s = store.create_session(title="x")
        for i in range(5):
            store.append_message(s.id, role="user", content=f"m{i}")
        contents = [m.content for m in store.list_messages(s.id)]
        assert contents == [f"m{i}" for i in range(5)]


class TestPersistence:
    def test_data_survives_reopen(self, tmp_path: Path) -> None:
        path = tmp_path / "chat.sqlite3"
        s1 = ChatStore(db_path=path).create_session(title="persist")
        s1_id = s1.id
        # New store instance, same file.
        s2 = ChatStore(db_path=path)
        sessions = s2.list_sessions()
        assert [s.id for s in sessions] == [s1_id]


class TestSchemaIdempotency:
    def test_ensure_schema_safe_to_call_repeatedly(self, store: ChatStore) -> None:
        store.ensure_schema()
        store.ensure_schema()
        # No exception → safe.

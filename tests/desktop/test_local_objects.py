"""Tests for ``app.local.local_objects``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.local.local_objects import LocalObjectStore


@pytest.fixture
def store(tmp_data_dir: Path) -> LocalObjectStore:
    return LocalObjectStore("media", root=tmp_data_dir / "objects")


class TestPutAndGet:
    def test_round_trip_bytes(self, store: LocalObjectStore) -> None:
        meta = store.put_object("a/b/c.bin", b"hello-world")
        assert meta.key == "a/b/c.bin"
        assert meta.size == 11
        assert store.get_object("a/b/c.bin") == b"hello-world"

    def test_round_trip_from_path(self, store: LocalObjectStore, tmp_path: Path) -> None:
        src = tmp_path / "src.bin"
        src.write_bytes(b"x" * 1024)
        meta = store.put_object("from-path.bin", src)
        assert meta.size == 1024
        assert store.get_object("from-path.bin") == b"x" * 1024

    def test_etag_is_md5(self, store: LocalObjectStore) -> None:
        import hashlib
        store.put_object("e.txt", b"abc")
        meta = next(store.list_objects("e."))
        assert meta.etag == hashlib.md5(b"abc").hexdigest()

    def test_get_missing_raises(self, store: LocalObjectStore) -> None:
        with pytest.raises(FileNotFoundError):
            store.get_object("nope")

    def test_exists_and_delete(self, store: LocalObjectStore) -> None:
        store.put_object("k", b"v")
        assert store.exists("k") is True
        assert store.delete("k") is True
        assert store.exists("k") is False
        assert store.delete("k") is False


class TestAtomicity:
    def test_put_is_atomic(self, store: LocalObjectStore) -> None:
        # Two consecutive writes must always observe a complete payload.
        store.put_object("p", b"first-payload")
        assert store.get_object("p") == b"first-payload"
        store.put_object("p", b"replaced-payload")
        assert store.get_object("p") == b"replaced-payload"

    def test_no_stray_tmp_files_after_put(self, store: LocalObjectStore) -> None:
        store.put_object("clean", b"x")
        for entry in (store._root.rglob("*.tmp.*")):
            raise AssertionError(f"stray tmp file: {entry}")


class TestListAndPresign:
    def test_list_with_prefix(self, store: LocalObjectStore) -> None:
        store.put_object("img/cat.jpg", b"1")
        store.put_object("img/dog.jpg", b"2")
        store.put_object("vid/v.mp4", b"3")
        keys = sorted(m.key for m in store.list_objects("img/"))
        assert keys == ["img/cat.jpg", "img/dog.jpg"]

    def test_list_empty_prefix_yields_all(self, store: LocalObjectStore) -> None:
        store.put_object("a", b"1")
        store.put_object("b/c", b"2")
        keys = sorted(m.key for m in store.list_objects(""))
        assert keys == ["a", "b/c"]

    def test_presigned_url_is_file_uri(self, store: LocalObjectStore) -> None:
        store.put_object("x", b"v")
        url = store.get_presigned_url("x")
        assert url.startswith("file://")
        assert url.endswith("x")


class TestSecurity:
    def test_path_traversal_rejected(self, store: LocalObjectStore) -> None:
        with pytest.raises(ValueError):
            store.put_object("../escape", b"x")
        with pytest.raises(ValueError):
            store.get_object("../etc/passwd")
        with pytest.raises(ValueError):
            store.delete("../../oops")

    def test_absolute_key_rejected(self, store: LocalObjectStore) -> None:
        with pytest.raises(ValueError):
            store.put_object("/abs/key", b"x")

    def test_invalid_bucket_rejected(self, tmp_data_dir: Path) -> None:
        with pytest.raises(ValueError):
            LocalObjectStore("bad/bucket", root=tmp_data_dir / "objects")
        with pytest.raises(ValueError):
            LocalObjectStore("", root=tmp_data_dir / "objects")

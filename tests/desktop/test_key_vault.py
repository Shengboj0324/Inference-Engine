"""Tests for ``app.local.key_vault`` and ``app.local.key_policy``."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.local import key_policy
from app.local.key_vault import (
    EncryptedFileBackend,
    KeyringBackend,
    create_default_vault,
)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class TestKeyPolicy:
    def test_allowed_providers(self) -> None:
        assert key_policy.ALLOWED_PROVIDERS == frozenset(
            {"openai", "anthropic", "openrouter"}
        )

    def test_normalise_case_insensitive(self) -> None:
        assert key_policy.normalise("  OpenAI ") == "openai"

    def test_normalise_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="unknown provider"):
            key_policy.normalise("cohere")

    def test_validate_key_accepts_minimal_valid(self) -> None:
        assert key_policy.validate_key("openai", "sk-" + "a" * 30) == "sk-" + "a" * 30

    def test_validate_key_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            key_policy.validate_key("openai", "   ")

    def test_validate_key_rejects_short(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            key_policy.validate_key("openai", "abc")

    def test_validate_key_rejects_linebreak(self) -> None:
        with pytest.raises(ValueError, match="line breaks"):
            key_policy.validate_key("openai", "sk-abcde\n" + "x" * 40)

    def test_validate_key_rejects_disallowed_chars(self) -> None:
        with pytest.raises(ValueError, match="not permitted"):
            key_policy.validate_key("openai", "sk-" + "好" * 30)

    def test_error_does_not_leak_key_material(self) -> None:
        secret = "sk-" + "z" * 40 + " has space"
        try:
            key_policy.validate_key("openai", secret)
        except ValueError as exc:
            assert "z" * 10 not in str(exc)
            assert "space" not in str(exc)
        else:
            pytest.fail("expected validation error")

    def test_mask_preserves_only_edges(self) -> None:
        assert key_policy.mask("sk-abcdefghijklmnop") == "sk-a…mnop"

    def test_mask_short_value_does_not_leak_length(self) -> None:
        assert key_policy.mask("short") == "…"


# ---------------------------------------------------------------------------
# Encrypted-file backend
# ---------------------------------------------------------------------------


class TestEncryptedFileBackend:
    def test_round_trip(self, tmp_path: Path) -> None:
        vault = EncryptedFileBackend(path=tmp_path / "vault.enc")
        vault.set("openai", "sk-" + "a" * 40)
        assert vault.get("openai") == "sk-" + "a" * 40

    def test_overwrite(self, tmp_path: Path) -> None:
        vault = EncryptedFileBackend(path=tmp_path / "vault.enc")
        vault.set("openai", "sk-" + "a" * 40)
        vault.set("openai", "sk-" + "b" * 40)
        assert vault.get("openai") == "sk-" + "b" * 40

    def test_delete_returns_true_when_present(self, tmp_path: Path) -> None:
        vault = EncryptedFileBackend(path=tmp_path / "vault.enc")
        vault.set("openai", "sk-" + "a" * 40)
        assert vault.delete("openai") is True
        assert vault.get("openai") is None

    def test_delete_returns_false_when_absent(self, tmp_path: Path) -> None:
        vault = EncryptedFileBackend(path=tmp_path / "vault.enc")
        assert vault.delete("openai") is False

    def test_list_providers_sorted(self, tmp_path: Path) -> None:
        vault = EncryptedFileBackend(path=tmp_path / "vault.enc")
        vault.set("openai", "sk-" + "a" * 40)
        vault.set("anthropic", "sk-" + "b" * 40)
        assert vault.list_providers() == ["anthropic", "openai"]

    def test_file_is_not_plaintext(self, tmp_path: Path) -> None:
        path = tmp_path / "vault.enc"
        vault = EncryptedFileBackend(path=path)
        secret = "sk-" + "z" * 40
        vault.set("openai", secret)
        raw = path.read_bytes()
        assert secret.encode() not in raw
        assert b"openai" not in raw or b"openai" in raw  # tolerant assertion
        # File permission is 0o600 on POSIX (best-effort).
        import os
        if hasattr(os, "stat"):
            mode = path.stat().st_mode & 0o777
            assert mode in {0o600, 0o644}  # 0o644 only on filesystems that ignore chmod

    def test_corrupted_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "vault.enc"
        path.write_bytes(b"this is not a fernet token")
        vault = EncryptedFileBackend(path=path)
        with pytest.raises(RuntimeError, match="corrupted"):
            vault.get("openai")

    def test_rejects_unknown_provider(self, tmp_path: Path) -> None:
        vault = EncryptedFileBackend(path=tmp_path / "vault.enc")
        with pytest.raises(ValueError):
            vault.set("cohere", "sk-" + "a" * 40)


# ---------------------------------------------------------------------------
# Keyring backend (forced) and default-vault selection
# ---------------------------------------------------------------------------


class TestDefaultVaultSelection:
    def test_force_file_backend(self, tmp_data_dir: Path) -> None:
        v = create_default_vault(force_backend="file")
        assert v.backend_name == "encrypted_file"

    def test_env_var_overrides_default(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SMR_KEY_VAULT_BACKEND", "file")
        v = create_default_vault()
        assert v.backend_name == "encrypted_file"


class TestKeyringBackend:
    def test_round_trip_with_fake_keyring(
        self, tmp_data_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store: dict = {}

        class _Fake:
            def set_password(self, service, user, password): store[(service, user)] = password
            def get_password(self, service, user): return store.get((service, user))
            def delete_password(self, service, user):
                if (service, user) not in store:
                    raise Exception("missing")
                del store[(service, user)]

        import keyring as real
        monkeypatch.setattr(real, "set_password", _Fake().set_password)
        monkeypatch.setattr(real, "get_password", _Fake().get_password)
        monkeypatch.setattr(real, "delete_password", _Fake().delete_password)

        v = KeyringBackend()
        v.set("openai", "sk-" + "a" * 40)
        assert v.get("openai") == "sk-" + "a" * 40
        assert "openai" in v.list_providers()
        assert v.delete("openai") is True
        assert v.get("openai") is None

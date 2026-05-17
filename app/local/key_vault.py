"""Local-first encrypted vault for user-supplied API keys.

Two backends are provided, both implementing the same :class:`KeyVault`
protocol so callers do not branch on which is in use:

* :class:`KeyringBackend` — wraps the :mod:`keyring` library and stores
  keys in the OS keychain (macOS Keychain, Windows Credential Manager,
  Secret Service on Linux desktops).  This is the default whenever a
  usable backend is present.
* :class:`EncryptedFileBackend` — Fernet-encrypted JSON file stored
  inside the user data directory.  Used on headless Linux / CI / any
  environment where :mod:`keyring` has no functional backend.

The vault stores **only** raw provider keys.  Higher-level metadata
(rotation timestamps, labels, last-used) lives in the chat store and is
keyed by provider id so the two surfaces can move independently.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.core.config import settings
from app.local.key_policy import normalise
from app.local.user_data_dir import get_user_data_dir

logger = logging.getLogger(__name__)

_SERVICE_NAME = "SocialMediaRadar"
_KEY_FILE_NAME = "keys.enc"
_KDF_SALT = b"smr-local-key-vault\x00v1"
_KDF_ITERATIONS = 200_000


class KeyVault(Protocol):
    """Protocol implemented by every storage backend."""

    backend_name: str

    def set(self, provider: str, key: str) -> None: ...
    def get(self, provider: str) -> Optional[str]: ...
    def delete(self, provider: str) -> bool: ...
    def list_providers(self) -> List[str]: ...


class KeyringBackend:
    """OS keychain-backed vault using the cross-platform ``keyring`` lib."""

    backend_name = "keyring"

    def __init__(self, *, service: str = _SERVICE_NAME) -> None:
        import keyring  # local import keeps the dependency optional at import time
        self._keyring = keyring
        self._service = service
        # Track which providers we have stored so list_providers() works
        # without iterating an opaque OS keychain.  Index lives alongside
        # the encrypted-file fallback so the two backends can coexist.
        self._index_path = get_user_data_dir() / "keyring_index.json"

    def _load_index(self) -> List[str]:
        if not self._index_path.exists():
            return []
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            return [str(p) for p in data.get("providers", [])]
        except (json.JSONDecodeError, OSError):
            return []

    def _save_index(self, providers: Iterable[str]) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"providers": sorted(set(providers))}),
            encoding="utf-8",
        )
        os.replace(tmp, self._index_path)

    def set(self, provider: str, key: str) -> None:
        p = normalise(provider)
        self._keyring.set_password(self._service, p, key)
        providers = self._load_index()
        if p not in providers:
            providers.append(p)
            self._save_index(providers)

    def get(self, provider: str) -> Optional[str]:
        p = normalise(provider)
        return self._keyring.get_password(self._service, p)

    def delete(self, provider: str) -> bool:
        p = normalise(provider)
        try:
            self._keyring.delete_password(self._service, p)
            deleted = True
        except Exception:  # keyring raises PasswordDeleteError on miss
            deleted = False
        providers = [x for x in self._load_index() if x != p]
        self._save_index(providers)
        return deleted

    def list_providers(self) -> List[str]:
        # Filter the index against what is actually present so a stale
        # entry from a manual keychain deletion does not haunt the UI.
        return [p for p in self._load_index() if self.get(p) is not None]


class EncryptedFileBackend:
    """Fernet-encrypted JSON file inside the user data directory.

    The Fernet key is derived from ``settings.encryption_key`` via PBKDF2
    with a fixed module-level salt — this is acceptable because the file
    itself never leaves the user's machine and the master secret is the
    only thing protecting it.
    """

    backend_name = "encrypted_file"

    def __init__(self, *, path: Optional[Path] = None) -> None:
        self._path = path or (get_user_data_dir() / _KEY_FILE_NAME)
        self._cipher = self._build_cipher()

    @staticmethod
    def _build_cipher() -> Fernet:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=_KDF_SALT,
            iterations=_KDF_ITERATIONS,
        )
        derived = kdf.derive(settings.encryption_key.encode("utf-8"))
        return Fernet(base64.urlsafe_b64encode(derived))

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            plaintext = self._cipher.decrypt(self._path.read_bytes())
            data = json.loads(plaintext.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except (InvalidToken, json.JSONDecodeError, OSError) as exc:
            logger.error(
                "Failed to decrypt key vault at %s (%s); refusing to overwrite. "
                "Move or repair the file to recover.",
                self._path,
                type(exc).__name__,
            )
            raise RuntimeError("key vault is corrupted or master key changed")

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        ciphertext = self._cipher.encrypt(json.dumps(data).encode("utf-8"))
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_bytes(ciphertext)
        os.replace(tmp, self._path)
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass

    def set(self, provider: str, key: str) -> None:
        p = normalise(provider)
        data = self._load()
        data[p] = key
        self._save(data)

    def get(self, provider: str) -> Optional[str]:
        p = normalise(provider)
        return self._load().get(p)

    def delete(self, provider: str) -> bool:
        p = normalise(provider)
        data = self._load()
        if p not in data:
            return False
        del data[p]
        self._save(data)
        return True

    def list_providers(self) -> List[str]:
        return sorted(self._load().keys())


def create_default_vault(*, force_backend: Optional[str] = None) -> KeyVault:
    """Return the best-available vault for the current environment.

    Resolution order:
      1. ``force_backend`` argument (``"keyring"`` or ``"file"``) when set.
      2. ``SMR_KEY_VAULT_BACKEND`` env var with the same values.
      3. :class:`KeyringBackend` if :mod:`keyring` is importable **and**
         exposes a non-null backend.
      4. :class:`EncryptedFileBackend` as the universal fallback.
    """
    pick = force_backend or os.environ.get("SMR_KEY_VAULT_BACKEND")
    if pick == "file":
        return EncryptedFileBackend()
    if pick == "keyring":
        return KeyringBackend()
    try:
        import keyring
        kr = keyring.get_keyring()
        # Fail backends advertise themselves with names containing "fail" or
        # "null"; treat those as unusable and fall back to the file backend.
        name = type(kr).__name__.lower()
        if "fail" in name or "null" in name:
            raise RuntimeError(f"keyring backend unusable: {name}")
        return KeyringBackend()
    except Exception as exc:
        logger.info("keyring unavailable (%s); using encrypted-file vault", exc)
        return EncryptedFileBackend()


_global_vault: Optional[KeyVault] = None


def get_key_vault() -> KeyVault:
    """Return the process-wide vault singleton (constructed on first call)."""
    global _global_vault
    if _global_vault is None:
        _global_vault = create_default_vault()
    return _global_vault


def reset_key_vault() -> None:
    """Drop the cached singleton (test-only helper)."""
    global _global_vault
    _global_vault = None

#!/usr/bin/env python3
"""Re-encrypt PlatformConfigDB credentials from the old static-salt scheme
to the new per-credential random-salt scheme introduced in M1 (Session 2).

BACKGROUND
----------
Before M1, ``CredentialEncryption`` derived a Fernet key using PBKDF2-HMAC-SHA256
with a fixed salt ``b"social_media_radar_salt"`` and stored
``base64url(fernet_token)``.

After M1, each call to ``encrypt()`` generates a fresh 16-byte random salt,
prepends it to the Fernet token, and stores ``base64url(salt || fernet_token)``.

Any row written before M1 cannot be decrypted by the new code.  This script:
  1. Reads every ``PlatformConfigDB`` row that has ``encrypted_credentials``.
  2. Tries to detect which scheme the stored value uses (length heuristic).
  3. Attempts old-scheme decryption; skips the row silently if it fails
     (already migrated, or corrupted data that would fail in either case).
  4. Re-encrypts with the new scheme and writes back.
  5. Is fully idempotent — safe to run multiple times.

USAGE
-----
    python scripts/migrate_credentials.py

Run with the same environment variables that the API uses (DATABASE_URL,
ENCRYPTION_KEY).  The script uses the *synchronous* SQLAlchemy engine so it
can run outside the asyncio event loop.
"""

import base64
import json
import logging
import sys
from pathlib import Path

# Allow running from the repository root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.db_models import PlatformConfigDB
from app.core.security import CredentialEncryption

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ── Old-scheme constants (must match the pre-M1 implementation exactly) ────
_OLD_STATIC_SALT = b"social_media_radar_salt"
_OLD_KDF_ITERATIONS = 100_000
_NEW_SALT_BYTES = 16  # bytes prepended by the new scheme


def _old_scheme_decrypt(encrypted_data: str, master_key: str) -> dict:
    """Attempt to decrypt a value written by the pre-M1 CredentialEncryption.

    The old scheme stored: base64url(fernet_token).
    The Fernet key was derived via PBKDF2(master_key, static_salt).

    Raises:
        InvalidToken: if the ciphertext is not a valid old-scheme token.
        Exception: on any other decryption error.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_OLD_STATIC_SALT,
        iterations=_OLD_KDF_ITERATIONS,
    )
    fernet_key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
    cipher = Fernet(fernet_key)

    raw = base64.urlsafe_b64decode(encrypted_data.encode())
    plaintext = cipher.decrypt(raw)
    return json.loads(plaintext.decode())


def _looks_like_new_scheme(encrypted_data: str) -> bool:
    """Heuristic: new-scheme blobs are ≥ 16 salt bytes + Fernet overhead.

    Fernet tokens are at minimum ~148 bytes when base64-encoded.  The new
    scheme adds 16 salt bytes before the Fernet token, so the minimum
    base64url-decoded length is ~164 bytes.  Old-scheme Fernet tokens decode
    to a minimum of ~148 bytes.

    A reliable differentiator: try new-scheme decrypt first; if it raises
    ``ValueError`` (salt extraction check) the caller should try old-scheme.
    This function is just an optimisation to skip obviously-old values.
    """
    try:
        raw = base64.urlsafe_b64decode(encrypted_data.encode())
        return len(raw) > _NEW_SALT_BYTES + 45  # 45 = minimum Fernet token
    except Exception:
        return False


def migrate(dry_run: bool = False) -> None:
    engine = create_engine(settings.database_sync_url)
    enc = CredentialEncryption()
    master_key = settings.encryption_key

    with Session(engine) as session:
        rows = session.scalars(
            select(PlatformConfigDB).where(
                PlatformConfigDB.encrypted_credentials.isnot(None)
            )
        ).all()

        log.info("Found %d platform_config rows with encrypted_credentials.", len(rows))
        migrated = skipped = already_new = errors = 0

        for row in rows:
            ciphertext: str = row.encrypted_credentials

            # ── 1. Try new-scheme decode first (idempotency check) ────────
            try:
                enc.decrypt(ciphertext)
                already_new += 1
                log.debug("Row %s (platform=%s): already new scheme — skipping.",
                          row.id, row.platform)
                continue
            except Exception:
                pass  # Not new-scheme — proceed to old-scheme attempt.

            # ── 2. Try old-scheme decode ──────────────────────────────────
            try:
                plaintext_dict = _old_scheme_decrypt(ciphertext, master_key)
            except (InvalidToken, Exception) as exc:
                log.warning(
                    "Row %s (platform=%s): old-scheme decrypt failed (%s) — "
                    "skipping (data may be corrupt or use an unknown scheme).",
                    row.id, row.platform, exc,
                )
                errors += 1
                continue

            # ── 3. Re-encrypt with new scheme ─────────────────────────────
            new_ciphertext = enc.encrypt(plaintext_dict)

            if dry_run:
                log.info("[DRY RUN] Would re-encrypt row %s (platform=%s).",
                         row.id, row.platform)
            else:
                row.encrypted_credentials = new_ciphertext
                migrated += 1
                log.info("Re-encrypted row %s (platform=%s).", row.id, row.platform)

        if not dry_run:
            session.commit()

    log.info(
        "Migration complete. migrated=%d  already_new=%d  skipped/errors=%d",
        migrated, already_new, errors,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be migrated without writing to the database.",
    )
    args = parser.parse_args()
    migrate(dry_run=args.dry_run)


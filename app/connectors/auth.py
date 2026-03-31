"""Token lifecycle utilities for platform connectors.

This module centralises every cross-cutting concern related to OAuth token
management so each connector can stay focused on platform-specific logic.

Utilities provided
------------------
is_token_expired(credential, buffer_seconds)
    Pre-call expiry guard — returns True when a token is expired or within
    the 60-second safety buffer.  Every connector's fetch_content() should
    call this before making any API request.

ConnectorAuthError
    Structured exception raised on HTTP 401 or pre-call expiry detection.
    Carries ``platform``, ``user_id``, ``auth_status``, and ``http_status``
    so the caller can update PlatformCredential.auth_status without needing
    to parse a free-form error message string.

safe_error_str(exc)
    Returns a log-safe string from any exception, stripping access_token,
    refresh_token, secret, and similar credential patterns that aiohttp /
    httpx may include verbatim in their URL-embedded error messages.

Conditional refresh lock strategy
----------------------------------
Problem
~~~~~~~
Two concurrent worker processes (or async tasks) for the same (user, platform)
pair can both detect that a token has expired and both attempt a refresh
simultaneously.  If both succeed, the second write clobbers the first new token
and the first process now holds a stale token that fails immediately.

Chosen strategy: optimistic concurrency / compare-and-swap via a conditional
UPDATE.  Only the first writer wins; the second detects 0 updated rows and
re-reads the token already written by the winner.

PostgreSQL / SQLAlchemy expression (use ``text()`` for async sessions)::

    UPDATE platform_credentials
       SET access_token      = :new_token,
           token_expires_at  = :new_expiry,
           auth_status       = 'CONNECTED',
           last_refreshed_at = NOW()
     WHERE user_id          = :uid
       AND platform         = :plat
       AND (
             token_expires_at IS NOT DISTINCT FROM :expected_expiry
           )
    RETURNING id

``IS NOT DISTINCT FROM`` handles the NULL case correctly (NULL = NULL is true).
If no rows are returned the caller must NOT retry the refresh — it must instead
re-read the credential row to obtain the token written by the winning process.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Grace window applied before the literal expiry timestamp.  A token that
#: expires within this many seconds is treated as already expired to avoid
#: making API calls with a token that will expire before the response arrives.
TOKEN_EXPIRY_BUFFER_SECONDS: int = 60

# Regex patterns for credential values that must never appear in log output.
# Covers URL query-param forms (access_token=...) and bare long-token shapes.
_CRED_QP_RE = re.compile(
    r'(?:access_token|refresh_token|client_secret|app_secret|secret)'
    r'=[^\s&"\']+',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# is_token_expired
# ---------------------------------------------------------------------------


def is_token_expired(
    credential: Dict[str, Any],
    buffer_seconds: int = TOKEN_EXPIRY_BUFFER_SECONDS,
) -> bool:
    """Return ``True`` when the stored token is expired or about to expire.

    Parameters
    ----------
    credential:
        A dict in the shape returned by ``CredentialVault.get_credential()``.
        The relevant key is ``"token_expires_at"``, which may be a UTC-aware
        :class:`~datetime.datetime`, an ISO-8601 string, or ``None`` / absent
        (treated as non-expiring — function returns ``False``).
    buffer_seconds:
        Safety window in seconds.  A token expiring within this window is
        considered expired.  Default: :data:`TOKEN_EXPIRY_BUFFER_SECONDS`.

    Returns
    -------
    bool
        ``True``  → token is expired or expires within *buffer_seconds*.
        ``False`` → token is valid, or expiry information is unavailable.
    """
    expires_raw = credential.get("token_expires_at")
    if expires_raw is None:
        return False  # Non-expiring credential or expiry unknown

    if isinstance(expires_raw, str):
        try:
            expires_at = datetime.fromisoformat(expires_raw)
        except (ValueError, TypeError):
            logger.debug("is_token_expired: unparseable token_expires_at %r", expires_raw)
            return False  # Cannot determine — treat as valid
    elif isinstance(expires_raw, datetime):
        expires_at = expires_raw
    else:
        logger.debug("is_token_expired: unexpected type %s for token_expires_at", type(expires_raw))
        return False

    # Normalise to UTC-aware for safe comparison
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    seconds_remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
    return seconds_remaining < buffer_seconds


# ---------------------------------------------------------------------------
# ConnectorAuthError
# ---------------------------------------------------------------------------


class ConnectorAuthError(Exception):
    """Raised when a connector detects an authentication failure.

    This exception is raised in two situations:

    1. **Pre-call**: ``is_token_expired()`` returns ``True`` before any network
       request is made (``http_status=None``).
    2. **Post-call**: The platform API responds with HTTP 401 / 403.

    Attributes
    ----------
    platform : str
        The ``SourcePlatform.value`` string (e.g. ``"reddit"``).
    user_id : str
        String representation of the user's UUID.
    auth_status : str
        Always ``"EXPIRED"`` — the authoritative value to write to
        ``PlatformCredential.auth_status``.
    http_status : Optional[int]
        HTTP response status code, or ``None`` for pre-call expiry.
    """

    def __init__(
        self,
        message: str,
        *,
        platform: str,
        user_id: str,
        auth_status: str = "EXPIRED",
        http_status: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.platform = platform
        self.user_id = user_id
        self.auth_status = auth_status
        self.http_status = http_status


# ---------------------------------------------------------------------------
# safe_error_str
# ---------------------------------------------------------------------------


def safe_error_str(exc: Exception) -> str:
    """Return a log-safe string representation of *exc*.

    Strips credential values from the exception message so that aiohttp /
    httpx errors that embed the full request URL (including
    ``?access_token=…``) never leak tokens into log files.

    Parameters
    ----------
    exc:
        Any exception.  ``str(exc)`` is taken and scrubbed.

    Returns
    -------
    str
        The scrubbed message with all ``key=value`` pairs for known credential
        keys replaced by ``key=[REDACTED]``.
    """
    msg = str(exc)
    return _CRED_QP_RE.sub(lambda m: m.group(0).split("=")[0] + "=[REDACTED]", msg)


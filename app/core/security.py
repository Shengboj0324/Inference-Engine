"""Security utilities and credential encryption."""

import base64
import hashlib
import secrets
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from app.core.config import settings


class CredentialEncryption:
    """Encrypt and decrypt sensitive credentials."""

    def __init__(self, encryption_key: Optional[str] = None):
        """Initialize encryption.

        Args:
            encryption_key: Base64-encoded encryption key
        """
        key = encryption_key or settings.encryption_key

        # Derive Fernet key from encryption key
        if len(key) < 32:
            # Derive key using PBKDF2HMAC
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"social_media_radar_salt",  # In production, use random salt per user
                iterations=100000,
            )
            derived_key = kdf.derive(key.encode())
            fernet_key = base64.urlsafe_b64encode(derived_key)
        else:
            fernet_key = key.encode()

        self.cipher = Fernet(fernet_key)

    def encrypt(self, data: Dict[str, Any]) -> str:
        """Encrypt credentials dictionary.

        Args:
            data: Credentials to encrypt

        Returns:
            Encrypted string
        """
        import json

        json_data = json.dumps(data)
        encrypted = self.cipher.encrypt(json_data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt credentials.

        Args:
            encrypted_data: Encrypted string

        Returns:
            Decrypted credentials dictionary
        """
        import json

        decoded = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = self.cipher.decrypt(decoded)
        return json.loads(decrypted.decode())


class TokenManager:
    """Manage API tokens and secrets."""

    @staticmethod
    def generate_api_key(prefix: str = "smr") -> str:
        """Generate a secure API key.

        Args:
            prefix: Key prefix

        Returns:
            API key string
        """
        random_bytes = secrets.token_bytes(32)
        key = base64.urlsafe_b64encode(random_bytes).decode().rstrip("=")
        return f"{prefix}_{key}"

    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage.

        Args:
            api_key: API key to hash

        Returns:
            Hashed key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    @staticmethod
    def verify_api_key(api_key: str, hashed_key: str) -> bool:
        """Verify API key against hash.

        Args:
            api_key: API key to verify
            hashed_key: Stored hash

        Returns:
            True if key matches
        """
        return TokenManager.hash_api_key(api_key) == hashed_key


class RateLimiter:
    """Rate limiting for API endpoints."""

    def __init__(self, redis_client):
        """Initialize rate limiter.

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client

    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> bool:
        """Check if request is within rate limit.

        Args:
            key: Rate limit key (e.g., user_id, ip_address)
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            True if within limit, False otherwise
        """
        import time

        current_time = int(time.time())
        window_start = current_time - window_seconds

        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)

        # Count requests in window
        request_count = await self.redis.zcard(key)

        if request_count >= max_requests:
            return False

        # Add current request
        await self.redis.zadd(key, {str(current_time): current_time})
        await self.redis.expire(key, window_seconds)

        return True

    async def get_remaining_requests(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> int:
        """Get remaining requests in current window.

        Args:
            key: Rate limit key
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            Number of remaining requests
        """
        import time

        current_time = int(time.time())
        window_start = current_time - window_seconds

        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)

        # Count requests in window
        request_count = await self.redis.zcard(key)

        return max(0, max_requests - request_count)


class InputSanitizer:
    """Sanitize user inputs to prevent injection attacks."""

    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input.

        Args:
            input_str: Input string
            max_length: Maximum allowed length

        Returns:
            Sanitized string
        """
        # Truncate
        sanitized = input_str[:max_length]

        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")

        # Strip whitespace
        sanitized = sanitized.strip()

        return sanitized

    @staticmethod
    def sanitize_url(url: str) -> str:
        """Sanitize URL input.

        Args:
            url: URL to sanitize

        Returns:
            Sanitized URL

        Raises:
            ValueError: If URL is invalid
        """
        from urllib.parse import urlparse

        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ["http", "https"]:
            raise ValueError("Invalid URL scheme")

        # Check for suspicious patterns
        suspicious_patterns = ["javascript:", "data:", "file:", "vbscript:"]
        url_lower = url.lower()
        for pattern in suspicious_patterns:
            if pattern in url_lower:
                raise ValueError("Suspicious URL pattern detected")

        return url

    @staticmethod
    def sanitize_html(html: str) -> str:
        """Sanitize HTML content.

        Args:
            html: HTML to sanitize

        Returns:
            Sanitized HTML
        """
        import bleach

        # Allow only safe tags
        allowed_tags = [
            "p",
            "br",
            "strong",
            "em",
            "u",
            "a",
            "ul",
            "ol",
            "li",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "blockquote",
            "code",
            "pre",
        ]

        allowed_attributes = {"a": ["href", "title"], "img": ["src", "alt"]}

        return bleach.clean(
            html, tags=allowed_tags, attributes=allowed_attributes, strip=True
        )


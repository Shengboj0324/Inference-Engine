"""Authentication routes."""

import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException, Header, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.db import get_db
from app.core.db_models import User
from app.core.models import UserProfile

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# Redis-backed token blacklist
# ---------------------------------------------------------------------------
# Each revoked token is stored in Redis as  blacklist:<sha256(token)>  with
# a TTL equal to the remaining lifetime of the JWT so that entries expire
# automatically — no manual housekeeping needed.
#
# If Redis is temporarily unavailable the blacklist check fails *open*
# (allows the request) and the add silently logs a warning.  This is an
# acceptable trade-off: a brief Redis outage does not lock out all users.
# ---------------------------------------------------------------------------

_redis_client: Optional[aioredis.Redis] = None
_BLACKLIST_PREFIX = "blacklist:"


def _get_redis() -> aioredis.Redis:
    """Return the module-level Redis client, creating it on first call."""
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
    return _redis_client


def _blacklist_key(token: str) -> str:
    """Return the Redis key for a token (hashed to keep key size bounded)."""
    return _BLACKLIST_PREFIX + hashlib.sha256(token.encode()).hexdigest()


async def add_token_to_blacklist(token: str) -> None:
    """Revoke a JWT by storing its hash in Redis with an appropriate TTL.

    The TTL is the remaining valid lifetime of the token as decoded from its
    ``exp`` claim.  If the token is already expired the key is not stored
    (nothing to revoke).  Falls back gracefully if Redis is unavailable.

    Args:
        token: Raw JWT string to revoke.
    """
    # Compute TTL from the token's exp claim.
    ttl_seconds: int = settings.jwt_access_token_expire_minutes * 60  # safe default
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
        exp = payload.get("exp")
        if exp is not None:
            remaining = int(exp) - int(time.time())
            if remaining <= 0:
                logger.debug("Token already expired; skipping blacklist entry.")
                return
            ttl_seconds = remaining
    except JWTError:
        pass  # Malformed/expired token — still try to blacklist with the default TTL.

    try:
        redis = _get_redis()
        await redis.setex(_blacklist_key(token), ttl_seconds, "1")
        logger.info("Token added to Redis blacklist (TTL=%ds)", ttl_seconds)
    except Exception as exc:
        logger.warning("Redis blacklist add failed (fail-open): %s", exc)


async def is_token_blacklisted(token: str) -> bool:
    """Return True if the token has been revoked.

    Falls back gracefully (returns False) if Redis is unavailable so that
    a transient Redis outage does not lock out all authenticated users.

    Args:
        token: Raw JWT string to check.

    Returns:
        True if the token is in the blacklist, False otherwise.
    """
    try:
        redis = _get_redis()
        exists = await redis.exists(_blacklist_key(token))
        return exists > 0
    except Exception as exc:
        logger.warning("Redis blacklist check failed (fail-open): %s", exc)
        return False

# JWT settings — read from the central Settings object so that the
# jwt_access_token_expire_minutes environment variable is actually honoured.
# The previous hardcoded 7-day value contradicted the 30-minute default in
# config.py and made stolen tokens valid far longer than operators expected.
SECRET_KEY = settings.secret_key
ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.jwt_access_token_expire_minutes


class UserCreate(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str


class UserLogin(BaseModel):
    """User login request."""

    email: EmailStr
    password: str


class Token(BaseModel):
    """Authentication token response."""

    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@router.post("/register", response_model=Token)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user.

    Args:
        user: User registration data
        db: Database session

    Returns:
        Authentication token
    """
    try:
        # Check if user already exists
        result = await db.execute(select(User).where(User.email == user.email))
        existing_user = result.scalar_one_or_none()

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Validate password strength
        if len(user.password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long",
            )

        # Create new user
        hashed_password = get_password_hash(user.password)
        new_user = User(
            email=user.email,
            hashed_password=hashed_password,
            is_active=True,
        )

        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        # Generate access token
        access_token = create_access_token(
            data={"sub": str(new_user.id), "email": new_user.email}
        )

        logger.info(f"New user registered: {user.email}")

        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=str(new_user.id),
            email=new_user.email,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user",
        )


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return token.

    Args:
        credentials: User login credentials
        db: Database session

    Returns:
        Authentication token
    """
    try:
        # Find user by email
        result = await db.execute(select(User).where(User.email == credentials.email))
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Verify password
        if not verify_password(credentials.password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive",
            )

        # Generate access token
        access_token = create_access_token(
            data={"sub": str(user.id), "email": user.email}
        )

        logger.info(f"User logged in: {user.email}")

        return Token(
            access_token=access_token,
            token_type="bearer",
            user_id=str(user.id),
            email=user.email,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to authenticate user",
        )


async def get_current_user_from_token(
    token: str,
    db: AsyncSession,
) -> User:
    """Authenticate a raw JWT token string and return the User ORM object.

    This variant is used by WebSocket endpoints, which cannot carry
    ``Authorization`` headers and must pass the token as a query parameter.

    Args:
        token: Raw JWT access token string.
        db: Async database session.

    Returns:
        :class:`~app.core.db_models.User` ORM instance.

    Raises:
        HTTPException 401: If the token is invalid or the user cannot be found.
        HTTPException 403: If the account is inactive.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError as exc:
        logger.warning("WebSocket JWT validation error: %s", exc)
        raise credentials_exception

    try:
        result = await db.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()
        if user is None:
            raise credentials_exception
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive",
            )
        return user
    except (ValueError, HTTPException):
        raise
    except Exception as exc:
        logger.error("Error loading user for WebSocket auth: %s", exc)
        raise credentials_exception


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Logout user (invalidate token).

    Args:
        credentials: HTTP Bearer token

    Returns:
        Success message
    """
    try:
        token = credentials.credentials

        # Add token to Redis blacklist (async, fail-open)
        await add_token_to_blacklist(token)

        logger.info("User logged out successfully")
        return {
            "message": "Logged out successfully",
            "token_invalidated": True,
        }

    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to logout",
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> UserProfile:
    """Get current authenticated user from JWT token.

    This is a dependency function used in protected routes.

    Args:
        credentials: HTTP Bearer token
        db: Database session

    Returns:
        Current user profile

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode JWT token
        token = credentials.credentials

        # Check if token is blacklisted (async, fail-open)
        if await is_token_blacklisted(token):
            logger.warning("Attempted use of blacklisted token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been invalidated",
                headers={"WWW-Authenticate": "Bearer"},
            )

        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")

        if user_id is None or email is None:
            raise credentials_exception

    except JWTError as e:
        logger.warning(f"JWT validation error: {e}")
        raise credentials_exception

    try:
        # Load user from database
        result = await db.execute(select(User).where(User.id == UUID(user_id)))
        user = result.scalar_one_or_none()

        if user is None:
            raise credentials_exception

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive",
            )

        # Return user profile
        return UserProfile(
            id=user.id,
            email=user.email,
        )

    except ValueError:
        # Invalid UUID format
        raise credentials_exception
    except Exception as e:
        logger.error(f"Error loading user: {e}")
        raise credentials_exception


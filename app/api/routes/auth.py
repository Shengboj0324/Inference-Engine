"""Authentication routes."""

import logging
from datetime import datetime, timedelta
from typing import Optional, Set
from uuid import UUID

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

# Token blacklist (in production, use Redis or database)
# This is a simple in-memory implementation for now
_token_blacklist: Set[str] = set()
_blacklist_max_size = 10000  # Prevent memory exhaustion


def add_token_to_blacklist(token: str) -> None:
    """Add token to blacklist with size limit protection.

    Args:
        token: JWT token to blacklist
    """
    global _token_blacklist

    # Prevent memory exhaustion
    if len(_token_blacklist) >= _blacklist_max_size:
        # Remove oldest 20% of tokens (FIFO approximation)
        tokens_to_remove = list(_token_blacklist)[:_blacklist_max_size // 5]
        _token_blacklist -= set(tokens_to_remove)
        logger.warning(f"Token blacklist size limit reached, removed {len(tokens_to_remove)} tokens")

    _token_blacklist.add(token)
    logger.info(f"Token added to blacklist (total: {len(_token_blacklist)})")


def is_token_blacklisted(token: str) -> bool:
    """Check if token is blacklisted.

    Args:
        token: JWT token to check

    Returns:
        True if token is blacklisted
    """
    return token in _token_blacklist

# JWT settings
SECRET_KEY = settings.secret_key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days


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

        # Add token to blacklist
        add_token_to_blacklist(token)

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

        # Check if token is blacklisted
        if is_token_blacklisted(token):
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


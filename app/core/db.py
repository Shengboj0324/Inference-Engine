"""Database configuration and session management with health checks and recovery."""

import logging
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import Pool

from app.core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


# Connection pool event listeners for monitoring
@event.listens_for(Pool, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log database connections."""
    logger.debug("Database connection established")


@event.listens_for(Pool, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Log connection checkout from pool."""
    logger.debug("Connection checked out from pool")


@event.listens_for(Pool, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """Log connection checkin to pool."""
    logger.debug("Connection checked in to pool")


# Async engine for FastAPI with enhanced configuration
async_engine = create_async_engine(
    settings.database_url,
    echo=settings.log_level == "DEBUG",
    pool_pre_ping=True,  # Verify connections before using
    pool_size=10,  # Number of connections to maintain
    max_overflow=20,  # Additional connections when pool is full
    pool_timeout=30,  # Timeout for getting connection from pool
    pool_recycle=3600,  # Recycle connections after 1 hour
    connect_args={
        "server_settings": {
            "application_name": "social_media_radar",
            "jit": "off",  # Disable JIT for better performance on simple queries
        },
        "command_timeout": 60,  # Query timeout in seconds
        "timeout": 10,  # Connection timeout in seconds
    },
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Sync engine for Celery and migrations
sync_engine = create_engine(
    settings.database_sync_url,
    echo=settings.log_level == "DEBUG",
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=3600,
    connect_args={
        "application_name": "social_media_radar_sync",
        "connect_timeout": 10,
    },
)

# Sync session factory
SessionLocal = sessionmaker(
    sync_engine,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting async database sessions with error handling.

    Yields:
        AsyncSession: Database session

    Raises:
        OperationalError: If database connection fails
    """
    session: Optional[AsyncSession] = None
    try:
        session = AsyncSessionLocal()
        yield session
        await session.commit()
    except OperationalError as e:
        logger.error(f"Database operational error: {e}")
        if session:
            await session.rollback()
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        if session:
            await session.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database session: {e}")
        if session:
            await session.rollback()
        raise
    finally:
        if session:
            await session.close()


def get_sync_db():
    """Get synchronous database session for Celery tasks with error handling.

    Yields:
        Session: Database session

    Raises:
        OperationalError: If database connection fails
    """
    db = None
    try:
        db = SessionLocal()
        yield db
        db.commit()
    except OperationalError as e:
        logger.error(f"Database operational error: {e}")
        if db:
            db.rollback()
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        if db:
            db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error in database session: {e}")
        if db:
            db.rollback()
        raise
    finally:
        if db:
            db.close()


async def check_database_health() -> bool:
    """Check database health.

    Returns:
        bool: True if database is healthy, False otherwise
    """
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
            logger.info("Database health check: OK")
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


def check_database_health_sync() -> bool:
    """Check database health synchronously.

    Returns:
        bool: True if database is healthy, False otherwise
    """
    try:
        with SessionLocal() as session:
            session.execute(text("SELECT 1"))
            logger.info("Database health check (sync): OK")
            return True
    except Exception as e:
        logger.error(f"Database health check (sync) failed: {e}")
        return False


async def close_database_connections() -> None:
    """Close all database connections gracefully."""
    logger.info("Closing database connections...")
    await async_engine.dispose()
    sync_engine.dispose()
    logger.info("Database connections closed")


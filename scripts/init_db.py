#!/usr/bin/env python3
"""Initialize database with pgvector extension and run migrations."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from sqlalchemy import text

from app.core.config import settings
from app.core.db import async_engine, sync_engine


async def init_database():
    """Initialize database with required extensions."""
    print("Initializing database...")

    # Create pgvector extension
    print("Creating pgvector extension...")
    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        print("✓ pgvector extension created")

    print("\nDatabase initialization complete!")
    print("\nNext steps:")
    print("1. Run migrations: alembic upgrade head")
    print("2. Start the API server: uvicorn app.api.main:app --reload")


if __name__ == "__main__":
    asyncio.run(init_database())


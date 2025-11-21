#!/usr/bin/env python3
"""Create a new user account."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from getpass import getpass

from sqlalchemy import select

from app.core.db import AsyncSessionLocal
from app.core.db_models import User
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def create_user(email: str, password: str):
    """Create a new user account.

    Args:
        email: User email
        password: User password
    """
    async with AsyncSessionLocal() as session:
        # Check if user exists
        result = await session.execute(select(User).where(User.email == email))
        existing_user = result.scalar_one_or_none()

        if existing_user:
            print(f"Error: User with email {email} already exists")
            return

        # Create user
        hashed_password = pwd_context.hash(password)
        user = User(email=email, hashed_password=hashed_password, is_active=True)

        session.add(user)
        await session.commit()

        print(f"✓ User created successfully!")
        print(f"  Email: {email}")
        print(f"  ID: {user.id}")


async def main():
    """Main function."""
    print("Create New User")
    print("=" * 50)

    email = input("Email: ").strip()
    if not email:
        print("Error: Email is required")
        return

    password = getpass("Password: ")
    if not password:
        print("Error: Password is required")
        return

    password_confirm = getpass("Confirm password: ")
    if password != password_confirm:
        print("Error: Passwords do not match")
        return

    await create_user(email, password)


if __name__ == "__main__":
    asyncio.run(main())


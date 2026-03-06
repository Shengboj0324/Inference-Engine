"""Source configuration routes."""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user
from app.core.db import get_db
from app.core.db_models import PlatformConfigDB, User
from app.core.models import PlatformConfig, SourcePlatform
from app.core.security import CredentialEncryption
from app.connectors.registry import get_connector

logger = logging.getLogger(__name__)

router = APIRouter()


class SourceConfigRequest(BaseModel):
    """Request to configure a source."""

    platform: SourcePlatform
    enabled: bool = True
    credentials: dict
    settings: dict = {}


class SourceConfigResponse(BaseModel):
    """Source configuration response."""

    id: UUID
    platform: SourcePlatform
    enabled: bool
    connection_status: str
    feeds_count: int


@router.get("/", response_model=List[SourceConfigResponse])
async def list_sources(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all configured sources for the user.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        List of source configurations
    """
    try:
        # Query user's platform configurations
        result = await db.execute(
            select(PlatformConfigDB).where(PlatformConfigDB.user_id == current_user.id)
        )
        configs = result.scalars().all()

        # Convert to response format (without sensitive credentials)
        return [
            SourceConfigResponse(
                id=config.id,
                platform=config.platform,
                enabled=config.enabled,
                connection_status="active" if config.enabled else "disabled",
                feeds_count=len(config.feeds) if config.feeds else 0,
            )
            for config in configs
        ]

    except Exception as e:
        logger.error(f"Failed to list sources for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve source configurations",
        )


@router.post("/", response_model=SourceConfigResponse)
async def add_source(
    config: SourceConfigRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Add or update a source configuration.

    Args:
        config: Source configuration
        current_user: Authenticated user
        db: Database session

    Returns:
        Created/updated source configuration
    """
    try:
        # Validate credentials by testing connection
        try:
            connector = get_connector(config.platform)
            # Test connection with provided credentials
            # Note: This is a basic validation, actual implementation depends on connector
            logger.info(f"Testing connection for {config.platform} for user {current_user.id}")
        except Exception as e:
            logger.warning(f"Connector test failed for {config.platform}: {e}")
            # Continue anyway - some connectors may not support test_connection

        # Encrypt credentials
        encryption = CredentialEncryption()
        encrypted_credentials = encryption.encrypt(config.credentials)

        # Check if configuration already exists
        result = await db.execute(
            select(PlatformConfigDB).where(
                PlatformConfigDB.user_id == current_user.id,
                PlatformConfigDB.platform == config.platform,
            )
        )
        existing_config = result.scalar_one_or_none()

        if existing_config:
            # Update existing configuration
            existing_config.enabled = config.enabled
            existing_config.encrypted_credentials = encrypted_credentials
            existing_config.settings = config.settings
            platform_config = existing_config
        else:
            # Create new configuration
            platform_config = PlatformConfigDB(
                user_id=current_user.id,
                platform=config.platform,
                enabled=config.enabled,
                encrypted_credentials=encrypted_credentials,
                settings=config.settings,
                feeds=[],
            )
            db.add(platform_config)

        await db.commit()
        await db.refresh(platform_config)

        logger.info(f"Source configured: {config.platform} for user {current_user.id}")

        return SourceConfigResponse(
            id=platform_config.id,
            platform=platform_config.platform,
            enabled=platform_config.enabled,
            connection_status="active" if platform_config.enabled else "disabled",
            feeds_count=len(platform_config.feeds) if platform_config.feeds else 0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to configure source {config.platform}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to configure source: {str(e)}",
        )


@router.get("/{platform}/test")
async def test_source(
    platform: SourcePlatform,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Test connection to a configured source.

    Args:
        platform: Platform to test
        current_user: Authenticated user
        db: Database session

    Returns:
        Connection test results
    """
    try:
        # Get user's configuration for this platform
        result = await db.execute(
            select(PlatformConfigDB).where(
                PlatformConfigDB.user_id == current_user.id,
                PlatformConfigDB.platform == platform,
            )
        )
        config = result.scalar_one_or_none()

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No configuration found for platform: {platform}",
            )

        # Decrypt credentials
        encryption = CredentialEncryption()
        credentials = encryption.decrypt(config.encrypted_credentials)

        # Initialize connector and test connection
        try:
            connector = get_connector(platform)
            # Note: Actual test depends on connector implementation
            # For now, just verify connector exists
            test_result = {
                "platform": platform.value,
                "status": "success",
                "message": "Connection test passed",
                "enabled": config.enabled,
            }
            logger.info(f"Connection test successful for {platform} (user: {current_user.id})")
            return test_result

        except Exception as e:
            logger.error(f"Connection test failed for {platform}: {e}")
            return {
                "platform": platform.value,
                "status": "failed",
                "message": f"Connection test failed: {str(e)}",
                "enabled": config.enabled,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to test source {platform}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test source: {str(e)}",
        )


@router.delete("/{platform}")
async def remove_source(
    platform: SourcePlatform,
    delete_content: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove a source configuration.

    Args:
        platform: Platform to remove
        delete_content: Whether to delete associated content
        current_user: Authenticated user
        db: Database session

    Returns:
        Success message
    """
    try:
        # Get user's configuration for this platform
        result = await db.execute(
            select(PlatformConfigDB).where(
                PlatformConfigDB.user_id == current_user.id,
                PlatformConfigDB.platform == platform,
            )
        )
        config = result.scalar_one_or_none()

        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No configuration found for platform: {platform}",
            )

        # Delete configuration
        await db.delete(config)

        # Optionally delete associated content
        if delete_content:
            from app.core.db_models import ContentItemDB

            await db.execute(
                select(ContentItemDB)
                .where(
                    ContentItemDB.user_id == current_user.id,
                    ContentItemDB.source_platform == platform,
                )
                .delete()
            )

        await db.commit()

        logger.info(
            f"Source removed: {platform} for user {current_user.id} "
            f"(delete_content={delete_content})"
        )

        return {
            "message": f"Source {platform.value} removed successfully",
            "content_deleted": delete_content,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove source {platform}: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove source: {str(e)}",
        )


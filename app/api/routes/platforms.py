"""Platform connection and management API routes.

Simplified onboarding flow - users just click "Connect" buttons!
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.auth import get_current_user
from app.core.credential_vault import CredentialVault
from app.core.db import get_db
from app.core.models import SourcePlatform, UserProfile
from app.oauth.oauth_proxy import OAuthProxyService


router = APIRouter(prefix="/platforms", tags=["platforms"])


class PlatformInfo(BaseModel):
    """Platform information."""
    
    platform: SourcePlatform
    name: str
    description: str
    requires_oauth: bool
    requires_api_key: bool
    is_connected: bool
    connection_url: Optional[str] = None


class ConnectPlatformResponse(BaseModel):
    """Response for platform connection initiation."""
    
    platform: SourcePlatform
    authorization_url: str
    message: str


class PlatformConnectionStatus(BaseModel):
    """Platform connection status."""
    
    platform: SourcePlatform
    is_connected: bool
    credential_id: Optional[str] = None
    connected_at: Optional[str] = None
    last_accessed: Optional[str] = None


# Platform metadata
PLATFORM_INFO = {
    SourcePlatform.REDDIT: {
        "name": "Reddit",
        "description": "Connect your Reddit account to monitor subreddits and posts",
        "requires_oauth": True,
        "requires_api_key": False,
    },
    SourcePlatform.YOUTUBE: {
        "name": "YouTube",
        "description": "Connect YouTube to track channels and videos",
        "requires_oauth": True,
        "requires_api_key": False,
    },
    SourcePlatform.TIKTOK: {
        "name": "TikTok",
        "description": "Connect TikTok to monitor trending videos and hashtags",
        "requires_oauth": True,
        "requires_api_key": False,
    },
    SourcePlatform.FACEBOOK: {
        "name": "Facebook",
        "description": "Connect Facebook to track pages and posts",
        "requires_oauth": True,
        "requires_api_key": False,
    },
    SourcePlatform.INSTAGRAM: {
        "name": "Instagram",
        "description": "Connect Instagram Business account to monitor posts and stories",
        "requires_oauth": True,
        "requires_api_key": False,
    },
    SourcePlatform.WECHAT: {
        "name": "WeChat",
        "description": "Connect WeChat Official Account to track articles",
        "requires_oauth": True,
        "requires_api_key": False,
    },
    SourcePlatform.NYTIMES: {
        "name": "New York Times",
        "description": "Connect NYTimes API to access articles and news",
        "requires_oauth": False,
        "requires_api_key": True,
    },
    SourcePlatform.GOOGLE_NEWS: {
        "name": "Google News",
        "description": "Monitor Google News RSS feeds (no authentication required)",
        "requires_oauth": False,
        "requires_api_key": False,
    },
    SourcePlatform.WSJ: {
        "name": "Wall Street Journal",
        "description": "Monitor WSJ RSS feeds (no authentication required)",
        "requires_oauth": False,
        "requires_api_key": False,
    },
    SourcePlatform.ABC_NEWS: {
        "name": "ABC News",
        "description": "Monitor ABC News RSS feeds (no authentication required)",
        "requires_oauth": False,
        "requires_api_key": False,
    },
    SourcePlatform.APPLE_NEWS: {
        "name": "Apple News",
        "description": "Monitor Apple Newsroom (no authentication required)",
        "requires_oauth": False,
        "requires_api_key": False,
    },
}


@router.get("/", response_model=List[PlatformInfo])
async def list_platforms(
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all available platforms and their connection status.
    
    This endpoint shows users which platforms they can connect to.
    """
    vault = CredentialVault(db)
    
    # Get user's connected platforms
    credentials = await vault.list_credentials(current_user.id)
    connected_platforms = {cred["platform"] for cred in credentials}
    
    platforms = []
    for platform, info in PLATFORM_INFO.items():
        platforms.append(PlatformInfo(
            platform=platform,
            name=info["name"],
            description=info["description"],
            requires_oauth=info["requires_oauth"],
            requires_api_key=info["requires_api_key"],
            is_connected=platform.value in connected_platforms,
        ))
    
    return platforms


@router.post("/connect/{platform}", response_model=ConnectPlatformResponse)
async def connect_platform(
    platform: SourcePlatform,
    request: Request,
    current_user: UserProfile = Depends(get_current_user),
):
    """Initiate platform connection (Step 1 of OAuth flow).
    
    This is the "Connect [Platform]" button endpoint!
    Users just call this and get a URL to visit.
    
    Args:
        platform: Platform to connect
        
    Returns:
        Authorization URL for user to visit
    """
    # Get OAuth proxy service (injected via dependency in production)
    from app.core.config import settings
    import os

    # Load app credentials from environment variables
    app_credentials = {
        "twitter": {
            "client_id": os.getenv("TWITTER_CLIENT_ID", ""),
            "client_secret": os.getenv("TWITTER_CLIENT_SECRET", ""),
        },
        "reddit": {
            "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
            "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
        },
        "linkedin": {
            "client_id": os.getenv("LINKEDIN_CLIENT_ID", ""),
            "client_secret": os.getenv("LINKEDIN_CLIENT_SECRET", ""),
        },
    }

    oauth_service = OAuthProxyService(
        credential_vault=None,  # Will be injected
        app_credentials=app_credentials,
        base_redirect_uri=f"{settings.api_base_url}/api/v1/platforms/callback"
    )
    
    try:
        auth_url = oauth_service.get_authorization_url(
            user_id=current_user.id,
            platform=platform
        )
        
        return ConnectPlatformResponse(
            platform=platform,
            authorization_url=auth_url,
            message=f"Visit the URL to authorize {PLATFORM_INFO[platform]['name']}"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/callback/{platform}")
async def oauth_callback(
    platform: SourcePlatform,
    code: str = Query(..., description="Authorization code"),
    state: str = Query(..., description="State parameter"),
    request: Request = None,
    db: AsyncSession = Depends(get_db)
):
    """Handle OAuth callback (Step 2 of OAuth flow).
    
    This endpoint is called automatically by the platform after user authorizes.
    Users don't need to interact with this directly!
    
    Args:
        platform: Platform
        code: Authorization code
        state: State parameter
        
    Returns:
        Success message and redirect to app
    """
    # Get user from session state
    # The state parameter contains the user_id encoded during authorization
    import json
    import base64

    try:
        # Decode state to get user_id
        state_data = json.loads(base64.b64decode(state))
        user_id = state_data.get("user_id")

        # Get user password from secure session or credential vault
        # Note: In production, use session management or secure token exchange
        from app.core.config import settings
        user_password = settings.secret_key  # Use app secret for encryption

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid state parameter: {e}"
        )

    vault = CredentialVault(db)

    # Load app credentials from environment
    import os
    app_credentials = {
        "twitter": {
            "client_id": os.getenv("TWITTER_CLIENT_ID", ""),
            "client_secret": os.getenv("TWITTER_CLIENT_SECRET", ""),
        },
        "reddit": {
            "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
            "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
        },
        "linkedin": {
            "client_id": os.getenv("LINKEDIN_CLIENT_ID", ""),
            "client_secret": os.getenv("LINKEDIN_CLIENT_SECRET", ""),
        },
    }

    from app.core.config import settings
    oauth_service = OAuthProxyService(
        credential_vault=vault,
        app_credentials=app_credentials,
        base_redirect_uri=f"{settings.api_base_url}/api/v1/platforms/callback"
    )
    
    try:
        result = await oauth_service.handle_callback(
            platform=platform,
            code=code,
            state=state,
            user_password=user_password
        )
        
        # Redirect to success page
        return {
            "success": True,
            "message": f"{PLATFORM_INFO[platform]['name']} connected successfully!",
            "credential_id": result["credential_id"],
            "platform": result["platform"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status", response_model=List[PlatformConnectionStatus])
async def get_connection_status(
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get connection status for all platforms.
    
    Shows which platforms are connected and when they were last used.
    """
    vault = CredentialVault(db)
    credentials = await vault.list_credentials(current_user.id)
    
    # Create status for all platforms
    statuses = []
    for platform in PLATFORM_INFO.keys():
        cred = next((c for c in credentials if c["platform"] == platform.value), None)
        
        statuses.append(PlatformConnectionStatus(
            platform=platform,
            is_connected=cred is not None,
            credential_id=cred["id"] if cred else None,
            connected_at=cred["created_at"] if cred else None,
            last_accessed=cred["last_accessed"] if cred else None,
        ))
    
    return statuses


@router.delete("/disconnect/{platform}")
async def disconnect_platform(
    platform: SourcePlatform,
    current_user: UserProfile = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Disconnect a platform.
    
    Removes stored credentials for the platform.
    """
    vault = CredentialVault(db)
    credentials = await vault.list_credentials(current_user.id, platform=platform.value)
    
    if not credentials:
        raise HTTPException(status_code=404, detail="Platform not connected")
    
    # Delete all credentials for this platform
    for cred in credentials:
        await vault.delete_credential(UUID(cred["id"]))
    
    return {
        "success": True,
        "message": f"{PLATFORM_INFO[platform]['name']} disconnected successfully"
    }


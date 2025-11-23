"""Application configuration using Pydantic settings."""

from typing import List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://radar:radar_password@localhost:5432/social_radar"
    )
    database_sync_url: str = Field(
        default="postgresql://radar:radar_password@localhost:5432/social_radar"
    )

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Celery
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")

    # Object Storage
    s3_endpoint: str = Field(default="http://localhost:9000")
    s3_access_key: str = Field(default="minioadmin")
    s3_secret_key: str = Field(default="minioadmin")
    s3_bucket: str = Field(default="radar-content")

    # LLM Providers
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Embedding Configuration
    embedding_provider: str = Field(default="openai")
    embedding_model: str = Field(default="text-embedding-3-large")
    embedding_dimension: int = Field(default=1536)

    # LLM Configuration
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4-turbo-preview")
    llm_temperature: float = Field(default=0.7)

    # Local Model Settings
    local_model_path: Optional[str] = None
    vllm_endpoint: Optional[str] = None

    # Security
    secret_key: str = Field(default="change-this-in-production")
    encryption_key: str = Field(default="change-this-32-byte-key-base64==")

    # API Settings
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_workers: int = Field(default=4)
    cors_origins: List[str] = Field(default=["http://localhost:3000"])

    # MCP Server
    mcp_host: str = Field(default="0.0.0.0")
    mcp_port: int = Field(default=8001)

    # Ingestion Settings
    ingestion_interval_minutes: int = Field(default=15)
    max_items_per_fetch: int = Field(default=100)

    # Content Settings
    max_content_age_days: int = Field(default=30)
    cluster_min_similarity: float = Field(default=0.7)
    max_clusters_per_digest: int = Field(default=20)

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True)
    rate_limit_per_minute: int = Field(default=60)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # Feature Flags
    enable_video_transcription: bool = Field(default=True)
    enable_auto_clustering: bool = Field(default=True)
    enable_personalization: bool = Field(default=True)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.strip("[]").split(",")]
        return v


# Global settings instance
settings = Settings()


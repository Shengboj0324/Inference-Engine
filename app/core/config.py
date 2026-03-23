"""Application configuration using Pydantic settings with comprehensive validation."""

import os
import secrets
from typing import List, Optional, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: str = Field(default="development")

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

    # Ollama (offline local LLM) — competitive_analysis.md §5.2
    local_llm_url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL for the Ollama REST API (e.g. http://localhost:11434). "
            "When set, LLMRouter will prefer OllamaProvider for non-frontier signals."
        ),
    )
    local_llm_model: str = Field(
        default="llama3.1:8b",
        description="Ollama model tag to use for inference and embeddings.",
    )

    # Fine-tuned model — competitive_analysis.md §5.3 / Step 3
    fine_tuned_model_id: Optional[str] = Field(
        default=None,
        description=(
            "OpenAI fine-tuned model ID (e.g. ft:gpt-4o-mini-2024-07-18:...). "
            "When set, LLMRouter routes non-frontier signal types to this model."
        ),
    )

    # Security
    secret_key: str = Field(default="change-this-in-production")
    encryption_key: str = Field(default="change-this-32-byte-key-base64==")

    # JWT Settings
    jwt_algorithm: str = Field(default="HS256")
    jwt_access_token_expire_minutes: int = Field(default=30)
    jwt_refresh_token_expire_days: int = Field(default=7)

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

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        allowed = ["development", "staging", "production", "test"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.strip("[]").split(",")]
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v_upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        allowed = ["json", "plain"]
        if v not in allowed:
            raise ValueError(f"Log format must be one of {allowed}")
        return v

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate LLM temperature."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("cluster_min_similarity")
    @classmethod
    def validate_similarity(cls, v: float) -> float:
        """Validate cluster similarity threshold."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity must be between 0.0 and 1.0")
        return v

    @field_validator("api_port", "mcp_port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v

    @field_validator("api_workers")
    @classmethod
    def validate_workers(cls, v: int) -> int:
        """Validate number of workers."""
        if v < 1:
            raise ValueError("Workers must be at least 1")
        if v > 32:
            raise ValueError("Workers should not exceed 32")
        return v

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Validate production-specific settings."""
        if self.environment == "production":
            # Check for default secrets
            if self.secret_key == "change-this-in-production":
                raise ValueError(
                    "SECRET_KEY must be changed in production! "
                    "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )

            if self.encryption_key == "change-this-32-byte-key-base64==":
                raise ValueError(
                    "ENCRYPTION_KEY must be changed in production! "
                    "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
                )

            # Warn about localhost in production
            if "localhost" in self.database_url:
                import warnings
                warnings.warn("Using localhost database in production!")

            if "localhost" in self.redis_url:
                import warnings
                warnings.warn("Using localhost Redis in production!")

        return self

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"

    @property
    def is_testing(self) -> bool:
        """Check if running in test mode."""
        return self.environment == "test"


# Global settings instance (singleton — constructed once at import time)
settings = Settings()


"""Initial schema — all base tables except actionable_signals.

Revision ID: 000_initial
Revises:
Create Date: 2026-03-23

Creates:
  - pgvector extension
  - Enum types: sourceplatform, mediatype
  - users
  - interest_profiles
  - platform_configs
  - content_items     (with vector(1536) embedding)
  - clusters
  - raw_observations
  - normalized_observations  (with vector(1536) embedding)
  - signal_inferences
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "000_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _enum(*values, name: str):
    return postgresql.ENUM(*values, name=name, create_type=True)


# ---------------------------------------------------------------------------
# upgrade
# ---------------------------------------------------------------------------

def upgrade() -> None:
    # pgvector extension (safe to call on any existing DB)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── Enums ─────────────────────────────────────────────────────────────
    sourceplatform = _enum(
        "reddit", "youtube", "tiktok", "facebook", "instagram", "wechat",
        "rss", "newsapi", "nytimes", "wsj", "abc_news", "abc_news_au",
        "google_news", "apple_news",
        name="sourceplatform",
    )
    sourceplatform.create(op.get_bind(), checkfirst=True)

    mediatype = _enum("text", "video", "image", "mixed", name="mediatype")
    mediatype.create(op.get_bind(), checkfirst=True)

    # ── users ─────────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True, if_not_exists=True)

    # ── interest_profiles ─────────────────────────────────────────────────
    op.create_table(
        "interest_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False, unique=True),
        sa.Column("interest_topics", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("negative_filters", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("interest_embedding", sa.Text(), nullable=True),  # stored as text; pgvector via ORM
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )

    # ── platform_configs ──────────────────────────────────────────────────
    op.create_table(
        "platform_configs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("platform", sourceplatform, nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("encrypted_credentials", sa.Text(), nullable=True),
        sa.Column("settings", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("last_fetch_time", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )



    # ── content_items ─────────────────────────────────────────────────────
    op.create_table(
        "content_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_platform", sourceplatform, nullable=False),
        sa.Column("source_id", sa.String(500), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("author", sa.String(255), nullable=True),
        sa.Column("channel", sa.String(255), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("media_type", mediatype, nullable=False),
        sa.Column("media_urls", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("topics", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("lang", sa.String(10), nullable=True),
        sa.Column("embedding", sa.Text(), nullable=True),
        sa.Column("metadata", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "source_platform", "source_id", name="uq_user_platform_source"),
    )
    op.create_index("ix_content_items_user_id", "content_items", ["user_id"], if_not_exists=True)
    op.create_index("ix_content_items_source_platform", "content_items", ["source_platform"], if_not_exists=True)
    op.create_index("ix_content_items_published_at", "content_items", ["published_at"], if_not_exists=True)

    # ── clusters ──────────────────────────────────────────────────────────
    op.create_table(
        "clusters",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("topic", sa.String(500), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("keywords", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("item_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("relevance_score", sa.Float(), nullable=False),
        sa.Column("diversity_score", sa.Float(), nullable=True),
        sa.Column("platforms_represented", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("perspective_summary", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_clusters_user_id", "clusters", ["user_id"], if_not_exists=True)
    op.create_index("ix_clusters_created_at", "clusters", ["created_at"], if_not_exists=True)


    # ── raw_observations ──────────────────────────────────────────────────
    op.create_table(
        "raw_observations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_platform", sourceplatform, nullable=False),
        sa.Column("source_id", sa.String(500), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("author", sa.String(255), nullable=True),
        sa.Column("author_id", sa.String(255), nullable=True),
        sa.Column("channel", sa.String(255), nullable=True),
        sa.Column("channel_id", sa.String(255), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("media_type", mediatype, nullable=False),
        sa.Column("media_urls", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("platform_metadata", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "source_platform", "source_id", name="uq_raw_user_platform_source"),
    )
    op.create_index("ix_raw_observations_user_id", "raw_observations", ["user_id"], if_not_exists=True)
    op.create_index("ix_raw_observations_source_platform", "raw_observations", ["source_platform"], if_not_exists=True)
    op.create_index("ix_raw_observations_published_at", "raw_observations", ["published_at"], if_not_exists=True)

    # ── normalized_observations ───────────────────────────────────────────
    op.create_table(
        "normalized_observations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("raw_observation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_platform", sourceplatform, nullable=False),
        sa.Column("source_id", sa.String(500), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("author", sa.String(255), nullable=True),
        sa.Column("channel", sa.String(255), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=True),
        sa.Column("original_language", sa.String(10), nullable=True),
        sa.Column("translated_text", sa.Text(), nullable=True),
        sa.Column("media_type", mediatype, nullable=False),
        sa.Column("media_urls", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{}"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("normalized_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("entities", postgresql.ARRAY(postgresql.JSON()), nullable=False, server_default="{}"),
        sa.Column("topics", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("keywords", postgresql.ARRAY(sa.String()), nullable=False, server_default="{}"),
        sa.Column("sentiment", sa.String(50), nullable=True),
        sa.Column("sentiment_score", sa.Float(), nullable=True),
        sa.Column("thread_context", postgresql.JSON(), nullable=True),
        sa.Column("quality", sa.String(50), nullable=False, server_default="unknown"),
        sa.Column("quality_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("completeness_score", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("engagement_velocity", sa.Float(), nullable=True),
        sa.Column("virality_score", sa.Float(), nullable=True),
        sa.Column("embedding", sa.Text(), nullable=True),
        sa.Column("normalization_metadata", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["raw_observation_id"], ["raw_observations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_normalized_obs_raw_id", "normalized_observations", ["raw_observation_id"], if_not_exists=True)
    op.create_index("ix_normalized_obs_user_id", "normalized_observations", ["user_id"], if_not_exists=True)
    op.create_index("ix_normalized_obs_platform", "normalized_observations", ["source_platform"], if_not_exists=True)
    op.create_index("ix_normalized_obs_normalized_at", "normalized_observations", ["normalized_at"], if_not_exists=True)

    # ── signal_inferences ─────────────────────────────────────────────────
    op.create_table(
        "signal_inferences",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("normalized_observation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("predictions", postgresql.ARRAY(postgresql.JSON()), nullable=False, server_default="{}"),
        sa.Column("top_prediction", postgresql.JSON(), nullable=True),
        sa.Column("abstained", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("abstention_reason", sa.String(100), nullable=True),
        sa.Column("abstention_confidence", sa.Float(), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("evidence_summary", sa.Text(), nullable=True),
        sa.Column("calibration_metrics", postgresql.JSON(), nullable=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("inference_method", sa.String(100), nullable=False),
        sa.Column("inferred_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("inference_metadata", postgresql.JSON(), nullable=False, server_default="{}"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["normalized_observation_id"], ["normalized_observations.id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_signal_inferences_norm_obs_id", "signal_inferences", ["normalized_observation_id"], if_not_exists=True)
    op.create_index("ix_signal_inferences_user_id", "signal_inferences", ["user_id"], if_not_exists=True)
    op.create_index("ix_signal_inferences_abstained", "signal_inferences", ["abstained"], if_not_exists=True)
    op.create_index("ix_signal_inferences_inferred_at", "signal_inferences", ["inferred_at"], if_not_exists=True)


# ---------------------------------------------------------------------------
# downgrade
# ---------------------------------------------------------------------------

def downgrade() -> None:
    op.drop_table("signal_inferences")
    op.drop_table("normalized_observations")
    op.drop_table("raw_observations")
    op.drop_table("clusters")
    op.drop_table("content_items")
    op.drop_table("platform_configs")
    op.drop_table("interest_profiles")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS mediatype")
    op.execute("DROP TYPE IF EXISTS sourceplatform")

"""add actionable signals table

Revision ID: 001_signals
Revises: 
Create Date: 2026-03-07

This migration adds the core actionable_signals table which is the primary
product entity for the signal-to-action workflow.

Design decisions:
- Multiple indexes for fast queue retrieval (action_score, status, created_at, expires_at)
- Enum types for signal_type, action_type, signal_status, response_tone
- Foreign keys with appropriate cascade behavior
- ARRAY type for source_item_ids to support multi-item signals
- JSON type for metadata and outcome_feedback
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_signals'
down_revision: Union[str, None] = '000_initial'  # depends on base schema
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create actionable_signals table and related enums."""
    
    # Create enum types
    signal_type_enum = postgresql.ENUM(
        'lead_opportunity',
        'competitor_weakness',
        'influencer_amplification',
        'churn_risk',
        'misinformation_risk',
        'support_escalation',
        'product_confusion',
        'feature_request_pattern',
        'launch_moment',
        'trend_to_content',
        name='signaltype',
        create_type=True,
    )
    signal_type_enum.create(op.get_bind(), checkfirst=True)
    
    action_type_enum = postgresql.ENUM(
        'reply_public',
        'dm_outreach',
        'create_content',
        'internal_alert',
        'monitor',
        'escalate',
        name='actiontype',
        create_type=True,
    )
    action_type_enum.create(op.get_bind(), checkfirst=True)
    
    signal_status_enum = postgresql.ENUM(
        'new',
        'queued',
        'in_progress',
        'acted',
        'dismissed',
        'expired',
        name='signalstatus',
        create_type=True,
    )
    signal_status_enum.create(op.get_bind(), checkfirst=True)
    
    response_tone_enum = postgresql.ENUM(
        'helpful',
        'professional',
        'technical',
        'founder_voice',
        'supportive',
        'educational',
        name='responsetone',
        create_type=True,
    )
    response_tone_enum.create(op.get_bind(), checkfirst=True)
    
    # Create actionable_signals table
    op.create_table(
        'actionable_signals',
        # Primary key
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        
        # Ownership
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),

        # Links to inference layer (Phase 1)
        sa.Column('signal_inference_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('normalized_observation_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),

        # Team collaboration
        sa.Column('team_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('assigned_role', sa.String(50), nullable=True),

        # Signal classification
        sa.Column('signal_type', signal_type_enum, nullable=False, index=True),
        
        # Source tracking
        sa.Column('source_item_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False),
        sa.Column('source_platform', sa.String(50), nullable=False, index=True),
        sa.Column('source_url', sa.Text(), nullable=False),
        sa.Column('source_author', sa.String(255), nullable=True),
        
        # Signal content
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('context', sa.Text(), nullable=False),
        
        # Multi-dimensional scoring
        sa.Column('urgency_score', sa.Float(), nullable=False),
        sa.Column('impact_score', sa.Float(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('action_score', sa.Float(), nullable=False, index=True),
        
        # Recommended actions
        sa.Column('recommended_action', action_type_enum, nullable=False),
        sa.Column('suggested_channel', sa.String(50), nullable=False),
        sa.Column('suggested_tone', response_tone_enum, nullable=False),
        
        # Generated assets
        sa.Column('draft_response', sa.Text(), nullable=True),
        sa.Column('draft_post', sa.Text(), nullable=True),
        sa.Column('draft_dm', sa.Text(), nullable=True),
        sa.Column('positioning_angle', sa.Text(), nullable=True),
        
        # Workflow management
        sa.Column('status', signal_status_enum, nullable=False, server_default='new', index=True),
        sa.Column('assigned_to', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()'), index=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True, index=True),
        sa.Column('acted_at', sa.DateTime(), nullable=True),
        
        # Learning loop
        sa.Column('outcome_feedback', postgresql.JSON(), nullable=True),
        
        # Additional data
        sa.Column('metadata', postgresql.JSON(), nullable=False, server_default='{}'),
        
        # Foreign keys
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['assigned_to'], ['users.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['signal_inference_id'], ['signal_inferences.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['normalized_observation_id'], ['normalized_observations.id'], ondelete='SET NULL'),
    )

    # signal_feedback — human corrections for online calibration
    op.create_table(
        'signal_feedback',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('signal_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('predicted_type', sa.String(50), nullable=False),
        sa.Column('true_type', sa.String(50), nullable=False),
        sa.Column('predicted_confidence', sa.Float(), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), index=True),
        sa.ForeignKeyConstraint(['signal_id'], ['actionable_signals.id'], ondelete='CASCADE'),
    )


def downgrade() -> None:
    """Drop actionable_signals, signal_feedback, and related enums."""
    op.drop_table('signal_feedback')
    op.drop_table('actionable_signals')
    
    # Drop enum types
    op.execute('DROP TYPE IF EXISTS signaltype')
    op.execute('DROP TYPE IF EXISTS actiontype')
    op.execute('DROP TYPE IF EXISTS signalstatus')
    op.execute('DROP TYPE IF EXISTS responsetone')


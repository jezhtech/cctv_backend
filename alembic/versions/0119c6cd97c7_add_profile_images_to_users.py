"""add_profile_images_to_users

Revision ID: 0119c6cd97c7
Revises: 6f66e43a6c94
Create Date: 2025-08-19 14:02:59.948502

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0119c6cd97c7'
down_revision = '6f66e43a6c94'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add profile_images column to users table
    op.add_column('users', sa.Column('profile_images', postgresql.JSON, nullable=True))


def downgrade() -> None:
    # Remove profile_images column from users table
    op.drop_column('users', 'profile_images')

"""Add new columns to face_embeddings table

Revision ID: add_face_embedding_columns
Revises: 0119c6cd97c7
Create Date: 2024-08-19 17:55:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_face_embedding_columns'
down_revision = '0119c6cd97c7'
branch_labels = None
depends_on = None


def upgrade():
    # Add new columns to face_embeddings table
    op.add_column('face_embeddings', sa.Column('face_size', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    op.add_column('face_embeddings', sa.Column('face_angle', sa.Float(), nullable=True))
    op.add_column('face_embeddings', sa.Column('lighting_score', sa.Float(), nullable=True))
    op.add_column('face_embeddings', sa.Column('blur_score', sa.Float(), nullable=True))


def downgrade():
    # Remove the columns
    op.drop_column('face_embeddings', 'blur_score')
    op.drop_column('face_embeddings', 'lighting_score')
    op.drop_column('face_embeddings', 'face_angle')
    op.drop_column('face_embeddings', 'face_size')

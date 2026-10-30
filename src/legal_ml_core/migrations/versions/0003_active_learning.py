"""Active learning: rounds, per-dataset config, task score and holdout flag."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "lmc_0003_active_learning"
down_revision = "lmc_0002_llm"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("annotation_tasks", sa.Column("score", sa.Float()))
    op.add_column(
        "annotation_tasks",
        sa.Column("is_holdout", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_index(
        "idx_annotation_tasks_queue",
        "annotation_tasks",
        ["dataset_id", "status", "is_holdout", "score"],
    )

    op.create_table(
        "al_configs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_id", postgresql.UUID(), nullable=False, unique=True),
        sa.Column("model_id", postgresql.UUID()),
        sa.Column("model_type", sa.String(length=100), nullable=False),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("seed_source", sa.String(length=20), nullable=False),
        sa.Column("random_seed", sa.Integer(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False),
        sa.Column("min_per_class", sa.Integer(), nullable=False),
        sa.Column("holdout_percent", sa.Integer(), nullable=False),
        sa.Column("holdout_indices", sa.JSON()),
        sa.Column("file_label_counts", sa.JSON()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="SET NULL"),
    )

    op.create_table(
        "al_rounds",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_id", postgresql.UUID(), nullable=False),
        sa.Column("number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("strategy", sa.String(length=50), nullable=False),
        sa.Column("model_id", postgresql.UUID()),
        sa.Column("job_id", postgresql.UUID()),
        sa.Column("labels_used", sa.Integer()),
        sa.Column("metrics", sa.JSON()),
        sa.Column("error", sa.Text()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.Column("completed_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["job_id"], ["training_jobs.id"], ondelete="SET NULL"),
        sa.UniqueConstraint("dataset_id", "number", name="uq_al_round_number"),
    )


def downgrade():
    op.drop_table("al_rounds")
    op.drop_table("al_configs")
    op.drop_index("idx_annotation_tasks_queue", table_name="annotation_tasks")
    op.drop_column("annotation_tasks", "is_holdout")
    op.drop_column("annotation_tasks", "score")

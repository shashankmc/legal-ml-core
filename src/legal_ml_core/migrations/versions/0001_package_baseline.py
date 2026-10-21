"""Package-owned baseline schema for legal-ml-core tables.

These tables are owned by the package (not the host). Hosts that use
Flask-Migrate include this directory via ``version_locations`` so a single
``flask db upgrade`` covers both host and package tables.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "lmc_0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "datasets",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("owner_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("file_path", sa.String(length=500), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("columns", postgresql.JSONB()),
        sa.Column("task_type", sa.String(length=50), nullable=False),
        sa.Column("project_id", postgresql.UUID()),
        sa.Column("dataset_type", sa.String(length=50)),
        sa.Column("annotation_progress", sa.Integer()),
        sa.Column("text_column", sa.String(length=100)),
        sa.Column("label_column", sa.String(length=100)),
        sa.Column("labels", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_index("idx_datasets_owner_id", "datasets", ["owner_id"])

    op.create_table(
        "annotation_tasks",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("dataset_id", postgresql.UUID(), nullable=False),
        sa.Column("item_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("annotation_label", sa.String(length=1000)),
        sa.Column("status", sa.String(length=50)),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("dataset_id", "item_index", name="annotation_tasks_dataset_id_item_index_key"),
    )
    op.create_index("idx_annotation_tasks_dataset_id", "annotation_tasks", ["dataset_id"])

    op.create_table(
        "models",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("owner_id", sa.String(length=255), nullable=False),
        sa.Column("dataset_id", postgresql.UUID(), nullable=False),
        sa.Column("project_id", postgresql.UUID()),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("model_type", sa.String(length=100), nullable=False),
        sa.Column("parameters", postgresql.JSONB()),
        sa.Column("status", sa.String(length=50)),
        sa.Column("file_path", sa.String(length=500)),
        sa.Column("metrics", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_models_owner_id", "models", ["owner_id"])
    op.create_index("idx_models_dataset_id", "models", ["dataset_id"])

    op.create_table(
        "training_jobs",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("owner_id", sa.String(length=255), nullable=False),
        sa.Column("model_id", postgresql.UUID(), nullable=False),
        sa.Column("status", sa.String(length=50)),
        sa.Column("progress", sa.Integer()),
        sa.Column("metrics", postgresql.JSONB()),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime()),
        sa.Column("completed_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["model_id"], ["models.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_training_jobs_owner_id", "training_jobs", ["owner_id"])
    op.create_index("idx_training_jobs_model_id", "training_jobs", ["model_id"])

    op.create_table(
        "training_metric_points",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("training_job_id", postgresql.UUID(), nullable=False),
        sa.Column("step", sa.Integer(), nullable=False),
        sa.Column("epoch", sa.Float()),
        sa.Column("metric_name", sa.String(length=100), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["training_job_id"], ["training_jobs.id"], ondelete="CASCADE"),
    )
    op.create_index(
        "ix_training_metric_points_job_metric_step",
        "training_metric_points",
        ["training_job_id", "metric_name", "step"],
    )

    op.create_table(
        "model_comparisons",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("owner_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("model_ids", postgresql.ARRAY(postgresql.UUID()), nullable=False),
        sa.Column("comparison_metrics", postgresql.JSONB()),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("idx_model_comparisons_owner_id", "model_comparisons", ["owner_id"])

    op.create_table(
        "ml_projects",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("owner_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("idx_ml_projects_owner_id", "ml_projects", ["owner_id"])


def downgrade():
    for table in (
        "ml_projects",
        "model_comparisons",
        "training_metric_points",
        "training_jobs",
        "models",
        "annotation_tasks",
        "datasets",
    ):
        op.drop_table(table)

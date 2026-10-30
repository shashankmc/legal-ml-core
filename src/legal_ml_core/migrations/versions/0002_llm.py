"""LLM assistance: credentials, acknowledgements, prompts, suggestions, usage."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "lmc_0002_llm"
down_revision = "lmc_0001_baseline"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "annotation_tasks", sa.Column("label_source", sa.String(length=20))
    )

    op.create_table(
        "llm_credentials",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("owner_id", sa.String(length=255), nullable=False, unique=True),
        sa.Column("encrypted_key", sa.Text(), nullable=False),
        sa.Column("base_url", sa.String(length=500), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )

    op.create_table(
        "llm_acknowledgements",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("owner_id", sa.String(length=255), nullable=False),
        sa.Column("scope_id", sa.String(length=255), nullable=False),
        sa.Column("destination", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime()),
        sa.UniqueConstraint(
            "owner_id", "scope_id", "destination", name="uq_llm_acknowledgement"
        ),
    )

    op.create_table(
        "llm_prompts",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_id", postgresql.UUID(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("template", sa.Text(), nullable=False),
        sa.Column("definitions", sa.JSON()),
        sa.Column("created_at", sa.DateTime()),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("dataset_id", "version", name="uq_llm_prompt_version"),
    )

    op.create_table(
        "llm_suggestions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("task_id", postgresql.UUID(), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("prompt_version", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=1000), nullable=False),
        sa.Column("rationale", sa.Text()),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("bulk_agreement", sa.JSON()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.ForeignKeyConstraint(
            ["task_id"], ["annotation_tasks.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint(
            "task_id", "model", "prompt_version", name="uq_llm_suggestion"
        ),
    )
    op.create_index("idx_llm_suggestions_task_id", "llm_suggestions", ["task_id"])

    op.create_table(
        "llm_usage",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("owner_id", sa.String(length=255), nullable=False),
        sa.Column("period", sa.String(length=7), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False),
        sa.UniqueConstraint("owner_id", "period", name="uq_llm_usage_period"),
    )


def downgrade():
    op.drop_table("llm_usage")
    op.drop_index("idx_llm_suggestions_task_id", table_name="llm_suggestions")
    op.drop_table("llm_suggestions")
    op.drop_table("llm_prompts")
    op.drop_table("llm_acknowledgements")
    op.drop_table("llm_credentials")
    op.drop_column("annotation_tasks", "label_source")

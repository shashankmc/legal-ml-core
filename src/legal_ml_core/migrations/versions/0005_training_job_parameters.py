"""Each training run records the Model's settings when it starts (Previous runs)."""

from alembic import op
import sqlalchemy as sa


revision = "lmc_0005_training_job_parameters"
down_revision = "lmc_0004_holdout_file_counts"
branch_labels = None
depends_on = None


def upgrade():
    # Nullable with no backfill: older runs list as having no recorded settings.
    op.add_column("training_jobs", sa.Column("parameters", sa.JSON()))


def downgrade():
    op.drop_column("training_jobs", "parameters")

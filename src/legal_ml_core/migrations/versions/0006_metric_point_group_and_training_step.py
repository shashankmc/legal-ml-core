"""Each training metric point records its group and the trainer's own step."""

from alembic import op
import sqlalchemy as sa


revision = "lmc_0006_metric_point_group"
down_revision = "lmc_0005_training_job_parameters"
branch_labels = None
depends_on = None


def upgrade():
    # Nullable with no backfill: the API derives an older point's group from
    # its name, and older points have no training step.
    op.add_column("training_metric_points", sa.Column("metric_group", sa.String(20)))
    op.add_column("training_metric_points", sa.Column("training_step", sa.Integer()))


def downgrade():
    op.drop_column("training_metric_points", "training_step")
    op.drop_column("training_metric_points", "metric_group")

"""Per class, the labels a file gave the holdout: the state cannot read the file."""

from alembic import op
import sqlalchemy as sa


revision = "lmc_0004_holdout_file_counts"
down_revision = "lmc_0003_active_learning"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("al_configs", sa.Column("holdout_file_counts", sa.JSON()))


def downgrade():
    op.drop_column("al_configs", "holdout_file_counts")

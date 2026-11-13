"""Each annotation task stores the kit's Assignment; its label column widens (BAIGLE #137).

``annotation_label`` stays the label training reads, derived from the payload on
each save. NER entity lists outgrow 1000 characters, so it becomes Text. No
backfill: a task saved before this has no payload, and one is built from its
label when the task is opened (``annotation_format.assignment_from_task``).
"""

from alembic import op
import sqlalchemy as sa


revision = "lmc_0007_annotation_payload"
down_revision = "lmc_0006_metric_point_group"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("annotation_tasks") as batch:
        batch.add_column(sa.Column("annotation_payload", sa.JSON()))
        batch.alter_column("annotation_label", existing_type=sa.String(1000), type_=sa.Text())


def downgrade():
    # Postgres will not narrow a column over a longer value; cut such labels to fit.
    op.execute(
        "UPDATE annotation_tasks SET annotation_label = substr(annotation_label, 1, 1000) "
        "WHERE length(annotation_label) > 1000"
    )
    with op.batch_alter_table("annotation_tasks") as batch:
        batch.alter_column("annotation_label", existing_type=sa.Text(), type_=sa.String(1000))
        batch.drop_column("annotation_payload")

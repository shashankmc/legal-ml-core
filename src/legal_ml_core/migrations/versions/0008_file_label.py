"""Each annotation task keeps the label the file gave its row (BAIGLE #147).

Every row with text now has a task, and a row the file labels starts labelled
(``label_source = "file"``). ``file_label`` keeps the file's value after a person
changes the label, and holds an NER file's entity types, which are not spans.
Nullable, no backfill: tasks are added for file-labelled rows when a Dataset is
next read (``annotation_queue.ensure_project_tasks``).
"""

from alembic import op
import sqlalchemy as sa


revision = "lmc_0008_file_label"
down_revision = "lmc_0007_annotation_payload"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("annotation_tasks") as batch:
        batch.add_column(sa.Column("file_label", sa.Text()))


def downgrade():
    with op.batch_alter_table("annotation_tasks") as batch:
        batch.drop_column("file_label")

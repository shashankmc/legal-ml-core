"""The kit Assignment column and the wider label ship as a migration, both ways (#137)."""
import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

import legal_ml_core

VERSIONS = Path(legal_ml_core.__file__).parent / "migrations" / "versions"


def _migration():
    path = VERSIONS / "0007_annotation_payload.py"
    spec = importlib.util.spec_from_file_location("payload_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns(connection):
    return {c["name"]: c["type"] for c in sa.inspect(connection).get_columns("annotation_tasks")}


def test_migration_follows_the_metric_point_group():
    module = _migration()
    assert module.revision == "lmc_0007_annotation_payload"
    assert module.down_revision == "lmc_0006_metric_point_group"


def test_migration_adds_the_payload_and_widens_the_label_then_undoes_both(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'tasks.db'}")
    long_label = "x" * 5000
    with engine.begin() as connection:
        connection.execute(sa.text(
            "CREATE TABLE annotation_tasks (id VARCHAR(36) PRIMARY KEY, annotation_label VARCHAR(1000))"
        ))
        connection.execute(sa.text("INSERT INTO annotation_tasks VALUES ('old-task', 'Void')"))
        operations = Operations(MigrationContext.configure(connection))

        with Operations.context(operations.migration_context):
            _migration().upgrade()
        columns = _columns(connection)
        assert "annotation_payload" in columns
        assert isinstance(columns["annotation_label"], sa.Text)
        # A task from before keeps its label and has no payload; it is built on read.
        assert connection.execute(
            sa.text("SELECT annotation_label, annotation_payload FROM annotation_tasks")
        ).all() == [("Void", None)]
        connection.execute(sa.text("UPDATE annotation_tasks SET annotation_label = :l"), {"l": long_label})

        with Operations.context(operations.migration_context):
            _migration().downgrade()
        columns = _columns(connection)
        assert set(columns) == {"id", "annotation_label"}
        assert isinstance(columns["annotation_label"], sa.String)
        assert not isinstance(columns["annotation_label"], sa.Text)
        # Postgres refuses to narrow over a longer value, so downgrade cuts it to fit.
        assert connection.execute(sa.text("SELECT length(annotation_label) FROM annotation_tasks")).scalar() == 1000

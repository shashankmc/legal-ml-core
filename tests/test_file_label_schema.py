"""The file's own label per task ships as a migration, both ways (BAIGLE #147)."""
import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

import legal_ml_core

VERSIONS = Path(legal_ml_core.__file__).parent / "migrations" / "versions"


def _migration():
    spec = importlib.util.spec_from_file_location("file_label_migration", VERSIONS / "0008_file_label.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns(connection):
    return {c["name"] for c in sa.inspect(connection).get_columns("annotation_tasks")}


def test_migration_follows_the_annotation_payload():
    module = _migration()
    assert module.revision == "lmc_0008_file_label"
    assert module.down_revision == "lmc_0007_annotation_payload"


def test_migration_adds_and_removes_the_file_label(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'tasks.db'}")
    with engine.begin() as connection:
        connection.execute(sa.text("CREATE TABLE annotation_tasks (id VARCHAR(36) PRIMARY KEY, annotation_label TEXT)"))
        connection.execute(sa.text("INSERT INTO annotation_tasks VALUES ('old-task', 'Void')"))
        operations = Operations(MigrationContext.configure(connection))

        with Operations.context(operations.migration_context):
            _migration().upgrade()
        assert "file_label" in _columns(connection)
        assert connection.execute(sa.text("SELECT annotation_label, file_label FROM annotation_tasks")).all() == [("Void", None)]

        with Operations.context(operations.migration_context):
            _migration().downgrade()
        assert _columns(connection) == {"id", "annotation_label"}

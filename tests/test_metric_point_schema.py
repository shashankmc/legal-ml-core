"""The point group and training step ship as a migration, both ways (BAIGLE #88)."""
import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

import legal_ml_core

VERSIONS = Path(legal_ml_core.__file__).parent / "migrations" / "versions"


def _migration():
    path = VERSIONS / "0006_metric_point_group_and_training_step.py"
    spec = importlib.util.spec_from_file_location("point_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns(connection):
    return {column["name"] for column in sa.inspect(connection).get_columns("training_metric_points")}


def test_migration_follows_the_run_settings_snapshot():
    module = _migration()

    assert module.revision == "lmc_0006_metric_point_group"
    assert module.down_revision == "lmc_0005_training_job_parameters"


def test_migration_adds_and_removes_the_group_and_training_step_columns(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'points.db'}")
    with engine.begin() as connection:
        connection.execute(sa.text(
            "CREATE TABLE training_metric_points (id VARCHAR(36) PRIMARY KEY, metric_name VARCHAR(100))"
        ))
        connection.execute(sa.text("INSERT INTO training_metric_points VALUES ('old-point', 'loss')"))
        operations = Operations(MigrationContext.configure(connection))

        with Operations.context(operations.migration_context):
            _migration().upgrade()
        assert {"metric_group", "training_step"} <= _columns(connection)
        # A point from before the migration keeps both empty; the API derives its group.
        rows = connection.execute(
            sa.text("SELECT id, metric_group, training_step FROM training_metric_points")
        ).all()
        assert rows == [("old-point", None, None)]

        with Operations.context(operations.migration_context):
            _migration().downgrade()
        assert _columns(connection) == {"id", "metric_name"}

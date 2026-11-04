"""The run settings snapshot ships as a migration, both ways (BAIGLE #83)."""
import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations

import legal_ml_core

VERSIONS = Path(legal_ml_core.__file__).parent / "migrations" / "versions"


def _migration():
    path = VERSIONS / "0005_training_job_parameters.py"
    spec = importlib.util.spec_from_file_location("run_migration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _columns(connection):
    return {column["name"] for column in sa.inspect(connection).get_columns("training_jobs")}


def test_migration_follows_the_holdout_file_counts():
    module = _migration()

    assert module.revision == "lmc_0005_training_job_parameters"
    assert module.down_revision == "lmc_0004_holdout_file_counts"


def test_migration_adds_and_removes_the_parameters_column(tmp_path):
    engine = sa.create_engine(f"sqlite:///{tmp_path / 'runs.db'}")
    with engine.begin() as connection:
        connection.execute(sa.text("CREATE TABLE training_jobs (id VARCHAR(36) PRIMARY KEY, metrics JSON)"))
        connection.execute(sa.text("INSERT INTO training_jobs (id, metrics) VALUES ('old-run', '{}')"))
        operations = Operations(MigrationContext.configure(connection))

        with Operations.context(operations.migration_context):
            _migration().upgrade()
        assert "parameters" in _columns(connection)
        # A run from before the migration keeps no settings, and is not lost.
        rows = connection.execute(sa.text("SELECT id, parameters FROM training_jobs")).all()
        assert rows == [("old-run", None)]

        with Operations.context(operations.migration_context):
            _migration().downgrade()
        assert _columns(connection) == {"id", "metrics"}
        assert connection.execute(sa.text("SELECT id FROM training_jobs")).scalars().all() == ["old-run"]

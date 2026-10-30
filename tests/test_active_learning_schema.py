"""Active learning's tables ship as a migration that matches the models."""
import importlib.util
import io
import re
from pathlib import Path

from alembic.migration import MigrationContext
from alembic.operations import Operations

import legal_ml_core
from legal_ml_core import ALConfig, ALRound, AnnotationTask, Dataset, db
from tests.conftest import TEST_USER_ID

VERSIONS = Path(legal_ml_core.__file__).parent / "migrations" / "versions"


def _migration():
    spec = importlib.util.spec_from_file_location("al_migration", VERSIONS / "0003_active_learning.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _upgrade_sql():
    output = io.StringIO()
    context = MigrationContext.configure(
        dialect_name="postgresql", opts={"as_sql": True, "output_buffer": output}
    )
    with Operations.context(context):
        _migration().upgrade()
    return output.getvalue()


def test_migration_follows_the_llm_tables():
    module = _migration()

    assert module.revision == "lmc_0003_active_learning"
    assert module.down_revision == "lmc_0002_llm"


def test_migration_adds_the_score_and_holdout_flag_to_tasks():
    sql = _upgrade_sql()

    assert "ALTER TABLE annotation_tasks ADD COLUMN score FLOAT" in sql
    assert "ALTER TABLE annotation_tasks ADD COLUMN is_holdout BOOLEAN" in sql


def test_migration_creates_every_mapped_column():
    sql = _upgrade_sql()

    for model in (ALConfig, ALRound):
        table = sql.split(f"CREATE TABLE {model.__tablename__} (")[1].split(";")[0]
        for column in model.__table__.columns:
            assert re.search(rf"\b{column.name} [A-Z]", table), f"{model.__tablename__}.{column.name}"


def test_tasks_start_outside_the_holdout_with_no_score(app):
    dataset = Dataset(owner_id=TEST_USER_ID, name="D", file_path="x.csv", row_count=1, task_type="classification")
    db.session.add(dataset)
    db.session.flush()
    task = AnnotationTask(dataset_id=dataset.id, item_index=0, text="a")
    db.session.add(task)
    db.session.commit()

    assert task.score is None
    assert task.is_holdout is False


def test_deleting_a_dataset_removes_its_rounds_and_config(app):
    dataset = Dataset(owner_id=TEST_USER_ID, name="D", file_path="x.csv", row_count=1, task_type="classification")
    db.session.add(dataset)
    db.session.flush()
    db.session.add(ALConfig(dataset_id=dataset.id, strategy="prediction_entropy"))
    db.session.add(ALRound(dataset_id=dataset.id, number=0, strategy="random"))
    db.session.commit()

    db.session.delete(dataset)
    db.session.commit()

    assert ALConfig.query.count() == 0
    assert ALRound.query.count() == 0

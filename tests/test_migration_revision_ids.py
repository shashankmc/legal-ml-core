"""Every package revision ID fits Alembic's version table.

Alembic records the applied revision in `alembic_version.version_num`, a
VARCHAR(32). SQLite ignores the length, so a longer ID passes every SQLite
test and only fails when a host upgrades a Postgres database.
"""
import importlib.util
from pathlib import Path

import pytest

import legal_ml_core

VERSIONS = Path(legal_ml_core.__file__).parent / "migrations" / "versions"
ALEMBIC_VERSION_NUM_LENGTH = 32


def _revisions():
    for path in sorted(VERSIONS.glob("[0-9]*.py")):
        spec = importlib.util.spec_from_file_location(f"revision_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield path.name, module.revision


@pytest.mark.parametrize("filename, revision", list(_revisions()))
def test_revision_id_fits_the_alembic_version_table(filename, revision):
    assert len(revision) <= ALEMBIC_VERSION_NUM_LENGTH, (
        f"{filename}: revision {revision!r} is {len(revision)} characters; "
        f"alembic_version.version_num holds {ALEMBIC_VERSION_NUM_LENGTH}"
    )

import os

import fakeredis
import pytest
from flask import Flask
from flask_jwt_extended import JWTManager, create_access_token
from sqlalchemy import Column, String, Table

from legal_ml_core import db, init_ml_feature


def _ensure_host_stub_tables():
    """`users`/`projects` belong to the host app, not this package, but
    SQLAlchemy's DDL emission needs *some* table object to resolve the
    `ForeignKey("users.id")`/`ForeignKey("projects.id")` columns on this
    package's own models against — `db.create_all()` fails with
    `NoReferencedTableError` otherwise. A real host app supplies the real
    tables; tests supply minimal stand-ins with just the referenced columns.
    """
    if "users" not in db.metadata.tables:
        Table("users", db.metadata, Column("id", String(36), primary_key=True))
    if "projects" not in db.metadata.tables:
        Table(
            "projects",
            db.metadata,
            Column("id", String(36), primary_key=True),
            Column("user_id", String(36)),
        )


@pytest.fixture
def fake_redis():
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def app(fake_redis, tmp_path):
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        JWT_SECRET_KEY="test-secret",
        JWT_TOKEN_LOCATION=["headers"],
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        MODEL_FOLDER=str(tmp_path / "models"),
    )
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

    db.init_app(app)
    JWTManager(app)
    init_ml_feature(app, fake_redis, app.config["MODEL_FOLDER"], db=db)
    _ensure_host_stub_tables()

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


TEST_USER_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def auth_headers(app):
    with app.app_context():
        token = create_access_token(identity=TEST_USER_ID)
    return {"Authorization": f"Bearer {token}"}

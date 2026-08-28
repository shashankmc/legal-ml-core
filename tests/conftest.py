import os

import fakeredis
import pytest
from flask import Flask
from flask_jwt_extended import JWTManager, create_access_token

from legal_ml_core import db, init_ml_feature
from legal_ml_core.db_utils import ensure_external_fk_stub_tables


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
    ensure_external_fk_stub_tables(db)

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

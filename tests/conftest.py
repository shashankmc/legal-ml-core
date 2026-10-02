import json
import os

import fakeredis
import pytest
from flask import Flask

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
        AUTH_ISSUER="https://keycloak.test/realms/biss",
        AUTH_AUDIENCE="baigle",
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        MODEL_FOLDER=str(tmp_path / "models"),
    )
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)

    db.init_app(app)
    init_ml_feature(app, fake_redis, app.config["MODEL_FOLDER"], db=db)
    ensure_external_fk_stub_tables(db)

    # These route tests exercise ML endpoints, not the host's identity
    # provisioning; resolve every token to the fixed test user id.
    app.config["IDENTITY_RESOLVER"] = lambda payload: {"id": payload.get("sub")}

    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


TEST_USER_ID = "11111111-1111-1111-1111-111111111111"

# Test tokens use the same `test-token::<json claims>` convention as the host
# backend's suite; `verify_bearer_token` is patched below to read the claims
# directly, so no real Keycloak realm is needed.
_TEST_CLAIMS = {
    "sub": TEST_USER_ID,
    "iss": "https://keycloak.test/realms/biss",
    "aud": "baigle",
    "preferred_username": "tester",
    "email": "tester@example.com",
}


@pytest.fixture(autouse=True)
def _fake_oidc(monkeypatch):
    import legal_ml_core.auth as auth

    def verify(token):
        if not token.startswith("test-token::"):
            raise auth.AuthError("Invalid access token")
        return json.loads(token.split("::", 1)[1])

    monkeypatch.setattr(auth, "verify_bearer_token", verify)


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer test-token::{json.dumps(_TEST_CLAIMS)}"}

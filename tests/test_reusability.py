"""Proof that the package mounts into a foreign host with no local user model.

These tests deliberately do NOT import any host `User`/`Project` classes and do
NOT install an `IDENTITY_RESOLVER`. They exercise the two reuse paths named in
docs/legal-ml-core-reusability-plan.md:

- a Keycloak host whose owner id is the token `sub`, and
- an ML-only single-tenant deployment with no user management at all.
"""

import json
import os

import fakeredis
import pytest
from flask import Flask

from legal_ml_core import (
    KeycloakOwnerRef,
    NoOwnerRef,
    db,
    init_ml_feature,
)


def _base_app(tmp_path):
    app = Flask(__name__)
    app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        AUTH_ISSUER="https://keycloak.test/realms/other-app",
        AUTH_AUDIENCE="other-app",
        UPLOAD_FOLDER=str(tmp_path / "uploads"),
        MODEL_FOLDER=str(tmp_path / "models"),
    )
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
    return app


def _install_oidc_stub(monkeypatch):
    """Patch bearer verification to read `test-token::<json>` claims.

    Mirrors the existing suite, so no real realm is needed; crucially, no
    IDENTITY_RESOLVER is configured here.
    """
    import legal_ml_core.auth as auth

    def verify(token):
        if not token.startswith("test-token::"):
            raise auth.AuthError("Invalid access token")
        return json.loads(token.split("::", 1)[1])

    monkeypatch.setattr(auth, "verify_bearer_token", verify)


def _headers(sub):
    claims = {
        "sub": sub,
        "iss": "https://keycloak.test/realms/other-app",
        "aud": "other-app",
        "email": "someone@other-app.test",
    }
    return {"Authorization": f"Bearer test-token::{json.dumps(claims)}"}


def test_no_host_models_required(monkeypatch, tmp_path):
    """The package's tables exist with no host `users`/`projects` model defined."""
    _install_oidc_stub(monkeypatch)
    app = _base_app(tmp_path)
    db.init_app(app)
    init_ml_feature(app, fakeredis.FakeStrictRedis(decode_responses=True), app.config["MODEL_FOLDER"], db=db)

    with app.app_context():
        # create_all succeeds without any host model registered — the package
        # has no foreign keys into host tables any more.
        db.create_all()
        names = set(db.metadata.tables)
        assert "datasets" in names and "models" in names
        assert "users" not in names  # no host identity table
        assert "projects" not in names
        db.drop_all()


def test_keycloak_sub_is_the_owner(monkeypatch, tmp_path):
    """A Keycloak host scopes rows by the token `sub`; no local user row exists."""
    _install_oidc_stub(monkeypatch)
    app = _base_app(tmp_path)
    app.config["OWNER_REF"] = KeycloakOwnerRef()  # explicit, though this is the default
    db.init_app(app)
    init_ml_feature(app, fakeredis.FakeStrictRedis(decode_responses=True), app.config["MODEL_FOLDER"], db=db)

    with app.app_context():
        db.create_all()

    client = app.test_client()
    sub = "kc-user-123"
    # A route that requires auth and returns the caller's datasets.
    resp = client.get("/api/datasets/", headers=_headers(sub))
    assert resp.status_code == 200
    assert resp.get_json()["datasets"] == []

    with app.app_context():
        from legal_ml_core import Dataset

        db.session.add(
            Dataset(
                owner_id=sub,
                name="kc dataset",
                file_path="/tmp/x.csv",
                row_count=1,
                columns=[],
                task_type="classification",
            )
        )
        db.session.commit()

    resp = client.get("/api/datasets/", headers=_headers(sub))
    assert [d["owner_id"] for d in resp.get_json()["datasets"]] == [sub]

    # A different subject sees nothing — scoping is per-KC-sub, not global.
    resp = client.get("/api/datasets/", headers=_headers("someone-else"))
    assert resp.get_json()["datasets"] == []

    with app.app_context():
        db.drop_all()


def test_auth_none_requires_explicit_opt_in(monkeypatch, tmp_path):
    """AUTH_MODE='none' refuses to serve until LEGAL_ML_ALLOW_NO_AUTH is set."""
    app = _base_app(tmp_path)
    app.config["AUTH_MODE"] = "none"
    app.config["OWNER_REF"] = NoOwnerRef()
    db.init_app(app)
    init_ml_feature(app, fakeredis.FakeStrictRedis(decode_responses=True), app.config["MODEL_FOLDER"], db=db)
    with app.app_context():
        db.create_all()

    client = app.test_client()
    # No token, and the opt-in flag is absent -> refused.
    resp = client.get("/api/datasets/")
    assert resp.status_code == 500
    assert "LEGAL_ML_ALLOW_NO_AUTH" in resp.get_json()["error"]

    # With the explicit acknowledgement, requests succeed without a token.
    app.config["LEGAL_ML_ALLOW_NO_AUTH"] = True
    resp = client.get("/api/datasets/")
    assert resp.status_code == 200

    with app.app_context():
        db.drop_all()


def test_no_owner_ref_single_tenant(monkeypatch, tmp_path):
    """ML-only deployment: no users, one implicit owner, rows still scoped."""
    _install_oidc_stub(monkeypatch)
    app = _base_app(tmp_path)
    app.config["OWNER_REF"] = NoOwnerRef()
    db.init_app(app)
    init_ml_feature(app, fakeredis.FakeStrictRedis(decode_responses=True), app.config["MODEL_FOLDER"], db=db)

    with app.app_context():
        db.create_all()

    client = app.test_client()
    resp = client.get("/api/datasets/", headers=_headers("ignored"))
    assert resp.status_code == 200

    with app.app_context():
        from legal_ml_core import Dataset

        db.session.add(
            Dataset(
                owner_id=NoOwnerRef().owner_id,
                name="single tenant",
                file_path="/tmp/y.csv",
                row_count=1,
                columns=[],
                task_type="classification",
            )
        )
        db.session.commit()

    resp = client.get("/api/datasets/", headers=_headers("a-different-sub"))
    assert len(resp.get_json()["datasets"]) == 1

    with app.app_context():
        db.drop_all()


def test_package_table_project_scope(monkeypatch, tmp_path):
    """PROJECT_SCOPE="package_table": ml_projects validates references."""
    _install_oidc_stub(monkeypatch)
    app = _base_app(tmp_path)
    app.config["OWNER_REF"] = NoOwnerRef()
    app.config["PROJECT_SCOPE"] = "package_table"
    db.init_app(app)
    init_ml_feature(
        app, fakeredis.FakeStrictRedis(decode_responses=True),
        app.config["MODEL_FOLDER"], db=db,
    )

    from legal_ml_core.db_utils import project_belongs_to_owner
    from legal_ml_core import MLProject

    with app.app_context():
        db.create_all()
        assert "ml_projects" in db.metadata.tables
        p = MLProject(owner_id="owner-1", name="Alpha")
        db.session.add(p)
        db.session.commit()

        assert project_belongs_to_owner(db.session, p.id, "owner-1") is True
        assert project_belongs_to_owner(db.session, p.id, "owner-2") is False
        assert project_belongs_to_owner(db.session, "no-such-id", "owner-1") is False
        assert project_belongs_to_owner(db.session, "", "owner-1") is True
        db.drop_all()

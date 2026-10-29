"""AUTH_MODE="platform": a host signs a short-lived HS256 token per request.

The token `sub` is the owner, so each platform user's ML data is isolated
without Keycloak. These tests sign real tokens with PyJWT; the suite-wide
Keycloak stub (conftest `_fake_oidc`) is irrelevant to this mode.
"""

import time

import jwt

import legal_ml_core.auth as auth
from legal_ml_core import Dataset, NoOwnerRef, db

# Captured at import, before conftest's autouse stub replaces it per test.
_real_verify_bearer_token = auth.verify_bearer_token

SECRET = "platform-test-secret-0123456789abcdef"


def _token(sub="42", lifetime=30, secret=SECRET, algorithm="HS256", **claims):
    now = int(time.time())
    payload = {"sub": sub, "iat": now, "exp": now + lifetime, **claims}
    return jwt.encode(payload, secret, algorithm=algorithm)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _platform(app, secret=SECRET):
    app.config["AUTH_MODE"] = "platform"
    app.config["LEGAL_ML_PLATFORM_SECRET"] = secret


def _add_dataset(owner_id, name):
    db.session.add(
        Dataset(owner_id=owner_id, name=name, file_path="/tmp/x.csv",
                row_count=1, columns=[], task_type="classification")
    )
    db.session.commit()


def test_platform_token_subject_is_the_owner(app, client):
    _platform(app)
    _add_dataset("42", "mine")

    resp = client.get("/api/datasets/", headers=_auth(_token(sub="42")))

    assert resp.status_code == 200
    assert [d["name"] for d in resp.get_json()["datasets"]] == ["mine"]


def _rsa_private_pem():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def test_platform_rejects_rs256_token(app, client):
    _platform(app)
    token = _token(secret=_rsa_private_pem(), algorithm="RS256")

    resp = client.get("/api/datasets/", headers=_auth(token))

    assert resp.status_code == 401


def test_platform_rejects_token_signed_with_another_secret(app, client):
    _platform(app)

    resp = client.get("/api/datasets/", headers=_auth(_token(secret="x" * 40)))

    assert resp.status_code == 401


def test_platform_rejects_missing_token(app, client):
    _platform(app)

    resp = client.get("/api/datasets/")

    assert resp.status_code == 401


def test_platform_rejects_token_valid_for_more_than_60s(app, client):
    _platform(app)

    ok = client.get("/api/datasets/", headers=_auth(_token(lifetime=60)))
    too_long = client.get("/api/datasets/", headers=_auth(_token(lifetime=61)))

    assert ok.status_code == 200
    assert too_long.status_code == 401


def test_platform_requires_expiry(app, client):
    _platform(app)
    token = jwt.encode({"sub": "42", "iat": int(time.time())}, SECRET, algorithm="HS256")

    resp = client.get("/api/datasets/", headers=_auth(token))

    assert resp.status_code == 401


def test_platform_requires_issued_at(app, client):
    _platform(app)
    token = jwt.encode({"sub": "42", "exp": int(time.time()) + 30}, SECRET, algorithm="HS256")

    resp = client.get("/api/datasets/", headers=_auth(token))

    assert resp.status_code == 401


def test_platform_rejects_expired_token(app, client):
    _platform(app)
    now = int(time.time())
    token = jwt.encode({"sub": "42", "iat": now - 40, "exp": now - 10}, SECRET, algorithm="HS256")

    resp = client.get("/api/datasets/", headers=_auth(token))

    assert resp.status_code == 401


def test_platform_rejects_non_string_subject(app, client):
    _platform(app)
    now = int(time.time())
    token = jwt.encode({"sub": 42, "iat": now, "exp": now + 30}, SECRET, algorithm="HS256")

    resp = client.get("/api/datasets/", headers=_auth(token))

    assert resp.status_code == 401


def test_platform_secret_file_is_read_at_request_time(app, client, tmp_path):
    secret_file = tmp_path / "ml-secret"
    app.config["AUTH_MODE"] = "platform"
    app.config["LEGAL_ML_PLATFORM_SECRET_FILE"] = str(secret_file)

    # The host may generate the secret after this service started.
    secret_file.write_text("first-secret-0123456789abcdef0123\n")
    first = client.get("/api/datasets/", headers=_auth(_token(secret="first-secret-0123456789abcdef0123")))

    secret_file.write_text("rotated-secret-0123456789abcdef01")
    stale = client.get("/api/datasets/", headers=_auth(_token(secret="first-secret-0123456789abcdef0123")))
    rotated = client.get("/api/datasets/", headers=_auth(_token(secret="rotated-secret-0123456789abcdef01")))

    assert (first.status_code, stale.status_code, rotated.status_code) == (200, 401, 200)


def test_platform_missing_secret_file_is_503_naming_the_path(app, client, tmp_path, caplog):
    missing = tmp_path / "nowhere" / "ml-secret"
    app.config["AUTH_MODE"] = "platform"
    app.config["LEGAL_ML_PLATFORM_SECRET_FILE"] = str(missing)

    resp = client.get("/api/datasets/", headers=_auth(_token()))

    assert resp.status_code == 503
    assert str(missing) in resp.get_json()["error"]
    assert f"platform secret not found at {missing}" in caplog.text


def test_platform_empty_secret_file_is_503(app, client, tmp_path):
    secret_file = tmp_path / "ml-secret"
    secret_file.write_text("  \n")
    app.config["AUTH_MODE"] = "platform"
    app.config["LEGAL_ML_PLATFORM_SECRET_FILE"] = str(secret_file)

    resp = client.get("/api/datasets/", headers=_auth(_token()))

    assert resp.status_code == 503
    assert str(secret_file) in resp.get_json()["error"]


def test_platform_unconfigured_secret_is_503_naming_the_settings(app, client):
    app.config["AUTH_MODE"] = "platform"

    resp = client.get("/api/datasets/", headers=_auth(_token()))

    assert resp.status_code == 503
    error = resp.get_json()["error"]
    assert "LEGAL_ML_PLATFORM_SECRET" in error and "LEGAL_ML_PLATFORM_SECRET_FILE" in error


def test_platform_secret_value_wins_over_file(app, client, tmp_path):
    secret_file = tmp_path / "ml-secret"
    secret_file.write_text("file-secret-0123456789abcdef012345")
    _platform(app)
    app.config["LEGAL_ML_PLATFORM_SECRET_FILE"] = str(secret_file)

    resp = client.get("/api/datasets/", headers=_auth(_token()))

    assert resp.status_code == 200


def test_keycloak_mode_rejects_hs256_token(app, client, monkeypatch):
    """Even if key lookup hands back the HMAC secret, keycloak never accepts HS256."""

    class _KeyIsTheSecret:
        def get_signing_key_from_jwt(self, token):
            return type("Key", (), {"key": SECRET})()

    monkeypatch.setattr(auth, "verify_bearer_token", _real_verify_bearer_token)
    monkeypatch.setattr(auth, "_jwks_client", lambda uri: _KeyIsTheSecret())
    token = _token(iss="https://keycloak.test/realms/biss", aud="baigle")

    resp = client.get("/api/datasets/", headers=_auth(token))

    assert resp.status_code == 401


def _seed_owner(owner_id):
    from legal_ml_core import Model, TrainingJob

    ds = Dataset(owner_id=owner_id, name=f"ds-{owner_id}", file_path="/tmp/x.csv",
                 row_count=1, columns=[], task_type="classification")
    db.session.add(ds)
    db.session.flush()
    model = Model(owner_id=owner_id, dataset_id=ds.id, name=f"m-{owner_id}", model_type="svm")
    db.session.add(model)
    db.session.flush()
    job = TrainingJob(owner_id=owner_id, model_id=model.id)
    db.session.add(job)
    db.session.commit()
    return ds.id, model.id, job.id


def test_two_platform_owners_never_see_each_others_data(app, client):
    _platform(app)
    ds7, model7, job7 = _seed_owner("7")
    ds8, model8, job8 = _seed_owner("8")

    def visible(sub):
        h = _auth(_token(sub=sub))
        return (
            {d["id"] for d in client.get("/api/datasets/", headers=h).get_json()["datasets"]},
            {m["id"] for m in client.get("/api/models/", headers=h).get_json()["models"]},
            {j["id"] for j in client.get("/api/training/jobs", headers=h).get_json()["jobs"]},
        )

    assert visible("7") == ({ds7}, {model7}, {job7})
    assert visible("8") == ({ds8}, {model8}, {job8})
    assert client.get(f"/api/models/{model8}", headers=_auth(_token(sub="7"))).status_code == 404
    assert client.get(f"/api/training/status/{job8}", headers=_auth(_token(sub="7"))).status_code == 404


def test_platform_mode_refuses_a_shared_owner_strategy(app, client):
    """Platform mode never falls back to one shared owner for every student."""
    _platform(app)
    app.config["OWNER_REF"] = NoOwnerRef()

    resp = client.get("/api/datasets/", headers=_auth(_token(sub="7")))

    assert resp.status_code == 500
    assert "OWNER_REF" in resp.get_json()["error"]

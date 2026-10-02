import json

from legal_ml_core import Dataset, Model, db
from tests.conftest import TEST_USER_ID


def _make_dataset(app):
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="D1",
            file_path="/tmp/d1.csv",
            row_count=10,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.commit()
        return dataset.id


def test_get_available_models_catalog(client, auth_headers):
    resp = client.get("/api/models/available", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert "classification" in body["models"]
    assert "svm" in body["models"]["classification"]


def test_get_available_models_filtered_by_task_type(client, auth_headers):
    resp = client.get("/api/models/available?task_type=clustering", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert "kmeans" in body["models"]

    resp = client.get("/api/models/available?task_type=nonsense", headers=auth_headers)
    assert resp.status_code == 400


def test_create_model_validates_parameters(client, auth_headers, app):
    dataset_id = _make_dataset(app)

    resp = client.post(
        "/api/models/",
        json={
            "name": "My SVM",
            "dataset_id": dataset_id,
            "model_type": "svm",
            "parameters": {"kernel": "not-a-real-kernel"},
        },
        headers=auth_headers,
    )
    assert resp.status_code == 400

    resp = client.post(
        "/api/models/",
        json={
            "name": "My SVM",
            "dataset_id": dataset_id,
            "model_type": "svm",
            "parameters": {"kernel": "rbf", "C": 1.0},
        },
        headers=auth_headers,
    )
    assert resp.status_code == 201
    model = resp.get_json()["model"]
    assert model["model_type"] == "svm"
    assert model["status"] == "pending"


def test_train_model_queues_task_and_updates_status(client, auth_headers, app, fake_redis):
    dataset_id = _make_dataset(app)
    resp = client.post(
        "/api/models/",
        json={
            "name": "My SVM",
            "dataset_id": dataset_id,
            "model_type": "svm",
            "parameters": {"kernel": "rbf"},
        },
        headers=auth_headers,
    )
    model_id = resp.get_json()["model"]["id"]

    resp = client.post(f"/api/models/{model_id}/train", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["model"]["status"] == "training"
    assert body["job_id"]

    queued = fake_redis.brpop("training_queue", timeout=1)
    assert queued is not None
    task = json.loads(queued[1])
    assert task["model_id"] == model_id
    assert task["dataset_id"] == dataset_id


def test_predict_returns_504_when_worker_never_responds(client, auth_headers, app, fake_redis, monkeypatch):
    dataset_id = _make_dataset(app)
    with app.app_context():
        model = Model(
            owner_id=TEST_USER_ID,
            dataset_id=dataset_id,
            name="Trained SVM",
            model_type="svm",
            parameters={},
            status="completed",
        )
        db.session.add(model)
        db.session.commit()
        model_id = model.id

    # fakeredis's blpop really blocks for the given timeout (30s for a non-
    # transformer model) when nothing is pushed — force an immediate "nothing
    # arrived" response so this test exercises the timeout-handling branch
    # without an actual multi-second wait.
    monkeypatch.setattr(fake_redis, "blpop", lambda *a, **k: None)

    resp = client.post(
        f"/api/models/{model_id}/predict",
        json={"texts": ["some text"]},
        headers=auth_headers,
    )
    assert resp.status_code == 504


def test_available_models_for_sparse_task_type(client, auth_headers):
    """`ner` is a valid task with no trainable models yet — answer 200, not 400."""
    resp = client.get("/api/models/available?task_type=ner", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["models"] == {}


def test_create_model_rejects_incompatible_model(client, auth_headers, app):
    """A clustering model cannot be created on a classification dataset."""
    dataset_id = _make_dataset(app)
    resp = client.post(
        "/api/models/",
        json={
            "name": "Wrong model",
            "dataset_id": dataset_id,
            "model_type": "kmeans",
            "parameters": {},
        },
        headers=auth_headers,
    )
    assert resp.status_code == 400
    assert "cannot be trained" in resp.get_json()["error"]


def test_update_model_labels_persists_display_names(client, auth_headers, app):
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="topics",
            file_path="/tmp/topics.csv",
            row_count=10,
            columns=[],
            task_type="topic_modeling",
        )
        db.session.add(dataset)
        db.session.commit()
        dataset_id = dataset.id

    create = client.post(
        "/api/models/",
        json={
            "name": "Topic model",
            "dataset_id": dataset_id,
            "model_type": "lda",
            "parameters": {},
        },
        headers=auth_headers,
    )
    model_id = create.get_json()["model"]["id"]

    resp = client.put(
        f"/api/models/{model_id}/labels",
        json={"topics": {"0": "Contracts", "1": ""}},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    display = resp.get_json()["model"]["metrics"]["display_labels"]
    assert display["topics"]["0"] == "Contracts"
    assert "1" not in display["topics"]

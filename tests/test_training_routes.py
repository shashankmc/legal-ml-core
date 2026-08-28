import json

from legal_ml_core import Dataset, Model, TrainingJob, TrainingMetricPoint, db
from tests.conftest import TEST_USER_ID


def _make_model_and_job(app):
    with app.app_context():
        dataset = Dataset(
            user_id=TEST_USER_ID,
            name="D1",
            file_path="/tmp/d1.csv",
            row_count=10,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.flush()
        model = Model(
            user_id=TEST_USER_ID,
            dataset_id=dataset.id,
            name="M1",
            model_type="svm",
            parameters={},
            status="training",
        )
        db.session.add(model)
        db.session.flush()
        job = TrainingJob(user_id=TEST_USER_ID, model_id=model.id, status="running")
        db.session.add(job)
        db.session.commit()
        return model.id, job.id


def test_start_training_queues_task(client, auth_headers, app, fake_redis):
    with app.app_context():
        dataset = Dataset(
            user_id=TEST_USER_ID,
            name="D1",
            file_path="/tmp/d1.csv",
            row_count=10,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.flush()
        model = Model(
            user_id=TEST_USER_ID,
            dataset_id=dataset.id,
            name="M1",
            model_type="svm",
            parameters={},
            status="pending",
        )
        db.session.add(model)
        db.session.commit()
        model_id = model.id

    resp = client.post("/api/training/start", json={"model_id": model_id}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["job_id"]

    queued = fake_redis.brpop("training_queue", timeout=1)
    assert queued is not None
    task = json.loads(queued[1])
    assert task["model_id"] == model_id


def test_get_training_status(client, auth_headers, app):
    model_id, job_id = _make_model_and_job(app)

    resp = client.get(f"/api/training/status/{job_id}", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["job"]["id"] == job_id
    assert body["model"]["id"] == model_id


def test_training_metrics_since_step_cursor(client, auth_headers, app):
    _, job_id = _make_model_and_job(app)

    with app.app_context():
        for i in range(3):
            db.session.add(
                TrainingMetricPoint(
                    training_job_id=job_id,
                    step=i,
                    epoch=float(i),
                    metric_name="loss",
                    value=1.0 - i * 0.1,
                )
            )
        db.session.commit()

    resp = client.get(f"/api/training/{job_id}/metrics", headers=auth_headers)
    assert resp.status_code == 200
    points = resp.get_json()["points"]
    assert len(points) == 3
    assert [p["step"] for p in points] == [0, 1, 2]

    resp = client.get(f"/api/training/{job_id}/metrics?since_step=1", headers=auth_headers)
    points = resp.get_json()["points"]
    assert [p["step"] for p in points] == [2]


def test_stop_training(client, auth_headers, app):
    _, job_id = _make_model_and_job(app)

    resp = client.post(f"/api/training/stop/{job_id}", headers=auth_headers)
    assert resp.status_code == 200

    resp = client.get(f"/api/training/status/{job_id}", headers=auth_headers)
    assert resp.get_json()["job"]["status"] == "cancelled"

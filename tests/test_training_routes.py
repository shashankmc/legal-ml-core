import json

from legal_ml_core import Dataset, Model, TrainingJob, TrainingMetricPoint, db
from tests.conftest import TEST_USER_ID


def _make_model_and_job(app):
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
        db.session.flush()
        model = Model(
            owner_id=TEST_USER_ID,
            dataset_id=dataset.id,
            name="M1",
            model_type="svm",
            parameters={},
            status="training",
        )
        db.session.add(model)
        db.session.flush()
        job = TrainingJob(owner_id=TEST_USER_ID, model_id=model.id, status="running")
        db.session.add(job)
        db.session.commit()
        return model.id, job.id


def test_start_training_queues_task(client, auth_headers, app, fake_redis):
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
        db.session.flush()
        model = Model(
            owner_id=TEST_USER_ID,
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


def test_training_metrics_return_the_group_and_the_training_step(client, auth_headers, app):
    _, job_id = _make_model_and_job(app)

    with app.app_context():
        db.session.add(
            TrainingMetricPoint(
                training_job_id=job_id,
                step=0,
                epoch=0.5,
                training_step=10,
                metric_name="loss",
                metric_group="train",
                value=0.7,
            )
        )
        db.session.commit()

    point = client.get(f"/api/training/{job_id}/metrics", headers=auth_headers).get_json()["points"][0]
    assert (point["group"], point["training_step"], point["step"]) == ("train", 10, 0)


def test_points_stored_before_groups_existed_get_a_group_from_their_name(client, auth_headers, app):
    _, job_id = _make_model_and_job(app)

    with app.app_context():
        for i, name in enumerate(["eval_accuracy", "training_done", "loss", "grad_norm"]):
            db.session.add(
                TrainingMetricPoint(training_job_id=job_id, step=i, metric_name=name, value=1.0)
            )
        db.session.commit()

    points = client.get(f"/api/training/{job_id}/metrics", headers=auth_headers).get_json()["points"]
    assert [(p["metric_name"], p["group"], p["training_step"]) for p in points] == [
        ("eval_accuracy", "eval", None),
        ("training_done", "stage", None),
        ("loss", "train", None),
        ("grad_norm", "train", None),
    ]


def test_stop_training(client, auth_headers, app):
    _, job_id = _make_model_and_job(app)

    resp = client.post(f"/api/training/stop/{job_id}", headers=auth_headers)
    assert resp.status_code == 200

    resp = client.get(f"/api/training/status/{job_id}", headers=auth_headers)
    assert resp.get_json()["job"]["status"] == "cancelled"


def test_status_merges_live_redis_progress(client, auth_headers, app, fake_redis):
    """The UI stepper reads progress from Redis, since the DB row only reaches
    100 at the very end."""
    _, job_id = _make_model_and_job(app)
    fake_redis.setex(
        f"job_status_{job_id}",
        3600,
        json.dumps({"status": "running", "progress": 75}),
    )
    resp = client.get(f"/api/training/status/{job_id}", headers=auth_headers)
    assert resp.status_code == 200
    job = resp.get_json()["job"]
    assert job["progress"] == 75
    assert job["status"] == "running"


def _served(client, auth_headers, app, rows):
    """Stores these points, `(metric_name, group, epoch, training_step)`, and
    gives what the API serves for each."""
    _, job_id = _make_model_and_job(app)
    with app.app_context():
        for i, (name, group, epoch, training_step) in enumerate(rows):
            db.session.add(
                TrainingMetricPoint(
                    training_job_id=job_id,
                    step=i,
                    epoch=epoch,
                    training_step=training_step,
                    metric_name=name,
                    metric_group=group,
                    value=1.0,
                )
            )
        db.session.commit()
    return client.get(f"/api/training/{job_id}/metrics", headers=auth_headers).get_json()["points"]


def test_each_point_says_what_its_training_step_counts(client, auth_headers, app):
    points = _served(
        client,
        auth_headers,
        app,
        [
            ("loss", "train", 0.5, 10),  # fine-tuning: the trainer's step
            ("loss", "train", None, 3),  # logistic regression's loss by iteration
            ("eval_loss", "eval", None, 3),
            ("learning_curve_eval_accuracy", "eval", None, 16),  # training-set size
            ("memory_mb", "system", None, None),
            ("training_done", "stage", None, None),
        ],
    )

    assert [p["x_kind"] for p in points] == ["step", "iteration", "iteration", "training_examples", "step", "step"]


def test_a_resource_reading_says_its_unit_and_other_points_have_none(client, auth_headers, app):
    points = _served(
        client,
        auth_headers,
        app,
        [
            ("memory_mb", "system", None, None),
            ("gpu_memory_mb", "system", 0.5, 10),
            ("gpu_use_percent", "system", 0.5, 10),
            ("samples_per_second", "system", 0.5, 10),
            ("seconds_remaining", "system", 0.5, 10),
            ("train_samples_per_second", "train", 1.0, 20),
            ("loss", "train", 0.5, 10),
            ("training_done", "stage", None, None),
        ],
    )

    assert [p["unit"] for p in points] == ["MB", "MB", "%", "samples/s", "s", None, None, None]

"""One training job per run, whichever route starts it (BAIGLE #58)."""
import json
import time
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy import event
from sqlalchemy.orm.attributes import set_committed_value

from legal_ml_core import Dataset, Model, TrainingJob, db
from tests.conftest import TEST_USER_ID

ROUTES = ["/api/training/start", "/api/models/{id}/train"]


class _BrokenQueue:
    """Redis that accepts reads but refuses to queue anything."""

    def __init__(self, real):
        self._real = real

    def lpush(self, *args):
        raise ConnectionError("redis is down")

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def model_id(app):
    return _make_model(app, "pending")


def _make_model(app, status):
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
            status=status,
        )
        db.session.add(model)
        db.session.commit()
        return model.id


def _add_job(app, model_id, status, age_seconds=0):
    with app.app_context():
        job = TrainingJob(
            owner_id=TEST_USER_ID,
            model_id=model_id,
            status=status,
            created_at=datetime.utcnow() - timedelta(seconds=age_seconds),
        )
        db.session.add(job)
        db.session.commit()
        return job.id


def _post(client, headers, route, model_id):
    return client.post(
        route.format(id=model_id), json={"model_id": model_id}, headers=headers
    )


def _jobs(app, model_id):
    with app.app_context():
        return [
            (job.id, job.status)
            for job in TrainingJob.query.filter_by(model_id=model_id).all()
        ]


def _model_status(app, model_id):
    with app.app_context():
        return db.session.get(Model, model_id).status


@pytest.mark.parametrize("route", ROUTES)
def test_a_repeated_start_returns_the_active_job(
    client, auth_headers, app, fake_redis, model_id, route
):
    first = _post(client, auth_headers, route, model_id)
    second = _post(client, auth_headers, route, model_id)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.get_json()["job_id"] == first.get_json()["job_id"]
    assert fake_redis.llen("training_queue") == 1
    assert len(_jobs(app, model_id)) == 1


def test_the_two_routes_share_one_active_job(client, auth_headers, app, fake_redis, model_id):
    first = _post(client, auth_headers, ROUTES[0], model_id)
    second = _post(client, auth_headers, ROUTES[1], model_id)

    assert second.get_json()["job_id"] == first.get_json()["job_id"]
    assert fake_redis.llen("training_queue") == 1


@pytest.mark.parametrize("route", ROUTES)
def test_postgres_uuid_ids_can_be_queued(client, auth_headers, app, fake_redis, model_id, route):
    """Migrated Postgres columns are UUID, so psycopg2 hands back uuid.UUID
    objects though the ORM declares String(36). SQLite never does; make its
    rows behave the same."""

    def as_uuid(target, *_):
        set_committed_value(target, "id", uuid.UUID(target.id))

    for cls in (Dataset, TrainingJob):
        event.listen(cls, "load", as_uuid)
        event.listen(cls, "refresh", as_uuid)
    try:
        resp = _post(client, auth_headers, route, model_id)
    finally:
        for cls in (Dataset, TrainingJob):
            event.remove(cls, "load", as_uuid)
            event.remove(cls, "refresh", as_uuid)

    assert resp.status_code == 200
    task = json.loads(fake_redis.brpop("training_queue", timeout=1)[1])
    assert task["job_id"] == resp.get_json()["job_id"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("before", ["pending", "completed"])
def test_a_failed_start_leaves_no_pending_job(
    client, auth_headers, app, fake_redis, route, before, caplog
):
    model_id = _make_model(app, before)
    app.redis_client = _BrokenQueue(fake_redis)

    resp = _post(client, auth_headers, route, model_id)

    assert resp.status_code == 503
    assert resp.get_json()["error"] == "Training could not start. Try again."
    assert [status for _, status in _jobs(app, model_id)] == ["failed"]
    assert _model_status(app, model_id) == before
    assert "redis is down" in caplog.text


def test_a_failed_start_can_be_retried(client, auth_headers, app, fake_redis, model_id):
    app.redis_client = _BrokenQueue(fake_redis)
    _post(client, auth_headers, ROUTES[0], model_id)
    app.redis_client = fake_redis

    resp = _post(client, auth_headers, ROUTES[0], model_id)

    assert resp.status_code == 200
    assert fake_redis.llen("training_queue") == 1


def test_a_ghost_job_is_failed_when_the_jobs_list_is_read(
    client, auth_headers, app, model_id, fake_redis
):
    """Existing rows from the old 500: pending, never queued, long past pickup."""
    ghost = _add_job(app, model_id, "pending", age_seconds=600)

    jobs = client.get("/api/training/jobs", headers=auth_headers).get_json()["jobs"]

    assert [(j["id"], j["status"]) for j in jobs] == [(ghost, "failed")]
    assert jobs[0]["error_message"] == "Training could not start. Try again."
    assert _model_status(app, model_id) == "pending"


def test_a_ghost_job_does_not_touch_a_model_that_trained_anyway(
    client, auth_headers, app, fake_redis
):
    model_id = _make_model(app, "completed")
    _add_job(app, model_id, "pending", age_seconds=600)

    client.get("/api/training/jobs", headers=auth_headers)

    assert _model_status(app, model_id) == "completed"


def test_a_ghost_job_does_not_fail_a_model_with_a_live_run(
    client, auth_headers, app, fake_redis
):
    model_id = _make_model(app, "training")
    ghost = _add_job(app, model_id, "pending", age_seconds=600)
    running = _add_job(app, model_id, "running")
    fake_redis.set(f"job_heartbeat_{running}", time.time())

    client.get("/api/training/jobs", headers=auth_headers)

    assert dict(_jobs(app, model_id)) == {ghost: "failed", running: "running"}
    assert _model_status(app, model_id) == "training"


def test_a_ghost_job_on_a_stuck_training_model_fails_the_model(
    client, auth_headers, app, fake_redis
):
    model_id = _make_model(app, "training")
    _add_job(app, model_id, "pending", age_seconds=600)

    client.get("/api/training/jobs", headers=auth_headers)

    assert _model_status(app, model_id) == "failed"


def test_a_job_still_in_the_queue_keeps_waiting(client, auth_headers, app, model_id, fake_redis):
    job_id = _add_job(app, model_id, "pending", age_seconds=600)
    fake_redis.lpush("training_queue", json.dumps({"job_id": job_id}))

    client.get("/api/training/jobs", headers=auth_headers)

    assert _jobs(app, model_id) == [(job_id, "pending")]


def test_a_just_created_job_is_not_a_ghost(client, auth_headers, app, model_id, fake_redis):
    """The worker has popped it and has not yet marked it running."""
    job_id = _add_job(app, model_id, "pending")

    client.get("/api/training/jobs", headers=auth_headers)

    assert _jobs(app, model_id) == [(job_id, "pending")]


def test_an_unreadable_queue_fails_no_job(client, auth_headers, app, model_id, fake_redis):
    class NoRead(_BrokenQueue):
        def lrange(self, *args):
            raise ConnectionError("redis is down")

    job_id = _add_job(app, model_id, "pending", age_seconds=600)
    app.redis_client = NoRead(fake_redis)

    client.get("/api/training/jobs", headers=auth_headers)

    assert _jobs(app, model_id) == [(job_id, "pending")]


def test_a_ghost_job_does_not_block_a_new_start(client, auth_headers, app, model_id, fake_redis):
    ghost = _add_job(app, model_id, "pending", age_seconds=600)

    resp = _post(client, auth_headers, ROUTES[1], model_id)

    assert resp.status_code == 200
    assert resp.get_json()["job_id"] != ghost
    assert dict(_jobs(app, model_id))[ghost] == "failed"
    assert fake_redis.llen("training_queue") == 1
    assert _model_status(app, model_id) == "training"

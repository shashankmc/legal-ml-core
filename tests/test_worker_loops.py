"""Worker survives errors and double-clicks (#30; eng Q1/D5, OV-E10/D8,
OV-E11/D9).

The worker loops take injectable `sleep` and `stop` hooks so these run
without real waits: `stop` ends the loop once the test's queue is drained.
"""
import json
import logging

import pytest
import redis as redis_lib

from legal_ml_core import Model, TrainingJob, db
from legal_ml_core.queue_contract import job_status_key
from legal_ml_core.worker import (
    LOOP_ERROR_MESSAGE,
    _explanation_loop,
    _prediction_loop,
    _training_loop,
)
from tests.conftest import TEST_USER_ID
from tests.test_training_cancel import queued_job  # noqa: F401


def _until_drained(redis_client, queue):
    return lambda: redis_client.llen(queue) == 0


def test_malformed_training_task_fails_its_job_and_the_loop_keeps_serving(
    app, fake_redis, queued_job, caplog
):
    caplog.set_level(logging.INFO, logger="legal_ml_core")
    with app.app_context():
        broken = TrainingJob(
            owner_id=TEST_USER_ID, model_id=queued_job["model_id"], status="pending"
        )
        db.session.add(broken)
        db.session.commit()
        broken_id = broken.id

    # BRPOP pops from the right: the malformed task is served first.
    fake_redis.lpush("training_queue", json.dumps(queued_job))
    fake_redis.lpush("training_queue", json.dumps({"job_id": broken_id}))  # no model_id
    fake_redis.lpush("training_queue", "not json at all")

    _training_loop(
        fake_redis,
        app.config["MODEL_FOLDER"],
        app,
        stop=_until_drained(fake_redis, "training_queue"),
        sleep=lambda s: None,
    )

    with app.app_context():
        failed = db.session.get(TrainingJob, broken_id)
        assert failed.status == "failed"
        assert failed.error_message == LOOP_ERROR_MESSAGE == "Training could not start. Try again."
        assert db.session.get(TrainingJob, queued_job["job_id"]).status == "completed"
    live = json.loads(fake_redis.get(job_status_key(broken_id)))
    assert live == {"status": "failed", "error": LOOP_ERROR_MESSAGE}
    loop_logs = [r.getMessage() for r in caplog.records if "KeyError" in r.getMessage()]
    assert any(broken_id in line for line in loop_logs)


class _FlakyRedis:
    """BRPOP fails `down_for` times with a connection error, then works
    (empty queue). Counts calls so `stop` can end the loop after recovery."""

    def __init__(self, down_for):
        self.down_for = down_for
        self.calls = 0

    def brpop(self, queue, timeout=0):
        self.calls += 1
        if self.calls <= self.down_for:
            raise redis_lib.ConnectionError("Error 111 connecting to redis:6379")
        return None


@pytest.mark.parametrize(
    "loop, name",
    [(_training_loop, "training"), (_prediction_loop, "prediction"), (_explanation_loop, "explanation")],
)
def test_unreachable_redis_backs_off_1s_to_30s_and_logs_once_per_change(app, caplog, loop, name):
    caplog.set_level(logging.INFO, logger="legal_ml_core")
    flaky = _FlakyRedis(down_for=8)
    slept = []

    loop(flaky, app.config["MODEL_FOLDER"], app, sleep=slept.append, stop=lambda: flaky.calls >= 10)

    assert slept == [1, 2, 4, 8, 16, 30, 30, 30]
    down = [r for r in caplog.records if "unreachable" in r.getMessage()]
    recovered = [r for r in caplog.records if "reconnected" in r.getMessage()]
    assert len(down) == 1 and name in down[0].getMessage()
    assert len(recovered) == 1 and name in recovered[0].getMessage()


def test_backoff_resets_after_redis_recovers(app):
    class _DownUpDown(_FlakyRedis):
        def brpop(self, queue, timeout=0):
            self.calls += 1
            if self.calls in (1, 2, 4):
                raise redis_lib.ConnectionError("down")
            return None

    flaky = _DownUpDown(down_for=0)
    slept = []
    _training_loop(flaky, app.config["MODEL_FOLDER"], app, sleep=slept.append, stop=lambda: flaky.calls >= 5)
    assert slept == [1, 2, 1]


def test_two_rapid_starts_make_one_job(client, auth_headers, app, fake_redis, queued_job):
    with app.app_context():
        TrainingJob.query.delete()  # the fixture's job; start from a clean model
        db.session.commit()
    model_id = queued_job["model_id"]

    first = client.post("/api/training/start", json={"model_id": model_id}, headers=auth_headers)
    second = client.post("/api/training/start", json={"model_id": model_id}, headers=auth_headers)

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.get_json()["error"] == "This model is already training"
    assert fake_redis.llen("training_queue") == 1
    with app.app_context():
        assert TrainingJob.query.filter_by(model_id=model_id).count() == 1


@pytest.mark.parametrize("status, expected", [("running", 409), ("completed", 200), ("failed", 200), ("cancelled", 200)])
def test_start_is_refused_only_while_a_job_is_active(client, auth_headers, app, queued_job, status, expected):
    with app.app_context():
        job = db.session.get(TrainingJob, queued_job["job_id"])
        job.status = status
        db.session.get(Model, queued_job["model_id"]).status = "pending"  # model status is not the guard
        db.session.commit()

    resp = client.post(
        "/api/training/start", json={"model_id": queued_job["model_id"]}, headers=auth_headers
    )
    assert resp.status_code == expected


def test_redis_down_during_training_fails_the_job_with_a_plain_message(
    app, client, auth_headers, queued_job, caplog
):
    """S3: the worker's Redis is unreachable for the whole run. The job ends
    failed with the retry message, never raw Redis text, and the log names the
    job and its owner."""
    import fakeredis

    from legal_ml_core.compute.tasks import process_training_task

    caplog.set_level(logging.INFO, logger="legal_ml_core")
    server = fakeredis.FakeServer()
    server.connected = False
    down = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    job_id = queued_job["job_id"]

    process_training_task(queued_job, down, app.config["MODEL_FOLDER"])

    job = client.get(f"/api/training/status/{job_id}", headers=auth_headers).get_json()["job"]
    assert job["status"] == "failed"
    assert job["error_message"] == "Training could not start. Try again."
    with app.app_context():
        assert db.session.get(Model, queued_job["model_id"]).status == "failed"
    assert any(
        job_id in r.getMessage() and TEST_USER_ID in r.getMessage() and "Redis" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


def test_unreadable_cancel_flag_falls_back_to_the_db_and_training_continues(
    app, client, auth_headers, fake_redis, queued_job, monkeypatch
):
    from legal_ml_core.compute.tasks import process_training_task

    def down(*a, **k):
        raise redis_lib.ConnectionError("Error 111 connecting to redis:6379")

    monkeypatch.setattr(fake_redis, "exists", down)
    job_id = queued_job["job_id"]

    process_training_task(queued_job, fake_redis, app.config["MODEL_FOLDER"])

    job = client.get(f"/api/training/status/{job_id}", headers=auth_headers).get_json()["job"]
    assert job["status"] == "completed"
    assert job["error_message"] is None


def test_start_locks_the_model_row_before_checking_for_an_active_job(
    client, auth_headers, app, queued_job
):
    """S1: the 409 guard must serialise concurrent starts. SQLite has no row
    locks, so assert the statements as Postgres would run them: the model row
    is read FOR UPDATE, and the active-job check comes after, inside the
    same transaction, so a second start waits for the first to commit and
    then sees its job."""
    from sqlalchemy import event
    from sqlalchemy.dialects import postgresql

    with app.app_context():
        TrainingJob.query.delete()
        db.session.commit()

    statements = []

    def record(state):
        if state.is_select:
            statements.append(str(state.statement.compile(dialect=postgresql.dialect())))

    event.listen(db.session, "do_orm_execute", record)
    try:
        resp = client.post(
            "/api/training/start", json={"model_id": queued_job["model_id"]}, headers=auth_headers
        )
    finally:
        event.remove(db.session, "do_orm_execute", record)

    assert resp.status_code == 200
    model_lock = next(i for i, s in enumerate(statements) if "FROM models" in s and "FOR UPDATE" in s)
    job_check = next(i for i, s in enumerate(statements) if "FROM training_jobs" in s)
    assert model_lock < job_check

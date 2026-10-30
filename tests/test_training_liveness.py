"""Crashed or missing worker shows a clear message, not a spinner (#29; CEO R5,
D10, D12; eng OV-E3).

Timings are shrunk via LivenessTimings so these run in well under a second
each. "Killing the worker" is simulated by stopping the job's heartbeat while
the training call stays blocked -- exactly what process death looks like from
Redis and the API (no more beats, nothing else written).
"""
import json
import logging
import threading
import time

import pytest

from legal_ml_core import Model, TrainingJob, db
from legal_ml_core.compute.tasks import process_training_task
from legal_ml_core.liveness import (
    STALE_JOB_MESSAGE,
    WORKER_DOWN_MESSAGE,
    Heartbeat,
    LivenessTimings,
    start_worker_liveness,
)
from legal_ml_core.queue_contract import (
    WORKER_LIVENESS_KEY,
    job_heartbeat_key,
    job_status_key,
)
from tests.conftest import TEST_USER_ID
from tests.test_training_cancel import _paused_trainer, _status, queued_job  # noqa: F401

FAST = LivenessTimings(heartbeat_interval=0.05, stale_after=0.3, status_ttl=1)


@pytest.fixture
def fast_api(app):
    app.config["LEGAL_ML_HEARTBEAT_STALE_SECONDS"] = FAST.stale_after
    return app


def _wait_until(pred, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return False


def _jobs(client, auth_headers):
    resp = client.get("/api/training/jobs", headers=auth_headers)
    assert resp.status_code == 200
    return {j["id"]: j for j in resp.get_json()["jobs"]}


def test_heartbeat_is_written_while_a_job_runs_and_removed_after(
    app, fake_redis, queued_job
):
    job_id = queued_job["job_id"]
    seen = {}

    def during_fit():
        # Wait for several beats: the key keeps being rewritten.
        seen["first"] = fake_redis.get(job_heartbeat_key(job_id))
        time.sleep(FAST.heartbeat_interval * 4)
        seen["later"] = fake_redis.get(job_heartbeat_key(job_id))
        seen["ttl"] = fake_redis.pttl(job_heartbeat_key(job_id))

    process_training_task(
        queued_job,
        fake_redis,
        app.config["MODEL_FOLDER"],
        trainer_factory=_paused_trainer(on_fit=during_fit),
        liveness=FAST,
    )

    assert seen["first"] is not None
    assert seen["later"] is not None and seen["later"] != seen["first"]
    assert seen["ttl"] > 0  # dead workers' keys expire on their own
    assert fake_redis.get(job_heartbeat_key(job_id)) is None


def test_a_run_longer_than_the_status_lifetime_keeps_its_status(
    app, client, auth_headers, fake_redis, queued_job
):
    """Deterministic: the heartbeat thread never fires on its own (huge
    interval); the test beats explicitly, and stands in for elapsed time by
    cutting the status key's remaining TTL before each beat."""
    job_id = queued_job["job_id"]
    timings = LivenessTimings(heartbeat_interval=3600, stale_after=7200, status_ttl=60)
    lifetime_ms = timings.status_ttl * 1000
    heartbeats = []
    seen = {"ttl_after_beat": []}

    class ManualHeartbeat(Heartbeat):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            heartbeats.append(self)

    def long_fit():
        # Three rounds of "50s of a 60s lifetime have passed": 150s in total,
        # well past the status lifetime, yet each beat restores it.
        for _ in range(3):
            assert fake_redis.pexpire(job_status_key(job_id), lifetime_ms - 50_000)
            heartbeats[0].beat()
            seen["ttl_after_beat"].append(fake_redis.pttl(job_status_key(job_id)))
        seen["live"] = fake_redis.get(job_status_key(job_id))
        seen["job"] = _status(client, auth_headers, job_id)

    process_training_task(
        queued_job,
        fake_redis,
        app.config["MODEL_FOLDER"],
        trainer_factory=_paused_trainer(on_fit=long_fit),
        liveness=timings,
        heartbeat_factory=ManualHeartbeat,
    )

    assert all(ttl > lifetime_ms - 50_000 for ttl in seen["ttl_after_beat"]), seen
    assert json.loads(seen["live"])["progress"] == 25
    assert seen["job"]["status"] == "running"
    assert seen["job"]["progress"] == 25


class _Worker:
    """Runs one training task on a background thread, blocked inside fit until
    released, and exposes its heartbeat so the test can "kill" it."""

    def __init__(self, app, fake_redis, task_info, timings=FAST):
        self.release = threading.Event()
        self.in_fit = threading.Event()
        self.heartbeats = []
        self._redis = fake_redis
        self._job_id = task_info["job_id"]
        worker = self

        class RecordingHeartbeat(Heartbeat):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                worker.heartbeats.append(self)

        def blocking_fit():
            self.in_fit.set()
            assert self.release.wait(10)

        def run():
            with app.app_context():
                process_training_task(
                    task_info,
                    fake_redis,
                    app.config["MODEL_FOLDER"],
                    trainer_factory=_paused_trainer(on_fit=blocking_fit),
                    liveness=timings,
                    heartbeat_factory=RecordingHeartbeat,
                )

        self.thread = threading.Thread(target=run, daemon=True)

    def __enter__(self):
        self.thread.start()
        assert self.in_fit.wait(5)
        return self

    def kill(self):
        # Process death: beats stop and nothing cleans up the last one, so
        # put back the beat that stop() removed.
        key = job_heartbeat_key(self._job_id)
        last_beat = self._redis.get(key)
        for hb in self.heartbeats:
            hb.stop()
        self._redis.set(key, last_beat)

    def __exit__(self, *exc):
        self.release.set()
        self.thread.join(10)


def test_killing_the_worker_mid_job_fails_it_with_a_plain_message(
    fast_api, client, auth_headers, fake_redis, queued_job, caplog
):
    app = fast_api
    job_id = queued_job["job_id"]
    caplog.set_level(logging.INFO, logger="legal_ml_core")

    with _Worker(app, fake_redis, queued_job) as worker:
        # A blocking training call longer than the stale window is NOT stale
        # while the worker lives: the heartbeat runs on its own thread.
        time.sleep(FAST.stale_after * 2)
        assert _status(client, auth_headers, job_id)["status"] == "running"

        worker.kill()
        time.sleep(FAST.stale_after * 1.5)

        job = _status(client, auth_headers, job_id)
        assert job["status"] == "failed"
        assert job["error_message"] == STALE_JOB_MESSAGE
        listed = _jobs(client, auth_headers)[job_id]
        assert listed["status"] == "failed"
        assert listed["error_message"] == STALE_JOB_MESSAGE
        with app.app_context():
            assert Model.query.get(queued_job["model_id"]).status == "failed"

    # A worker that "comes back" (here: the released thread) cannot overwrite it.
    assert _status(client, auth_headers, job_id)["status"] == "failed"
    assert _status(client, auth_headers, job_id)["error_message"] == STALE_JOB_MESSAGE

    stale_logs = [r.getMessage() for r in caplog.records if "-> failed" in r.getMessage()]
    assert any(job_id in m and TEST_USER_ID in m for m in stale_logs), stale_logs


def test_jobs_list_alone_sweeps_a_dead_running_job(
    fast_api, client, auth_headers, fake_redis, queued_job
):
    job_id = queued_job["job_id"]
    with _Worker(fast_api, fake_redis, queued_job) as worker:
        worker.kill()
        time.sleep(FAST.stale_after * 1.5)
        listed = _jobs(client, auth_headers)[job_id]
        assert listed["status"] == "failed"
        assert listed["error_message"] == STALE_JOB_MESSAGE


def test_queued_job_says_waiting_when_no_worker_is_running(
    fast_api, client, auth_headers, queued_job
):
    job_id = queued_job["job_id"]

    job = _status(client, auth_headers, job_id)
    assert job["status"] == "pending"
    assert job["status_message"] == WORKER_DOWN_MESSAGE
    assert _jobs(client, auth_headers)[job_id]["status_message"] == WORKER_DOWN_MESSAGE


def test_queued_job_has_no_waiting_message_while_a_worker_is_alive(
    fast_api, client, auth_headers, fake_redis, queued_job
):
    job_id = queued_job["job_id"]
    fake_redis.lpush("training_queue", json.dumps({"job_id": job_id}))
    liveness = start_worker_liveness(fake_redis, FAST)
    try:
        time.sleep(FAST.stale_after * 2)  # survives past one stale window
        assert _status(client, auth_headers, job_id)["status_message"] is None
        assert _jobs(client, auth_headers)[job_id]["status_message"] is None
    finally:
        liveness.stop()

    time.sleep(FAST.stale_after * 1.5)  # worker gone
    assert _status(client, auth_headers, job_id)["status_message"] == WORKER_DOWN_MESSAGE


def test_run_ml_worker_publishes_liveness(monkeypatch, fake_redis):
    from legal_ml_core import worker

    monkeypatch.setattr(worker, "_connect_redis", lambda url: fake_redis)
    monkeypatch.setattr(worker, "_build_worker_app", lambda url: None)
    monkeypatch.setattr(worker, "_training_loop", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_prediction_loop", lambda *a, **k: None)
    monkeypatch.setattr(worker, "_explanation_loop", lambda *a, **k: None)

    started = []
    real_start = worker.start_worker_liveness
    monkeypatch.setattr(
        worker, "start_worker_liveness", lambda *a: started.append(real_start(*a)) or started[-1]
    )

    worker.run_ml_worker("redis://x", "sqlite://", "/tmp", block=False, liveness=FAST)
    try:
        assert fake_redis.get(WORKER_LIVENESS_KEY) is not None
        first = fake_redis.get(WORKER_LIVENESS_KEY)
        assert _wait_until(lambda: fake_redis.get(WORKER_LIVENESS_KEY) != first)
    finally:
        for beater in started:
            beater.stop()


def test_a_running_job_that_never_beat_is_swept_once_past_the_stale_window(
    fast_api, client, auth_headers, queued_job
):
    """Worker died between pickup and its first beat (or a pre-heartbeat row)."""
    from datetime import datetime, timedelta

    job_id = queued_job["job_id"]
    with fast_api.app_context():
        job = db.session.get(TrainingJob, job_id)
        job.status = "running"
        job.started_at = datetime.utcnow()
        db.session.commit()

    assert _status(client, auth_headers, job_id)["status"] == "running"  # grace

    with fast_api.app_context():
        db.session.get(TrainingJob, job_id).started_at = datetime.utcnow() - timedelta(
            seconds=FAST.stale_after * 2
        )
        db.session.commit()

    job = _status(client, auth_headers, job_id)
    assert job["status"] == "failed"
    assert job["error_message"] == STALE_JOB_MESSAGE


@pytest.mark.parametrize("terminal", ["completed", "cancelled"])
def test_sweep_never_overwrites_a_terminal_job(
    fast_api, client, auth_headers, fake_redis, queued_job, terminal
):
    job_id = queued_job["job_id"]
    with fast_api.app_context():
        db.session.get(TrainingJob, job_id).status = terminal
        db.session.commit()
    # Stale heartbeat left behind, as after a crash.
    fake_redis.set(job_heartbeat_key(job_id), repr(time.time() - 3600))

    assert _status(client, auth_headers, job_id)["status"] == terminal
    assert _jobs(client, auth_headers)[job_id]["status"] == terminal
    assert _status(client, auth_headers, job_id)["error_message"] is None


def test_each_transition_is_logged_with_job_id_and_owner(
    app, client, auth_headers, fake_redis, queued_job, caplog
):
    caplog.set_level(logging.INFO, logger="legal_ml_core")
    job_id = queued_job["job_id"]

    process_training_task(queued_job, fake_redis, app.config["MODEL_FOLDER"], liveness=FAST)

    lines = [r.getMessage() for r in caplog.records if job_id in r.getMessage()]
    assert any("-> running" in m and TEST_USER_ID in m for m in lines), lines
    assert any("-> completed" in m and TEST_USER_ID in m for m in lines), lines


def test_stop_is_logged_with_job_id_and_owner(client, auth_headers, queued_job, caplog):
    caplog.set_level(logging.INFO, logger="legal_ml_core")
    job_id = queued_job["job_id"]

    assert client.post(f"/api/training/stop/{job_id}", headers=auth_headers).status_code == 200

    lines = [r.getMessage() for r in caplog.records if job_id in r.getMessage()]
    assert any("-> cancelled" in m and TEST_USER_ID in m for m in lines), lines

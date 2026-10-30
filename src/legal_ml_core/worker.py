"""Worker entrypoint: consumes the training/prediction/explanation/LLM/scoring Redis queues.

Runs as its own process (typically its own Docker container), sharing no
Python objects with the Flask API process, but importing the exact same
`db`/ORM model classes from this package — the single source of truth the
`legal-ml-core` package exists to provide. Flask-SQLAlchemy's `db.session`
and `Model.query` work outside of a real HTTP request as long as an app
context is pushed, so this module builds a minimal, routeless Flask app
purely to get that context and a configured database connection; nothing
here is actually served over HTTP.

The unrelated learning-platform sandbox (arbitrary code execution / pip
install queues) is NOT part of this package — it stays in the host app's own
worker process, composed alongside `run_ml_worker` there.
"""
import json
import logging
import threading
import time
from datetime import datetime

import redis as redis_lib
from flask import Flask

from .active_learning.scoring import fail_scoring_round, process_scoring_task
from .compute.tasks import process_explanation_task, process_prediction_task, process_training_task
from .db import db, get_db, set_active_db
from .compute.tasks import REDIS_DOWN_ERRORS
from .job_state import ACTIVE_JOB_STATUSES, COULD_NOT_START_MESSAGE, transition
from .liveness import DEFAULT_TIMINGS, start_worker_liveness
from .llm.suggest import fail_llm_job, process_llm_annotation_task
from .queue_contract import (
    EXPLANATION_RESULT_TTL_SECONDS,
    JOB_STATUS_TTL_SECONDS,
    PREDICTION_RESULT_TTL_SECONDS,
    explanation_result_key,
    job_status_key,
    prediction_result_key,
)

logger = logging.getLogger(__name__)

LOOP_ERROR_MESSAGE = COULD_NOT_START_MESSAGE


def _build_worker_app(database_url):
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    from .storage import LocalStorageBackend

    app.config.setdefault("LEGAL_ML_STORAGE_BACKEND", LocalStorageBackend())
    # No host model definitions are needed: ownership is an opaque owner_id with
    # no foreign key into host tables (see ownership.py), so the package's models
    # resolve on their own. The worker uses the package's default handle.
    db = get_db()
    db.init_app(app)
    set_active_db(db)
    return app


def _connect_redis(redis_url, retries=None):
    attempt = 0
    while True:
        try:
            client = redis_lib.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            logger.info("Redis ready at %s", redis_url)
            return client
        except Exception as e:
            attempt += 1
            if retries is not None and attempt >= retries:
                raise
            logger.warning("Waiting for Redis (%s)...: %s", attempt, e)
            time.sleep(3)


class RedisBackoff:
    """D8: exponential backoff while Redis is unreachable (1s doubling to a
    30s cap), logging one line when it goes down and one when it recovers."""

    def __init__(self, name, initial=1, maximum=30):
        self.name = name
        self.initial = initial
        self.maximum = maximum
        self._next = None  # None => Redis is up

    def failed(self, exc):
        if self._next is None:
            logger.warning("%s loop: Redis unreachable (%s); backing off", self.name, exc)
            self._next = self.initial
        delay = self._next
        self._next = min(self._next * 2, self.maximum)
        return delay

    def succeeded(self):
        if self._next is not None:
            logger.info("%s loop: Redis reconnected", self.name)
            self._next = None


def _consume(redis_client, queue, handle, on_error, name, sleep=time.sleep, stop=None):
    """Shared queue loop: BRPOP a task, `handle(raw)` it inside a catch-all so
    one bad task never kills the loop; `on_error(raw, exc)` settles it. Redis
    connection errors on BRPOP back off via `sleep` instead of spinning."""
    backoff = RedisBackoff(name)
    while not (stop and stop()):
        try:
            task_data = redis_client.brpop(queue, timeout=10)
        except REDIS_DOWN_ERRORS as e:
            sleep(backoff.failed(e))
            continue
        backoff.succeeded()
        if not task_data:
            continue
        raw = task_data[1]
        try:
            handle(raw)
        except Exception as e:
            logger.exception("%s loop error (%s)", name, type(e).__name__)
            try:
                on_error(raw, e)
            except Exception:
                logger.exception("%s loop could not record the error", name)


def _parse_field(raw, field):
    try:
        return json.loads(raw).get(field)
    except Exception:
        return None


def _fail_training_job(raw, exc, redis_client, app):
    """D5: a loop-level error (malformed/unloadable task) fails the job with a
    plain message, when its id can be parsed; terminal jobs are left alone."""
    job_id = _parse_field(raw, "job_id")
    if not job_id:
        logger.error("training loop: unparseable task dropped (%s)", type(exc).__name__)
        return
    logger.error("training job %s: loop error %s", job_id, type(exc).__name__)
    with app.app_context():
        db.session.rollback()
        failed = transition(
            job_id,
            ACTIVE_JOB_STATUSES,
            "failed",
            model_status="failed",
            reason=f"loop error {type(exc).__name__}",
            error_message=LOOP_ERROR_MESSAGE,
            completed_at=datetime.utcnow(),
        )
    if failed:
        redis_client.setex(
            job_status_key(job_id),
            JOB_STATUS_TTL_SECONDS,
            json.dumps({"status": "failed", "error": LOOP_ERROR_MESSAGE}),
        )


def _training_loop(
    redis_client,
    model_storage_path,
    app,
    training_queue="training_queue",
    liveness=DEFAULT_TIMINGS,
    sleep=time.sleep,
    stop=None,
):
    def handle(raw):
        task_info = json.loads(raw)
        with app.app_context():
            process_training_task(task_info, redis_client, model_storage_path, liveness=liveness)

    _consume(
        redis_client,
        training_queue,
        handle,
        lambda raw, exc: _fail_training_job(raw, exc, redis_client, app),
        "training",
        sleep=sleep,
        stop=stop,
    )


def _prediction_loop(
    redis_client, model_storage_path, app, prediction_queue="prediction_queue", sleep=time.sleep, stop=None
):
    def handle(raw):
        task_info = json.loads(raw)
        prediction_id = task_info["prediction_id"]
        with app.app_context():
            result = process_prediction_task(task_info, model_storage_path)
        result_key = prediction_result_key(prediction_id)
        redis_client.lpush(result_key, json.dumps(result))
        redis_client.expire(result_key, PREDICTION_RESULT_TTL_SECONDS)
        logger.info("prediction completed (%s)", prediction_id)

    def on_error(raw, exc):
        prediction_id = _parse_field(raw, "prediction_id")
        if prediction_id:
            result_key = prediction_result_key(prediction_id)
            redis_client.lpush(result_key, json.dumps({"error": str(exc)}))
            redis_client.expire(result_key, PREDICTION_RESULT_TTL_SECONDS)

    _consume(redis_client, prediction_queue, handle, on_error, "prediction", sleep=sleep, stop=stop)


def _explanation_loop(
    redis_client, model_storage_path, app, explanation_queue="explanation_queue", sleep=time.sleep, stop=None
):
    def handle(raw):
        task_info = json.loads(raw)
        explanation_id = task_info["explanation_id"]
        with app.app_context():
            result = process_explanation_task(task_info, model_storage_path)
        redis_client.setex(
            explanation_result_key(explanation_id), EXPLANATION_RESULT_TTL_SECONDS, json.dumps(result)
        )
        logger.info(
            "%s explanation generated for model %s",
            result.get("method", "?").upper(),
            task_info.get("model_id"),
        )

    def on_error(raw, exc):
        explanation_id = _parse_field(raw, "explanation_id")
        if explanation_id:
            redis_client.setex(
                explanation_result_key(explanation_id),
                EXPLANATION_RESULT_TTL_SECONDS,
                json.dumps({"status": "failed", "error": str(exc)}),
            )

    _consume(redis_client, explanation_queue, handle, on_error, "explanation", sleep=sleep, stop=stop)


def _llm_loop(redis_client, app, llm_queue="llm_queue", sleep=time.sleep, stop=None):
    def handle(raw):
        task_info = json.loads(raw)
        with app.app_context():
            process_llm_annotation_task(task_info, redis_client)

    def on_error(raw, exc):
        with app.app_context():
            db.session.rollback()
            fail_llm_job(raw, redis_client)

    _consume(redis_client, llm_queue, handle, on_error, "llm", sleep=sleep, stop=stop)


def _scoring_loop(
    redis_client, model_storage_path, app, scoring_queue="scoring_queue", sleep=time.sleep, stop=None
):
    def handle(raw):
        task_info = json.loads(raw)
        with app.app_context():
            process_scoring_task(task_info, model_storage_path)

    def on_error(raw, exc):
        with app.app_context():
            db.session.rollback()
            fail_scoring_round(_parse_field(raw, "round_id"))

    _consume(redis_client, scoring_queue, handle, on_error, "scoring", sleep=sleep, stop=stop)


def run_ml_worker(
    redis_url,
    database_url,
    model_storage_path,
    block=True,
    training_queue="training_queue",
    prediction_queue="prediction_queue",
    explanation_queue="explanation_queue",
    liveness=DEFAULT_TIMINGS,
    llm_queue="llm_queue",
    usage_meter=None,
    scoring_queue="scoring_queue",
):
    """Starts the training/prediction/explanation queue-consumer threads.

    Set `block=False` to start the threads and return immediately (e.g. when
    the host app's own worker process wants to run this alongside its own
    unrelated queues in the same process).

    `liveness` (LivenessTimings) sets the heartbeat interval (15s), the stale
    window (120s) and the job-status TTL (3600s). The API's sweep reads its
    stale window from `LEGAL_ML_HEARTBEAT_STALE_SECONDS`; keep the two equal.

    `usage_meter` meters shared-key tokens for batch LLM jobs, and must be the
    same kind the API uses (the API's `USAGE_METER`). It defaults to the DB
    counter. The shared endpoint and encryption key come from the environment
    (see llm/config.py).
    """
    # No-op when the host already configured logging; otherwise the worker's
    # startup and job lines go to stderr like the print()s they replaced.
    logging.basicConfig(level=logging.INFO)
    redis_client = _connect_redis(redis_url)
    app = _build_worker_app(database_url)
    if usage_meter is not None:
        app.config["USAGE_METER"] = usage_meter

    threads = [
        threading.Thread(target=_training_loop, args=(redis_client, model_storage_path, app, training_queue, liveness), daemon=True),
        threading.Thread(target=_prediction_loop, args=(redis_client, model_storage_path, app, prediction_queue), daemon=True),
        threading.Thread(target=_explanation_loop, args=(redis_client, model_storage_path, app, explanation_queue), daemon=True),
        threading.Thread(target=_llm_loop, args=(redis_client, app, llm_queue), daemon=True),
        threading.Thread(target=_scoring_loop, args=(redis_client, model_storage_path, app, scoring_queue), daemon=True),
    ]
    for t in threads:
        t.start()
    # D12: lets the API tell queued jobs that no worker is running.
    start_worker_liveness(redis_client, liveness)

    logger.info(
        "legal_ml_core worker running (%s, %s, %s, %s, %s)",
        training_queue,
        prediction_queue,
        explanation_queue,
        llm_queue,
        scoring_queue,
    )

    if not block:
        return threads

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down legal_ml_core worker...")

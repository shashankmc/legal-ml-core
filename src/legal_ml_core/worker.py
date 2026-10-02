"""Worker entrypoint: consumes the training/prediction/explanation Redis queues.

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
import threading
import time

import redis as redis_lib
from flask import Flask

from .compute.tasks import process_explanation_task, process_prediction_task, process_training_task
from .db import db


def _build_worker_app(database_url):
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    # No host model definitions are needed: ownership is an opaque owner_id with
    # no foreign key into host tables (see ownership.py), so the package's models
    # resolve on their own.
    db.init_app(app)
    return app


def _connect_redis(redis_url, retries=None):
    attempt = 0
    while True:
        try:
            client = redis_lib.Redis.from_url(redis_url, decode_responses=True)
            client.ping()
            print(f"Redis ready at {redis_url}", flush=True)
            return client
        except Exception as e:
            attempt += 1
            if retries is not None and attempt >= retries:
                raise
            print(f"Waiting for Redis ({attempt})...: {e}", flush=True)
            time.sleep(3)


def _training_loop(redis_client, model_storage_path, app):
    while True:
        try:
            task_data = redis_client.brpop("training_queue", timeout=10)
            if not task_data:
                continue
            task_info = json.loads(task_data[1])
            with app.app_context():
                process_training_task(task_info, redis_client, model_storage_path)
        except Exception as e:
            print(f"Error in training loop: {str(e)}", flush=True)


def _prediction_loop(redis_client, model_storage_path):
    while True:
        prediction_id = None
        try:
            task_data = redis_client.brpop("prediction_queue", timeout=10)
            if not task_data:
                continue
            task_info = json.loads(task_data[1])
            prediction_id = task_info["prediction_id"]
            result = process_prediction_task(task_info, model_storage_path)
            result_key = f"prediction_result_{prediction_id}"
            redis_client.lpush(result_key, json.dumps(result))
            redis_client.expire(result_key, 300)
            print(f"Prediction completed ({prediction_id})", flush=True)
        except Exception as e:
            print(f"Error in prediction loop: {str(e)}", flush=True)
            if prediction_id:
                result_key = f"prediction_result_{prediction_id}"
                redis_client.lpush(result_key, json.dumps({"error": str(e)}))
                redis_client.expire(result_key, 300)


def _explanation_loop(redis_client, model_storage_path):
    while True:
        explanation_id = None
        try:
            task_data = redis_client.brpop("explanation_queue", timeout=10)
            if not task_data:
                continue
            task_info = json.loads(task_data[1])
            explanation_id = task_info["explanation_id"]
            result = process_explanation_task(task_info, model_storage_path)
            redis_client.setex(f"explanation_{explanation_id}", 3600, json.dumps(result))
            print(f"{result.get('method', '?').upper()} explanation generated for model {task_info.get('model_id')}", flush=True)
        except Exception as e:
            print(f"Error in explanation loop: {str(e)}", flush=True)
            if explanation_id:
                redis_client.setex(
                    f"explanation_{explanation_id}",
                    3600,
                    json.dumps({"status": "failed", "error": str(e)}),
                )


def run_ml_worker(redis_url, database_url, model_storage_path, block=True):
    """Starts the training/prediction/explanation queue-consumer threads.

    Set `block=False` to start the threads and return immediately (e.g. when
    the host app's own worker process wants to run this alongside its own
    unrelated queues in the same process).
    """
    redis_client = _connect_redis(redis_url)
    app = _build_worker_app(database_url)

    threads = [
        threading.Thread(target=_training_loop, args=(redis_client, model_storage_path, app), daemon=True),
        threading.Thread(target=_prediction_loop, args=(redis_client, model_storage_path), daemon=True),
        threading.Thread(target=_explanation_loop, args=(redis_client, model_storage_path), daemon=True),
    ]
    for t in threads:
        t.start()

    print("legal_ml_core worker running (training_queue, prediction_queue, explanation_queue)", flush=True)

    if not block:
        return threads

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down legal_ml_core worker...", flush=True)

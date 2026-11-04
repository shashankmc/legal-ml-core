"""Shared set-up for the monitoring tests: a Model with monitoring settings, its
run through the worker, and the points the run stored."""
import json

from legal_ml_core import TrainingJob, db
from legal_ml_core.compute.tasks import process_training_task
from legal_ml_core.compute.trainer import MLModelTrainer


def create_model(client, auth_headers, dataset_id, monitoring, model_type="logistic_regression"):
    return client.post(
        "/api/models/",
        json={"name": "M", "dataset_id": dataset_id, "model_type": model_type, "parameters": {"monitoring": monitoring}},
        headers=auth_headers,
    )


def start_run(client, auth_headers, dataset_id, monitoring):
    created = create_model(client, auth_headers, dataset_id, monitoring)
    assert created.status_code == 201, created.get_json()
    model_id = created.get_json()["model"]["id"]
    started = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)
    assert started.status_code == 200, started.get_json()
    return model_id


def run_classical(client, auth_headers, app, fake_redis, dataset_id, monitoring, trainer_factory=MLModelTrainer):
    """A logistic regression run with these monitoring settings, through the
    worker to the end. Gives the points it stored, as the API returns them."""
    start_run(client, auth_headers, dataset_id, monitoring)
    task = json.loads(fake_redis.rpop("training_queue"))
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"], trainer_factory=trainer_factory)
    assert db.session.get(TrainingJob, task["job_id"]).status == "completed"
    return stored_points(client, auth_headers, task["job_id"])


def stored_points(client, auth_headers, job_id):
    return client.get(f"/api/training/{job_id}/metrics", headers=auth_headers).get_json()["points"]

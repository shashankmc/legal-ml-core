"""Shared set-up for the active learning API tests."""
import json

import pandas as pd

from legal_ml_core import AnnotationTask, Dataset, Model, db
from legal_ml_core.compute.tasks import process_training_task
from legal_ml_core.worker import _scoring_loop
from tests.conftest import TEST_USER_ID

PROJECT_ID = "proj-al"
CIVIL = "the contract breach caused damages and the claim for compensation"
CRIMINAL = "the defendant committed murder and theft and was sentenced to prison"


def link_dataset(app, tmp_path, texts, labels=None, task_type="classification", dataset_labels=None):
    path = tmp_path / f"data-{len(texts)}.csv"
    rows = {"text": texts}
    if labels is not None:
        rows["label"] = labels
    pd.DataFrame(rows).to_csv(path, index=False)
    app.config.setdefault("_known_projects", set()).add((PROJECT_ID, TEST_USER_ID))
    dataset = Dataset(
        owner_id=TEST_USER_ID,
        name="Cases",
        file_path=str(path),
        row_count=len(texts),
        columns=[{"name": "text"}] + ([{"name": "label"}] if labels is not None else []),
        task_type=task_type,
        project_id=PROJECT_ID,
        labels=dataset_labels,
    )
    db.session.add(dataset)
    db.session.commit()
    return dataset.id


def start(client, auth_headers, dataset_id, **options):
    return client.post(
        f"/api/active-learning/datasets/{dataset_id}/start", json=options, headers=auth_headers
    )


def state(client, auth_headers, dataset_id):
    resp = client.get(f"/api/active-learning/datasets/{dataset_id}", headers=auth_headers)
    assert resp.status_code == 200
    return resp.get_json()


def retrain(client, auth_headers, dataset_id):
    return client.post(f"/api/active-learning/datasets/{dataset_id}/retrain", headers=auth_headers)


def next_item(client, auth_headers, dataset_id, **params):
    resp = client.get(
        f"/api/annotation/projects/{PROJECT_ID}/queue/next",
        query_string={"dataset": dataset_id, **params},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    return resp.get_json()


def label(client, auth_headers, task_id, value):
    resp = client.post(
        f"/api/annotation/tasks/{task_id}/save", json={"annotation_label": value}, headers=auth_headers
    )
    assert resp.status_code == 200


def label_next(client, auth_headers, dataset_id, value_for, count, **params):
    """Label the next ``count`` queue items with ``value_for(text)``."""
    served = []
    for _ in range(count):
        task = next_item(client, auth_headers, dataset_id, **params)["task"]
        label(client, auth_headers, task["id"], value_for(task["text"]))
        served.append(task)
    return served


def run_training(app, fake_redis):
    """What the worker does with the queued job: train, then rescore the pool."""
    process_training_task(json.loads(fake_redis.rpop("training_queue")), fake_redis, app.config["MODEL_FOLDER"])
    db.session.expire_all()


def run_scoring(app, fake_redis):
    _scoring_loop(
        fake_redis,
        app.config["MODEL_FOLDER"],
        app,
        stop=lambda: fake_redis.llen("scoring_queue") == 0,
        sleep=lambda s: None,
    )
    db.session.expire_all()


def by_topic(text):
    return "civil" if "contract" in text else "criminal"


def balanced_texts(pairs):
    return [text for i in range(pairs) for text in (f"{CIVIL} {i}", f"{CRIMINAL} {i}")]


def task_for(dataset_id, item_index):
    return AnnotationTask.query.filter_by(dataset_id=dataset_id, item_index=item_index).first()


def loop_model(dataset_id):
    return Model.query.filter_by(dataset_id=dataset_id).first()

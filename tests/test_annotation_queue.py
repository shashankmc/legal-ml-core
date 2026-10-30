"""A dataset on a project offers its unlabelled rows for annotation."""
import pandas as pd

from legal_ml_core import AnnotationTask, Dataset, Model, TrainingJob, db
from legal_ml_core.compute.tasks import process_training_task
from tests.conftest import TEST_USER_ID

PROJECT_ID = "proj-queue"


def _link_dataset(app, tmp_path, rows, task_type="classification", name="data.csv"):
    """Write ``rows`` to a file and link a dataset on it to the test project."""
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    app.config.setdefault("_known_projects", set()).add((PROJECT_ID, TEST_USER_ID))
    dataset = Dataset(
        owner_id=TEST_USER_ID,
        name=name,
        file_path=str(path),
        row_count=len(next(iter(rows.values()))),
        columns=[],
        task_type=task_type,
        project_id=PROJECT_ID,
    )
    db.session.add(dataset)
    db.session.commit()
    return dataset.id


def _next(client, auth_headers):
    resp = client.get(f"/api/annotation/projects/{PROJECT_ID}/queue/next", headers=auth_headers)
    assert resp.status_code == 200
    return resp.get_json()


def _tasks(client, auth_headers):
    resp = client.get(f"/api/annotation/projects/{PROJECT_ID}/tasks", headers=auth_headers)
    return resp.get_json()["tasks"]


def test_unlabelled_rows_are_queued_in_row_order(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b", "c"], "label": ["x", None, None]})

    body = _next(client, auth_headers)

    assert body["task"]["text"] == "b"
    assert body["task"]["item_index"] == 1
    assert body["stats"] == {"total": 2, "annotated": 0, "percentage": 0}


def test_whitespace_label_counts_as_blank(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "label": ["   ", "x"]})

    assert [t["text"] for t in _tasks(client, auth_headers)] == ["a"]


def test_file_without_a_label_column_queues_every_row(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"]})

    assert _next(client, auth_headers)["stats"]["total"] == 2


def test_fully_labelled_dataset_queues_nothing(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "label": ["x", "y"]})

    body = _next(client, auth_headers)

    assert body["task"] is None
    assert body["stats"]["total"] == 0
    assert body["message"] == "Nothing to annotate"


def test_column_names_match_case_insensitively(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"Text": ["a", "b"], "Label": ["x", None]})

    assert [t["text"] for t in _tasks(client, auth_headers)] == ["b"]


def test_ner_rows_without_entities_are_queued(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "entities": ["[]", None]}, task_type="ner")

    assert [t["text"] for t in _tasks(client, auth_headers)] == ["b"]


def test_task_types_without_labels_queue_nothing(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"]}, task_type="topic_modeling")

    assert _tasks(client, auth_headers) == []


def test_summarization_is_not_queued(client, auth_headers, app, tmp_path):
    # No trainer learns from summaries, so writing them here would go nowhere.
    rows = {"text": ["a", "b"], "summary": [None, None]}
    _link_dataset(app, tmp_path, rows, task_type="summarization")

    assert _tasks(client, auth_headers) == []


def test_rows_with_blank_text_are_left_out(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", None], "label": [None, None]})

    assert [t["text"] for t in _tasks(client, auth_headers)] == ["a"]


def test_repeated_reads_do_not_duplicate_tasks(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "label": [None, None]})

    _next(client, auth_headers)
    _next(client, auth_headers)

    assert len(_tasks(client, auth_headers)) == 2


def test_saved_and_skipped_work_survives_later_reads(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b", "c"], "label": [None, None, None]})
    first = _next(client, auth_headers)["task"]["id"]
    client.post(
        f"/api/annotation/tasks/{first}/save", json={"annotation_label": "x"}, headers=auth_headers
    )
    second = _next(client, auth_headers)["task"]["id"]
    client.post(f"/api/annotation/tasks/{second}/skip", headers=auth_headers)

    body = _next(client, auth_headers)

    assert body["task"]["text"] == "c"
    assert body["stats"] == {"total": 3, "annotated": 1, "percentage": 33}
    assert sorted(t["status"] for t in _tasks(client, auth_headers)) == [
        "annotated",
        "pending",
        "skipped",
    ]


def test_stats_count_rows_before_the_queue_is_opened(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a", "b"], "label": [None, None]})

    resp = client.get(f"/api/annotation/projects/{PROJECT_ID}/queue/stats", headers=auth_headers)

    assert resp.get_json()["stats"]["pending"] == 2


def _queued_task(app, tmp_path, client, auth_headers):
    _link_dataset(app, tmp_path, {"text": ["a"], "label": [None]})
    return _next(client, auth_headers)["task"]["id"]


def test_saving_an_empty_label_is_refused(client, auth_headers, app, tmp_path):
    task_id = _queued_task(app, tmp_path, client, auth_headers)

    resp = client.post(
        f"/api/annotation/tasks/{task_id}/save", json={"annotation_label": ""}, headers=auth_headers
    )

    assert resp.status_code == 400
    assert db.session.get(AnnotationTask, task_id).status == "pending"


def test_saving_a_whitespace_label_is_refused(client, auth_headers, app, tmp_path):
    task_id = _queued_task(app, tmp_path, client, auth_headers)

    resp = client.post(
        f"/api/annotation/tasks/{task_id}/save",
        json={"annotation_label": "   "},
        headers=auth_headers,
    )

    assert resp.status_code == 400
    assert db.session.get(AnnotationTask, task_id).status == "pending"


def test_a_passage_with_no_entities_can_be_saved(client, auth_headers, app, tmp_path):
    _link_dataset(app, tmp_path, {"text": ["a"], "entities": [None]}, task_type="ner")
    task_id = _next(client, auth_headers)["task"]["id"]

    resp = client.post(
        f"/api/annotation/tasks/{task_id}/save",
        json={"annotation_label": "[]"},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    assert db.session.get(AnnotationTask, task_id).status == "annotated"


def test_unreadable_file_leaves_the_queue_empty(client, auth_headers, app, tmp_path):
    dataset_id = _link_dataset(app, tmp_path, {"text": ["a"], "label": [None]})
    (tmp_path / "data.csv").unlink()

    assert _next(client, auth_headers)["stats"]["total"] == 0
    assert db.session.get(Dataset, dataset_id) is not None


def test_another_owners_dataset_on_the_project_is_not_queued(client, auth_headers, app, tmp_path):
    dataset_id = _link_dataset(app, tmp_path, {"text": ["a"], "label": [None]})
    dataset = db.session.get(Dataset, dataset_id)
    dataset.owner_id = "22222222-2222-2222-2222-222222222222"
    db.session.commit()

    assert _next(client, auth_headers)["stats"]["total"] == 0
    assert AnnotationTask.query.count() == 0


def _train(app, fake_redis, dataset_id, task_type="classification"):
    dataset = db.session.get(Dataset, dataset_id)
    model = Model(
        owner_id=TEST_USER_ID,
        dataset_id=dataset_id,
        name="M",
        model_type="naive_bayes",
        parameters={},
        status="training",
    )
    db.session.add(model)
    db.session.flush()
    job = TrainingJob(owner_id=TEST_USER_ID, model_id=model.id, status="pending")
    db.session.add(job)
    db.session.commit()
    process_training_task(
        {
            "job_id": job.id,
            "model_id": model.id,
            "dataset_id": dataset_id,
            "model_type": "naive_bayes",
            "parameters": {},
            "dataset_path": dataset.file_path,
            "task_type": task_type,
            "preprocessing_options": {},
        },
        fake_redis,
        app.config["MODEL_FOLDER"],
    )
    return db.session.get(Model, model.id), db.session.get(TrainingJob, job.id)


def _sentences(n):
    return [f"clause number {i} about the contract" for i in range(n)]


def test_saved_annotations_become_training_labels(client, auth_headers, app, fake_redis, tmp_path):
    rows = {"text": _sentences(16), "label": ["civil", "criminal"] * 6 + [None] * 4}
    dataset_id = _link_dataset(app, tmp_path, rows)
    for _ in range(3):
        task = _next(client, auth_headers)["task"]
        client.post(
            f"/api/annotation/tasks/{task['id']}/save",
            json={"annotation_label": "labour"},
            headers=auth_headers,
        )

    model, job = _train(app, fake_redis, dataset_id)

    assert job.status == "completed"
    # The last row stays unlabelled: it must be left out, not become a class.
    assert model.metrics["confusion_matrix_labels"] == ["civil", "criminal", "labour"]


def test_file_without_a_label_column_trains_on_saved_annotations(
    client, auth_headers, app, fake_redis, tmp_path
):
    dataset_id = _link_dataset(app, tmp_path, {"text": _sentences(10)})
    for label in ["civil", "criminal"] * 5:
        task = _next(client, auth_headers)["task"]
        client.post(
            f"/api/annotation/tasks/{task['id']}/save",
            json={"annotation_label": label},
            headers=auth_headers,
        )

    model, job = _train(app, fake_redis, dataset_id)

    assert job.status == "completed"
    assert model.metrics["confusion_matrix_labels"] == ["civil", "criminal"]


def test_training_uses_the_annotated_column_not_the_second_one(
    client, auth_headers, app, fake_redis, tmp_path
):
    rows = {"text": _sentences(10), "court": ["Hoge Raad", "Rechtbank"] * 5}
    dataset_id = _link_dataset(app, tmp_path, rows)
    for label in ["civil", "criminal"] * 5:
        task = _next(client, auth_headers)["task"]
        client.post(
            f"/api/annotation/tasks/{task['id']}/save",
            json={"annotation_label": label},
            headers=auth_headers,
        )

    model, job = _train(app, fake_redis, dataset_id)

    assert job.status == "completed"
    assert model.metrics["confusion_matrix_labels"] == ["civil", "criminal"]


def test_training_reads_the_text_column_wherever_it_sits(
    client, auth_headers, app, fake_redis, tmp_path
):
    rows = {
        "court": ["Hoge Raad"] * 10,
        "text": ["contract breach"] * 5 + ["theft and robbery"] * 5,
        "label": ["civil"] * 5 + ["criminal"] * 5,
    }
    dataset_id = _link_dataset(app, tmp_path, rows)

    model, job = _train(app, fake_redis, dataset_id)

    assert job.status == "completed"
    assert model.metrics["confusion_matrix_labels"] == ["civil", "criminal"]

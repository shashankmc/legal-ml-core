from legal_ml_core import AnnotationTask, Dataset, db
from tests.conftest import TEST_USER_ID


def _make_project_and_dataset(app, project_id="proj-1", owner_id=TEST_USER_ID):
    """Projects are a host concern; register this (project, owner) pair with the
    conftest project validator, then create the dataset/task it scopes.
    """
    with app.app_context():
        app.config.setdefault("_known_projects", set()).add((project_id, owner_id))
        dataset = Dataset(
            owner_id=owner_id,
            name="Annotated dataset",
            file_path="/tmp/a.csv",
            row_count=2,
            columns=[],
            task_type="classification",
            project_id=project_id,
        )
        db.session.add(dataset)
        db.session.flush()
        task = AnnotationTask(dataset_id=dataset.id, item_index=0, text="clause text", status="pending")
        db.session.add(task)
        db.session.commit()
        return project_id, dataset.id, task.id


def test_get_next_annotation_task_and_save(client, auth_headers, app):
    project_id, dataset_id, task_id = _make_project_and_dataset(app)

    resp = client.get(f"/api/annotation/projects/{project_id}/queue/next", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["task"]["id"] == task_id
    assert body["stats"]["total"] == 1

    resp = client.post(
        f"/api/annotation/tasks/{task_id}/save",
        json={"annotation_label": "positive"},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    assert resp.get_json()["task"]["status"] == "annotated"

    resp = client.get(f"/api/annotation/projects/{project_id}/queue/next", headers=auth_headers)
    assert resp.get_json()["task"] is None
    assert "All items annotated" in resp.get_json()["message"]


def test_skip_annotation(client, auth_headers, app):
    _, _, task_id = _make_project_and_dataset(app)

    resp = client.post(f"/api/annotation/tasks/{task_id}/skip", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["task"]["status"] == "skipped"


def test_annotation_queue_requires_owned_project(client, auth_headers, app):
    project_id, _, _ = _make_project_and_dataset(app, project_id="proj-2", owner_id="someone-else")

    resp = client.get(f"/api/annotation/projects/{project_id}/queue/next", headers=auth_headers)
    assert resp.status_code == 404


def test_reset_annotation_tasks(client, auth_headers, app):
    project_id, _, task_id = _make_project_and_dataset(app, project_id="proj-3")

    client.post(f"/api/annotation/tasks/{task_id}/save", json={"annotation_label": "x"}, headers=auth_headers)

    resp = client.post(f"/api/annotation/projects/{project_id}/tasks/reset", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["reset_count"] == 1

    resp = client.get(f"/api/annotation/tasks/{task_id}", headers=auth_headers)
    assert resp.get_json()["task"]["status"] == "pending"

"""Opening and saving a task as the kit's Assignment (#137)."""
import json

from legal_ml_core import AnnotationTask, Dataset, db
from tests.conftest import TEST_USER_ID

OTHER_USER_ID = "22222222-2222-2222-2222-222222222222"
TEXT = "Smith sued Acme in Paris."


def _task(app, task_type="classification", labels=("Void", "Valid"), owner_id=TEST_USER_ID, **task_fields):
    with app.app_context():
        dataset = Dataset(
            owner_id=owner_id,
            name="Contracts",
            file_path="/tmp/a.csv",
            row_count=1,
            columns=[],
            task_type=task_type,
            labels=list(labels),
            project_id="proj-1",
        )
        db.session.add(dataset)
        db.session.flush()
        task = AnnotationTask(dataset_id=dataset.id, item_index=0, text=TEXT, status="pending", **task_fields)
        db.session.add(task)
        db.session.commit()
        return task.id


def _assignment(tags=(), spans=(), metadata=None):
    return {
        "annotator": TEST_USER_ID,
        "order": 1,
        "status": "done",
        "confidence": 4,
        "annotations": list(spans),
        "document_annotations": [
            {"id": -(i + 1), "label": label, "confidence": 5, "metadata": metadata} for i, label in enumerate(tags)
        ],
        "document_relations": [],
    }


def _url(task_id):
    return f"/api/annotation/tasks/{task_id}/assignment"


def test_a_task_opens_as_a_kit_assignment_with_its_labels_and_level(client, auth_headers, app):
    task_id = _task(app)

    resp = client.get(_url(task_id), headers=auth_headers)

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["document"] == {"name": "Contracts, item 1", "full_text": TEXT}
    assert body["labels"] == ["Void", "Valid"]
    assert body["annotation_level"] == "document"
    assert body["assignment"]["id"] == task_id
    assert body["assignment"]["status"] == "pending"
    assert body["assignment"]["document_annotations"] == []


def test_an_ner_task_opens_at_word_level(client, auth_headers, app):
    task_id = _task(app, task_type="ner", labels=("PERSON", "ORG"))
    assert client.get(_url(task_id), headers=auth_headers).get_json()["annotation_level"] == "word"


def test_saving_an_assignment_labels_the_task_and_keeps_what_the_kit_sent(client, auth_headers, app):
    task_id = _task(app)

    resp = client.put(_url(task_id), json=_assignment(tags=["Void"], metadata="clause 4"), headers=auth_headers)

    assert resp.status_code == 200
    task = resp.get_json()["task"]
    assert (task["annotation_label"], task["status"], task["label_source"]) == ("Void", "annotated", "human")
    reopened = client.get(_url(task_id), headers=auth_headers).get_json()["assignment"]
    assert reopened["status"] == "done"
    assert reopened["confidence"] == 4
    assert reopened["document_annotations"][0]["metadata"] == "clause 4"


def test_saving_ner_spans_stores_the_entity_list(client, auth_headers, app):
    task_id = _task(app, task_type="ner", labels=("PERSON", "ORG"))
    span = {"id": -1, "start": 0, "end": 5, "label": "PERSON", "text": "Smith", "relations": [], "confidence": 0, "metadata": None}

    resp = client.put(_url(task_id), json=_assignment(spans=[span]), headers=auth_headers)

    assert resp.status_code == 200
    assert json.loads(resp.get_json()["task"]["annotation_label"]) == [
        {"text": "Smith", "start": 0, "end": 5, "type": "PERSON"}
    ]


def test_an_assignment_that_breaks_the_rules_is_refused_with_the_reason(client, auth_headers, app):
    task_id = _task(app)

    resp = client.put(_url(task_id), json=_assignment(tags=["Void", "Valid"]), headers=auth_headers)

    assert resp.status_code == 400
    assert "only one label" in resp.get_json()["error"]
    reopened = client.get(_url(task_id), headers=auth_headers).get_json()["assignment"]
    assert reopened["status"] == "pending" and reopened["document_annotations"] == []


def test_a_task_type_with_no_item_labels_is_refused(client, auth_headers, app):
    task_id = _task(app, task_type="summarization", labels=())
    resp = client.put(_url(task_id), json=_assignment(tags=["x"]), headers=auth_headers)
    assert resp.status_code == 400


def test_a_dataset_with_typed_labels_offers_the_labels_saved_so_far(client, auth_headers, app):
    first = _task(app, labels=())
    client.put(_url(first), json=_assignment(tags=["Upheld"]), headers=auth_headers)

    assert client.get(_url(first), headers=auth_headers).get_json()["labels"] == ["Upheld"]


def test_another_owners_task_can_be_neither_opened_nor_saved(client, auth_headers, app):
    task_id = _task(app, owner_id=OTHER_USER_ID)

    assert client.get(_url(task_id), headers=auth_headers).status_code == 403
    assert client.put(_url(task_id), json=_assignment(tags=["Void"]), headers=auth_headers).status_code == 403


def test_an_unknown_task_is_not_found(client, auth_headers):
    assert client.get(_url("nope"), headers=auth_headers).status_code == 404
    assert client.put(_url("nope"), json=_assignment(tags=["Void"]), headers=auth_headers).status_code == 404


def _opened_tags(client, auth_headers, task_id):
    assignment = client.get(_url(task_id), headers=auth_headers).get_json()["assignment"]
    return assignment["status"], [(t["label"], t["metadata"]) for t in assignment["document_annotations"]]


def test_saving_a_plain_label_afterwards_replaces_the_assignment(client, auth_headers, app):
    task_id = _task(app)
    client.put(_url(task_id), json=_assignment(tags=["Void"], metadata="old note"), headers=auth_headers)

    client.post(f"/api/annotation/tasks/{task_id}/save", json={"annotation_label": "Valid"}, headers=auth_headers)

    assert _opened_tags(client, auth_headers, task_id) == ("done", [("Valid", None)])


def test_resetting_tasks_clears_the_assignment(client, auth_headers, app):
    task_id = _task(app)
    app.config.setdefault("_known_projects", set()).add(("proj-1", TEST_USER_ID))
    client.put(_url(task_id), json=_assignment(tags=["Void"], metadata="note"), headers=auth_headers)

    client.post("/api/annotation/projects/proj-1/tasks/reset", headers=auth_headers)

    assert _opened_tags(client, auth_headers, task_id) == ("pending", [])

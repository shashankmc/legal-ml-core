"""Exporting annotations as kit JSON and importing a kit file as a Dataset (#138)."""
import io
import json

from legal_ml_core import AnnotationTask, Dataset, db
from tests.conftest import TEST_USER_ID

OTHER_USER_ID = "22222222-2222-2222-2222-222222222222"
TEXT = "Smith sued Acme in Paris."


def _span(start, end, label, id_, relations=()):
    return {
        "id": id_,
        "start": start,
        "end": end,
        "label": label,
        "text": TEXT[start:end],
        "relations": list(relations),
        "confidence": 4,
        "metadata": f"note {id_}",
    }


def _ner_file(**overrides):
    smith = _span(0, 5, "PERSON", 1, relations=[{"to": 2, "direction": "right", "labels": ["Has a"]}])
    acme = _span(11, 15, "ORG", 2)
    return {
        "name": "Imported",
        "desc": "",
        "labelset": {"name": "L", "desc": "", "labels": [{"name": "PERSON", "color": "#000"}, {"name": "ORG", "color": "#111"}]},
        "annotation_level": "word",
        "documents": [
            {
                "name": "a.txt",
                "full_text": TEXT,
                "assignments": [
                    {
                        "annotator": 7,
                        "order": 1,
                        "status": "done",
                        "confidence": 5,
                        "annotations": [smith, acme],
                        "document_annotations": [],
                        "document_relations": [],
                    }
                ],
            },
            {"name": "b.txt", "full_text": "Nobody labelled this.", "assignments": []},
        ],
        **overrides,
    }


def _import(client, auth_headers, data, name="Imported cases"):
    raw = data if isinstance(data, bytes) else json.dumps(data).encode()
    return client.post(
        "/api/datasets/import/legal-annotation-kit",
        data={"file": (io.BytesIO(raw), "task.json"), "name": name, "description": "from Legal Blocks"},
        content_type="multipart/form-data",
        headers=auth_headers,
    )


def _export(client, auth_headers, dataset_id, fmt="legal-annotation-kit"):
    return client.get(f"/api/datasets/{dataset_id}/annotations/export?format={fmt}", headers=auth_headers)


def test_a_kit_file_imports_as_a_dataset_whose_items_keep_their_assignments(client, auth_headers, app):
    resp = _import(client, auth_headers, _ner_file())

    assert resp.status_code == 201
    body = resp.get_json()
    assert body["dataset"]["task_type"] == "ner"
    assert body["dataset"]["labels"] == ["PERSON", "ORG"]
    assert (body["labelled"], body["set_aside"]) == (1, 0)
    tasks = AnnotationTask.query.filter_by(dataset_id=body["dataset"]["id"]).order_by(AnnotationTask.item_index).all()
    assert [(t.status, t.label_source) for t in tasks] == [("annotated", "human"), ("pending", None)]
    assert json.loads(tasks[0].annotation_label)[0] == {"text": "Smith", "start": 0, "end": 5, "type": "PERSON"}


def test_export_then_import_keeps_relations_confidence_and_metadata(client, auth_headers, app):
    first = _import(client, auth_headers, _ner_file()).get_json()["dataset"]["id"]
    exported = _export(client, auth_headers, first)
    assert exported.status_code == 200
    assert "attachment" in exported.headers["Content-Disposition"]

    second = _import(client, auth_headers, exported.get_json()).get_json()["dataset"]["id"]
    again = _export(client, auth_headers, second).get_json()

    assignment = again["documents"][0]["assignments"][0]
    assert assignment["confidence"] == 5
    smith = assignment["annotations"][0]
    assert (smith["confidence"], smith["metadata"]) == (4, "note 1")
    assert smith["relations"] == [{"to": 2, "direction": "right", "labels": ["Has a"]}]
    assert again["documents"][1]["assignments"] == []
    assert again["counts"]["relations"] == 1


def test_document_tags_import_as_a_classification_dataset_ready_to_train(client, auth_headers, app):
    data = {
        **_ner_file(),
        "annotation_level": "document",
        "labelset": {"name": "L", "desc": "", "labels": [{"name": "Void", "color": "#000"}]},
        "documents": [
            {
                "name": "a",
                "full_text": TEXT,
                "assignments": [
                    {
                        "annotator": 1, "order": 1, "status": "done", "confidence": 0, "annotations": [],
                        "document_annotations": [{"id": 1, "label": "Void", "confidence": 0, "metadata": None}],
                        "document_relations": [],
                    }
                ],
            }
        ],
    }
    body = _import(client, auth_headers, data).get_json()

    assert body["dataset"]["task_type"] == "classification"
    task = AnnotationTask.query.filter_by(dataset_id=body["dataset"]["id"]).one()
    assert task.annotation_label == "Void"


def test_a_malformed_file_is_refused_naming_the_field_and_leaves_nothing(client, auth_headers, app):
    resp = _import(client, auth_headers, _ner_file(annotation_level="chapter"))

    assert resp.status_code == 400
    assert "annotation_level" in resp.get_json()["error"]
    assert Dataset.query.count() == 0


def test_a_file_that_is_not_json_is_refused(client, auth_headers):
    resp = _import(client, auth_headers, b"text,label\nx,y\n")
    assert resp.status_code == 400
    assert "JSON" in resp.get_json()["error"]


def test_an_assignment_that_breaks_the_rules_is_refused_naming_its_document(client, auth_headers, app):
    data = _ner_file()
    data["documents"][0]["assignments"][0]["annotations"][1]["label"] = "PLACE"

    resp = _import(client, auth_headers, data)

    assert resp.status_code == 400
    assert "documents[0]" in resp.get_json()["error"]
    assert Dataset.query.count() == 0


def test_the_row_cap_applies_to_imports(client, auth_headers, app):
    app.config["LEGAL_ML_MAX_ROWS"] = 1

    resp = _import(client, auth_headers, _ner_file())

    assert resp.status_code == 400
    assert "maximum" in resp.get_json()["error"]


def test_a_name_is_required(client, auth_headers):
    assert _import(client, auth_headers, _ner_file(), name="  ").status_code == 400


def test_another_owners_dataset_cannot_be_exported(client, auth_headers, app):
    dataset_id = _import(client, auth_headers, _ner_file()).get_json()["dataset"]["id"]
    with app.app_context():
        db.session.get(Dataset, dataset_id).owner_id = OTHER_USER_ID
        db.session.commit()

    assert _export(client, auth_headers, dataset_id).status_code == 404


def test_an_unknown_export_format_is_refused(client, auth_headers):
    dataset_id = _import(client, auth_headers, _ner_file()).get_json()["dataset"]["id"]
    assert _export(client, auth_headers, dataset_id, fmt="label-studio").status_code == 400


def test_a_failure_after_the_dataset_is_made_leaves_no_dataset_or_file(client, auth_headers, app, monkeypatch):
    import pytest
    from pathlib import Path

    import legal_ml_core.routes.datasets as routes

    def broken(dataset):
        raise RuntimeError("tasks could not be made")

    monkeypatch.setattr(routes, "create_annotation_tasks", broken)

    with pytest.raises(RuntimeError):
        _import(client, auth_headers, _ner_file())

    db.session.rollback()
    assert Dataset.query.count() == 0
    assert list(Path(app.config["UPLOAD_FOLDER"]).iterdir()) == []

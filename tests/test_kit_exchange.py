"""A Dataset's annotations as vue-legal-annotation-kit TaskData, both ways (#138)."""
import json
from types import SimpleNamespace

import pytest

from legal_ml_core.kit_exchange import ExchangeError, rows_from_task_data, task_data_from_dataset

TEXT = "Smith sued Acme in Paris."


def _dataset(task_type="classification", labels=("Void", "Valid")):
    return SimpleNamespace(
        id="ds-1", name="Contracts", description="Clauses", owner_id="owner-1", task_type=task_type, labels=list(labels)
    )


def _task(index, label=None, payload=None, status=None, text=TEXT):
    return SimpleNamespace(
        id=f"task-{index}",
        item_index=index,
        text=text,
        annotation_label=label,
        annotation_payload=payload,
        status=status or ("annotated" if label else "pending"),
    )


def _tag(label, metadata=None, confidence=0):
    return {"id": -1, "label": label, "confidence": confidence, "metadata": metadata}


def _span(start, end, label, id_=-1, relations=()):
    return {
        "id": id_,
        "start": start,
        "end": end,
        "label": label,
        "text": TEXT[start:end],
        "relations": list(relations),
        "confidence": 2,
        "metadata": None,
    }


def _assignment(tags=(), spans=(), status="done", annotator="owner-1", order=1):
    return {
        "annotator": annotator,
        "order": order,
        "status": status,
        "confidence": 3,
        "annotations": list(spans),
        "document_annotations": list(tags),
        "document_relations": [],
    }


# Export


def test_export_has_one_document_per_task_and_assignments_for_labelled_ones():
    saved = {**_assignment(tags=[_tag("Void", metadata="clause 4", confidence=5)]), "id": "task-0"}
    data = task_data_from_dataset(_dataset(), [_task(0, "Void", saved), _task(1)])

    assert data["name"] == "Contracts"
    assert data["annotation_level"] == "document"
    assert [label["name"] for label in data["labelset"]["labels"]] == ["Void", "Valid"]
    assert all(label["color"].startswith("#") for label in data["labelset"]["labels"])
    assert [d["name"] for d in data["documents"]] == ["item 1", "item 2"]
    assert data["documents"][0]["full_text"] == TEXT
    tag = data["documents"][0]["assignments"][0]["document_annotations"][0]
    assert (tag["label"], tag["metadata"], tag["confidence"]) == ("Void", "clause 4", 5)
    assert data["documents"][1]["assignments"] == []
    assert data["counts"] == {"documents": 2, "assignments": 1, "annotators": 1, "annotations": 1, "relations": 0}


def test_export_builds_assignments_for_tasks_labelled_before_the_kit():
    entities = [{"text": "Smith", "start": 0, "end": 5, "type": "PERSON"}]
    data = task_data_from_dataset(_dataset("ner", ("PERSON", "ORG")), [_task(0, json.dumps(entities))])

    assert data["annotation_level"] == "word"
    span = data["documents"][0]["assignments"][0]["annotations"][0]
    assert (span["start"], span["end"], span["label"]) == (0, 5, "PERSON")


def test_export_offers_saved_labels_when_the_dataset_names_none():
    data = task_data_from_dataset(_dataset(labels=()), [_task(0, "Upheld"), _task(1, "Quashed"), _task(2, "Upheld")])
    assert [label["name"] for label in data["labelset"]["labels"]] == ["Upheld", "Quashed"]


# Import


def _task_data(documents, level="document", labels=("Void", "Valid")):
    return {
        "name": "Imported",
        "desc": "",
        "labelset": {"name": "L", "desc": "", "labels": [{"name": n, "color": "#000"} for n in labels]},
        "annotation_level": level,
        "documents": documents,
    }


def _doc(text=TEXT, *assignments):
    return {"name": "doc.txt", "full_text": text, "assignments": list(assignments)}


def test_document_tags_import_as_a_classification_dataset():
    plan = rows_from_task_data(
        _task_data([_doc(TEXT, _assignment(tags=[_tag("Void", metadata="kept")])), _doc("Unlabelled.")])
    )

    assert plan.task_type == "classification"
    assert plan.labels == ["Void", "Valid"]
    assert [text for text, _ in plan.items] == [TEXT, "Unlabelled."]
    assert plan.items[0][1]["document_annotations"][0]["metadata"] == "kept"
    assert plan.items[1][1] is None
    assert plan.set_aside == 0


@pytest.mark.parametrize("level", ["character", "word", "sentence", "paragraph"])
def test_spans_import_as_an_ner_dataset(level):
    plan = rows_from_task_data(
        _task_data([_doc(TEXT, _assignment(spans=[_span(0, 5, "PERSON")]))], level=level, labels=("PERSON",))
    )
    assert plan.task_type == "ner"


def test_documents_with_several_tags_are_refused_as_multi_label():
    data = _task_data([_doc(TEXT, _assignment(tags=[_tag("Void"), _tag("Valid")]))])
    with pytest.raises(ExchangeError, match="one tag"):
        rows_from_task_data(data)


def test_of_several_annotators_the_finished_one_with_most_work_is_used():
    pending_rich = _assignment(spans=[_span(0, 5, "PERSON"), _span(11, 15, "ORG", -2)], status="pending", annotator=1)
    done_small = _assignment(spans=[_span(0, 5, "PERSON")], annotator=2)
    done_big = _assignment(spans=[_span(0, 5, "PERSON"), _span(19, 24, "LOC", -3)], annotator=3)
    empty = _assignment(status="pending", annotator=4)
    plan = rows_from_task_data(
        _task_data([_doc(TEXT, pending_rich, done_small, done_big, empty)], level="word", labels=("PERSON", "ORG", "LOC"))
    )

    assert plan.items[0][1]["annotator"] == 3
    assert plan.set_aside == 2  # the two others with work; the empty one is not counted


@pytest.mark.parametrize(
    "data, field",
    [
        ("not json object", "file"),
        ({"documents": []}, "documents"),
        ({**_task_data([]), "documents": "x"}, "documents"),
        (_task_data([{"name": "a", "assignments": []}]), "documents[0].full_text"),
        (_task_data([_doc(TEXT, "x")]), "documents[0].assignments[0]"),
        (_task_data([_doc(TEXT), _doc("   ")]), "documents[1].full_text"),
        ({**_task_data([_doc()]), "annotation_level": "chapter"}, "annotation_level"),
        ({**_task_data([_doc()]), "labelset": {"labels": [{"color": "#000"}]}}, "labelset.labels[0].name"),
    ],
)
def test_a_malformed_file_is_refused_naming_the_field(data, field):
    with pytest.raises(ExchangeError, match=field.replace("[", r"\[").replace("]", r"\]")):
        rows_from_task_data(data)

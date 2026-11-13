"""The kit's Assignment as the stored annotation, and the label derived from it (#137)."""
import json

import pytest

from types import SimpleNamespace

from legal_ml_core.annotation_format import AssignmentError, assignment_from_task, label_from_assignment


def _assignment(tags=(), spans=()):
    return {
        "id": "t1",
        "annotator": "u1",
        "order": 1,
        "status": "done",
        "confidence": 0,
        "annotations": list(spans),
        "document_annotations": [
            {"id": -(i + 1), "label": label, "confidence": 0, "metadata": None} for i, label in enumerate(tags)
        ],
        "document_relations": [],
    }


def test_a_single_label_task_saves_its_one_tag_as_the_label():
    label = label_from_assignment("classification", _assignment(tags=["Void"]), "text", ["Void", "Valid"])
    assert label == "Void"


@pytest.mark.parametrize(
    "tags, reason",
    [([], "Choose a label"), (["Void", "Valid"], "only one label")],
)
def test_a_single_label_task_needs_exactly_one_tag(tags, reason):
    with pytest.raises(AssignmentError, match=reason):
        label_from_assignment("classification", _assignment(tags=tags), "text", ["Void", "Valid"])


def test_a_label_the_dataset_does_not_name_is_refused():
    with pytest.raises(AssignmentError, match="Maybe"):
        label_from_assignment("classification", _assignment(tags=["Maybe"]), "text", ["Void", "Valid"])


def test_a_dataset_with_typed_labels_takes_any_label():
    assert label_from_assignment("judgment_prediction", _assignment(tags=["Upheld"]), "text", []) == "Upheld"


def test_a_blank_label_is_refused():
    with pytest.raises(AssignmentError, match="blank"):
        label_from_assignment("classification", _assignment(tags=["  "]), "text", [])


TEXT = "Smith sued Acme in Paris."


def _span(start, end, label, id_=-1):
    return {
        "id": id_,
        "start": start,
        "end": end,
        "label": label,
        "text": TEXT[start:end],
        "relations": [],
        "confidence": 3,
        "metadata": "note",
    }


def test_an_ner_task_saves_its_spans_as_the_entity_list_in_text_order():
    assignment = _assignment(spans=[_span(11, 15, "ORG", -2), _span(0, 5, "PERSON", -1)])
    label = label_from_assignment("ner", assignment, TEXT, ["PERSON", "ORG"])
    assert json.loads(label) == [
        {"text": "Smith", "start": 0, "end": 5, "type": "PERSON"},
        {"text": "Acme", "start": 11, "end": 15, "type": "ORG"},
    ]


def test_an_ner_task_may_save_no_entities():
    assert json.loads(label_from_assignment("named_entity_recognition", _assignment(), TEXT, [])) == []


@pytest.mark.parametrize(
    "spans, reason",
    [
        ([_span(0, 5, "PERSON"), _span(3, 9, "ORG", -2)], "overlap"),
        ([{**_span(0, 5, "PERSON"), "end": 99}], "outside the text"),
        ([{**_span(0, 5, "PERSON"), "start": 5}], "outside the text"),
        ([_span(0, 5, "PLACE")], "PLACE"),
    ],
)
def test_ner_spans_must_sit_in_the_text_apart_with_known_labels(spans, reason):
    with pytest.raises(AssignmentError, match=reason):
        label_from_assignment("ner", _assignment(spans=spans), TEXT, ["PERSON", "ORG"])


def test_a_task_type_with_no_item_labels_is_refused():
    with pytest.raises(AssignmentError, match="summarization"):
        label_from_assignment("summarization", _assignment(tags=["x"]), TEXT, [])


@pytest.mark.parametrize(
    "assignment",
    [None, "Void", {}, {"document_annotations": "Void"}, {"document_annotations": [{"name": "Void"}]}],
)
def test_a_malformed_assignment_is_refused_not_crashed_on(assignment):
    with pytest.raises(AssignmentError):
        label_from_assignment("classification", assignment, "text", [])


@pytest.mark.parametrize(
    "spans",
    [
        [{"start": 0}],
        ["Smith"],
        [_span(0, 5, "PERSON"), {**_span(11, 15, "ORG", -2), "start": "11"}],
        [{**_span(0, 5, "PERSON"), "start": True}],
    ],
)
def test_malformed_spans_are_refused_not_crashed_on(spans):
    with pytest.raises(AssignmentError):
        label_from_assignment("ner", {**_assignment(), "annotations": spans}, TEXT, [])


def _task(label=None, status="pending", payload=None, text=TEXT):
    return SimpleNamespace(
        id="task-1", item_index=4, text=text, annotation_label=label, status=status, annotation_payload=payload
    )


def _dataset(task_type="classification"):
    return SimpleNamespace(owner_id="owner-1", task_type=task_type)


def test_an_unlabelled_task_opens_as_an_empty_pending_assignment():
    assert assignment_from_task(_task(), _dataset()) == {
        "id": "task-1",
        "annotator": "owner-1",
        "order": 5,
        "status": "pending",
        "confidence": 0,
        "annotations": [],
        "document_annotations": [],
        "document_relations": [],
    }


def test_a_task_labelled_before_the_kit_opens_with_its_label_as_the_tag():
    assignment = assignment_from_task(_task(label="Void", status="annotated"), _dataset())
    assert assignment["status"] == "done"
    assert assignment["document_annotations"] == [{"id": 1, "label": "Void", "confidence": 0, "metadata": None}]
    assert label_from_assignment("classification", assignment, TEXT, []) == "Void"


def test_an_ner_task_labelled_before_the_kit_opens_with_its_entities_as_spans():
    entities = [{"text": "Smith", "start": 0, "end": 5, "type": "PERSON"}]
    assignment = assignment_from_task(_task(label=json.dumps(entities), status="annotated"), _dataset("ner"))
    assert assignment["annotations"] == [
        {"id": 1, "start": 0, "end": 5, "label": "PERSON", "text": "Smith", "relations": [], "confidence": 0, "metadata": None}
    ]
    assert json.loads(label_from_assignment("ner", assignment, TEXT, [])) == entities


def test_a_stored_payload_opens_as_saved_with_the_task_status():
    saved = _assignment(tags=["Void"])
    saved["document_annotations"][0]["metadata"] = "kept"
    opened = assignment_from_task(_task(label="Void", status="annotated", payload=saved), _dataset())
    assert opened["document_annotations"][0]["metadata"] == "kept"
    assert opened["id"] == "task-1" and opened["status"] == "done"


def test_a_skipped_task_opens_as_pending():
    assert assignment_from_task(_task(status="skipped"), _dataset())["status"] == "pending"

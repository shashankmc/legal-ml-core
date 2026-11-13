"""The stored annotation is the kit's Assignment; the label training reads is
derived from it here, and only here (BAIGLE ADR 0007).

`vue-legal-annotation-kit` hands back an Assignment: document tags, spans, and
whatever else the annotator added. Training, active learning and the LLM code
read ``AnnotationTask.annotation_label``, so each save turns the Assignment into
that label, and a task saved before the kit (no payload) is turned back into an
Assignment when it is opened.
"""

import json

SINGLE_LABEL_TYPES = {"classification", "document_classification", "judgment_prediction"}
SPAN_TYPES = {"ner", "named_entity_recognition"}


def annotation_level(task_type):
    """The kit's ``annotation_level`` for a task type: whole-item tags for the
    single-label types, word-snapped spans for NER; None when not annotated."""
    if task_type in SINGLE_LABEL_TYPES:
        return "document"
    if task_type in SPAN_TYPES:
        return "word"
    return None


def offered_labels(dataset, tasks):
    """The labels an item can be given: those the Dataset names, else (typed
    labels) every single label saved among ``tasks``, in their order."""
    if dataset.labels:
        return list(dataset.labels)
    if dataset.task_type not in SINGLE_LABEL_TYPES:
        return []
    return list(dict.fromkeys(t.annotation_label for t in tasks if t.status == "annotated" and t.annotation_label))


class AssignmentError(ValueError):
    """The Assignment cannot be saved as it is; the message says why."""


def label_from_assignment(task_type, assignment, text, allowed_labels):
    """The ``annotation_label`` an Assignment saves as, for a task of ``task_type``
    over ``text``. ``allowed_labels`` empty means the Dataset's labels are typed."""
    _check_shape(assignment)
    if task_type in SINGLE_LABEL_TYPES:
        return _single_label(assignment, allowed_labels)
    if task_type in SPAN_TYPES:
        return _entities(assignment, text, allowed_labels)
    raise AssignmentError(f"A {task_type} Dataset is not annotated item by item.")


def _check_label(label, allowed_labels):
    if not isinstance(label, str) or not label.strip():
        raise AssignmentError("A label cannot be blank.")
    if allowed_labels and label not in allowed_labels:
        raise AssignmentError(f"{label!r} is not one of this Dataset's labels.")
    return label


def _single_label(assignment, allowed_labels):
    tags = assignment["document_annotations"]
    if not tags:
        raise AssignmentError("Choose a label for this item.")
    if len(tags) > 1:
        raise AssignmentError("Choose only one label for this item.")
    return _check_label(tags[0]["label"], allowed_labels)


def _entities(assignment, text, allowed_labels):
    """The spans as the entity list NER Datasets store: ``[{text, start, end,
    type}]`` in text order, the format the annotation screen saved before the kit."""
    spans = assignment["annotations"]
    for span in spans:
        start, end = span["start"], span["end"]
        if not (_is_offset(start) and _is_offset(end) and 0 <= start < end <= len(text)):
            raise AssignmentError(f"A span from {start!r} to {end!r} is outside the text.")
    entities = []
    for span in sorted(spans, key=lambda a: (a["start"], a["end"])):
        start, end = span["start"], span["end"]
        if entities and start < entities[-1]["end"]:
            raise AssignmentError("Entities may not overlap.")
        entities.append(
            {"text": text[start:end], "start": start, "end": end, "type": _check_label(span["label"], allowed_labels)}
        )
    return json.dumps(entities)


def _is_offset(value):
    return isinstance(value, int) and not isinstance(value, bool)


def rename_tag(payload, old, new):
    """``payload`` with document tags called ``old`` called ``new``, all else kept."""
    if not payload:
        return payload
    tags = [{**t, "label": new} if t.get("label") == old else t for t in payload.get("document_annotations", [])]
    return {**payload, "document_annotations": tags}


def _check_shape(assignment):
    """Enough of the Assignment's shape that the rules above can read it."""
    if not isinstance(assignment, dict):
        raise AssignmentError("Send the assignment as an object.")
    for key, fields in (("document_annotations", ("label",)), ("annotations", ("start", "end", "label"))):
        items = assignment.get(key)
        if not isinstance(items, list) or not all(
            isinstance(item, dict) and all(field in item for field in fields) for item in items
        ):
            raise AssignmentError(f"{key} must be a list of objects with {', '.join(fields)}.")


def assignment_from_task(task, dataset):
    """The Assignment to open ``task`` with: its stored payload, or one built from
    ``annotation_label`` for a task saved before the kit. Its id and status always
    come from the task, so they never drift from the queue."""
    status = "done" if task.status == "annotated" else "pending"
    if task.annotation_payload:
        return {**task.annotation_payload, "id": task.id, "status": status}

    tags, spans = [], []
    if task.annotation_label and dataset.task_type in SINGLE_LABEL_TYPES:
        tags = [{"id": 1, "label": task.annotation_label, "confidence": 0, "metadata": None}]
    elif task.annotation_label and dataset.task_type in SPAN_TYPES:
        spans = [
            {
                "id": i + 1,
                "start": e["start"],
                "end": e["end"],
                "label": e["type"],
                "text": task.text[e["start"]:e["end"]],
                "relations": [],
                "confidence": 0,
                "metadata": None,
            }
            for i, e in enumerate(_stored_entities(task.annotation_label))
        ]
    return {
        "id": task.id,
        "annotator": dataset.owner_id,
        "order": task.item_index + 1,
        "status": status,
        "confidence": 0,
        "annotations": spans,
        "document_annotations": tags,
        "document_relations": [],
    }


def _stored_entities(label):
    """The entity list an NER label holds; anything unreadable opens as none."""
    try:
        entities = json.loads(label)
    except (TypeError, ValueError):
        return []
    if not isinstance(entities, list):
        return []
    return [
        e for e in entities
        if isinstance(e, dict) and isinstance(e.get("start"), int) and isinstance(e.get("end"), int) and e.get("type")
    ]

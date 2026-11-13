"""A Dataset's annotations as vue-legal-annotation-kit TaskData, and back (BAIGLE #138).

Kit JSON is the exchange format between the kit's hosts: a Dataset's tasks
export as one document each, with an assignment for every labelled one; a kit
file imports as a Dataset whose items keep their Assignments, so relations,
confidence and metadata survive the round trip.
"""

from dataclasses import dataclass

from .annotation_format import (
    AssignmentError,
    annotation_level,
    assignment_from_task,
    label_from_assignment,
    offered_labels,
)

SPAN_LEVELS = {"character", "word", "sentence", "paragraph"}

# Label colours for the export's labelset, in label order. The kit needs one per
# label; a host that themes labels its own way replaces them.
PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
    "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5",
]


class ExchangeError(ValueError):
    """The kit file cannot become a Dataset; the message names what is wrong."""


def task_data_from_dataset(dataset, tasks):
    """``dataset``'s annotation tasks as kit TaskData: one document per task, in
    item order, with one assignment for each labelled task."""
    names = offered_labels(dataset, sorted(tasks, key=lambda t: t.item_index))
    documents = []
    for task in sorted(tasks, key=lambda t: t.item_index):
        assignments = [assignment_from_task(task, dataset)] if task.status == "annotated" else []
        documents.append({"name": f"item {task.item_index + 1}", "full_text": task.text, "assignments": assignments})

    assignments = [a for d in documents for a in d["assignments"]]
    return {
        "name": dataset.name,
        "desc": dataset.description or "",
        "labelset": {
            "name": dataset.name,
            "desc": "",
            "labels": [{"name": n, "color": PALETTE[i % len(PALETTE)]} for i, n in enumerate(names)],
        },
        "ann_guidelines": "",
        "annotation_level": annotation_level(dataset.task_type),
        "documents": documents,
        "counts": {
            "documents": len(documents),
            "assignments": len(assignments),
            "annotators": len({str(a["annotator"]) for a in assignments}),
            "annotations": sum(len(a["annotations"]) + len(a["document_annotations"]) for a in assignments),
            "relations": sum(
                len(a["document_relations"]) + sum(len(s.get("relations", [])) for s in a["annotations"])
                for a in assignments
            ),
        },
    }


@dataclass
class ImportPlan:
    """What a kit file becomes: the task type, its labels, and per document its
    text and the Assignment kept for it (None when nobody labelled it)."""

    task_type: str
    labels: list
    items: list
    set_aside: int


def _work(assignment):
    return len(assignment.get("annotations") or []) + len(assignment.get("document_annotations") or [])


def _chosen(assignments):
    """The assignment a document keeps: finished before pending, then the one
    with most annotations; and how many others with work were set aside."""
    with_work = [a for a in assignments if _work(a)]
    if not with_work:
        return None, 0
    best = max(with_work, key=lambda a: (a.get("status") == "done", _work(a)))
    return best, len(with_work) - 1


def _check_file(data):
    if not isinstance(data, dict):
        raise ExchangeError("The file must hold one kit task (a JSON object).")
    documents = data.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ExchangeError("documents must be a list with at least one document.")
    level = data.get("annotation_level")
    if level != "document" and level not in SPAN_LEVELS:
        raise ExchangeError(f"annotation_level must be document or one of {', '.join(sorted(SPAN_LEVELS))}.")
    labels = (data.get("labelset") or {}).get("labels")
    if not isinstance(labels, list):
        raise ExchangeError("labelset.labels must be a list.")
    for i, label in enumerate(labels):
        if not isinstance(label, dict) or not isinstance(label.get("name"), str) or not label["name"].strip():
            raise ExchangeError(f"labelset.labels[{i}].name must be text.")
    for i, document in enumerate(documents):
        if not isinstance(document, dict) or not isinstance(document.get("full_text"), str):
            raise ExchangeError(f"documents[{i}].full_text must be text.")
        if not document["full_text"].strip():
            raise ExchangeError(f"documents[{i}].full_text is empty; every document needs text.")
        assignments = document.get("assignments", [])
        if not isinstance(assignments, list):
            raise ExchangeError(f"documents[{i}].assignments must be a list.")
        for j, assignment in enumerate(assignments):
            if not isinstance(assignment, dict):
                raise ExchangeError(f"documents[{i}].assignments[{j}] must be an object.")


def rows_from_task_data(data):
    """Plan the Dataset a kit file becomes. Document-level tasks become
    classification, span tasks NER; every kept Assignment is checked by the same
    rules as a save, so a bad one is refused naming its document."""
    _check_file(data)
    task_type = "classification" if data["annotation_level"] == "document" else "ner"
    labels = [label["name"] for label in data["labelset"]["labels"]]

    items, set_aside = [], 0
    for i, document in enumerate(data["documents"]):
        assignment, others = _chosen(document.get("assignments", []))
        set_aside += others
        if assignment is not None:
            if task_type == "classification" and len(assignment.get("document_annotations") or []) > 1:
                raise ExchangeError(
                    f"documents[{i}] has several tags; each document needs one tag, "
                    "as legal-ml-core has no multi-label Datasets."
                )
            try:
                label_from_assignment(task_type, assignment, document["full_text"], labels)
            except AssignmentError as e:
                raise ExchangeError(f"documents[{i}]: {e}") from None
        items.append((document["full_text"], assignment))
    return ImportPlan(task_type=task_type, labels=labels, items=items, set_aside=set_aside)

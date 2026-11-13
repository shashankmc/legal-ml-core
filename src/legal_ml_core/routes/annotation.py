"""Annotation API routes for managing annotation tasks."""
from flask import Blueprint, request, jsonify
from ..auth import jwt_required, get_jwt_identity
from datetime import datetime
import json
from ..db import db
from ..models import Dataset, AnnotationTask
from ..active_learning.loop import reset_loops
from ..annotation_queue import ensure_project_tasks
from ..annotation_format import (
    SINGLE_LABEL_TYPES,
    AssignmentError,
    annotation_level,
    assignment_from_task,
    label_from_assignment,
)
from ..db_utils import project_belongs_to_owner

# url_prefix is supplied by init_ml_feature at registration time, alongside the
# other five blueprints, rather than baked in here.
annotation_bp = Blueprint("annotation", __name__)


def _get_owned_dataset(dataset_id, owner_id):
    """The dataset with ``dataset_id`` owned by ``owner_id``, else ``None``.

    Not-found and not-yours are deliberately indistinguishable to the caller.
    """
    return Dataset.query.filter_by(id=dataset_id, owner_id=str(owner_id)).first()


def _owned_tasks(project_id, owner_id):
    """Query over annotation tasks of the caller's own datasets on ``project_id``.

    ``project_id`` alone is not an ownership check: another owner's dataset can
    reference the same project id, so the dataset owner is filtered too.
    """
    return AnnotationTask.query.join(Dataset).filter(
        Dataset.project_id == project_id, Dataset.owner_id == str(owner_id)
    )


@annotation_bp.route("/projects/<project_id>/queue/next", methods=["GET"])
@jwt_required()
def get_next_annotation_task(project_id):
    owner_id = get_jwt_identity()

    if not project_belongs_to_owner(db.session, project_id, owner_id):
        return jsonify({"success": False, "error": "Project not found"}), 404

    dataset_id = request.args.get("dataset")
    if dataset_id and not _get_owned_dataset(dataset_id, owner_id):
        return jsonify({"success": False, "error": "Dataset not found"}), 404

    ensure_project_tasks(project_id, owner_id)

    tasks = _owned_tasks(project_id, owner_id)
    if dataset_id:
        tasks = tasks.filter(AnnotationTask.dataset_id == dataset_id)

    # The holdout is never in the active learning queue. It is labelled only on
    # request, to grade the model, and rows skipped before the draw count: nothing
    # else reaches them. Otherwise scored items come first, best first, and rows
    # no loop has scored stay in row order.
    holdout_mode = request.args.get("holdout") in ("1", "true")
    statuses = ("pending", "skipped") if holdout_mode else ("pending",)
    candidates = tasks.filter(
        AnnotationTask.status.in_(statuses), AnnotationTask.is_holdout.is_(holdout_mode)
    )
    task = candidates.order_by(AnnotationTask.score.desc().nulls_last(), AnnotationTask.item_index).first()
    total_tasks = tasks.count()
    annotated_tasks = tasks.filter(AnnotationTask.status == "annotated").count()

    percentage = int((annotated_tasks / total_tasks * 100) if total_tasks > 0 else 0)
    stats = {
        "total": total_tasks,
        "annotated": annotated_tasks,
        "percentage": percentage,
    }
    holdout_remaining = tasks.filter(
        AnnotationTask.is_holdout.is_(True), AnnotationTask.status != "annotated"
    ).count()
    if holdout_remaining:
        stats["holdout_remaining"] = holdout_remaining

    if not task:
        message = "All items annotated" if total_tasks else "Nothing to annotate"
        return jsonify({"success": True, "task": None, "stats": stats, "message": message}), 200

    return jsonify({
        "success": True,
        "task": {
            "id": task.id,
            "dataset_id": task.dataset_id,
            "item_index": task.item_index,
            "text": task.text,
            "status": task.status,
        },
        "stats": stats,
    }), 200


@annotation_bp.route("/projects/<project_id>/queue/stats", methods=["GET"])
@jwt_required()
def get_annotation_queue_stats(project_id):
    owner_id = get_jwt_identity()

    if not project_belongs_to_owner(db.session, project_id, owner_id):
        return jsonify({"success": False, "error": "Project not found"}), 404

    ensure_project_tasks(project_id, owner_id)

    tasks = _owned_tasks(project_id, owner_id)
    total_tasks = tasks.count()
    pending_tasks = tasks.filter(AnnotationTask.status == "pending").count()
    annotated_tasks = tasks.filter(AnnotationTask.status == "annotated").count()
    skipped_tasks = tasks.filter(AnnotationTask.status == "skipped").count()

    return jsonify({
        "success": True,
        "stats": {
            "total": total_tasks,
            "pending": pending_tasks,
            "annotated": annotated_tasks,
            "skipped": skipped_tasks,
            "percentage_complete": int((annotated_tasks / total_tasks * 100) if total_tasks > 0 else 0),
        },
    }), 200


@annotation_bp.route("/tasks/<task_id>/save", methods=["POST"])
@jwt_required()
def save_annotation(task_id):
    owner_id = get_jwt_identity()
    data = request.get_json()

    task = AnnotationTask.query.get(task_id)
    if not task:
        return jsonify({"success": False, "error": "Task not found"}), 404

    dataset = Dataset.query.get(task.dataset_id)
    if not dataset:
        return jsonify({"success": False, "error": "Dataset not found for this task"}), 404

    if str(dataset.owner_id) != str(owner_id):
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    try:
        annotation_label = data.get("annotation_label")
        if annotation_label is None:
            return jsonify({"success": False, "error": "annotation_label is required"}), 400

        if isinstance(annotation_label, str) and not annotation_label.strip():
            return jsonify({"success": False, "error": "annotation_label must not be empty"}), 400

        if isinstance(annotation_label, (dict, list)):
            task.annotation_label = json.dumps(annotation_label)
        else:
            task.annotation_label = str(annotation_label)
        # A plain label replaces any Assignment; it is rebuilt from the label.
        task.annotation_payload = None

        task.status = "annotated"
        task.label_source = "human"
        task.updated_at = datetime.utcnow()
        dataset.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({"success": True, "message": "Annotation saved successfully", "task": task.to_dict()}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@annotation_bp.route("/tasks/<task_id>/skip", methods=["POST"])
@jwt_required()
def skip_annotation(task_id):
    owner_id = get_jwt_identity()

    task = AnnotationTask.query.get(task_id)
    if not task:
        return jsonify({"success": False, "error": "Task not found"}), 404

    dataset = Dataset.query.get(task.dataset_id)
    if not dataset or str(dataset.owner_id) != str(owner_id):
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    try:
        task.status = "skipped"
        task.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({"success": True, "message": "Task skipped successfully", "task": task.to_dict()}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


@annotation_bp.route("/tasks/<task_id>", methods=["GET"])
@jwt_required()
def get_annotation_task(task_id):
    owner_id = get_jwt_identity()

    task = AnnotationTask.query.get(task_id)
    if not task:
        return jsonify({"success": False, "error": "Task not found"}), 404

    dataset = Dataset.query.get(task.dataset_id)
    if not dataset or str(dataset.owner_id) != str(owner_id):
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    return jsonify({"success": True, "task": task.to_dict()}), 200


def _owned_task(task_id, owner_id):
    """``(task, dataset, error_response)`` for a task of the caller's own Dataset."""
    task = AnnotationTask.query.get(task_id)
    if not task:
        return None, None, (jsonify({"success": False, "error": "Task not found"}), 404)
    dataset = Dataset.query.get(task.dataset_id)
    if not dataset or str(dataset.owner_id) != str(owner_id):
        return None, None, (jsonify({"success": False, "error": "Unauthorized"}), 403)
    return task, dataset, None


def _offered_labels(dataset):
    """The labels an item can be given: those the Dataset names, else (typed
    labels) every single label saved on it so far, in the order first saved."""
    if dataset.labels:
        return list(dataset.labels)
    if dataset.task_type not in SINGLE_LABEL_TYPES:
        return []
    rows = (
        db.session.query(AnnotationTask.annotation_label)
        .filter(AnnotationTask.dataset_id == dataset.id, AnnotationTask.status == "annotated")
        .order_by(AnnotationTask.updated_at)
        .all()
    )
    return list(dict.fromkeys(label for (label,) in rows if label))


@annotation_bp.route("/tasks/<task_id>/assignment", methods=["GET"])
@jwt_required()
def get_assignment(task_id):
    """The task as the kit opens it: the item's text, its Assignment, the labels
    it can be given and the annotation level."""
    task, dataset, error = _owned_task(task_id, get_jwt_identity())
    if error:
        return error
    return jsonify({
        "success": True,
        "document": {"name": f"{dataset.name}, item {task.item_index + 1}", "full_text": task.text},
        "assignment": assignment_from_task(task, dataset),
        "labels": _offered_labels(dataset),
        "annotation_level": annotation_level(dataset.task_type),
    }), 200


@annotation_bp.route("/tasks/<task_id>/assignment", methods=["PUT"])
@jwt_required()
def save_assignment(task_id):
    """Save the kit's Assignment: stored as sent, with the label training reads
    derived from it. A broken rule is a 400 that says which."""
    task, dataset, error = _owned_task(task_id, get_jwt_identity())
    if error:
        return error
    assignment = request.get_json(silent=True)
    try:
        label = label_from_assignment(
            dataset.task_type, assignment, task.text, list(dataset.labels or [])
        )
    except AssignmentError as e:
        return jsonify({"success": False, "error": str(e)}), 400

    task.annotation_payload = {**assignment, "id": task.id, "status": "done"}
    task.annotation_label = label
    task.status = "annotated"
    task.label_source = "human"
    task.updated_at = datetime.utcnow()
    dataset.updated_at = datetime.utcnow()
    db.session.commit()
    return jsonify({"success": True, "task": task.to_dict()}), 200


@annotation_bp.route("/projects/<project_id>/tasks", methods=["GET"])
@jwt_required()
def list_annotation_tasks(project_id):
    owner_id = get_jwt_identity()

    if not project_belongs_to_owner(db.session, project_id, owner_id):
        return jsonify({"success": False, "error": "Project not found"}), 404

    ensure_project_tasks(project_id, owner_id)

    status = request.args.get("status")
    dataset_id = request.args.get("dataset_id")

    query = _owned_tasks(project_id, owner_id).filter(AnnotationTask.is_holdout.is_(False))
    if status:
        query = query.filter(AnnotationTask.status == status)
    if dataset_id:
        query = query.filter(AnnotationTask.dataset_id == dataset_id)

    tasks = query.all()
    return jsonify({"success": True, "tasks": [task.to_dict() for task in tasks], "count": len(tasks)}), 200


@annotation_bp.route("/projects/<project_id>/tasks/reset", methods=["POST"])
@jwt_required()
def reset_annotation_tasks(project_id):
    owner_id = get_jwt_identity()

    if not project_belongs_to_owner(db.session, project_id, owner_id):
        return jsonify({"success": False, "error": "Project not found"}), 404

    try:
        tasks = _owned_tasks(project_id, owner_id).all()
        for task in tasks:
            task.status = "pending"
            task.annotation_label = None
            task.annotation_payload = None
            task.label_source = None
            task.updated_at = datetime.utcnow()

        reset_loops(tasks)
        db.session.commit()
        return jsonify({"success": True, "message": f"Reset {len(tasks)} annotation tasks", "reset_count": len(tasks)}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500

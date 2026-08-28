"""Annotation API routes for managing annotation tasks."""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import json
from ..db import db
from ..models import Dataset, AnnotationTask
from ..db_utils import project_belongs_to_user

# url_prefix is supplied by init_ml_feature at registration time, alongside the
# other five blueprints, rather than baked in here.
annotation_bp = Blueprint("annotation", __name__)


@annotation_bp.route("/projects/<project_id>/queue/next", methods=["GET"])
@jwt_required()
def get_next_annotation_task(project_id):
    user_id = get_jwt_identity()

    if not project_belongs_to_user(db.session, project_id, user_id):
        return jsonify({"success": False, "error": "Project not found"}), 404

    query = AnnotationTask.query.join(Dataset).filter(
        Dataset.project_id == project_id, AnnotationTask.status == "pending"
    )

    dataset_id = request.args.get("dataset")
    if dataset_id:
        query = query.filter(Dataset.id == dataset_id)

    task = query.first()

    if dataset_id:
        total_tasks = AnnotationTask.query.filter(
            AnnotationTask.dataset_id == dataset_id
        ).count()
        annotated_tasks = AnnotationTask.query.filter(
            AnnotationTask.dataset_id == dataset_id,
            AnnotationTask.status == "annotated",
        ).count()
    else:
        total_tasks = (
            AnnotationTask.query.join(Dataset)
            .filter(Dataset.project_id == project_id)
            .count()
        )
        annotated_tasks = (
            AnnotationTask.query.join(Dataset)
            .filter(
                Dataset.project_id == project_id, AnnotationTask.status == "annotated"
            )
            .count()
        )

    percentage = int((annotated_tasks / total_tasks * 100) if total_tasks > 0 else 0)
    stats = {
        "total": total_tasks,
        "annotated": annotated_tasks,
        "percentage": percentage,
    }

    if not task:
        return jsonify({"success": True, "task": None, "stats": stats, "message": "All items annotated"}), 200

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
    user_id = get_jwt_identity()

    if not project_belongs_to_user(db.session, project_id, user_id):
        return jsonify({"success": False, "error": "Project not found"}), 404

    total_tasks = (
        AnnotationTask.query.join(Dataset)
        .filter(Dataset.project_id == project_id)
        .count()
    )
    pending_tasks = (
        AnnotationTask.query.join(Dataset)
        .filter(Dataset.project_id == project_id, AnnotationTask.status == "pending")
        .count()
    )
    annotated_tasks = (
        AnnotationTask.query.join(Dataset)
        .filter(Dataset.project_id == project_id, AnnotationTask.status == "annotated")
        .count()
    )
    skipped_tasks = (
        AnnotationTask.query.join(Dataset)
        .filter(Dataset.project_id == project_id, AnnotationTask.status == "skipped")
        .count()
    )

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
    user_id = get_jwt_identity()
    data = request.get_json()

    task = AnnotationTask.query.get(task_id)
    if not task:
        return jsonify({"success": False, "error": "Task not found"}), 404

    dataset = Dataset.query.get(task.dataset_id)
    if not dataset:
        return jsonify({"success": False, "error": "Dataset not found for this task"}), 404

    if str(dataset.user_id) != str(user_id):
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    try:
        annotation_label = data.get("annotation_label")
        if annotation_label is None:
            return jsonify({"success": False, "error": "annotation_label is required"}), 400

        if isinstance(annotation_label, (dict, list)):
            task.annotation_label = json.dumps(annotation_label)
        else:
            task.annotation_label = str(annotation_label)

        task.status = "annotated"
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
    user_id = get_jwt_identity()

    task = AnnotationTask.query.get(task_id)
    if not task:
        return jsonify({"success": False, "error": "Task not found"}), 404

    dataset = Dataset.query.get(task.dataset_id)
    if not dataset or str(dataset.user_id) != str(user_id):
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
    user_id = get_jwt_identity()

    task = AnnotationTask.query.get(task_id)
    if not task:
        return jsonify({"success": False, "error": "Task not found"}), 404

    dataset = Dataset.query.get(task.dataset_id)
    if not dataset or str(dataset.user_id) != str(user_id):
        return jsonify({"success": False, "error": "Unauthorized"}), 403

    return jsonify({"success": True, "task": task.to_dict()}), 200


@annotation_bp.route("/projects/<project_id>/tasks", methods=["GET"])
@jwt_required()
def list_annotation_tasks(project_id):
    user_id = get_jwt_identity()

    if not project_belongs_to_user(db.session, project_id, user_id):
        return jsonify({"success": False, "error": "Project not found"}), 404

    status = request.args.get("status")
    dataset_id = request.args.get("dataset_id")

    query = AnnotationTask.query.join(Dataset).filter(Dataset.project_id == project_id)
    if status:
        query = query.filter(AnnotationTask.status == status)
    if dataset_id:
        query = query.filter(AnnotationTask.dataset_id == dataset_id)

    tasks = query.all()
    return jsonify({"success": True, "tasks": [task.to_dict() for task in tasks], "count": len(tasks)}), 200


@annotation_bp.route("/projects/<project_id>/tasks/reset", methods=["POST"])
@jwt_required()
def reset_annotation_tasks(project_id):
    user_id = get_jwt_identity()

    if not project_belongs_to_user(db.session, project_id, user_id):
        return jsonify({"success": False, "error": "Project not found"}), 404

    try:
        tasks = (
            AnnotationTask.query.join(Dataset)
            .filter(Dataset.project_id == project_id)
            .all()
        )
        for task in tasks:
            task.status = "pending"
            task.annotation_label = None
            task.updated_at = datetime.utcnow()

        db.session.commit()
        return jsonify({"success": True, "message": f"Reset {len(tasks)} annotation tasks", "reset_count": len(tasks)}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500

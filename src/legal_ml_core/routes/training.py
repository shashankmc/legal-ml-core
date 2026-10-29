from flask import Blueprint, request, jsonify, current_app
from ..auth import jwt_required, get_jwt_identity
from ..db import db
from ..models import Model, Dataset, TrainingJob, TrainingMetricPoint
from ..job_state import (
    ACTIVE_JOB_STATUSES,
    DEFAULT_STALE_SECONDS,
    TERMINAL_JOB_STATUSES,
    sweep_stale_job,
    transition,
    waiting_message,
    worker_is_alive,
)
from ..queue_contract import JOB_CANCEL_TTL_SECONDS, job_cancel_key, job_status_key
from .models import validate_model_task_compatibility
import json
import uuid
from datetime import datetime

training_bp = Blueprint("training", __name__)



def _stale_after():
    return current_app.config.get("LEGAL_ML_HEARTBEAT_STALE_SECONDS", DEFAULT_STALE_SECONDS)


def _liveness_view(jobs):
    """R5/D12 on read: sweep dead running jobs to failed, then return
    {job_id: waiting message} for queued jobs when no worker is alive."""
    redis_client = current_app.redis_client
    stale_after = _stale_after()
    for job in jobs:
        sweep_stale_job(job, redis_client, stale_after)
    if not any(job.status == "pending" for job in jobs):
        return {}
    alive = worker_is_alive(redis_client, stale_after)
    return {job.id: waiting_message(job, alive) for job in jobs}


def _job_dict(job, messages):
    payload = job.to_dict()
    payload["status_message"] = messages.get(job.id)
    return payload


@training_bp.route("/start", methods=["POST"])
@jwt_required()
def start_training():
    try:
        owner_id = get_jwt_identity()
        data = request.get_json()

        if "model_id" not in data:
            return jsonify({"error": "model_id is required"}), 400

        model_id = data["model_id"]

        # Get model and verify ownership
        model = Model.query.filter_by(id=model_id, owner_id=owner_id).first()
        if not model:
            return jsonify({"error": "Model not found"}), 404

        # D9: one active job per model, so a double-clicked Train is a no-op.
        active = TrainingJob.query.filter(
            TrainingJob.model_id == model_id, TrainingJob.status.in_(ACTIVE_JOB_STATUSES)
        ).first()
        if active:
            return jsonify({"error": "This model is already training", "job_id": active.id}), 409

        # Get dataset
        dataset = Dataset.query.get(model.dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Reject incompatible model/dataset pairings with an explanation before
        # queueing a job that would otherwise fail silently.
        ok, compat_error = validate_model_task_compatibility(
            model.model_type, dataset.task_type
        )
        if not ok:
            return jsonify({"error": compat_error}), 400

        # Create training job
        training_job = TrainingJob(owner_id=owner_id, model_id=model_id, status="pending")

        db.session.add(training_job)

        # Update model status
        model.status = "pending"

        db.session.commit()

        # Prepare training task data
        task_data = {
            "job_id": training_job.id,
            "model_id": model_id,
            "dataset_id": dataset.id,
            "model_type": model.model_type,
            "parameters": model.parameters,
            "dataset_path": dataset.file_path,
            "task_type": dataset.task_type,
            "preprocessing_options": data.get("preprocessing_options", {}),
            "test_size": data.get("test_size", 0.2),
            "random_state": data.get("random_state", 42),
        }

        # Add task to Redis queue
        current_app.redis_client.lpush(current_app.config.get("LEGAL_ML_TRAINING_QUEUE", "training_queue"), json.dumps(task_data))

        return (
            jsonify(
                {
                    "message": "Training started",
                    "job_id": training_job.id,
                    "model_id": model_id,
                }
            ),
            200,
        )

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@training_bp.route("/status/<job_id>", methods=["GET"])
@jwt_required()
def get_training_status(job_id):
    try:
        owner_id = get_jwt_identity()
        job = TrainingJob.query.filter_by(id=job_id, owner_id=owner_id).first()

        if not job:
            return jsonify({"error": "Training job not found"}), 404

        job_payload = _job_dict(job, _liveness_view([job]))
        # The worker only writes fine-grained progress to Redis (the DB row's
        # progress is set to 100 once, at the end). Merge the live value so the
        # UI stepper actually advances while training runs.
        #
        # Precedence (A2): a terminal DB state (completed/failed/cancelled) is
        # final and Redis is not consulted at all; Redis only refines a
        # non-terminal (pending/running) job.
        try:
            raw = (
                None
                if job.status in TERMINAL_JOB_STATUSES
                else current_app.redis_client.get(job_status_key(job_id))
            )
            if raw:
                live = json.loads(raw)
                if live.get("status"):
                    job_payload["status"] = live["status"]
                if live.get("progress") is not None:
                    job_payload["progress"] = live["progress"]
                if live.get("error"):
                    job_payload["error_message"] = live["error"]
        except Exception:
            # Live status is best-effort; the DB row still answers the request.
            pass

        return (
            jsonify(
                {
                    "job": job_payload,
                    "model": job.model.to_dict() if job.model else None,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/<job_id>/metrics", methods=["GET"])
@jwt_required()
def get_training_metrics(job_id):
    """Live/historical metric points for a training run (experiment monitoring).

    Polled by the frontend on an interval while a job is running. Real-valued
    per-step curves only exist for the HuggingFace Trainer path (LegalBERT/
    RoBERTa); classical sklearn algorithms only log coarse phase markers —
    see legal_ml_core.compute.tasks.MetricLogger.
    """
    try:
        owner_id = get_jwt_identity()
        job = TrainingJob.query.filter_by(id=job_id, owner_id=owner_id).first()

        if not job:
            return jsonify({"error": "Training job not found"}), 404

        since_step = request.args.get("since_step", default=-1, type=int)

        points = (
            TrainingMetricPoint.query.filter(
                TrainingMetricPoint.training_job_id == job_id,
                TrainingMetricPoint.step > since_step,
            )
            .order_by(TrainingMetricPoint.step.asc())
            .all()
        )

        return jsonify({"points": [p.to_dict() for p in points]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/jobs", methods=["GET"])
@jwt_required()
def get_training_jobs():
    try:
        owner_id = get_jwt_identity()
        model_id = request.args.get("model_id")

        query = TrainingJob.query.filter_by(owner_id=owner_id)
        if model_id:
            query = query.filter_by(model_id=model_id)

        jobs = query.order_by(TrainingJob.created_at.desc()).limit(50).all()

        messages = _liveness_view(jobs)
        return jsonify({"jobs": [_job_dict(job, messages) for job in jobs]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/stop/<job_id>", methods=["POST"])
@jwt_required()
def stop_training(job_id):
    try:
        owner_id = get_jwt_identity()
        job = TrainingJob.query.filter_by(id=job_id, owner_id=owner_id).first()

        if not job:
            return jsonify({"error": "Training job not found"}), 404

        if job.status not in ["pending", "running"]:
            return jsonify({"error": "Training job is not active"}), 400

        # Guarded write: if the worker's final write landed between the check
        # above and here, the job stays completed and Stop is rejected.
        if not transition(
            job_id,
            ("pending", "running"),
            "cancelled",
            model_status="failed",
            reason="stop requested",
            completed_at=datetime.utcnow(),
        ):
            return jsonify({"error": "Training job is not active"}), 400

        # Cooperative cancel (D4/OV-E4): the worker polls this flag at pickup,
        # between stages, and inside HF training.
        current_app.redis_client.setex(
            job_cancel_key(job_id), JOB_CANCEL_TTL_SECONDS, "1"
        )

        return jsonify({"message": "Training stopped successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@training_bp.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    try:
        owner_id = get_jwt_identity()
        data = request.get_json()

        if "model_id" not in data or "text" not in data:
            return jsonify({"error": "model_id and text are required"}), 400

        model_id = data["model_id"]
        text = data["text"]

        # Get model and verify ownership
        model = Model.query.filter_by(id=model_id, owner_id=owner_id).first()
        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.status != "completed":
            return jsonify({"error": "Model is not ready for prediction"}), 400

        # Add prediction task to queue
        prediction_data = {"model_id": model_id, "text": text, "owner_id": owner_id}

        prediction_id = str(uuid.uuid4())
        redis_client = current_app.redis_client
        redis_client.setex(
            f"prediction_{prediction_id}", 300, json.dumps(prediction_data)
        )  # 5 min TTL
        redis_client.lpush(
            current_app.config.get("LEGAL_ML_PREDICTION_QUEUE", "prediction_queue"),
            json.dumps({"prediction_id": prediction_id, **prediction_data}),
        )

        return (
            jsonify({"message": "Prediction queued", "prediction_id": prediction_id}),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/prediction/<prediction_id>", methods=["GET"])
@jwt_required()
def get_prediction(prediction_id):
    try:
        owner_id = get_jwt_identity()

        # Get prediction result from Redis
        result_key = f"prediction_result_{prediction_id}"
        result = current_app.redis_client.get(result_key)

        if not result:
            return jsonify({"error": "Prediction not found or expired"}), 404

        result_data = json.loads(result)

        # Verify user ownership
        if result_data.get("owner_id") != owner_id:
            return jsonify({"error": "Unauthorized"}), 403

        return jsonify({"prediction": result_data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/batch-predict", methods=["POST"])
@jwt_required()
def batch_predict():
    try:
        owner_id = get_jwt_identity()
        data = request.get_json()

        if "model_id" not in data or "texts" not in data:
            return jsonify({"error": "model_id and texts are required"}), 400

        model_id = data["model_id"]
        texts = data["texts"]

        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "texts must be a non-empty list"}), 400

        if len(texts) > 1000:
            return (
                jsonify({"error": "Maximum 1000 texts allowed for batch prediction"}),
                400,
            )

        # Get model and verify ownership
        model = Model.query.filter_by(id=model_id, owner_id=owner_id).first()
        if not model:
            return jsonify({"error": "Model not found"}), 404

        if model.status != "completed":
            return jsonify({"error": "Model is not ready for prediction"}), 400

        # Add batch prediction task to queue
        batch_prediction_data = {
            "model_id": model_id,
            "texts": texts,
            "owner_id": owner_id,
        }

        batch_id = str(uuid.uuid4())
        redis_client = current_app.redis_client
        redis_client.setex(
            f"batch_prediction_{batch_id}", 1800, json.dumps(batch_prediction_data)
        )  # 30 min TTL
        redis_client.lpush(
            current_app.config.get("LEGAL_ML_BATCH_PREDICTION_QUEUE", "batch_prediction_queue"),
            json.dumps({"batch_id": batch_id, **batch_prediction_data}),
        )

        return (
            jsonify({"message": "Batch prediction queued", "batch_id": batch_id}),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/batch-prediction/<batch_id>", methods=["GET"])
@jwt_required()
def get_batch_prediction(batch_id):
    try:
        owner_id = get_jwt_identity()

        # Get batch prediction result from Redis
        result_key = f"batch_prediction_result_{batch_id}"
        result = current_app.redis_client.get(result_key)

        if not result:
            return jsonify({"error": "Batch prediction not found or expired"}), 404

        result_data = json.loads(result)

        # Verify user ownership
        if result_data.get("owner_id") != owner_id:
            return jsonify({"error": "Unauthorized"}), 403

        return jsonify({"batch_prediction": result_data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

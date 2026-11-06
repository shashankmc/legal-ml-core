"""Active learning API: start a dataset's loop, read its state, retrain."""
import json
import uuid

from flask import Blueprint, current_app, jsonify, request

from ..active_learning.loop import (
    LoopError,
    llm_seed_sample,
    llm_seed_texts,
    loop_state,
    rename_label,
    request_retrain,
    request_verification,
    set_strategy,
    settings,
    start_loop,
)
from ..auth import get_jwt_identity, jwt_required
from ..db import db
from ..models import Dataset
from ..queue_contract import LLM_JOB_STATUS_TTL_SECONDS, llm_job_status_key, llm_seed_job_key

active_learning_bp = Blueprint("active_learning", __name__)


def _owned_dataset(dataset_id):
    return Dataset.query.filter_by(id=dataset_id, owner_id=str(get_jwt_identity())).first()


def _not_found():
    return jsonify({"success": False, "error": "Dataset not found"}), 404


def _refused(error):
    db.session.rollback()
    return jsonify({"success": False, "error": error.message}), error.status


@active_learning_bp.route("/settings", methods=["GET"])
@jwt_required()
def get_settings():
    return jsonify({"success": True, **settings()}), 200


@active_learning_bp.route("/datasets/<dataset_id>", methods=["GET"])
@jwt_required()
def get_state(dataset_id):
    dataset = _owned_dataset(dataset_id)
    if not dataset:
        return _not_found()
    return jsonify({"success": True, **loop_state(dataset)}), 200


@active_learning_bp.route("/datasets/<dataset_id>/start", methods=["POST"])
@jwt_required()
def start(dataset_id):
    from .llm import ApiError

    dataset = _owned_dataset(dataset_id)
    if not dataset:
        return _not_found()
    options = request.get_json(silent=True) or {}
    try:
        credential = _llm_seed_credential(dataset, options) if options.get("seed_source") == "llm" else None
        start_loop(dataset, options)
        seed_job = _llm_seed_job(dataset, credential) if credential else None
    except LoopError as error:
        return _refused(error)
    except ApiError as error:
        db.session.rollback()
        return jsonify({"success": False, **error.body}), error.status
    db.session.commit()
    if seed_job:
        _queue_llm_seed(dataset, seed_job)
    return jsonify({"success": True, **loop_state(dataset)}), 201


@active_learning_bp.route("/datasets/<dataset_id>/seed-estimate", methods=["POST"])
@jwt_required()
def seed_estimate(dataset_id):
    """What an LLM-suggested first batch would send and cost, before starting."""
    from types import SimpleNamespace

    from ..llm.suggest import estimate_job_tokens, remaining_tokens
    from .llm import ApiError, _credential, _labelable_dataset

    dataset = _owned_dataset(dataset_id)
    if not dataset:
        return _not_found()
    try:
        _labelable_dataset(dataset.id)
        credential = _credential()
        texts = llm_seed_texts(dataset, request.get_json(silent=True) or {})
    except LoopError as error:
        return _refused(error)
    except ApiError as error:
        return jsonify({"success": False, **error.body}), error.status
    estimated = estimate_job_tokens(dataset, [SimpleNamespace(text=text) for text in texts])
    remaining = remaining_tokens(str(get_jwt_identity())) if credential.mode == "shared" else None
    return jsonify(
        {
            "tasks": len(texts),
            "estimated_tokens": estimated,
            "remaining_tokens": remaining,
            "within_budget": remaining is None or estimated <= remaining,
            "destination": credential.destination,
        }
    ), 200


def _llm_seed_credential(dataset, options):
    """Everything an LLM seed needs, checked before the loop starts."""
    from .llm import ApiError, _labelable_dataset, _ready_credential

    if options.get("confirm_estimate") is not True:
        raise ApiError(400, "Review the estimate, then send confirm_estimate: true.")
    _labelable_dataset(dataset.id)
    return _ready_credential(dataset)


def _llm_seed_job(dataset, credential):
    """The worker message for the seed, refused when over the monthly limit."""
    from .llm import ApiError, _estimate

    # start_loop adds the config without the dataset seeing it yet.
    db.session.flush()
    db.session.refresh(dataset)
    tasks, estimate = _estimate(dataset, credential, [task.id for task in llm_seed_sample(dataset)])
    if not estimate["within_budget"]:
        raise ApiError(
            429,
            "Choosing the first batch is larger than your remaining monthly limit.",
            code="budget_exhausted",
            byok_available=True,
            estimate=estimate,
        )
    return {
        "job_id": str(uuid.uuid4()),
        "dataset_id": dataset.id,
        "owner_id": str(get_jwt_identity()),
        "mode": credential.mode,
        "task_ids": [task.id for task in tasks],
        "seed_batch_size": dataset.al_config.batch_size,
    }


def _queue_llm_seed(dataset, job):
    redis_client = current_app.redis_client
    redis_client.setex(
        llm_job_status_key(job["job_id"]),
        LLM_JOB_STATUS_TTL_SECONDS,
        json.dumps({"owner_id": job["owner_id"], "status": "queued", "total": len(job["task_ids"]), "done": 0}),
    )
    redis_client.setex(llm_seed_job_key(dataset.id), LLM_JOB_STATUS_TTL_SECONDS, job["job_id"])
    redis_client.lpush("llm_queue", json.dumps(job))


@active_learning_bp.route("/datasets/<dataset_id>/strategy", methods=["PUT"])
@jwt_required()
def change_strategy(dataset_id):
    dataset = _owned_dataset(dataset_id)
    if not dataset:
        return _not_found()
    try:
        set_strategy(dataset, (request.get_json(silent=True) or {}).get("strategy"))
    except LoopError as error:
        return _refused(error)
    db.session.commit()
    return jsonify({"success": True, **loop_state(dataset)}), 200


@active_learning_bp.route("/datasets/<dataset_id>/labels", methods=["PUT"])
@jwt_required()
def correct_label(dataset_id):
    dataset = _owned_dataset(dataset_id)
    if not dataset:
        return _not_found()
    body = request.get_json(silent=True) or {}
    try:
        rename_label(dataset, body.get("from"), body.get("to"))
    except LoopError as error:
        return _refused(error)
    db.session.commit()
    return jsonify({"success": True, **loop_state(dataset)}), 200


@active_learning_bp.route("/datasets/<dataset_id>/retrain", methods=["POST"])
@jwt_required()
def retrain(dataset_id):
    dataset = _owned_dataset(dataset_id)
    if not dataset:
        return _not_found()
    try:
        body, status = request_retrain(dataset, str(get_jwt_identity()))
    except LoopError as error:
        return _refused(error)
    return jsonify({"success": status < 400, **body}), status


@active_learning_bp.route("/datasets/<dataset_id>/verify", methods=["POST"])
@jwt_required()
def verify(dataset_id):
    dataset = _owned_dataset(dataset_id)
    if not dataset:
        return _not_found()
    try:
        request_verification(dataset)
    except LoopError as error:
        return _refused(error)
    return jsonify({"success": True, "message": "Checking the score"}), 202

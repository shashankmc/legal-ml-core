"""Active learning API: start a dataset's loop, read its state, retrain."""
from flask import Blueprint, jsonify, request

from ..active_learning.loop import (
    LoopError,
    loop_state,
    request_retrain,
    request_verification,
    set_strategy,
    settings,
    start_loop,
)
from ..auth import get_jwt_identity, jwt_required
from ..db import db
from ..models import Dataset

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
    dataset = _owned_dataset(dataset_id)
    if not dataset:
        return _not_found()
    try:
        start_loop(dataset, request.get_json(silent=True) or {})
    except LoopError as error:
        return _refused(error)
    db.session.commit()
    return jsonify({"success": True, **loop_state(dataset)}), 201


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

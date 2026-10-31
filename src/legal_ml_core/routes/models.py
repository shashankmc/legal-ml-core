from flask import Blueprint, request, jsonify, current_app
from ..auth import jwt_required, get_jwt_identity
from ..db import db
from ..active_learning.loop import training_refusal
from ..models import ALRound, Model, Dataset, TrainingJob
from ..db_utils import project_belongs_to_owner
from ..job_state import (
    ACTIVE_JOB_STATUSES,
    COULD_NOT_START_MESSAGE,
    DEFAULT_STALE_SECONDS,
    sweep_lost_job,
    transition,
)
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

models_bp = Blueprint("models", __name__)

# Rare labels count for more in training. Multinomial naive Bayes has no such option.
CLASS_WEIGHT_PARAMETER = {
    "type": "select",
    "options": ["none", "balanced"],
    "default": "none",
    "description": "Give rare labels more weight in training",
}

# Available model types and their parameters: what the wizard offers and what
# parameter_refusal accepts, read from the same table. A rule limits `parameter`
# to `options` when the values in its `when` are chosen, and `reason` says why.
AVAILABLE_MODELS = {
    "classification": {
        "svm": {
            "name": "Support Vector Machine",
            "description": "Linear and non-linear classification using support vectors",
            "parameters": {
                "kernel": {
                    "type": "select",
                    "options": ["linear", "rbf", "poly", "sigmoid"],
                    "default": "rbf",
                    "description": "Kernel type",
                },
                "C": {
                    "type": "float",
                    "min": 0.001,
                    "max": 100.0,
                    "default": 1.0,
                    "description": "Regularisation parameter",
                },
                "gamma": {
                    "type": "select",
                    "options": ["scale", "auto"],
                    "default": "scale",
                    "description": "Kernel coefficient",
                },
                "degree": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "default": 3,
                    "description": "Degree for polynomial kernel",
                },
                "class_weight": CLASS_WEIGHT_PARAMETER,
            },
        },
        "decision_tree": {
            "name": "Decision Tree",
            "description": "Tree-based classification with interpretable rules",
            "parameters": {
                "max_depth": {
                    "type": "int",
                    "min": 1,
                    "max": 50,
                    "default": None,
                    "description": "Maximum tree depth",
                },
                "min_samples_split": {
                    "type": "int",
                    "min": 2,
                    "max": 100,
                    "default": 2,
                    "description": "Minimum samples to split",
                },
                "min_samples_leaf": {
                    "type": "int",
                    "min": 1,
                    "max": 50,
                    "default": 1,
                    "description": "Minimum samples per leaf",
                },
                "criterion": {
                    "type": "select",
                    "options": ["gini", "entropy"],
                    "default": "gini",
                    "description": "Split criterion",
                },
                "class_weight": CLASS_WEIGHT_PARAMETER,
            },
        },
        "random_forest": {
            "name": "Random Forest",
            "description": "Ensemble of decision trees for robust classification",
            "parameters": {
                "n_estimators": {
                    "type": "int",
                    "min": 1,
                    "max": 1000,
                    "default": 100,
                    "description": "Number of trees",
                },
                "max_depth": {
                    "type": "int",
                    "min": 1,
                    "max": 50,
                    "default": None,
                    "description": "Maximum tree depth",
                },
                "min_samples_split": {
                    "type": "int",
                    "min": 2,
                    "max": 100,
                    "default": 2,
                    "description": "Minimum samples to split",
                },
                "min_samples_leaf": {
                    "type": "int",
                    "min": 1,
                    "max": 50,
                    "default": 1,
                    "description": "Minimum samples per leaf",
                },
                "class_weight": CLASS_WEIGHT_PARAMETER,
            },
        },
        "naive_bayes": {
            "name": "Naive Bayes",
            "description": "Probabilistic classifier based on Bayes theorem",
            "parameters": {
                "alpha": {
                    "type": "float",
                    "min": 0.01,
                    "max": 10.0,
                    "default": 1.0,
                    "description": "Smoothing parameter",
                },
                "fit_prior": {
                    "type": "boolean",
                    "default": True,
                    "description": "Learn class prior probabilities",
                },
            },
        },
        "logistic_regression": {
            "name": "Logistic Regression",
            "description": "Linear classification with probabilistic output",
            "parameters": {
                "C": {
                    "type": "float",
                    "min": 0.001,
                    "max": 100.0,
                    "default": 1.0,
                    "description": "Inverse regularisation strength",
                },
                "penalty": {
                    "type": "select",
                    "options": ["l2", "l1", "elasticnet", "none"],
                    "default": "l2",
                    "description": "Regularization penalty",
                },
                "solver": {
                    "type": "select",
                    "options": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
                    "default": "lbfgs",
                    "description": "Optimisation algorithm",
                },
                "l1_ratio": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.5,
                    "description": "Share of l1 in the elasticnet penalty",
                },
                "class_weight": CLASS_WEIGHT_PARAMETER,
            },
            "rules": [
                {
                    "when": {"penalty": "l1"},
                    "parameter": "solver",
                    "options": ["liblinear", "saga"],
                    "reason": "The l1 penalty needs the liblinear or saga solver.",
                },
                {
                    "when": {"penalty": "elasticnet"},
                    "parameter": "solver",
                    "options": ["saga"],
                    "reason": "The elasticnet penalty needs the saga solver.",
                },
                {
                    "when": {"penalty": "none"},
                    "parameter": "solver",
                    "options": ["lbfgs", "newton-cg", "sag", "saga"],
                    "reason": "Training with no penalty cannot use the liblinear solver.",
                },
            ],
        },
    },
    "clustering": {
        "kmeans": {
            "name": "K-Means",
            "description": "Partition-based clustering algorithm",
            "parameters": {
                "n_clusters": {
                    "type": "int",
                    "min": 2,
                    "max": 20,
                    "default": 8,
                    "description": "Number of clusters",
                },
                "init": {
                    "type": "select",
                    "options": ["k-means++", "random"],
                    "default": "k-means++",
                    "description": "Initialisation method",
                },
                "max_iter": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "default": 300,
                    "description": "Maximum iterations",
                },
                "random_state": {
                    "type": "int",
                    "min": 0,
                    "max": 1000,
                    "default": 42,
                    "description": "Random seed",
                },
            },
        },
        "hierarchical": {
            "name": "Hierarchical Clustering",
            "description": "Tree-based clustering with different linkage methods",
            "parameters": {
                "n_clusters": {
                    "type": "int",
                    "min": 2,
                    "max": 20,
                    "default": 8,
                    "description": "Number of clusters",
                },
                "linkage": {
                    "type": "select",
                    "options": ["ward", "complete", "average", "single"],
                    "default": "ward",
                    "description": "Linkage criterion",
                },
                "affinity": {
                    "type": "select",
                    "options": ["euclidean", "l1", "l2", "manhattan", "cosine"],
                    "default": "euclidean",
                    "description": "Distance metric",
                },
            },
            "rules": [
                {
                    "when": {"linkage": "ward"},
                    "parameter": "affinity",
                    "options": ["euclidean", "l2"],
                    "reason": "Ward linkage needs euclidean or l2 distance.",
                },
            ],
        },
        "dbscan": {
            "name": "DBSCAN",
            "description": "Density-based clustering with noise detection",
            "parameters": {
                "eps": {
                    "type": "float",
                    "min": 0.01,
                    "max": 10.0,
                    "default": 0.5,
                    "description": "Maximum distance between samples",
                },
                "min_samples": {
                    "type": "int",
                    "min": 1,
                    "max": 100,
                    "default": 5,
                    "description": "Minimum samples in neighbourhood",
                },
                "metric": {
                    "type": "select",
                    "options": ["euclidean", "manhattan", "cosine"],
                    "default": "euclidean",
                    "description": "Distance metric",
                },
            },
        },
    },
    "topic_modeling": {
        "lda": {
            "name": "Latent Dirichlet Allocation",
            "description": "Probabilistic topic modelling using LDA",
            "parameters": {
                "n_components": {
                    "type": "int",
                    "min": 2,
                    "max": 50,
                    "default": 10,
                    "description": "Number of topics",
                },
                "max_iter": {
                    "type": "int",
                    "min": 10,
                    "max": 1000,
                    "default": 100,
                    "description": "Maximum iterations",
                },
                "learning_decay": {
                    "type": "float",
                    "min": 0.5,
                    "max": 1.0,
                    "default": 0.7,
                    "description": "Learning rate decay",
                },
                "random_state": {
                    "type": "int",
                    "min": 0,
                    "max": 1000,
                    "default": 42,
                    "description": "Random seed",
                },
            },
        },
        "nmf": {
            "name": "Non-negative Matrix Factorisation",
            "description": "Topic modelling using NMF decomposition",
            "parameters": {
                "n_components": {
                    "type": "int",
                    "min": 2,
                    "max": 50,
                    "default": 10,
                    "description": "Number of topics",
                },
                "max_iter": {
                    "type": "int",
                    "min": 100,
                    "max": 2000,
                    "default": 1000,
                    "description": "Maximum iterations",
                },
                "alpha": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.0,
                    "description": "Regularisation strength",
                },
                "random_state": {
                    "type": "int",
                    "min": 0,
                    "max": 1000,
                    "default": 42,
                    "description": "Random seed",
                },
            },
        },
    },
    "transformer": {
        "legalbert": {
            "name": "LegalBERT",
            "description": "BERT model fine-tuned on legal text",
            "parameters": {
                "model_name": {
                    "type": "select",
                    "options": ["nlpaueb/legal-bert-base-uncased"],
                    "default": "nlpaueb/legal-bert-base-uncased",
                    "description": "LegalBERT variant",
                },
                "max_length": {
                    "type": "int",
                    "min": 128,
                    "max": 512,
                    "default": 256,
                    "description": "Maximum sequence length",
                },
                "learning_rate": {
                    "type": "float",
                    "min": 1e-6,
                    "max": 1e-2,
                    "default": 2e-5,
                    "description": "Learning rate",
                },
                "batch_size": {
                    "type": "int",
                    "min": 4,
                    "max": 32,
                    "default": 16,
                    "description": "Training batch size",
                },
                "epochs": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "default": 3,
                    "description": "Number of training epochs",
                },
            },
        },
        "roberta": {
            "name": "RoBERTa",
            "description": "Robustly optimised BERT approach",
            "parameters": {
                "model_name": {
                    "type": "select",
                    "options": ["roberta-base", "roberta-large"],
                    "default": "roberta-base",
                    "description": "RoBERTa model size",
                },
                "max_length": {
                    "type": "int",
                    "min": 128,
                    "max": 512,
                    "default": 256,
                    "description": "Maximum sequence length",
                },
                "learning_rate": {
                    "type": "float",
                    "min": 1e-6,
                    "max": 1e-2,
                    "default": 2e-5,
                    "description": "Learning rate",
                },
                "batch_size": {
                    "type": "int",
                    "min": 4,
                    "max": 32,
                    "default": 16,
                    "description": "Training batch size",
                },
                "epochs": {
                    "type": "int",
                    "min": 1,
                    "max": 10,
                    "default": 3,
                    "description": "Number of training epochs",
                },
            },
        },
    },
}


# Dataset task types whose training is not implemented in the worker yet. Kept
# explicit so uploads/selection can explain the limitation instead of failing
# silently at training time.
UNSUPPORTED_TASK_TYPES = {"ner", "named_entity_recognition", "summarization"}

# Which model categories each dataset task type may use. Transformer models
# (LegalBERT/RoBERTa) are supervised text classifiers, so they are offered for
# every supervised text task.
_TASK_CATEGORIES = {
    "classification": ["classification", "transformer"],
    "document_classification": ["classification", "transformer"],
    "judgment_prediction": ["classification", "transformer"],
    "topic_modeling": ["topic_modeling", "clustering"],
    "clustering": ["clustering", "topic_modeling"],
}


def model_category_for(model_type):
    for category, models in AVAILABLE_MODELS.items():
        if model_type in models:
            return category
    return None


def compatible_models_for_task(task_type):
    """Return {model_type: config} of models that can train this task type."""
    categories = _TASK_CATEGORIES.get(task_type, [])
    result = {}
    for category in categories:
        result.update(AVAILABLE_MODELS.get(category, {}))
    return result


def validate_model_task_compatibility(model_type, task_type):
    """Return (ok, error_message). Explains *why* a pairing is invalid."""
    if task_type in UNSUPPORTED_TASK_TYPES or task_type not in _TASK_CATEGORIES:
        return False, (
            f"Training is not yet supported for '{task_type}' datasets. "
            "Choose a classification, clustering or topic-modelling dataset."
        )
    allowed = compatible_models_for_task(task_type)
    if model_type not in allowed:
        return False, (
            f"'{model_type}' cannot be trained on a '{task_type}' dataset. "
            f"Compatible models: {', '.join(sorted(allowed))}."
        )
    return True, None


def _number_refusal(label, value, field):
    """Why this value does not fit its catalogue entry, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{label} must be a number."
    if field["type"] == "int" and not isinstance(value, int):
        return f"{label} must be a whole number."
    # Written so that NaN, which compares false, is refused too.
    if not field["min"] <= value <= field["max"]:
        return f"{label} must be between {field['min']:g} and {field['max']:g}."
    return None


def _field_refusal(name, value, field):
    label = field.get("description", name)
    # "No limit" (e.g. max_depth) is a choice, and the wizard's default.
    if value is None and field["default"] is None:
        return None
    if field["type"] in ("int", "float"):
        return _number_refusal(label, value, field)
    if field["type"] == "boolean" and not isinstance(value, bool):
        return f"{label} must be true or false."
    if field["type"] == "select" and value not in field["options"]:
        return f'{label} must be one of: {", ".join(field["options"])}.'
    return None


def parameter_refusal(model_type, parameters):
    """Why these parameters cannot train this model type, or None. Everything
    comes from the model's catalogue entry."""
    if not isinstance(parameters, dict):
        return "parameters must be an object"

    info = AVAILABLE_MODELS[model_category_for(model_type)][model_type]
    fields = info["parameters"]
    for name, value in parameters.items():
        if name in fields:
            refusal = _field_refusal(name, value, fields[name])
            if refusal:
                return refusal

    chosen = {name: field["default"] for name, field in fields.items()} | parameters

    for rule in info.get("rules", []):
        holds = all(chosen.get(name) == value for name, value in rule["when"].items())
        if holds and chosen.get(rule["parameter"]) not in rule["options"]:
            return rule["reason"]
    return None


@models_bp.route("/available", methods=["GET"])
@jwt_required()
def get_available_models():
    try:
        task_type = request.args.get("task_type")

        if task_type:
            if task_type in UNSUPPORTED_TASK_TYPES:
                # A valid task with no trainable models yet — answer clearly
                # rather than 400 ("Invalid task type").
                return (
                    jsonify(
                        {
                            "models": {},
                            "message": (
                                f"Training for '{task_type}' datasets is not available yet."
                            ),
                        }
                    ),
                    200,
                )
            if task_type in AVAILABLE_MODELS and task_type not in _TASK_CATEGORIES:
                # Backwards-compatible: category name used directly.
                return jsonify({"models": AVAILABLE_MODELS[task_type]}), 200
            if task_type in _TASK_CATEGORIES:
                return jsonify({"models": compatible_models_for_task(task_type)}), 200
            return jsonify({"error": "Invalid task type"}), 400
        else:
            return jsonify({"models": AVAILABLE_MODELS}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/", methods=["GET"])
@jwt_required()
def get_models():
    try:
        owner_id = get_jwt_identity()
        dataset_id = request.args.get("dataset_id")

        query = Model.query.filter_by(owner_id=owner_id)
        if dataset_id:
            query = query.filter_by(dataset_id=dataset_id)

        models = query.order_by(Model.created_at.desc()).all()

        return jsonify({"models": [model.to_dict() for model in models]}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>", methods=["GET"])
@jwt_required()
def get_model(model_id):
    try:
        owner_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, owner_id=owner_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        return jsonify({"model": model.to_dict()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/", methods=["POST"])
@jwt_required()
def create_model():
    try:
        owner_id = get_jwt_identity()
        data = request.get_json()

        # Validate required fields
        required_fields = ["name", "dataset_id", "model_type", "parameters"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} is required"}), 400

        # Check if dataset exists and belongs to user
        dataset = Dataset.query.filter_by(
            id=data["dataset_id"], owner_id=owner_id
        ).first()
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Creating a Model starts its training, so say so here, not after.
        refusal = training_refusal(dataset)
        if refusal:
            return jsonify({"error": refusal}), 409

        # Validate model type
        model_type = data["model_type"]

        # Determine which category the model belongs to
        model_category = None
        for category, models in AVAILABLE_MODELS.items():
            if model_type in models:
                model_category = category
                break

        if not model_category:
            return jsonify({"error": "Invalid model type"}), 400

        # Reject incompatible dataset/model pairings at creation time, with an
        # explanation, rather than failing later during training.
        ok, compat_error = validate_model_task_compatibility(
            model_type, dataset.task_type
        )
        if not ok:
            return jsonify({"error": compat_error}), 400

        # Validate parameters
        refusal = parameter_refusal(model_type, data["parameters"])
        if refusal:
            return jsonify({"error": refusal}), 400

        # Resolve project_id: explicit payload value takes precedence over dataset's project.
        # `Project` belongs to the host app, not this package, so existence is checked via
        # a direct query against the `projects` table rather than importing its ORM class.
        explicit_project_id = data.get("project_id")
        if explicit_project_id:
            if not project_belongs_to_owner(db.session, explicit_project_id, owner_id):
                return jsonify({"error": "Project not found"}), 404
            resolved_project_id = explicit_project_id
        else:
            resolved_project_id = dataset.project_id

        # Create model
        model = Model(
            owner_id=owner_id,
            dataset_id=data["dataset_id"],
            project_id=resolved_project_id,
            name=data["name"],
            model_type=model_type,
            parameters=data["parameters"],
        )

        db.session.add(model)
        db.session.commit()

        return (
            jsonify(
                {"message": "Model created successfully", "model": model.to_dict()}
            ),
            201,
        )

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>", methods=["PUT"])
@jwt_required()
def update_model(model_id):
    try:
        owner_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, owner_id=owner_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        data = request.get_json()

        # Update name if provided
        if "name" in data:
            model.name = data["name"].strip()

        # Update parameters if provided
        if "parameters" in data:
            refusal = parameter_refusal(model.model_type, data["parameters"])
            if refusal:
                return jsonify({"error": refusal}), 400
            model.parameters = data["parameters"]

        db.session.commit()

        return (
            jsonify(
                {"message": "Model updated successfully", "model": model.to_dict()}
            ),
            200,
        )

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/labels", methods=["PUT"])
@jwt_required()
def update_model_labels(model_id):
    """Persist human-readable names for unsupervised topics/clusters.

    Stored under `metrics.display_labels` so no schema change is needed; a
    label map for a topic/cluster id overrides the numeric "Topic 3"/"Cluster 2"
    presentation in the UI.
    """
    try:
        owner_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, owner_id=owner_id).first()
        if not model:
            return jsonify({"error": "Model not found"}), 404

        data = request.get_json() or {}
        metrics = dict(model.metrics or {})
        display = dict(metrics.get("display_labels") or {})

        for key in ("topics", "clusters"):
            incoming = data.get(key)
            if isinstance(incoming, dict):
                merged = dict(display.get(key) or {})
                for label_id, name in incoming.items():
                    name = str(name).strip()
                    if name:
                        merged[str(label_id)] = name
                    else:
                        merged.pop(str(label_id), None)
                display[key] = merged

        metrics["display_labels"] = display
        model.metrics = metrics
        db.session.commit()

        return jsonify({"message": "Labels updated", "model": model.to_dict()}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>", methods=["DELETE"])
@jwt_required()
def delete_model(model_id):
    try:
        owner_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, owner_id=owner_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        # Delete model file if it exists
        if model.file_path and os.path.exists(model.file_path):
            os.remove(model.file_path)

        # Delete model record
        db.session.delete(model)
        db.session.commit()

        return jsonify({"message": "Model deleted successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/download", methods=["GET"])
@jwt_required()
def download_model(model_id):
    try:
        owner_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, owner_id=owner_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if not model.file_path or not os.path.exists(model.file_path):
            return jsonify({"error": "Model file not found"}), 404

        if model.status != "completed":
            return jsonify({"error": "Model is not ready for download"}), 400

        # Return file path for download (handled by frontend)
        return (
            jsonify(
                {
                    "download_url": f"/api/models/{model_id}/file",
                    "filename": f"{model.name}_{model.model_type}.pkl",
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/file", methods=["GET"])
@jwt_required()
def get_model_file(model_id):
    try:
        owner_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, owner_id=owner_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        if not model.file_path or not os.path.exists(model.file_path):
            return jsonify({"error": "Model file not found"}), 404

        from flask import send_file

        return send_file(
            model.file_path,
            as_attachment=True,
            download_name=f"{model.name}_{model.model_type}.pkl",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/explain", methods=["POST"])
@jwt_required()
def explain_model(model_id):
    """
    Generate model explanation using LIME or SHAP
    Body: {
        "text": "text to explain",
        "method": "lime" or "shap"
    }

    Fixed during extraction: the original app registered this at
    "/models/<int:model_id>/explain" under a blueprint already prefixed with
    "/api/models" (giving "/api/models/models/<int:id>/explain") and typed
    model_id as int while every model id is actually a string UUID — so the
    route could never match a real request. Corrected to "/<model_id>/explain"
    to match the rest of this blueprint, since a frontend package meant to
    actually exercise this endpoint needs it to be reachable.
    """
    try:
        owner_id = get_jwt_identity()
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Text is required"}), 400

        text = data["text"]
        method = data.get("method", "lime")  # Default to LIME

        if method not in ["lime", "shap"]:
            return jsonify({"error": 'Method must be "lime" or "shap"'}), 400

        # Get model
        model = Model.query.get(model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404

        # Check if user owns the model or it's public. `is_public` isn't a real
        # column on Model (pre-existing gap in the original app) — default to
        # False via getattr rather than letting this raise AttributeError.
        if model.owner_id != owner_id and not getattr(model, "is_public", False):
            return jsonify({"error": "Unauthorized"}), 403

        # Check if model is trained
        if model.status != "completed":
            return jsonify({"error": "Model is not trained yet"}), 400

        # Generate unique explanation ID
        import uuid

        explanation_id = str(uuid.uuid4())

        # Queue explanation task
        task_data = {
            "explanation_id": explanation_id,
            "model_id": model_id,
            "text": text,
            "method": method,
            "owner_id": owner_id,
        }

        current_app.redis_client.lpush(current_app.config.get("LEGAL_ML_EXPLANATION_QUEUE", "explanation_queue"), json.dumps(task_data))

        return (
            jsonify(
                {
                    "explanation_id": explanation_id,
                    "status": "processing",
                    "message": f"{method.upper()} explanation is being generated",
                }
            ),
            202,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/explanations/<explanation_id>", methods=["GET"])
@jwt_required()
def get_explanation(explanation_id):
    """Get explanation result"""
    try:
        # Get result from Redis
        result = current_app.redis_client.get(f"explanation_{explanation_id}")

        if not result:
            return (
                jsonify(
                    {
                        "status": "processing",
                        "message": "Explanation is still being generated",
                    }
                ),
                202,
            )

        result_data = json.loads(result)

        if result_data.get("status") == "failed":
            return (
                jsonify(
                    {
                        "status": "failed",
                        "error": result_data.get("error", "Unknown error"),
                    }
                ),
                500,
            )

        return jsonify(result_data), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/explain-batch", methods=["POST"])
@jwt_required()
def explain_model_batch(model_id):
    """
    Generate explanations for multiple texts
    Body: {
        "texts": ["text1", "text2", ...],
        "method": "lime" or "shap"
    }

    Same pre-existing routing fix as explain_model above.
    """
    try:
        owner_id = get_jwt_identity()
        data = request.get_json()

        if not data or "texts" not in data:
            return jsonify({"error": "Texts array is required"}), 400

        texts = data["texts"]
        method = data.get("method", "lime")

        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "Texts must be a non-empty array"}), 400

        if len(texts) > 10:
            return jsonify({"error": "Maximum 10 texts per batch"}), 400

        if method not in ["lime", "shap"]:
            return jsonify({"error": 'Method must be "lime" or "shap"'}), 400

        # Get model
        model = Model.query.get(model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404

        # Check if user owns the model or it's public
        if model.owner_id != owner_id and not getattr(model, "is_public", False):
            return jsonify({"error": "Unauthorized"}), 403

        # Check if model is trained
        if model.status != "completed":
            return jsonify({"error": "Model is not trained yet"}), 400

        # Queue explanation tasks for each text
        import uuid

        explanation_ids = []

        for text in texts:
            explanation_id = str(uuid.uuid4())
            explanation_ids.append(explanation_id)

            task_data = {
                "explanation_id": explanation_id,
                "model_id": model_id,
                "text": text,
                "method": method,
                "owner_id": owner_id,
            }

            current_app.redis_client.lpush(current_app.config.get("LEGAL_ML_EXPLANATION_QUEUE", "explanation_queue"), json.dumps(task_data))

        return (
            jsonify(
                {
                    "explanation_ids": explanation_ids,
                    "status": "processing",
                    "message": f"{len(texts)} {method.upper()} explanations are being generated",
                }
            ),
            202,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/<model_id>/predict", methods=["POST"])
@jwt_required()
def predict(model_id):
    """
    Run inference on new text samples.
    Body: { "texts": ["text1", "text2", ...] }
    Delegates to the worker via Redis queue so TextPreprocessor and sklearn objects
    are always deserialized in the correct container.
    """
    try:
        import uuid

        owner_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, owner_id=owner_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404
        if model.status != "completed":
            return jsonify({"error": "Model has not completed training"}), 400

        data = request.get_json()
        if not data or "texts" not in data:
            return jsonify({"error": "texts is required"}), 400

        texts = data["texts"]
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "texts must be a non-empty array"}), 400
        if len(texts) > 50:
            return jsonify({"error": "Maximum 50 texts per request"}), 400

        prediction_id = str(uuid.uuid4())
        task = {
            "prediction_id": prediction_id,
            "model_id": str(model_id),
            "model_type": model.model_type,
            "texts": texts,
            "owner_id": str(owner_id),
        }

        TRANSFORMER_MODELS = {"legalbert", "roberta"}
        timeout = 120 if model.model_type in TRANSFORMER_MODELS else 30

        r = current_app.redis_client
        r.lpush(current_app.config.get("LEGAL_ML_PREDICTION_QUEUE", "prediction_queue"), json.dumps(task))

        result_key = f"prediction_result_{prediction_id}"
        raw = r.blpop(result_key, timeout=timeout)

        if raw is None:
            return jsonify({"error": "Prediction timed out — worker may be busy"}), 504

        result = json.loads(raw[1])

        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        return jsonify(result), 200

    except Exception as e:
        current_app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def training_queue():
    return current_app.config.get("LEGAL_ML_TRAINING_QUEUE", "training_queue")


def stale_after():
    return current_app.config.get("LEGAL_ML_HEARTBEAT_STALE_SECONDS", DEFAULT_STALE_SECONDS)


def _lock_model_and_find_active_job(model_id, owner_id):
    """FOR UPDATE (Postgres; SQLite ignores it) holds the model row until the
    caller commits or rolls back, so two concurrent starts serialise here and
    the second one's active-job check sees the first one's job."""
    while True:
        model = (
            Model.query.filter_by(id=model_id, owner_id=owner_id).with_for_update().first()
        )
        if not model:
            return None, None
        active = TrainingJob.query.filter(
            TrainingJob.model_id == model_id, TrainingJob.status.in_(ACTIVE_JOB_STATUSES)
        ).first()
        # A lost job is not a run to return. Sweeping commits, which releases
        # the lock, so look again.
        if not (
            active
            and sweep_lost_job(active, current_app.redis_client, training_queue(), stale_after())
        ):
            return model, active


MODEL_BUSY_MESSAGE = "This Model is already training. Retrain when it finishes."


def start_training_job(owner_id, model_id, data, al_round_id=None):
    """The one way a training run starts; both start routes call it. Returns
    (response body, status). Starting a model with an active job returns that
    job, and a run that cannot be queued leaves a failed job, never a pending
    one nothing will pick up. ``al_round_id`` ties the run to an active learning
    round, so the worker rescores the pool after it; it never comes from a body.
    A round cannot join a job already running: that job has no round to rescore
    and did not see the round's labels, so the round is refused instead."""
    model, active = _lock_model_and_find_active_job(model_id, owner_id)
    if not model:
        return {"error": "Model not found"}, 404

    # D9: one active job per model, so a double-clicked Train is a no-op.
    if active:
        job_id = str(active.id)
        db.session.rollback()  # release the model row lock
        if al_round_id:
            return {"error": MODEL_BUSY_MESSAGE, "job_id": job_id}, 409
        return (
            {"message": "Training already started", "job_id": job_id, "model_id": str(model_id)},
            200,
        )

    dataset = db.session.get(Dataset, model.dataset_id)
    if not dataset:
        return {"error": "Dataset not found"}, 404

    refusal = training_refusal(dataset)
    if refusal:
        db.session.rollback()  # release the model row lock
        return {"error": refusal}, 409

    # Fail fast, with a reason, if this model cannot train on this dataset.
    ok, compat_error = validate_model_task_compatibility(model.model_type, dataset.task_type)
    if not ok:
        return {"error": compat_error}, 400

    # A Model saved before a combination was refused must not reach the worker.
    refusal = parameter_refusal(model.model_type, model.parameters or {})
    if refusal:
        db.session.rollback()  # release the model row lock
        return {"error": refusal}, 400

    # Create training job
    training_job = TrainingJob(owner_id=owner_id, model_id=model_id, status="pending")
    db.session.add(training_job)
    if al_round_id:
        # With the job, so no crash can leave a running job its round doesn't know.
        db.session.flush()
        ALRound.query.filter_by(id=al_round_id).update({"job_id": training_job.id})

    # A retrain that cannot start must not cost the model its finished state.
    status_before = "failed" if model.status == "training" else model.status
    model.status = "training"

    db.session.commit()
    job_id = str(training_job.id)

    try:
        # ids are uuid.UUID on Postgres; the queue message is JSON.
        task_data = {
            "job_id": job_id,
            "model_id": str(model_id),
            "dataset_id": str(dataset.id),
            "model_type": model.model_type,
            "parameters": model.parameters,
            "dataset_path": dataset.file_path,
            "task_type": dataset.task_type,
            "preprocessing_options": data.get("preprocessing_options", {}),
            "test_size": data.get("test_size", 0.2),
            "random_state": data.get("random_state", 42),
        }
        if al_round_id:
            task_data["al_round_id"] = str(al_round_id)
        current_app.redis_client.lpush(training_queue(), json.dumps(task_data))
    except Exception:
        logger.exception("training job %s (owner %s): could not be queued", job_id, owner_id)
        transition(
            job_id,
            ACTIVE_JOB_STATUSES,
            "failed",
            model_status=status_before,
            reason="could not queue",
            error_message=COULD_NOT_START_MESSAGE,
            completed_at=datetime.utcnow(),
        )
        return {"error": COULD_NOT_START_MESSAGE}, 503

    return (
        {
            "message": "Training started",
            "job_id": job_id,
            "model_id": str(model_id),
            "model": model.to_dict(),
        },
        200,
    )


@models_bp.route("/<model_id>/train", methods=["POST"])
@jwt_required()
def train_model(model_id):
    """Start training a model"""
    try:
        owner_id = get_jwt_identity()
        data = request.get_json() if request.is_json else {}
        body, status = start_training_job(owner_id, model_id, data)
        return jsonify(body), status

    except Exception:
        db.session.rollback()
        logger.exception("could not start training for model %s", model_id)
        return jsonify({"error": COULD_NOT_START_MESSAGE}), 500

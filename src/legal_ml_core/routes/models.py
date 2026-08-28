from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from ..db import db
from ..models import Model, Dataset, TrainingJob
from ..db_utils import project_belongs_to_user
import os
import json

models_bp = Blueprint("models", __name__)

# Available model types and their parameters
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
                    "min": 0.1,
                    "max": 100.0,
                    "default": 1.0,
                    "description": "Regularization parameter",
                },
                "gamma": {
                    "type": "select",
                    "options": ["scale", "auto"],
                    "default": "scale",
                    "description": "Kernel coefficient",
                },
                "degree": {
                    "type": "int",
                    "min": 2,
                    "max": 10,
                    "default": 3,
                    "description": "Degree for polynomial kernel",
                },
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
            },
        },
        "random_forest": {
            "name": "Random Forest",
            "description": "Ensemble of decision trees for robust classification",
            "parameters": {
                "n_estimators": {
                    "type": "int",
                    "min": 10,
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
            },
        },
        "naive_bayes": {
            "name": "Naive Bayes",
            "description": "Probabilistic classifier based on Bayes theorem",
            "parameters": {
                "alpha": {
                    "type": "float",
                    "min": 0.1,
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
                    "min": 0.1,
                    "max": 100.0,
                    "default": 1.0,
                    "description": "Inverse regularization strength",
                },
                "penalty": {
                    "type": "select",
                    "options": ["l1", "l2", "elasticnet"],
                    "default": "l2",
                    "description": "Regularization penalty",
                },
                "solver": {
                    "type": "select",
                    "options": ["liblinear", "lbfgs", "saga"],
                    "default": "lbfgs",
                    "description": "Optimization algorithm",
                },
            },
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
                    "description": "Initialization method",
                },
                "max_iter": {
                    "type": "int",
                    "min": 100,
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
                    "options": ["euclidean", "l1", "l2", "manhattan"],
                    "default": "euclidean",
                    "description": "Distance metric",
                },
            },
        },
        "dbscan": {
            "name": "DBSCAN",
            "description": "Density-based clustering with noise detection",
            "parameters": {
                "eps": {
                    "type": "float",
                    "min": 0.1,
                    "max": 10.0,
                    "default": 0.5,
                    "description": "Maximum distance between samples",
                },
                "min_samples": {
                    "type": "int",
                    "min": 1,
                    "max": 100,
                    "default": 5,
                    "description": "Minimum samples in neighborhood",
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
            "description": "Probabilistic topic modeling using LDA",
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
            "name": "Non-negative Matrix Factorization",
            "description": "Topic modeling using NMF decomposition",
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
                    "description": "L1 regularization",
                },
                "l1_ratio": {
                    "type": "float",
                    "min": 0.0,
                    "max": 1.0,
                    "default": 0.0,
                    "description": "L1/L2 regularization ratio",
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
                    "options": ["nlpaueb/legal-bert-base-uncased", "lex_glue"],
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
                    "min": 8,
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
            "description": "Robustly optimized BERT approach",
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
                    "min": 8,
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


@models_bp.route("/available", methods=["GET"])
@jwt_required()
def get_available_models():
    try:
        task_type = request.args.get("task_type")

        if task_type:
            if task_type in AVAILABLE_MODELS:
                return jsonify({"models": AVAILABLE_MODELS[task_type]}), 200
            else:
                return jsonify({"error": "Invalid task type"}), 400
        else:
            return jsonify({"models": AVAILABLE_MODELS}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/", methods=["GET"])
@jwt_required()
def get_models():
    try:
        user_id = get_jwt_identity()
        dataset_id = request.args.get("dataset_id")

        query = Model.query.filter_by(user_id=user_id)
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
        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        return jsonify({"model": model.to_dict()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@models_bp.route("/", methods=["POST"])
@jwt_required()
def create_model():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()

        # Validate required fields
        required_fields = ["name", "dataset_id", "model_type", "parameters"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"{field} is required"}), 400

        # Check if dataset exists and belongs to user
        dataset = Dataset.query.filter_by(
            id=data["dataset_id"], user_id=user_id
        ).first()
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

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

        # Validate parameters
        if model_type in AVAILABLE_MODELS.get(model_category, {}):
            expected_params = AVAILABLE_MODELS[model_category][model_type]["parameters"]
            provided_params = data["parameters"]

            # Validate parameter types and ranges
            for param_name, param_value in provided_params.items():
                if param_name in expected_params:
                    param_config = expected_params[param_name]
                    param_type = param_config["type"]

                    if param_type == "int" and not isinstance(param_value, int):
                        return (
                            jsonify(
                                {"error": f"Parameter {param_name} must be an integer"}
                            ),
                            400,
                        )
                    elif param_type == "float" and not isinstance(
                        param_value, (int, float)
                    ):
                        return (
                            jsonify(
                                {"error": f"Parameter {param_name} must be a number"}
                            ),
                            400,
                        )
                    elif param_type == "boolean" and not isinstance(param_value, bool):
                        return (
                            jsonify(
                                {"error": f"Parameter {param_name} must be a boolean"}
                            ),
                            400,
                        )
                    elif (
                        param_type == "select"
                        and param_value not in param_config["options"]
                    ):
                        return (
                            jsonify(
                                {
                                    "error": f'Parameter {param_name} must be one of: {", ".join(param_config["options"])}'
                                }
                            ),
                            400,
                        )

        # Resolve project_id: explicit payload value takes precedence over dataset's project.
        # `Project` belongs to the host app, not this package, so existence is checked via
        # a direct query against the `projects` table rather than importing its ORM class.
        explicit_project_id = data.get("project_id")
        if explicit_project_id:
            if not project_belongs_to_user(db.session, explicit_project_id, user_id):
                return jsonify({"error": "Project not found"}), 404
            resolved_project_id = explicit_project_id
        else:
            resolved_project_id = dataset.project_id

        # Create model
        model = Model(
            user_id=user_id,
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
        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

        if not model:
            return jsonify({"error": "Model not found"}), 404

        data = request.get_json()

        # Update name if provided
        if "name" in data:
            model.name = data["name"].strip()

        # Update parameters if provided
        if "parameters" in data:
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


@models_bp.route("/<model_id>", methods=["DELETE"])
@jwt_required()
def delete_model(model_id):
    try:
        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

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
        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

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
        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

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
        user_id = get_jwt_identity()
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
        if model.user_id != user_id and not getattr(model, "is_public", False):
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
            "user_id": user_id,
        }

        current_app.redis_client.lpush("explanation_queue", json.dumps(task_data))

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
        user_id = get_jwt_identity()
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
        if model.user_id != user_id and not getattr(model, "is_public", False):
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
                "user_id": user_id,
            }

            current_app.redis_client.lpush("explanation_queue", json.dumps(task_data))

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

        user_id = get_jwt_identity()
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()

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
            "user_id": str(user_id),
        }

        TRANSFORMER_MODELS = {"legalbert", "roberta"}
        timeout = 120 if model.model_type in TRANSFORMER_MODELS else 30

        r = current_app.redis_client
        r.lpush("prediction_queue", json.dumps(task))

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


@models_bp.route("/<model_id>/train", methods=["POST"])
@jwt_required()
def train_model(model_id):
    """Start training a model"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json() if request.is_json else {}

        # Get model and verify ownership
        model = Model.query.filter_by(id=model_id, user_id=user_id).first()
        if not model:
            return jsonify({"error": "Model not found"}), 404

        # Check if model is already training
        if model.status == "training":
            return jsonify({"error": "Model is already being trained"}), 400

        # Get dataset
        dataset = Dataset.query.get(model.dataset_id)
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Create training job
        training_job = TrainingJob(user_id=user_id, model_id=model_id, status="pending")

        db.session.add(training_job)

        # Update model status
        model.status = "training"

        db.session.commit()

        # Prepare training task data
        task_data = {
            "job_id": str(training_job.id),
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

        # Try to add task to Redis queue if available
        try:
            current_app.redis_client.lpush("training_queue", json.dumps(task_data))
        except Exception as redis_err:
            # Log Redis error but continue - we still have the job in database
            current_app.logger.warning(
                f"Redis error while queuing training task: {str(redis_err)}"
            )

        return (
            jsonify(
                {
                    "message": "Training started successfully",
                    "job_id": str(training_job.id),
                    "model_id": str(model_id),
                    "model": model.to_dict(),
                }
            ),
            200,
        )

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

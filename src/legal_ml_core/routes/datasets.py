from flask import Blueprint, request, jsonify, current_app
from ..auth import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from ..db import db
from ..models import Dataset
import pandas as pd
import os
import uuid
import json

datasets_bp = Blueprint("datasets", __name__)

ALLOWED_EXTENSIONS = {"csv", "xlsx", "json"}
MAX_ROWS = 10000

# Single source of truth for upload task types. The frontend renders this
# catalogue (via GET /datasets/task-types) so its task guidance cannot drift
# from what the backend actually accepts and validates.
TASK_TYPES = [
    {
        "value": "classification",
        "label": "Text Classification",
        "summary": "Give each short piece of text one label.",
        "when_to_use": (
            "Use when every row is a passage (a clause, sentence or paragraph) "
            "and you want one category for it, e.g. deontic type or sentiment."
        ),
        "label_meaning": (
            "The single category you want predicted for the text (e.g. "
            "Obligation, Permission, Prohibition)."
        ),
        "required_columns": ["text", "label"],
        "example": 'text="Member States shall adopt these laws", label="Obligation"',
    },
    {
        "value": "document_classification",
        "label": "Document Classification",
        "summary": "Give each whole document one label.",
        "when_to_use": (
            "Same mechanics as Text Classification, but each row is an entire "
            "document (a full ruling, contract or judgment) rather than a "
            "passage."
        ),
        "label_meaning": (
            "The single category for the whole document (e.g. case outcome or "
            "subject area)."
        ),
        "required_columns": ["text", "label"],
        "example": 'text="<full court ruling>", label="criminal"',
    },
    {
        "value": "ner",
        "label": "Named Entity Recognition",
        "summary": "Find and label spans of text (people, organisations, dates).",
        "when_to_use": (
            "Use when you need to locate specific spans inside the text — not "
            "one label for the whole text. This is the main difference from "
            "classification: the output marks entities, not the row."
        ),
        "label_meaning": (
            "Entity types per span (e.g. PERSON, ORG, DATE), stored in the "
            "'entities' column."
        ),
        "required_columns": ["text", "entities"],
        "example": 'text="John Smith v. ABC Corp", entities="PERSON, ORG"',
    },
    {
        "value": "named_entity_recognition",
        "label": "Named Entity Recognition (Alt)",
        "summary": "Alias of Named Entity Recognition for imported datasets.",
        "when_to_use": (
            "Identical to Named Entity Recognition; choose it only if your "
            "existing data uses this task name."
        ),
        "label_meaning": "Same as Named Entity Recognition.",
        "required_columns": ["text", "entities"],
        "example": 'text="Legal case text", entities="PERSON, ORG, DATE"',
    },
    {
        "value": "summarization",
        "label": "Text Summarization",
        "summary": "Produce a short summary from a longer text.",
        "when_to_use": (
            "Use when you have long documents and the matching short summary "
            "you want the model to learn to produce."
        ),
        "label_meaning": "The target summary text, stored in the 'summary' column.",
        "required_columns": ["text", "summary"],
        "example": 'text="<long legal document>", summary="<brief summary>"',
    },
    {
        "value": "topic_modeling",
        "label": "Topic Modeling",
        "summary": "Discover recurring themes; no labels needed.",
        "when_to_use": (
            "Unsupervised: use when you have text but no labels and want to "
            "discover themes. You can name the topics after training."
        ),
        "label_meaning": "None — unsupervised, so there is no label column.",
        "required_columns": ["text"],
        "example": 'text="<legal document content>"',
    },
    {
        "value": "clustering",
        "label": "Document Clustering",
        "summary": "Group similar documents; no labels needed.",
        "when_to_use": (
            "Unsupervised: use when you want documents grouped by similarity "
            "rather than by theme keywords. You can name the clusters after "
            "training."
        ),
        "label_meaning": "None — unsupervised, so there is no label column.",
        "required_columns": ["text"],
        "example": 'text="<document content>"',
    },
    {
        "value": "judgment_prediction",
        "label": "Judgment Prediction",
        "summary": "Predict a case outcome from the facts.",
        "when_to_use": (
            "Use when each row is a case's facts and you want to predict the "
            "outcome."
        ),
        "label_meaning": (
            "The case outcome to predict (e.g. guilty / not_guilty), stored in "
            "the 'judgment' column."
        ),
        "required_columns": ["text", "judgment"],
        "example": 'text="<case facts>", judgment="guilty"',
    },
]

VALID_TASK_TYPES = [t["value"] for t in TASK_TYPES]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_dataset_file(file_path, task_type, annotation_mode=False):
    """Validate dataset file and return basic info

    Args:
        file_path: Path to the dataset file
        task_type: Type of task (classification, ner, etc.)
        annotation_mode: If True, label columns are optional. If False, they are required.
    """
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        elif file_path.endswith(".json"):
            df = pd.read_json(file_path)
        else:
            return None, "Unsupported file format"

        # Check row count limit
        if len(df) > MAX_ROWS:
            return None, f"Dataset exceeds maximum allowed rows ({MAX_ROWS})"

        # Enhanced validation based on task type
        columns = [col.lower() for col in df.columns]

        if task_type in ["classification", "document_classification"]:
            # Must have text column
            if "text" not in columns:
                return None, "Classification datasets must have a 'text' column"
            # Label column is required only for training
            if not annotation_mode and "label" not in columns:
                return (
                    None,
                    "Classification datasets for training must have a 'label' column",
                )

        elif task_type in ["ner", "named_entity_recognition"]:
            # Must have text column
            if "text" not in columns:
                return None, "NER datasets must have a 'text' column"
            # Entities column is required only for training
            if not annotation_mode and "entities" not in columns:
                return None, "NER datasets for training must have an 'entities' column"

        elif task_type in ["summarization"]:
            # Should have text column
            if "text" not in columns:
                return None, "Summarization datasets must have a 'text' column"
            # Summary column required only for training
            if not annotation_mode and "summary" not in columns:
                return (
                    None,
                    "Summarization datasets for training must have a 'summary' column",
                )

        elif task_type in ["topic_modeling", "clustering"]:
            # Should have text column (no label needed for these)
            if "text" not in columns:
                return (
                    None,
                    "Topic modeling/clustering datasets must have a 'text' column",
                )

        elif task_type in ["judgment_prediction"]:
            # Should have text column
            if "text" not in columns:
                return None, "Judgment prediction datasets must have a 'text' column"
            # Judgment column required only for training
            if not annotation_mode and "judgment" not in columns:
                return (
                    None,
                    "Judgment prediction datasets for training must have a 'judgment' column",
                )

        return df, None

    except Exception as e:
        return None, f"Error reading file: {str(e)}"


@datasets_bp.route("/", methods=["GET"])
@jwt_required()
def get_datasets():
    try:
        owner_id = get_jwt_identity()
        # Get all datasets for the user that are not already associated with a project
        # (optional: can include all datasets by removing the project_id filter)
        datasets = (
            Dataset.query.filter_by(owner_id=owner_id)
            .order_by(Dataset.created_at.desc())
            .all()
        )

        return (
            jsonify(
                {
                    "success": True,
                    "datasets": [dataset.to_dict() for dataset in datasets],
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@datasets_bp.route("/<dataset_id>", methods=["GET"])
@jwt_required()
def get_dataset(dataset_id):
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()

        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        return jsonify({"dataset": dataset.to_dict()}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/upload", methods=["POST"])
@jwt_required()
def upload_dataset():
    try:
        owner_id = get_jwt_identity()

        # Check if file is present
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return (
                jsonify(
                    {
                        "error": "File type not allowed. Supported formats: CSV, XLSX, JSON"
                    }
                ),
                400,
            )

        # Get form data
        name = request.form.get("name", "").strip()
        description = request.form.get("description", "").strip()
        task_type = request.form.get("task_type", "").strip()
        annotation_mode = request.form.get("annotation_mode", "false").lower() == "true"
        labels_str = request.form.get(
            "labels", ""
        )  # Comma-separated labels or JSON array

        if not name:
            return jsonify({"error": "Dataset name is required"}), 400

        if not task_type:
            return jsonify({"error": "Task type is required"}), 400

        # Parse labels
        labels = []
        if labels_str:
            try:
                # Try to parse as JSON array first
                if labels_str.startswith("["):
                    labels = json.loads(labels_str)
                else:
                    # Otherwise treat as comma-separated values
                    labels = [
                        label.strip()
                        for label in labels_str.split(",")
                        if label.strip()
                    ]
            except (json.JSONDecodeError, ValueError):
                labels = [
                    label.strip() for label in labels_str.split(",") if label.strip()
                ]

        # Validate task type
        if task_type not in VALID_TASK_TYPES:
            return (
                jsonify(
                    {
                        "error": f'Invalid task type. Valid types: {", ".join(VALID_TASK_TYPES)}'
                    }
                ),
                400,
            )

        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(file_path)

        # Validate dataset (pass annotation_mode to validation function)
        df, error = validate_dataset_file(
            file_path, task_type, annotation_mode=annotation_mode
        )
        if error:
            os.remove(file_path)  # Clean up file
            return jsonify({"error": error}), 400

        # Get column information
        columns_info = []
        for col in df.columns:
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "unique_count": int(df[col].nunique()),
            }
            columns_info.append(col_info)

        # Create dataset record
        dataset = Dataset(
            owner_id=owner_id,
            name=name,
            description=description,
            file_path=file_path,
            row_count=len(df),
            columns=columns_info,
            task_type=task_type,
            labels=labels if labels else None,
        )

        db.session.add(dataset)
        db.session.commit()

        return (
            jsonify(
                {
                    "message": "Dataset uploaded successfully",
                    "dataset": dataset.to_dict(),
                }
            ),
            201,
        )

    except Exception as e:
        db.session.rollback()
        if "file_path" in locals() and os.path.exists(file_path):
            os.remove(file_path)  # Clean up file
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/task-types", methods=["GET"])
def get_task_types():
    """Return the upload task-type catalogue (single source of truth).

    Unauthenticated on purpose: it is static guidance used to render the upload
    form, so it can be fetched before the user is signed in. The frontend must
    not hard-code this list.
    """
    return jsonify({"task_types": TASK_TYPES}), 200


@datasets_bp.route("/<dataset_id>/preview", methods=["GET"])
@jwt_required()
def preview_dataset(dataset_id):
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()

        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Read and return preview of dataset
        if dataset.file_path.endswith(".csv"):
            df = pd.read_csv(dataset.file_path)
        elif dataset.file_path.endswith(".xlsx"):
            df = pd.read_excel(dataset.file_path)
        elif dataset.file_path.endswith(".json"):
            df = pd.read_json(dataset.file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        # Return first 10 rows
        preview_data = df.head(10).to_dict("records")

        return (
            jsonify(
                {
                    "preview": preview_data,
                    "total_rows": len(df),
                    "columns": list(df.columns),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/<dataset_id>/rows", methods=["GET"])
@jwt_required()
def paginated_dataset_rows(dataset_id):
    """Paginated full-data browsing (HuggingFace-style dataset viewer).

    `preview_dataset` only returns the first 10 rows; this serves any window of
    the file so the UI can navigate the whole dataset without loading it all.
    """
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()
        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        offset = max(0, request.args.get("offset", default=0, type=int))
        limit = request.args.get("limit", default=50, type=int)
        limit = min(max(1, limit), 200)

        if dataset.file_path.endswith(".csv"):
            df = pd.read_csv(dataset.file_path)
        elif dataset.file_path.endswith(".xlsx"):
            df = pd.read_excel(dataset.file_path)
        elif dataset.file_path.endswith(".json"):
            df = pd.read_json(dataset.file_path)
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        total = len(df)
        window = df.iloc[offset : offset + limit]
        # NaN is not valid JSON; pandas' `to_dict` keeps it as float('nan').
        rows = json.loads(window.to_json(orient="records", date_format="iso"))

        return (
            jsonify(
                {
                    "rows": rows,
                    "columns": list(df.columns),
                    "total_rows": total,
                    "offset": offset,
                    "limit": limit,
                    "has_more": offset + limit < total,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/<dataset_id>/preprocessing-options", methods=["GET"])
@jwt_required()
def get_preprocessing_options(dataset_id):
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()

        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Return available preprocessing options based on task type
        preprocessing_options = {
            "text_cleaning": {
                "remove_html": {
                    "type": "boolean",
                    "default": True,
                    "description": "Remove HTML tags",
                },
                "remove_urls": {
                    "type": "boolean",
                    "default": True,
                    "description": "Remove URLs",
                },
                "remove_emails": {
                    "type": "boolean",
                    "default": True,
                    "description": "Remove email addresses",
                },
                "remove_numbers": {
                    "type": "boolean",
                    "default": False,
                    "description": "Remove numbers",
                },
                "remove_special_chars": {
                    "type": "boolean",
                    "default": False,
                    "description": "Remove special characters",
                },
                "lowercase": {
                    "type": "boolean",
                    "default": True,
                    "description": "Convert to lowercase",
                },
            },
            "tokenization": {
                "method": {
                    "type": "select",
                    "options": ["word", "sentence", "subword"],
                    "default": "word",
                    "description": "Tokenization method",
                },
                "max_length": {
                    "type": "number",
                    "min": 50,
                    "max": 512,
                    "default": 256,
                    "description": "Maximum sequence length",
                },
            },
            "feature_extraction": {
                "method": {
                    "type": "select",
                    "options": ["tfidf", "count", "hash"],
                    "default": "tfidf",
                    "description": "Feature extraction method",
                },
                "max_features": {
                    "type": "number",
                    "min": 100,
                    "max": 10000,
                    "default": 5000,
                    "description": "Maximum number of features",
                },
                "ngram_range": {
                    "type": "range",
                    "min": 1,
                    "max": 3,
                    "default": [1, 2],
                    "description": "N-gram range",
                },
            },
        }

        # Add task-specific options
        if dataset.task_type in ["ner", "named_entity_recognition"]:
            preprocessing_options["ner_specific"] = {
                "entity_types": {
                    "type": "multiselect",
                    "options": ["PERSON", "ORG", "GPE", "LAW", "CASE"],
                    "default": ["PERSON", "ORG", "GPE"],
                    "description": "Entity types to extract",
                },
                "use_spacy": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use spaCy for NER preprocessing",
                },
            }

        return jsonify({"preprocessing_options": preprocessing_options}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/<dataset_id>", methods=["DELETE"])
@jwt_required()
def delete_dataset(dataset_id):
    try:
        owner_id = get_jwt_identity()
        dataset = Dataset.query.filter_by(id=dataset_id, owner_id=owner_id).first()

        if not dataset:
            return jsonify({"error": "Dataset not found"}), 404

        # Delete file if it exists
        if os.path.exists(dataset.file_path):
            os.remove(dataset.file_path)

        # Delete dataset record (cascade will handle related models)
        db.session.delete(dataset)
        db.session.commit()

        return jsonify({"message": "Dataset deleted successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

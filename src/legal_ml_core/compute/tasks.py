import json
import os

import joblib
import pandas as pd

from ..db import db
from ..models import Model, TrainingJob, TrainingMetricPoint
from ..storage import get_model_path, get_transformer_output_dir
from .explain import generate_lime_explanation, generate_shap_explanation
from .trainer import MLModelTrainer


class MetricLogger:
    """Appends TrainingMetricPoint rows for one training job.

    `step` is a per-instance monotonically increasing polling cursor (used only
    by the `/training/<job_id>/metrics?since_step=` endpoint) — it is NOT the
    algorithm's own notion of step/epoch. The semantic x-axis for charts is
    `epoch`, which is populated for the HuggingFace Trainer path (via
    TrainingMetricCallback) and left None for the classical sklearn algorithms'
    coarse phase markers.
    """

    def __init__(self, training_job_id):
        self.training_job_id = training_job_id
        self._counter = 0

    def log(self, metric_name, value, epoch=None):
        step = self._counter
        self._counter += 1
        point = TrainingMetricPoint(
            training_job_id=self.training_job_id,
            step=step,
            epoch=epoch,
            metric_name=metric_name,
            value=float(value),
        )
        db.session.add(point)
        db.session.commit()
        return step


def process_training_task(task_info, redis_client, model_storage_path):
    """Runs one training task. Must be called with a Flask app context pushed
    (db.session/Model.query need it) — see legal_ml_core.worker.run_ml_worker.

    This replaces the original ml_worker's raw `psycopg2` UPDATE statements
    against the `models`/`training_jobs` tables with normal SQLAlchemy writes
    through the same ORM classes the API process uses — the two processes now
    share one schema representation instead of two independent ones.
    """
    job_id = task_info["job_id"]
    model_id = task_info["model_id"]
    metric_logger = MetricLogger(job_id)

    try:
        print(f"Processing training task for model {model_id}")

        redis_client.setex(
            f"job_status_{job_id}", 3600, json.dumps({"status": "running", "progress": 0})
        )
        metric_logger.log("preprocessing_started", 1.0)

        # Load dataset
        df = pd.read_csv(task_info["dataset_path"])

        # Initialize trainer
        trainer = MLModelTrainer(redis_client=redis_client, metric_logger=metric_logger)

        # Preprocess data
        texts, labels, label_names = trainer.preprocess_data(
            df, task_info["task_type"], task_info.get("preprocessing_options", {})
        )

        redis_client.setex(
            f"job_status_{job_id}", 3600, json.dumps({"status": "running", "progress": 25})
        )
        metric_logger.log("preprocessing_done", 1.0)
        metric_logger.log("training_started", 1.0)

        # Train model based on task type
        if task_info["model_type"] in ["legalbert", "roberta"]:
            task_params = dict(task_info)
            task_params["output_dir"] = get_transformer_output_dir(
                model_storage_path, model_id
            )
            metrics = trainer.train_transformer_model(
                texts, labels, task_info["model_type"], task_params
            )
        elif task_info["task_type"] in ["classification", "document_classification"]:
            metrics = trainer.train_classification_model(
                texts, labels, task_info["model_type"], task_info
            )
        elif task_info["task_type"] in ["topic_modeling", "clustering"]:
            if task_info["model_type"] in ["kmeans", "hierarchical", "dbscan"]:
                metrics = trainer.train_clustering_model(
                    texts, task_info["model_type"], task_info
                )
            else:
                metrics = trainer.train_topic_modeling(
                    texts, task_info["model_type"], task_info
                )
        else:
            raise ValueError(f"Unsupported task type: {task_info['task_type']}")

        redis_client.setex(
            f"job_status_{job_id}", 3600, json.dumps({"status": "running", "progress": 75})
        )
        metric_logger.log("training_done", 1.0)

        # Save model
        model_path = get_model_path(model_storage_path, model_id)
        trainer.save_model(model_path)
        metric_logger.log("model_saved", 1.0)

        # Update job status to completed in Redis
        redis_client.setex(
            f"job_status_{job_id}",
            3600,
            json.dumps(
                {
                    "status": "completed",
                    "progress": 100,
                    "metrics": metrics,
                    "model_path": model_path,
                }
            ),
        )

        # Update model + training job in the database via the shared ORM models
        model = Model.query.get(model_id)
        job = TrainingJob.query.get(job_id)
        if model:
            model.status = "completed"
            model.metrics = metrics
            model.file_path = model_path
        if job:
            job.status = "completed"
            job.metrics = metrics
            job.progress = 100
            from datetime import datetime

            job.completed_at = datetime.utcnow()
        db.session.commit()

        print(f"Training completed for model {model_id}")

    except Exception as e:
        db.session.rollback()
        print(f"Error processing training task: {str(e)}")
        redis_client.setex(
            f"job_status_{job_id}", 3600, json.dumps({"status": "failed", "error": str(e)})
        )
        try:
            metric_logger.log("training_failed", 1.0)
        except Exception:
            pass

        model = Model.query.get(model_id)
        job = TrainingJob.query.get(job_id)
        if model:
            model.status = "failed"
        if job:
            from datetime import datetime

            job.status = "failed"
            job.completed_at = datetime.utcnow()
        db.session.commit()


def process_prediction_task(task_info, model_storage_path):
    """Runs one prediction task and returns the result dict for the caller to
    push into Redis. Does not touch the database."""
    prediction_id = task_info["prediction_id"]
    model_id = task_info["model_id"]
    model_type = task_info["model_type"]
    texts = task_info["texts"]  # always a list

    print(f"Processing prediction for model {model_id} ({len(texts)} samples)")

    model_path = get_model_path(model_storage_path, model_id)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model_data = joblib.load(model_path)
    preprocessor = model_data.get("preprocessor")
    vectorizer = model_data.get("vectorizer")
    clf = model_data.get("model")
    label_encoder = model_data.get("label_encoder")

    cleaned = [preprocessor.clean_text(t) if preprocessor else t for t in texts]
    predictions = []

    if model_type in ("svm", "decision_tree", "random_forest", "naive_bayes", "logistic_regression"):
        X = vectorizer.transform(cleaned)
        raw_preds = clf.predict(X)
        has_proba = hasattr(clf, "predict_proba")
        probas = clf.predict_proba(X) if has_proba else None
        classes = label_encoder.classes_ if label_encoder else None

        for i, pred in enumerate(raw_preds):
            label = label_encoder.inverse_transform([pred])[0] if label_encoder else str(pred)
            entry = {"text": texts[i], "prediction": label}
            if has_proba and classes is not None:
                entry["confidence"] = round(float(probas[i][pred]), 4)
                entry["all_confidences"] = {
                    cls: round(float(probas[i][j]), 4)
                    for j, cls in enumerate(classes)
                }
            predictions.append(entry)

    elif model_type in ("kmeans", "hierarchical", "dbscan"):
        X = vectorizer.transform(cleaned)
        if hasattr(X, "toarray"):
            X = X.toarray()
        cluster_ids = clf.predict(X) if hasattr(clf, "predict") else clf.fit_predict(X)
        for i, cid in enumerate(cluster_ids):
            predictions.append({
                "text": texts[i],
                "prediction": f"Cluster {int(cid)}",
                "cluster_id": int(cid),
            })

    elif model_type in ("lda", "nmf"):
        X = vectorizer.transform(cleaned)
        topic_weights = clf.transform(X)
        for i, weights in enumerate(topic_weights):
            dominant = int(weights.argmax())
            predictions.append({
                "text": texts[i],
                "prediction": f"Topic {dominant}",
                "dominant_topic": dominant,
                "topic_weights": [round(float(w), 4) for w in weights],
            })

    elif model_type in ("legalbert", "roberta"):
        import torch

        tokenizer = model_data.get("tokenizer")
        if tokenizer is None:
            raise ValueError("Tokenizer not found in saved model")

        clf.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf.to(device)

        for i, text in enumerate(cleaned):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = clf(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_idx = int(torch.argmax(logits, dim=-1).item())

            label = (
                label_encoder.inverse_transform([pred_idx])[0]
                if label_encoder
                else str(pred_idx)
            )
            classes = label_encoder.classes_ if label_encoder else list(range(len(probs)))
            predictions.append({
                "text": texts[i],
                "prediction": label,
                "confidence": round(probs[pred_idx], 4),
                "all_confidences": {
                    str(cls): round(probs[j], 4)
                    for j, cls in enumerate(classes)
                },
            })

    else:
        raise ValueError(f"Prediction not supported for model type: {model_type}")

    return {"predictions": predictions, "model_type": model_type}


def process_explanation_task(task_info, model_storage_path):
    """Runs one explanation task and returns the result dict for the caller to
    push into Redis. Does not touch the database."""
    model_id = task_info["model_id"]
    text = task_info["text"]
    method = task_info.get("method", "lime")  # 'lime' or 'shap'

    print(f"Processing {method.upper()} explanation for model {model_id}")

    model_path = get_model_path(model_storage_path, model_id)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_data = joblib.load(model_path)

    if method == "lime":
        explanation = generate_lime_explanation(model_data, text)
    elif method == "shap":
        explanation = generate_shap_explanation(model_data, text)
    else:
        raise ValueError(f"Unknown explanation method: {method}")

    return {
        "explanation": explanation,
        "text": text,
        "method": method,
        "model_id": model_id,
        "status": "completed",
    }

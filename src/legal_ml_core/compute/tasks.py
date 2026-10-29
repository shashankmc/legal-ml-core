import json
import os
from datetime import datetime

import joblib
import pandas as pd

from ..db import db
from ..models import Model, TrainingJob, TrainingMetricPoint
from ..queue_contract import is_cancel_requested
from ..storage import get_storage_backend
from .explain import generate_lime_explanation, generate_shap_explanation
from .trainer import MLModelTrainer, TrainingCancelled


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


ACTIVE_JOB_STATUSES = ("pending", "running")


def _guarded_job_update(job_id, from_statuses, **values):
    """UPDATE training_jobs SET ... WHERE id=? AND status IN from_statuses.

    Returns True when the row was still in one of `from_statuses` and was
    updated. Terminal states (completed/failed/cancelled) are never in
    `from_statuses`, so a guarded write can never overwrite them.
    """
    updated = (
        TrainingJob.query.filter(
            TrainingJob.id == job_id, TrainingJob.status.in_(from_statuses)
        ).update(values, synchronize_session=False)
    )
    db.session.commit()
    return updated == 1


def _mark_running(job_id, model_id):
    """Pickup: pending -> running + started_at in the DB (OV-E3)."""
    if not _guarded_job_update(
        job_id, ACTIVE_JOB_STATUSES, status="running", started_at=datetime.utcnow()
    ):
        return False
    model = Model.query.get(model_id)
    if model:
        model.status = "training"
        db.session.commit()
    return True


def _mark_cancelled(job_id, redis_client):
    """Settle a stopped job as cancelled. /stop normally already wrote the DB
    row; this guarded write only covers the flag-landed-first window."""
    if _guarded_job_update(
        job_id, ACTIVE_JOB_STATUSES, status="cancelled", completed_at=datetime.utcnow()
    ):
        job = TrainingJob.query.get(job_id)
        if job and job.model:
            job.model.status = "failed"  # same as /stop
            db.session.commit()
    redis_client.setex(f"job_status_{job_id}", 3600, json.dumps({"status": "cancelled"}))


def process_training_task(
    task_info, redis_client, model_storage_path, trainer_factory=MLModelTrainer
):
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

    def cancel_requested():
        return is_cancel_requested(redis_client, job_id)

    def checkpoint():
        """Between-stage check. Raises TrainingCancelled after a Stop.

        This is the one place every stage boundary passes through, so later
        liveness work (heartbeat, #29) can hook in here too.
        """
        if cancel_requested():
            raise TrainingCancelled(job_id)

    def set_live(payload):
        redis_client.setex(f"job_status_{job_id}", 3600, json.dumps(payload))

    model_path = None
    try:
        print(f"Processing training task for model {model_id}")

        if cancel_requested():
            _mark_cancelled(job_id, redis_client)
            print(f"Training job {job_id} was stopped before pickup; skipping")
            return
        if not _mark_running(job_id, model_id):
            print(f"Training job {job_id} is no longer active; skipping")
            return

        set_live({"status": "running", "progress": 0})
        metric_logger.log("preprocessing_started", 1.0)

        # Load dataset
        df = pd.read_csv(task_info["dataset_path"])

        # Initialize trainer
        trainer = trainer_factory(
            redis_client=redis_client,
            metric_logger=metric_logger,
            cancel_check=cancel_requested,
        )

        # Preprocess data
        texts, labels, label_names = trainer.preprocess_data(
            df, task_info["task_type"], task_info.get("preprocessing_options", {})
        )
        checkpoint()

        set_live({"status": "running", "progress": 25})
        metric_logger.log("preprocessing_done", 1.0)
        metric_logger.log("training_started", 1.0)

        # Train model based on task type
        if task_info["model_type"] in ["legalbert", "roberta"]:
            task_params = dict(task_info)
            task_params["output_dir"] = get_storage_backend().get_transformer_output_dir(
                model_storage_path, model_id
            )
            # Checks the cancel flag every HF step and raises TrainingCancelled
            # itself, skipping evaluate/predict.
            metrics = trainer.train_transformer_model(
                texts, labels, task_info["model_type"], task_params
            )
        elif task_info["task_type"] in [
            "classification",
            "document_classification",
            "judgment_prediction",
        ]:
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
        checkpoint()

        set_live({"status": "running", "progress": 75})
        metric_logger.log("training_done", 1.0)

        # Save model
        model_path = get_storage_backend().get_model_path(model_storage_path, model_id)
        trainer.save_model(model_path)

        # Guarded final write (D4): only a still-active job becomes completed.
        # If Stop landed at any point up to here, the write misses and the
        # model file just written is removed.
        if not _guarded_job_update(
            job_id,
            ACTIVE_JOB_STATUSES,
            status="completed",
            metrics=metrics,
            progress=100,
            completed_at=datetime.utcnow(),
        ):
            raise TrainingCancelled(job_id)

        model = Model.query.get(model_id)
        if model:
            model.status = "completed"
            model.metrics = metrics
            model.file_path = model_path
        db.session.commit()
        metric_logger.log("model_saved", 1.0)

        set_live(
            {
                "status": "completed",
                "progress": 100,
                "metrics": metrics,
                "model_path": model_path,
            }
        )
        print(f"Training completed for model {model_id}")

    except Exception as e:
        db.session.rollback()
        # A Stop (or any error raised while a Stop is in flight) settles as
        # cancelled, never failed (OV-E4).
        if isinstance(e, TrainingCancelled) or _safe(cancel_requested, default=False):
            print(f"Training job {job_id} cancelled")
            _discard_model_file(model_path)
            _mark_cancelled(job_id, redis_client)
            _safe(lambda: metric_logger.log("training_cancelled", 1.0))
            return

        print(f"Error processing training task: {str(e)}")
        _discard_model_file(model_path)
        _safe(lambda: metric_logger.log("training_failed", 1.0))
        # Guarded like the success write: a job that went terminal meanwhile
        # (e.g. cancelled by /stop) keeps its state.
        if _guarded_job_update(
            job_id,
            ACTIVE_JOB_STATUSES,
            status="failed",
            # Persist the reason so the UI can explain the failure instead of
            # showing a generic "Training failed".
            error_message=str(e),
            completed_at=datetime.utcnow(),
        ):
            model = Model.query.get(model_id)
            if model:
                model.status = "failed"
                db.session.commit()
            set_live({"status": "failed", "error": str(e)})


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        db.session.rollback()
        return default


def _discard_model_file(model_path):
    if model_path and os.path.exists(model_path):
        try:
            os.remove(model_path)
        except OSError:
            pass


def process_prediction_task(task_info, model_storage_path):
    """Runs one prediction task and returns the result dict for the caller to
    push into Redis. Does not touch the database."""
    prediction_id = task_info["prediction_id"]
    model_id = task_info["model_id"]
    model_type = task_info["model_type"]
    texts = task_info["texts"]  # always a list

    print(f"Processing prediction for model {model_id} ({len(texts)} samples)")

    model_path = get_storage_backend().get_model_path(model_storage_path, model_id)
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

    model_path = get_storage_backend().get_model_path(model_storage_path, model_id)
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

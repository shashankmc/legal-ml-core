import json
import logging
import os
from datetime import datetime

import joblib
import pandas as pd
import redis as redis_lib

from ..active_learning.scoring import queue_pool_scoring, verify_model
from ..annotation_queue import apply_saved_annotations
from ..db import db
from ..job_state import ACTIVE_JOB_STATUSES, COULD_NOT_START_MESSAGE, transition
from ..models import Dataset, Model, TrainingJob, TrainingMetricPoint
from ..liveness import DEFAULT_TIMINGS, Heartbeat
from ..queue_contract import JOB_STATUS_TTL_SECONDS, is_cancel_requested, job_status_key
from ..storage import get_storage_backend
from .explain import generate_lime_explanation, generate_shap_explanation
from .trainer import MLModelTrainer, TrainingCancelled

logger = logging.getLogger(__name__)

# Redis being unreachable (vs. a bad command): reported as a retryable failure.
REDIS_DOWN_ERRORS = (redis_lib.ConnectionError, redis_lib.TimeoutError)


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


def _mark_running(job_id):
    """Pickup: pending -> running + started_at in the DB (OV-E3)."""
    return transition(
        job_id,
        ACTIVE_JOB_STATUSES,
        "running",
        model_status="training",
        started_at=datetime.utcnow(),
    )


def _mark_cancelled(job_id, redis_client):
    """Settle a stopped job as cancelled. /stop normally already wrote the DB
    row; this guarded write only covers the flag-landed-first window."""
    transition(
        job_id,
        ACTIVE_JOB_STATUSES,
        "cancelled",
        model_status="failed",  # same as /stop
        completed_at=datetime.utcnow(),
    )
    _write_live(redis_client, job_id, {"status": "cancelled"}, JOB_STATUS_TTL_SECONDS)


def _write_live(redis_client, job_id, payload, ttl_seconds):
    """Best-effort live status for the API (A2: it only refines progress). A
    Redis failure here never fails the job."""
    try:
        redis_client.psetex(
            job_status_key(job_id), max(1, int(ttl_seconds * 1000)), json.dumps(payload)
        )
    except Exception as e:
        logger.warning("training job %s: live status not written: %s", job_id, e)


def _job_left_active_in_db(job_id):
    """True once the DB row is no longer pending/running (cancelled by /stop,
    swept as failed, or gone). Reads the column, not the ORM identity map, so
    another process's commit is seen."""
    status = db.session.query(TrainingJob.status).filter(TrainingJob.id == job_id).scalar()
    return status not in ACTIVE_JOB_STATUSES


def process_training_task(
    task_info,
    redis_client,
    model_storage_path,
    trainer_factory=MLModelTrainer,
    liveness=DEFAULT_TIMINGS,
    heartbeat_factory=Heartbeat,
    scoring_queue="scoring_queue",
):
    """Runs one training task. Must be called with a Flask app context pushed
    (db.session/Model.query need it) — see legal_ml_core.worker.run_ml_worker.

    This replaces the original ml_worker's raw `psycopg2` UPDATE statements
    against the `models`/`training_jobs` tables with normal SQLAlchemy writes
    through the same ORM classes the API process uses — the two processes now
    share one schema representation instead of two independent ones.

    While the job runs, a heartbeat thread (`heartbeat_factory`, see
    liveness.py) beats every `liveness.heartbeat_interval` seconds; if this
    process dies, the beats stop and the API sweep marks the job failed (R5).
    """
    job_id = task_info["job_id"]
    model_id = task_info["model_id"]
    metric_logger = MetricLogger(job_id)

    def cancel_requested():
        """Stop check: the Redis flag (cheap, set by /stop), else the DB row,
        which /stop writes first and which also covers a flag that could not
        be set or read."""
        try:
            if is_cancel_requested(redis_client, job_id):
                return True
        except Exception as e:
            logger.warning("training job %s: cancel flag unreadable, using DB: %s", job_id, e)
        return _job_left_active_in_db(job_id)

    def checkpoint():
        """Between-stage check. Raises TrainingCancelled after a Stop.

        Liveness is not checked here: the heartbeat runs on its own thread
        because the stages between checkpoints block for minutes.
        """
        if cancel_requested():
            raise TrainingCancelled(job_id)

    def set_live(payload):
        _write_live(redis_client, job_id, payload, liveness.status_ttl)

    model_path = None
    heartbeat = None
    try:
        logger.info("training job %s: processing model %s", job_id, model_id)

        if cancel_requested():
            _mark_cancelled(job_id, redis_client)
            logger.info("training job %s was stopped before pickup; skipping", job_id)
            return
        if not _mark_running(job_id):
            logger.info("training job %s is no longer active; skipping", job_id)
            return
        heartbeat = heartbeat_factory(redis_client, job_id, liveness).start()

        set_live({"status": "running", "progress": 0})
        metric_logger.log("preprocessing_started", 1.0)

        # Load dataset
        df = pd.read_csv(task_info["dataset_path"])
        df = apply_saved_annotations(df, db.session.get(Dataset, task_info["dataset_id"]))

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
            # Checks for a Stop every HF step and raises TrainingCancelled
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
        metrics = verify_model(db.session.get(Dataset, task_info["dataset_id"]), model_path, metrics)

        # Guarded final write (D4): only a still-active job becomes completed.
        # If Stop landed at any point up to here, the write misses and the
        # model file just written is removed.
        if not transition(
            job_id,
            ACTIVE_JOB_STATUSES,
            "completed",
            model_status="completed",
            metrics=metrics,
            progress=100,
            completed_at=datetime.utcnow(),
        ):
            raise TrainingCancelled(job_id)

        model = db.session.get(Model, model_id)
        if model:
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
        logger.info("training job %s: completed model %s", job_id, model_id)

        if task_info.get("al_round_id"):
            queue_pool_scoring(task_info, redis_client, scoring_queue)

    except Exception as e:
        db.session.rollback()
        # A Stop (or any error raised while a Stop is in flight) settles as
        # cancelled, never failed (OV-E4).
        if isinstance(e, TrainingCancelled) or _call_swallowing_errors_with_rollback(
            cancel_requested, default=False
        ):
            logger.info("training job %s cancelled", job_id)
            _discard_model_file(model_path)
            _mark_cancelled(job_id, redis_client)
            _call_swallowing_errors_with_rollback(
                lambda: metric_logger.log("training_cancelled", 1.0)
            )
            return

        _discard_model_file(model_path)
        _call_swallowing_errors_with_rollback(lambda: metric_logger.log("training_failed", 1.0))
        if isinstance(e, REDIS_DOWN_ERRORS):
            # Infrastructure, not the user's data: never show raw Redis text.
            error_message = COULD_NOT_START_MESSAGE
            logger.error(
                "training job %s (owner %s): Redis unreachable during training: %s",
                job_id,
                _owner_of(job_id),
                e,
            )
        else:
            # Persist the reason so the UI can explain the failure instead of
            # showing a generic "Training failed".
            error_message = str(e)
            logger.warning("training job %s failed: %s", job_id, e)
        # Guarded like the success write: a job that went terminal meanwhile
        # (e.g. cancelled by /stop) keeps its state.
        if transition(
            job_id,
            ACTIVE_JOB_STATUSES,
            "failed",
            model_status="failed",
            error_message=error_message,
            completed_at=datetime.utcnow(),
        ):
            set_live({"status": "failed", "error": error_message})
    finally:
        if heartbeat is not None:
            heartbeat.stop()


def _owner_of(job_id):
    job = _call_swallowing_errors_with_rollback(lambda: db.session.get(TrainingJob, job_id))
    return job.owner_id if job else None


def _call_swallowing_errors_with_rollback(fn, default=None):
    """Run `fn`; on any error roll the session back and return `default`."""
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

    logger.info("prediction for model %s (%d samples)", model_id, len(texts))

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

    logger.info("%s explanation for model %s", method.upper(), model_id)

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

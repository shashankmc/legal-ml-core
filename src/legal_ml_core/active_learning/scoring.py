"""Rescoring the pool on the worker, once a round's model is saved."""
import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ..annotation_queue import holdout_examples, read_rows
from ..db import db
from ..models import ALConfig, ALRound, AnnotationTask, Dataset, Model, TrainingJob
from ..storage import get_storage_backend
from .strategies import score_pool

logger = logging.getLogger(__name__)

SCORING_FAILED_MESSAGE = "The queue could not be reordered. Try retraining."

# Metrics without a holdout behind them: the verdict must not grade these.
UNVERIFIED = "not_independently_verified"
VERIFIED = "holdout"
# Fewer held-back labels than this (or than the holdout holds) grade nothing.
MIN_HOLDOUT_LABELS = 10


def class_probabilities(model_data, texts):
    """``(probabilities, class names)`` for ``texts``, one row per text."""
    preprocessor = model_data.get("preprocessor")
    cleaned = [preprocessor.clean_text(t) if preprocessor else t for t in texts]
    probabilities = model_data["model"].predict_proba(model_data["vectorizer"].transform(cleaned))

    encoder = model_data.get("label_encoder")
    classes = [str(c) for c in encoder.classes_] if encoder else None
    return probabilities, classes


def _embeddings(model_data, texts):
    """The model's own vectors."""
    preprocessor = model_data.get("preprocessor")
    cleaned = [preprocessor.clean_text(t) if preprocessor else t for t in texts]
    return model_data["vectorizer"].transform(cleaned)


def _holdout_metrics(model_data, texts, labels):
    probabilities, classes = class_probabilities(model_data, texts)
    predicted = [classes[i] for i in probabilities.argmax(axis=1)]
    names = sorted(set(labels) | set(predicted))
    return {
        "accuracy": accuracy_score(labels, predicted),
        "precision": precision_score(labels, predicted, average="weighted", zero_division=0),
        "recall": recall_score(labels, predicted, average="weighted", zero_division=0),
        "f1_score": f1_score(labels, predicted, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=names).tolist(),
        "confusion_matrix_labels": names,
        "classification_report": classification_report(labels, predicted, labels=names, zero_division=0),
    }


def grade_on_holdout(dataset, model_data, metrics):
    """``metrics`` graded on the holdout, or marked as not independently verified
    when it is off or holds too few labels to mean anything. Grading needs
    ``predict_proba`` over a vectorizer, so a transformer trained on the same
    Dataset reads as not independently verified too."""
    texts, labels = holdout_examples(read_rows(dataset.file_path), dataset)
    holdout_size = len(dataset.al_config.holdout_indices or [])
    if model_data.get("vectorizer") is None or not texts or len(texts) < min(MIN_HOLDOUT_LABELS, holdout_size):
        return {**metrics, "verification": UNVERIFIED, "holdout_labels": len(texts)}
    return {
        **metrics,
        **_holdout_metrics(model_data, texts, labels),
        "verification": VERIFIED,
        "holdout_labels": len(texts),
    }


def verify_model(dataset, model_path, metrics):
    """Grade a just-saved model when its dataset has a loop, else leave ``metrics``."""
    if dataset is None or dataset.al_config is None:
        return metrics
    return grade_on_holdout(dataset, joblib.load(model_path), metrics)


def _process_verification(task_info, model_storage_path):
    model = db.session.get(Model, task_info["model_id"])
    dataset = db.session.get(Dataset, task_info["dataset_id"])
    if model is None or dataset is None or dataset.al_config is None:
        return
    model_path = get_storage_backend().get_model_path(model_storage_path, model.id)
    model.metrics = verify_model(dataset, model_path, model.metrics or {})
    db.session.commit()


def _pool_tasks(dataset_id):
    return (
        AnnotationTask.query.filter_by(dataset_id=dataset_id, is_holdout=False)
        .filter(AnnotationTask.status != "annotated")
        .order_by(AnnotationTask.item_index)
        .all()
    )


def _scores(strategy, tasks, model_data, config, round_number):
    texts = [task.text for task in tasks]
    seed = config.random_seed + round_number
    if strategy == "random":
        return score_pool("random", size=len(tasks), seed=seed)
    if not tasks:
        return []
    if strategy == "embedding_kmeans":
        return score_pool(
            strategy,
            embeddings=_embeddings(model_data, texts),
            batch_size=config.batch_size,
            seed=seed,
        )
    probabilities, _ = class_probabilities(model_data, texts)
    return score_pool(strategy, probabilities=probabilities)


def process_scoring_task(task_info, model_storage_path):
    """Score the pool and move the loop to its next round. Must be called with a
    Flask app context pushed. Scores and the round change commit together, so
    the queue never reorders without the round saying so."""
    if task_info.get("action") == "verify":
        _process_verification(task_info, model_storage_path)
        return

    round_row = db.session.get(ALRound, task_info["round_id"])
    if round_row is None or round_row.status != "scoring":
        logger.info("scoring round %s is no longer waiting; skipping", task_info["round_id"])
        return

    model_id = task_info["model_id"]
    model_path = get_storage_backend().get_model_path(model_storage_path, model_id)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model_data = joblib.load(model_path)

    config = ALConfig.query.filter_by(dataset_id=round_row.dataset_id).first()
    tasks = _pool_tasks(round_row.dataset_id)
    scores = _scores(task_info["strategy"], tasks, model_data, config, round_row.number)
    for task, score in zip(tasks, scores):
        task.score = float(score)

    round_row.status = "completed"
    round_row.completed_at = datetime.utcnow()
    db.session.add(
        ALRound(
            dataset_id=round_row.dataset_id,
            number=round_row.number + 1,
            strategy=config.strategy,
            model_id=round_row.model_id,
        )
    )
    db.session.commit()
    logger.info("scored %d pool items for round %s", len(tasks), round_row.id)


def queue_pool_scoring(task_info, redis_client, scoring_queue):
    """A round's training finished: hand the pool to the scoring queue. A failure
    here must not touch the finished training job, so the round takes it. The
    worker and the read-time sweep can both get here, so only the one that moves
    the round on sends the message."""
    round_id = task_info["al_round_id"]
    job = db.session.get(TrainingJob, task_info["job_id"])
    claimed = ALRound.query.filter_by(id=round_id, status="training").update(
        {"status": "scoring", "metrics": job.metrics}
    )
    db.session.commit()
    if not claimed:
        return
    round_row = db.session.get(ALRound, round_id)
    config = ALConfig.query.filter_by(dataset_id=round_row.dataset_id).first()
    try:
        redis_client.lpush(
            scoring_queue,
            json.dumps(
                {
                    "round_id": str(round_id),
                    "dataset_id": str(round_row.dataset_id),
                    "model_id": str(task_info["model_id"]),
                    "strategy": config.strategy,
                }
            ),
        )
    except Exception as e:
        logger.warning("round %s: scoring could not be queued: %s", round_id, e)
        fail_scoring_round(round_id)


def fail_scoring_round(round_id):
    """Back to labelling with a reason. The user can retrain to score again."""
    round_row = db.session.get(ALRound, round_id) if round_id else None
    if round_row is None or round_row.status != "scoring":
        return
    round_row.status = "labelling"
    round_row.error = SCORING_FAILED_MESSAGE
    db.session.commit()

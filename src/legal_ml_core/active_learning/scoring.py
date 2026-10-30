"""Rescoring the pool on the worker, once a round's model is saved."""
import logging
import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..db import db
from ..models import ALConfig, ALRound, AnnotationTask
from ..storage import get_storage_backend
from .strategies import score_pool

logger = logging.getLogger(__name__)

SCORING_FAILED_MESSAGE = "The queue could not be reordered. Try retraining."

TRANSFORMER_TYPES = ("legalbert", "roberta")
TRANSFORMER_BATCH_SIZE = 32


def _transformer_probabilities(model_data, cleaned):
    import torch

    clf = model_data["model"]
    tokenizer = model_data["tokenizer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf.eval()
    clf.to(device)

    batches = []
    for start in range(0, len(cleaned), TRANSFORMER_BATCH_SIZE):
        inputs = tokenizer(
            cleaned[start : start + TRANSFORMER_BATCH_SIZE],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            batches.append(torch.softmax(clf(**inputs).logits, dim=-1).cpu().numpy())
    return np.concatenate(batches)


def class_probabilities(model_data, model_type, texts):
    """``(probabilities, class names)`` for ``texts``, one row per text."""
    preprocessor = model_data.get("preprocessor")
    cleaned = [preprocessor.clean_text(t) if preprocessor else t for t in texts]
    if model_type in TRANSFORMER_TYPES:
        probabilities = _transformer_probabilities(model_data, cleaned)
    else:
        clf = model_data["model"]
        probabilities = clf.predict_proba(model_data["vectorizer"].transform(cleaned))

    encoder = model_data.get("label_encoder")
    classes = [str(c) for c in encoder.classes_] if encoder else None
    return probabilities, classes


def _embeddings(model_data, texts):
    """The model's own vectors. Transformers save no vectorizer, so they get
    TF-IDF over the pool: this strategy needs similar texts close, nothing more."""
    preprocessor = model_data.get("preprocessor")
    cleaned = [preprocessor.clean_text(t) if preprocessor else t for t in texts]
    vectorizer = model_data.get("vectorizer")
    if vectorizer is not None:
        return vectorizer.transform(cleaned)
    return TfidfVectorizer(max_features=5000).fit_transform(cleaned)


def _pool_tasks(dataset_id):
    return (
        AnnotationTask.query.filter_by(dataset_id=dataset_id, is_holdout=False)
        .filter(AnnotationTask.status != "annotated")
        .order_by(AnnotationTask.item_index)
        .all()
    )


def _scores(strategy, tasks, model_data, model_type, config, round_number):
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
    probabilities, _ = class_probabilities(model_data, model_type, texts)
    return score_pool(strategy, probabilities=probabilities)


def process_scoring_task(task_info, model_storage_path):
    """Score the pool and move the loop to its next round. Must be called with a
    Flask app context pushed. Scores and the round change commit together, so
    the queue never reorders without the round saying so."""
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
    scores = _scores(task_info["strategy"], tasks, model_data, task_info["model_type"], config, round_row.number)
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


def fail_scoring_round(round_id):
    """Back to labelling with a reason. The user can retrain to score again."""
    round_row = db.session.get(ALRound, round_id) if round_id else None
    if round_row is None or round_row.status != "scoring":
        return
    round_row.status = "labelling"
    round_row.error = SCORING_FAILED_MESSAGE
    db.session.commit()

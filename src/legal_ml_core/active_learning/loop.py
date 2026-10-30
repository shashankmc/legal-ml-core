"""The loop's state: starting, retrain requests, and what the user sees next."""
import json
import logging
import secrets
from collections import Counter
from datetime import datetime, timedelta

import numpy as np

from flask import current_app

from ..annotation_queue import (
    TRAINED_ON_LABELS,
    create_annotation_tasks,
    file_labels,
    read_rows,
    rows_with_text,
)
from ..db import db
from ..job_state import ACTIVE_JOB_STATUSES, COULD_NOT_START_MESSAGE, sweep_lost_job, sweep_stale_job
from ..models import ALConfig, ALRound, AnnotationTask, Model, TrainingJob
from .scoring import MIN_HOLDOUT_LABELS, fail_scoring_round, queue_pool_scoring
from .strategies import DEFAULT_STRATEGY, STRATEGIES, score_pool

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 20
# Below this the trainer's stratified 80/20 split cannot hold every class in its
# test side, so it is the floor as well as the default.
MIN_PER_CLASS = 5
# How long a round may wait for the scoring worker before it is called lost. It
# is far above the heartbeat window, as a large pool takes a while to score.
DEFAULT_SCORING_STALE_SECONDS = 3600

DEFAULT_HOLDOUT_PERCENT = 20
MAX_HOLDOUT_PERCENT = 50

ACTIVE_ROUND_STATUSES = ("training", "scoring")


class LoopError(Exception):
    """A request the loop refuses, with its status and a plain reason."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def allowed_strategies(config=None):
    config = current_app.config if config is None else config
    return list(config.get("LEGAL_ML_AL_STRATEGIES") or STRATEGIES)


def default_strategy(config=None):
    config = current_app.config if config is None else config
    return config.get("LEGAL_ML_AL_DEFAULT_STRATEGY") or DEFAULT_STRATEGY


def validate_host_config(config):
    """Fail at import, not on a user's first click, on a bad host setting."""
    allowed = allowed_strategies(config)
    unknown = [name for name in allowed if name not in STRATEGIES]
    if unknown:
        raise ValueError(f"LEGAL_ML_AL_STRATEGIES has unknown strategies: {unknown}")
    if default_strategy(config) not in allowed:
        raise ValueError("LEGAL_ML_AL_DEFAULT_STRATEGY must be in LEGAL_ML_AL_STRATEGIES")


def _host_batch_size():
    return current_app.config.get("LEGAL_ML_AL_BATCH_SIZE", DEFAULT_BATCH_SIZE)


def _host_holdout_percent():
    return current_app.config.get("LEGAL_ML_AL_HOLDOUT_PERCENT", DEFAULT_HOLDOUT_PERCENT)


def _host_min_per_class():
    return current_app.config.get("LEGAL_ML_AL_MIN_PER_CLASS", MIN_PER_CLASS)


def settings():
    return {
        "strategies": allowed_strategies(),
        "default_strategy": default_strategy(),
        "default_batch_size": _host_batch_size(),
        "min_per_class": _host_min_per_class(),
        "default_holdout_percent": _host_holdout_percent(),
        "max_holdout_percent": MAX_HOLDOUT_PERCENT,
        # The LLM seed arrives with LLM assistance in the builder (BAIGLE #67).
        "seed_sources": {"random": True, "llm": False},
    }


def _checked_strategy(strategy):
    if strategy not in allowed_strategies():
        raise LoopError("That way of choosing items is not available here.")
    return strategy


def _checked_positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LoopError(f"{name} must be a whole number above zero.")
    return value


def _pool_tasks(dataset_id):
    return (
        AnnotationTask.query.filter_by(dataset_id=dataset_id, is_holdout=False)
        .filter(AnnotationTask.status != "annotated")
        .order_by(AnnotationTask.item_index)
        .all()
    )


def _draw_holdout(df, dataset, percent, seed):
    """Row positions held back: a random share of all rows with text."""
    rows = rows_with_text(df, dataset)
    size = int(round(len(rows) * percent / 100))
    if percent:
        size = max(size, 1)
    drawn = np.random.default_rng(seed).choice(rows, size=size, replace=False)
    return sorted(int(position) for position in drawn)


def _file_label_counts(df, dataset, holdout):
    """Per class, the labels the file gave outside the holdout: they train with
    the rest."""
    labels = file_labels(df, dataset)
    labels = labels[~labels.index.isin(holdout)]
    return {str(k): int(v) for k, v in labels.value_counts().items()}


def _shuffle_pool(tasks, seed):
    """Random scores: the order of the first batch, before any model exists."""
    for task, score in zip(tasks, score_pool("random", size=len(tasks), seed=seed)):
        task.score = float(score)


def start_loop(dataset, options):
    """Set the dataset up for active learning; caller commits."""
    if dataset.task_type not in TRAINED_ON_LABELS:
        raise LoopError("Active learning works on Text Classification Datasets only.")
    if dataset.al_config is not None:
        raise LoopError("Active learning has already started on this Dataset.", 409)

    strategy = _checked_strategy(options.get("strategy") or default_strategy())
    if options.get("seed_source", "random") == "llm":
        raise LoopError("LLM-suggested seed batches are not available yet.")
    if options.get("seed_source", "random") != "random":
        raise LoopError("Choose a random or LLM-suggested seed batch.")
    batch_size = _checked_positive_int(options.get("batch_size", _host_batch_size()), "Batch size")
    min_per_class = _checked_positive_int(options.get("min_per_class", _host_min_per_class()), "Minimum per label")
    if min_per_class < MIN_PER_CLASS:
        raise LoopError(f"Each label needs at least {MIN_PER_CLASS} examples to train on.")

    holdout_percent = options.get("holdout_percent", _host_holdout_percent())
    if (
        isinstance(holdout_percent, bool)
        or not isinstance(holdout_percent, int)
        or not 0 <= holdout_percent <= MAX_HOLDOUT_PERCENT
    ):
        raise LoopError(f"The held-back share must be between 0 and {MAX_HOLDOUT_PERCENT}%.")

    if not dataset.annotation_tasks:
        create_annotation_tasks(dataset)
        db.session.flush()
    df = read_rows(dataset.file_path)
    seed = secrets.randbelow(2**31)
    holdout = _draw_holdout(df, dataset, holdout_percent, seed)
    held_out = set(holdout)
    for task in AnnotationTask.query.filter_by(dataset_id=dataset.id):
        task.is_holdout = task.item_index in held_out
    db.session.flush()

    tasks = _pool_tasks(dataset.id)
    if not tasks:
        raise LoopError("This Dataset has no rows to label.")

    config = ALConfig(
        dataset_id=dataset.id,
        strategy=strategy,
        batch_size=batch_size,
        min_per_class=min_per_class,
        random_seed=seed,
        holdout_percent=holdout_percent,
        holdout_indices=holdout,
        file_label_counts=_file_label_counts(df, dataset, holdout),
    )
    db.session.add(config)
    _shuffle_pool(tasks, seed)
    db.session.add(ALRound(dataset_id=dataset.id, number=0, strategy="random"))


def set_strategy(dataset, strategy):
    """Takes effect at the next rescore; caller commits."""
    config = _config_of(dataset)
    config.strategy = _checked_strategy(strategy)


def _config_of(dataset):
    if dataset.al_config is None:
        raise LoopError("Active learning has not started on this Dataset.", 409)
    return dataset.al_config


def _current_round(dataset_id):
    return ALRound.query.filter_by(dataset_id=dataset_id).order_by(ALRound.number.desc()).first()


def _confirmed_label_counts(config):
    """Per class, every label training would use."""
    rows = (
        db.session.query(AnnotationTask.annotation_label, db.func.count())
        .filter(
            AnnotationTask.dataset_id == config.dataset_id,
            AnnotationTask.status == "annotated",
            AnnotationTask.is_holdout.is_(False),
        )
        .group_by(AnnotationTask.annotation_label)
        .all()
    )
    return Counter(dict(rows)) + Counter(config.file_label_counts or {})


def _labels_at_last_train(dataset_id):
    return (
        db.session.query(db.func.max(ALRound.labels_used))
        .filter(ALRound.dataset_id == dataset_id, ALRound.status != "labelling")
        .scalar()
        or 0
    )


def _scoring_queue():
    return current_app.config.get("LEGAL_ML_SCORING_QUEUE", "scoring_queue")


def _back_to_labelling(round_row, reason):
    round_row.status = "labelling"
    round_row.labels_used = None
    round_row.error = reason
    db.session.commit()


def _settle_training_round(round_row):
    redis_client = current_app.redis_client
    stale = current_app.config.get("LEGAL_ML_HEARTBEAT_STALE_SECONDS", 120)
    if round_row.job_id is None:
        # The claim commits before the job exists, so a crash between the two
        # leaves a round that nothing will ever finish.
        if datetime.utcnow() - round_row.updated_at > timedelta(seconds=stale):
            _back_to_labelling(round_row, COULD_NOT_START_MESSAGE)
        return

    job = db.session.get(TrainingJob, round_row.job_id)
    if job is not None and job.status in ACTIVE_JOB_STATUSES:
        sweep_stale_job(job, redis_client, stale)
        queue = current_app.config.get("LEGAL_ML_TRAINING_QUEUE", "training_queue")
        sweep_lost_job(job, redis_client, queue, stale)
    if job is None or job.status in ("failed", "cancelled"):
        _back_to_labelling(round_row, (job.error_message if job else None) or "Training was stopped.")
    elif job.status == "completed":
        # The worker hands over itself, unless it died just after finishing.
        queue_pool_scoring(
            {"al_round_id": round_row.id, "job_id": job.id, "model_id": job.model_id},
            redis_client,
            _scoring_queue(),
        )


def _settle_round(round_row):
    """Read-time sweep, like training's: a round whose job ended without finishing
    goes back to labelling with the reason, and one whose job finished goes on."""
    if round_row.status == "training":
        _settle_training_round(round_row)
    elif round_row.status == "scoring":
        limit = current_app.config.get("LEGAL_ML_AL_SCORING_STALE_SECONDS", DEFAULT_SCORING_STALE_SECONDS)
        if datetime.utcnow() - round_row.updated_at > timedelta(seconds=limit):
            fail_scoring_round(round_row.id)


def _retrain_state(dataset, config, round_row, counts):
    trained = sum(counts.values())
    new_labels = max(trained - _labels_at_last_train(config.dataset_id), 0)
    classes = set(dataset.labels or []) | set(counts)
    classes_short = {
        name: config.min_per_class - counts[name]
        for name in sorted(classes)
        if counts[name] < config.min_per_class
    }
    needs_second_label = len(classes) < 2
    running = round_row.status in ACTIVE_ROUND_STATUSES
    allowed = new_labels > 0 and not running and not classes_short and not needs_second_label
    return {
        "new_labels": new_labels,
        "allowed": allowed,
        "suggested": allowed and new_labels >= config.batch_size,
        "classes_short": classes_short,
        "needs_second_label": needs_second_label,
    }


def loop_state(dataset):
    config = dataset.al_config
    if config is None:
        return {"active": False}

    round_row = _current_round(dataset.id)
    _settle_round(round_row)
    counts = _confirmed_label_counts(config)
    return {
        "active": True,
        "strategy": config.strategy,
        "seed_source": config.seed_source,
        "batch_size": config.batch_size,
        "min_per_class": config.min_per_class,
        "model_id": config.model_id,
        "round": round_row.to_dict(),
        "label_counts": dict(counts),
        "retrain": _retrain_state(dataset, config, round_row, counts),
        "holdout": _holdout_state(config),
    }


def _holdout_labelled(config):
    """Rows labelled in the file have no task, so only the unlabelled are counted."""
    unlabelled = AnnotationTask.query.filter(
        AnnotationTask.dataset_id == config.dataset_id,
        AnnotationTask.is_holdout.is_(True),
        AnnotationTask.status != "annotated",
    ).count()
    return len(config.holdout_indices or []) - unlabelled


def _holdout_needed(config):
    return min(MIN_HOLDOUT_LABELS, len(config.holdout_indices or []))


def _holdout_state(config):
    model = db.session.get(Model, config.model_id) if config.model_id else None
    return {
        "percent": config.holdout_percent,
        "size": len(config.holdout_indices or []),
        "labelled": _holdout_labelled(config),
        "needed": _holdout_needed(config),
        "verification": ((model.metrics or {}) if model else {}).get("verification"),
    }


def _loop_model(dataset, config, owner_id):
    """One model per loop: each round retrains it, and rounds keep the history."""
    model = db.session.get(Model, config.model_id) if config.model_id else None
    if model is None:
        model = Model(
            owner_id=owner_id,
            dataset_id=dataset.id,
            project_id=dataset.project_id,
            name=f"{dataset.name} (active learning)",
            model_type=config.model_type,
            parameters={},
            status="pending",
        )
        db.session.add(model)
        db.session.flush()
        config.model_id = model.id
    return model


def request_retrain(dataset, owner_id):
    """Start a training run for the current round. Returns (body, status)."""
    # Imported here: routes.models imports this package's sibling compute code.
    from ..routes.models import start_training_job

    config = _config_of(dataset)
    round_row = _current_round(dataset.id)
    _settle_round(round_row)
    if round_row.status in ACTIVE_ROUND_STATUSES:
        return {"message": "Training already started", "job_id": round_row.job_id}, 200

    counts = _confirmed_label_counts(config)
    retrain = _retrain_state(dataset, config, round_row, counts)
    if not retrain["new_labels"]:
        raise LoopError("There are no new labels to train on.", 409)
    if retrain["classes_short"]:
        raise LoopError(f"Each label needs at least {config.min_per_class} examples before training.", 409)
    if retrain["needs_second_label"]:
        raise LoopError("Training needs examples of at least two labels.", 409)

    claimed = ALRound.query.filter_by(id=round_row.id, status="labelling").update(
        {"status": "training", "labels_used": sum(counts.values()), "error": None}
    )
    if not claimed:
        db.session.rollback()
        return {"message": "Training already started", "job_id": None}, 200

    model = _loop_model(dataset, config, owner_id)
    round_row.model_id = model.id
    db.session.commit()

    body, status = start_training_job(owner_id, model.id, {}, al_round_id=round_row.id)
    if status != 200:
        _back_to_labelling(round_row, body.get("error"))
        return body, status

    return {"message": "Training started", "job_id": body["job_id"], "model_id": str(model.id)}, 202


def request_verification(dataset):
    """Grade the saved model on the holdout, after labelling it. Runs on the
    worker, which already holds the model libraries."""
    config = _config_of(dataset)
    model = db.session.get(Model, config.model_id) if config.model_id else None
    if model is None or model.status != "completed":
        raise LoopError("Train a Model before checking its score.", 409)
    missing = _holdout_needed(config) - _holdout_labelled(config)
    if not config.holdout_indices:
        raise LoopError("This Dataset has no held-back items.", 409)
    if missing > 0:
        item = "item" if missing == 1 else "items"
        raise LoopError(f"Label {missing} more held-back {item} before checking the score.", 409)

    current_app.redis_client.lpush(
        _scoring_queue(),
        json.dumps(
            {
                "action": "verify",
                "dataset_id": str(dataset.id),
                "model_id": str(model.id),
            }
        ),
    )


def reset_loops(tasks):
    """After a reset the loop begins again, with the same holdout; caller commits."""
    configs = ALConfig.query.filter(
        ALConfig.dataset_id.in_({task.dataset_id for task in tasks})
    ).all()
    for task in tasks:
        task.score = None
    for config in configs:
        ALRound.query.filter_by(dataset_id=config.dataset_id).delete()
        db.session.add(ALRound(dataset_id=config.dataset_id, number=0, strategy="random"))
        _shuffle_pool(_pool_tasks(config.dataset_id), config.random_seed)

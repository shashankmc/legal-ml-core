"""The loop's state: starting, retrain requests, and what the user sees next."""
import logging
from collections import Counter
from datetime import datetime, timedelta

from flask import current_app

from ..annotation_queue import TRAINED_ON_LABELS, create_annotation_tasks
from ..db import db
from ..job_state import ACTIVE_JOB_STATUSES, sweep_lost_job, sweep_stale_job
from ..models import ALConfig, ALRound, AnnotationTask, Model, TrainingJob
from .scoring import fail_scoring_round
from .strategies import DEFAULT_STRATEGY, STRATEGIES, score_pool

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 20
# How long a round may wait for the scoring worker before it is called lost. It
# is far above the heartbeat window: a transformer scores a large pool slowly.
DEFAULT_SCORING_STALE_SECONDS = 3600

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


def settings():
    return {
        "strategies": allowed_strategies(),
        "default_strategy": default_strategy(),
        "default_batch_size": current_app.config.get("LEGAL_ML_AL_BATCH_SIZE", DEFAULT_BATCH_SIZE),
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
    batch_size = _checked_positive_int(
        options.get("batch_size", current_app.config.get("LEGAL_ML_AL_BATCH_SIZE", DEFAULT_BATCH_SIZE)),
        "Batch size",
    )

    if not dataset.annotation_tasks:
        create_annotation_tasks(dataset)
        db.session.flush()
    tasks = _pool_tasks(dataset.id)
    if not tasks:
        raise LoopError("This Dataset has no rows to label.")

    config = ALConfig(dataset_id=dataset.id, strategy=strategy, batch_size=batch_size)
    db.session.add(config)
    _shuffle_pool(tasks, config.random_seed or 42)
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


def _confirmed_label_counts(dataset_id):
    rows = (
        db.session.query(AnnotationTask.annotation_label, db.func.count())
        .filter(
            AnnotationTask.dataset_id == dataset_id,
            AnnotationTask.status == "annotated",
            AnnotationTask.is_holdout.is_(False),
        )
        .group_by(AnnotationTask.annotation_label)
        .all()
    )
    return Counter(dict(rows))


def _labels_at_last_train(dataset_id):
    return (
        db.session.query(db.func.max(ALRound.labels_used))
        .filter(ALRound.dataset_id == dataset_id, ALRound.status != "labelling")
        .scalar()
        or 0
    )


def _settle_round(round_row):
    """Read-time sweep, like training's: a round whose job ended without
    finishing goes back to labelling with the reason."""
    if round_row.status == "training" and round_row.job_id:
        job = db.session.get(TrainingJob, round_row.job_id)
        if job is not None and job.status in ACTIVE_JOB_STATUSES:
            redis_client = current_app.redis_client
            stale = current_app.config.get("LEGAL_ML_HEARTBEAT_STALE_SECONDS", 120)
            sweep_stale_job(job, redis_client, stale)
            queue = current_app.config.get("LEGAL_ML_TRAINING_QUEUE", "training_queue")
            sweep_lost_job(job, redis_client, queue, stale)
        if job is None or job.status in ("failed", "cancelled"):
            round_row.status = "labelling"
            round_row.labels_used = None
            round_row.error = (job.error_message if job else None) or "Training was stopped."
            db.session.commit()
    elif round_row.status == "scoring":
        limit = current_app.config.get("LEGAL_ML_AL_SCORING_STALE_SECONDS", DEFAULT_SCORING_STALE_SECONDS)
        if datetime.utcnow() - round_row.updated_at > timedelta(seconds=limit):
            fail_scoring_round(round_row.id)


def _retrain_state(config, round_row, counts):
    trained = sum(counts.values())
    new_labels = max(trained - _labels_at_last_train(config.dataset_id), 0)
    running = round_row.status in ACTIVE_ROUND_STATUSES
    allowed = new_labels > 0 and not running
    return {
        "new_labels": new_labels,
        "allowed": allowed,
        "suggested": allowed and new_labels >= config.batch_size,
    }


def loop_state(dataset):
    config = dataset.al_config
    if config is None:
        return {"active": False}

    round_row = _current_round(dataset.id)
    _settle_round(round_row)
    counts = _confirmed_label_counts(dataset.id)
    return {
        "active": True,
        "strategy": config.strategy,
        "seed_source": config.seed_source,
        "batch_size": config.batch_size,
        "model_id": config.model_id,
        "round": round_row.to_dict(),
        "label_counts": dict(counts),
        "retrain": _retrain_state(config, round_row, counts),
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

    counts = _confirmed_label_counts(dataset.id)
    if not _retrain_state(config, round_row, counts)["allowed"]:
        raise LoopError("There are no new labels to train on.", 409)

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
        round_row.status = "labelling"
        round_row.labels_used = None
        round_row.error = body.get("error")
        db.session.commit()
        return body, status

    round_row.job_id = body["job_id"]
    db.session.commit()
    return {"message": "Training started", "job_id": body["job_id"], "model_id": str(model.id)}, 202


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

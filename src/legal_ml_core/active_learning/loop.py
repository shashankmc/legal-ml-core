"""The loop's state: starting, retrain requests, and what the user sees next."""
import json
import logging
import secrets
from collections import Counter
from datetime import datetime, timedelta

import numpy as np

from flask import current_app

from ..annotation_queue import (
    LLM_LABEL_SOURCES,
    MIN_TO_SPLIT,
    TRAINED_ON_LABELS,
    create_annotation_tasks,
    file_labels,
    read_rows,
    rows_with_text,
)
from ..db import db
from ..job_state import ACTIVE_JOB_STATUSES, COULD_NOT_START_MESSAGE, sweep_lost_job, sweep_stale_job
from ..models import ALConfig, ALRound, AnnotationTask, LLMSuggestion, Model, TrainingJob
from .scoring import MIN_HOLDOUT_LABELS, fail_scoring_round, queue_pool_scoring
from .strategies import DEFAULT_STRATEGY, STRATEGIES, score_pool

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 20
# Below this the trainer's stratified 80/20 split cannot hold every class in its
# test side, so it is the floor as well as the default.
MIN_PER_CLASS = 5
# The cap on labels, in batches: past it, a rare class no longer holds training
# back. It is what ends a loop on a class the Dataset has too few of.
DEFAULT_CAP_BATCHES = 5
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


def _cap_batches(config=None):
    config = current_app.config if config is None else config
    return config.get("LEGAL_ML_AL_CAP_BATCHES") or DEFAULT_CAP_BATCHES


def validate_host_config(config):
    """Fail at import, not on a user's first click, on a bad host setting."""
    cap_batches = config.get("LEGAL_ML_AL_CAP_BATCHES")
    if cap_batches is not None and (not isinstance(cap_batches, int) or cap_batches < 1):
        raise ValueError("LEGAL_ML_AL_CAP_BATCHES must be a whole number above zero")
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
        "cap_batches": _cap_batches(),
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


def _rows_the_llm_touched(dataset_id):
    """Rows with a suggestion or an LLM label: neither may reach the holdout."""
    rows = (
        db.session.query(AnnotationTask.item_index)
        .outerjoin(LLMSuggestion, LLMSuggestion.task_id == AnnotationTask.id)
        .filter(
            AnnotationTask.dataset_id == dataset_id,
            db.or_(LLMSuggestion.id.isnot(None), AnnotationTask.label_source.in_(LLM_LABEL_SOURCES)),
        )
    )
    return {row.item_index for row in rows}


def _labels_so_far(df, dataset):
    """Row position to label, for every row with a label from the file or saved."""
    labels = {int(row): str(value) for row, value in file_labels(df, dataset).items()}
    saved = AnnotationTask.query.filter_by(dataset_id=dataset.id, status="annotated")
    labels.update({task.item_index: task.annotation_label for task in saved})
    return labels


def _shares(sizes, total):
    """``total`` split across ``sizes`` by largest remainder, so it adds up. The
    rows beyond the whole shares go to the largest remainders, a group of one
    last: with no second example, it cannot train if it is held back."""
    grand_total = sum(sizes.values())
    exact = {key: size * total / grand_total for key, size in sizes.items()}
    shares = {key: int(value) for key, value in exact.items()}
    by_claim = sorted(sizes, key=lambda key: (sizes[key] == 1, shares[key] - exact[key]))
    for key in by_claim[: total - sum(shares.values())]:
        shares[key] += 1
    return shares


def _draw_holdout(df, dataset, percent, seed):
    """Row positions held back: a share of all rows with text, drawn from those no
    LLM touched, so it can hold fewer rows than the share, or none. Each class that
    has labels keeps its share of the draw, and unlabelled rows stay random."""
    rows = rows_with_text(df, dataset)
    size = int(round(len(rows) * percent / 100))
    if percent:
        size = max(size, 1)
    touched = _rows_the_llm_touched(dataset.id)
    labels = _labels_so_far(df, dataset)
    groups = {}
    for row in rows:
        if row not in touched:
            groups.setdefault(labels.get(row), []).append(row)
    if not groups:
        return []

    rng = np.random.default_rng(seed)
    size = min(size, sum(len(group) for group in groups.values()))
    shares = _shares({key: len(group) for key, group in groups.items()}, size)
    drawn = [
        position
        for key, group in groups.items()
        for position in rng.choice(group, size=shares[key], replace=False)
    ]
    return sorted(int(position) for position in drawn)


def _file_label_counts(df, dataset, rows, held_back):
    """Per class, the labels the file gave ``rows`` in or out of the holdout."""
    labels = file_labels(df, dataset)
    labels = labels[labels.index.isin(rows) == held_back]
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
        file_label_counts=_file_label_counts(df, dataset, holdout, held_back=False),
        holdout_file_counts=_file_label_counts(df, dataset, holdout, held_back=True),
    )
    db.session.add(config)
    _shuffle_pool(tasks, seed)
    db.session.add(ALRound(dataset_id=dataset.id, number=0, strategy="random"))


def set_strategy(dataset, strategy):
    """Takes effect at the next rescore; caller commits."""
    config = _config_of(dataset)
    config.strategy = _checked_strategy(strategy)


def rename_label(dataset, old, new):
    """Correct a label typed wrong, on every saved item that has it; caller
    commits. Only for typed labels: a Dataset that names its labels has no typos,
    and a label from the file stays as the file has it."""
    _config_of(dataset)
    if dataset.labels:
        raise LoopError("This Dataset names its labels, so they cannot be changed here.")
    if not isinstance(new, str) or not new.strip():
        raise LoopError("Type the label to use instead.")
    changed = AnnotationTask.query.filter_by(
        dataset_id=dataset.id, status="annotated", annotation_label=old
    ).update({"annotation_label": new.strip()}, synchronize_session=False)
    if not changed:
        raise LoopError("No saved label has that name.", 404)


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


def _examples_to_split(classes):
    """The fewest examples whose 20% test side can hold each of ``classes`` once."""
    return 5 * classes - 4


def _training_readiness(dataset, config, counts):
    """What these confirmed labels allow. Only the counts decide: a class exists
    once it has a label, whatever it is called and however many there are."""
    seen = {str(name): n for name, n in counts.items() if n > 0}
    trainable = [n for n in seen.values() if n >= MIN_TO_SPLIT]
    return {
        "labels_total": sum(seen.values()),
        "label_cap": config.batch_size * _cap_batches(),
        "classes_short": {
            name: config.min_per_class - n for name, n in sorted(seen.items()) if n < config.min_per_class
        },
        "classes_left_out": sorted(name for name, n in seen.items() if n < MIN_TO_SPLIT),
        "classes_unseen": sorted(set(dataset.labels or []) - set(seen)),
        "needs_second_label": len(trainable) < 2,
        "classes_to_train": len(trainable),
        "labels_needed": max(_examples_to_split(len(trainable)) - sum(trainable), 0),
    }


def _refusal(config, readiness):
    """Why these labels cannot train, else ``None``."""
    if readiness["classes_short"] and readiness["labels_total"] < readiness["label_cap"]:
        return (
            f"Each label needs at least {config.min_per_class} examples before training, "
            f"or {readiness['label_cap']} labels in all."
        )
    if readiness["needs_second_label"]:
        return "Training needs at least 2 examples each of two labels."
    if readiness["labels_needed"]:
        classes = readiness["classes_to_train"]
        return f"With {classes} labels, training needs at least {_examples_to_split(classes)} examples in all."
    return None


def training_refusal(dataset):
    """Why this Dataset cannot train yet, else ``None``. Only a Dataset in a loop
    has a minimum, and it binds every way of starting a run."""
    config = ALConfig.query.filter_by(dataset_id=str(dataset.id)).first()
    if config is None:
        return None
    return _refusal(config, _training_readiness(dataset, config, _confirmed_label_counts(config)))


def _retrain_state(dataset, config, round_row, counts):
    trained = sum(counts.values())
    new_labels = max(trained - _labels_at_last_train(config.dataset_id), 0)
    readiness = _training_readiness(dataset, config, counts)
    running = round_row.status in ACTIVE_ROUND_STATUSES
    allowed = new_labels > 0 and not running and _refusal(config, readiness) is None
    return {
        "new_labels": new_labels,
        "allowed": allowed,
        "suggested": allowed and new_labels >= config.batch_size,
        **readiness,
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
        "typed_labels": not dataset.labels,
        "retrain": _retrain_state(dataset, config, round_row, counts),
        "holdout": _holdout_state(config),
    }


def _holdout_labelled(config):
    """Rows labelled in the file have no task, so only the unlabelled are counted.
    An LLM's label does not count: it could not grade the model."""
    unlabelled = AnnotationTask.query.filter(
        AnnotationTask.dataset_id == config.dataset_id,
        AnnotationTask.is_holdout.is_(True),
        db.or_(AnnotationTask.status != "annotated", AnnotationTask.label_source.in_(LLM_LABEL_SOURCES)),
    ).count()
    return len(config.holdout_indices or []) - unlabelled


def _holdout_needed(config):
    return min(MIN_HOLDOUT_LABELS, len(config.holdout_indices or []))


def _holdout_label_counts(config):
    """Per class, the held-back labels that could grade a Model."""
    rows = (
        db.session.query(AnnotationTask.annotation_label, db.func.count())
        .filter(
            AnnotationTask.dataset_id == config.dataset_id,
            AnnotationTask.is_holdout.is_(True),
            AnnotationTask.status == "annotated",
            db.or_(AnnotationTask.label_source.is_(None), AnnotationTask.label_source.notin_(LLM_LABEL_SOURCES)),
        )
        .group_by(AnnotationTask.annotation_label)
        .all()
    )
    return dict(Counter(dict(rows)) + Counter(config.holdout_file_counts or {}))


def _holdout_state(config):
    model = db.session.get(Model, config.model_id) if config.model_id else None
    return {
        "percent": config.holdout_percent,
        "size": len(config.holdout_indices or []),
        "labelled": _holdout_labelled(config),
        "label_counts": _holdout_label_counts(config),
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
    refusal = training_refusal(dataset)
    if refusal:
        raise LoopError(refusal, 409)

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

    message = {"action": "verify", "dataset_id": str(dataset.id), "model_id": str(model.id)}
    try:
        current_app.redis_client.lpush(_scoring_queue(), json.dumps(message))
    except Exception:
        logger.exception("dataset %s: the score check could not be queued", dataset.id)
        raise LoopError("The score could not be checked. Try again.", 503)


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

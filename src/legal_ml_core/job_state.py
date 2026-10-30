"""Training job state transitions shared by the worker and the API.

Every status change goes through `transition`, a guarded
`UPDATE ... WHERE status IN from_statuses`: terminal states
(completed/failed/cancelled) are never in `from_statuses`, so nothing --
worker, /stop, or the stale sweep -- can overwrite them. Each transition that
lands is logged with job id and owner.
"""
import json
import logging
from datetime import datetime, timedelta

from .db import db
from .liveness import DEFAULT_STALE_SECONDS, STALE_JOB_MESSAGE, WORKER_DOWN_MESSAGE, is_beat_stale
from .models import Model, TrainingJob
from .queue_contract import WORKER_LIVENESS_KEY, job_heartbeat_key

logger = logging.getLogger(__name__)

ACTIVE_JOB_STATUSES = ("pending", "running")
TERMINAL_JOB_STATUSES = ("completed", "failed", "cancelled")

# Shown when infrastructure (a loop error, Redis unreachable) ended the job:
# retryable, and never the raw internal error text.
COULD_NOT_START_MESSAGE = "Training could not start. Try again."


def transition(job_id, from_statuses, to_status, model_status=None, reason=None, **values):
    """Guarded status change. Returns True when the row was still in one of
    `from_statuses` and was updated; optionally sets the model's status too."""
    updated = TrainingJob.query.filter(
        TrainingJob.id == job_id, TrainingJob.status.in_(from_statuses)
    ).update(dict(values, status=to_status), synchronize_session=False)
    db.session.commit()
    if updated != 1:
        return False
    job = db.session.get(TrainingJob, job_id)
    owner_id = job.owner_id if job else None
    if model_status and job is not None:
        model = db.session.get(Model, job.model_id)
        if model:
            model.status = model_status
            db.session.commit()
    logger.info(
        "training job %s (owner %s): %s -> %s%s",
        job_id,
        owner_id,
        "|".join(from_statuses),
        to_status,
        f" ({reason})" if reason else "",
    )
    return True


def sweep_stale_job(job, redis_client, stale_after=DEFAULT_STALE_SECONDS):
    """R5 stale sweep: a `running` job whose heartbeat is older than
    `stale_after` seconds (worker died, often OOM) becomes `failed` with a
    plain-language message. Returns True if it swept the job (caller should
    re-read it). Redis errors never fail a job."""
    if job.status != "running":
        return False
    try:
        beat = redis_client.get(job_heartbeat_key(job.id))
    except Exception as e:
        logger.warning("stale sweep skipped for job %s: %s", job.id, e)
        return False
    if not is_beat_stale(beat, stale_after):
        return False
    # Grace for the pickup window, before the first beat lands (rows from
    # before started_at was written fall back to created_at).
    since = job.started_at or job.created_at
    if beat is None and since is not None:
        if datetime.utcnow() - since <= timedelta(seconds=stale_after):
            return False
    swept = transition(
        job.id,
        ("running",),
        "failed",
        model_status="failed",
        reason="no heartbeat",
        error_message=STALE_JOB_MESSAGE,
        completed_at=datetime.utcnow(),
    )
    if swept:
        db.session.refresh(job)
    return swept


def _queued_job_id(raw):
    try:
        return json.loads(raw).get("job_id")
    except Exception:
        return None


def sweep_lost_job(job, redis_client, queue, stale_after=DEFAULT_STALE_SECONDS):
    """A `pending` job past the pickup window that is not in the queue will
    never run: its start failed before queueing, or Redis lost it. It becomes
    `failed`, so it neither shows as stuck nor blocks a new start. Returns True
    if it swept the job. The model only follows when it is left claiming a run
    that does not exist; one that trained anyway keeps its state."""
    if job.status != "pending" or job.created_at is None:
        return False
    if datetime.utcnow() - job.created_at <= timedelta(seconds=stale_after):
        return False
    try:
        queued = redis_client.lrange(queue, 0, -1)
    except Exception as e:
        logger.warning("lost job sweep skipped for job %s: %s", job.id, e)
        return False
    if str(job.id) in {_queued_job_id(raw) for raw in queued}:
        return False
    model = db.session.get(Model, job.model_id)
    others_active = TrainingJob.query.filter(
        TrainingJob.model_id == job.model_id,
        TrainingJob.id != job.id,
        TrainingJob.status.in_(ACTIVE_JOB_STATUSES),
    ).count()
    stranded = model is not None and model.status == "training" and not others_active
    swept = transition(
        job.id,
        ("pending",),
        "failed",
        model_status="failed" if stranded else None,
        reason="never queued",
        error_message=COULD_NOT_START_MESSAGE,
        completed_at=datetime.utcnow(),
    )
    if swept:
        db.session.refresh(job)
    return swept


def worker_is_alive(redis_client, stale_after=DEFAULT_STALE_SECONDS):
    """True if a worker process has beaten within `stale_after`. Unknown
    (Redis error) counts as alive: no false "not running" message."""
    try:
        return not is_beat_stale(redis_client.get(WORKER_LIVENESS_KEY), stale_after)
    except Exception:
        return True


def waiting_message(job, worker_alive):
    """Message for a queued job when no worker is running, else None."""
    if job.status == "pending" and not worker_alive:
        return WORKER_DOWN_MESSAGE
    return None

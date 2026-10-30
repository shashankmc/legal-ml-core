"""Typed documentation of the Redis queue message shapes shared between the
Flask API process (producer) and the worker process (consumer). These are
`TypedDict`s for editor/type-checker support, not runtime-validated schemas —
the existing task-processing code already works and this package's job is to
make the previously-implicit dict shapes explicit, not to add a new
validation layer that could reject payloads the working code already handles.

Queue names (Redis lists, `LPUSH`/`BRPOP`):
  training_queue, prediction_queue, explanation_queue, llm_queue, scoring_queue

Result keys (Redis strings/lists, `SETEX`/`LPUSH` + `GET`/`BLPOP`):
  job_status_{job_id}, prediction_result_{prediction_id}, explanation_{explanation_id},
  llm_job_status_{job_id}

Control keys (Redis strings, `SETEX` by the API, `EXISTS` by the worker):
  job_cancel_{job_id} -- set by POST /training/stop/<job_id>; the worker checks
  it at pickup, between preprocessing/fit/save stages, and every HF Trainer
  step, and stops the job cooperatively (see compute/tasks.py).

Liveness keys (Redis strings with a TTL, written by the worker; see liveness.py):
  job_heartbeat_{job_id} -- rewritten every heartbeat while the job runs; each
  beat also refreshes job_status_{job_id}'s TTL. Absent for a `running` job
  => the worker died (the API sweep marks it failed).
  legal_ml_worker_alive -- rewritten while any worker process is up. Absent
  => queued jobs report that the training service isn't running.
"""
from typing import Any, List, Optional, TypedDict

# Long enough to outlive any queue wait + training run; the worker never deletes
# it, so a late-delivered task for a stopped job still sees the flag.
JOB_CANCEL_TTL_SECONDS = 24 * 3600

# job_status_{job_id} lifetime; heartbeats refresh it so long runs keep it.
JOB_STATUS_TTL_SECONDS = 3600

PREDICTION_RESULT_TTL_SECONDS = 300
BATCH_PREDICTION_RESULT_TTL_SECONDS = 1800
EXPLANATION_RESULT_TTL_SECONDS = 3600
LLM_JOB_STATUS_TTL_SECONDS = 24 * 3600

WORKER_LIVENESS_KEY = "legal_ml_worker_alive"


def job_status_key(job_id):
    return f"job_status_{job_id}"


def job_heartbeat_key(job_id):
    return f"job_heartbeat_{job_id}"


def job_cancel_key(job_id):
    return f"job_cancel_{job_id}"


def prediction_result_key(prediction_id):
    return f"prediction_result_{prediction_id}"


def batch_prediction_result_key(batch_id):
    return f"batch_prediction_result_{batch_id}"


def explanation_result_key(explanation_id):
    return f"explanation_{explanation_id}"


def llm_job_status_key(job_id):
    return f"llm_job_status_{job_id}"


def is_cancel_requested(redis_client, job_id):
    return bool(redis_client.exists(job_cancel_key(job_id)))


class TrainingTaskPayload(TypedDict, total=False):
    job_id: str
    model_id: str
    dataset_id: str
    model_type: str
    parameters: dict
    dataset_path: str
    task_type: str
    preprocessing_options: dict
    test_size: float
    random_state: int
    al_round_id: str  # set on an active learning round's run: rescore after it


class ScoringTaskPayload(TypedDict):
    """Sent once a round's model is saved. The worker rescores the pool."""

    round_id: str
    dataset_id: str
    model_id: str
    strategy: str


class VerifyTaskPayload(TypedDict):
    """Sent by POST /datasets/<id>/verify, on the same queue. The worker grades
    the saved model on the holdout."""

    action: str  # always "verify"
    dataset_id: str
    model_id: str


class JobStatusUpdate(TypedDict, total=False):
    status: str  # "running" | "completed" | "failed" | "cancelled"
    progress: int  # 0-100
    metrics: Optional[dict]
    model_path: Optional[str]
    error: Optional[str]


class PredictionTaskPayload(TypedDict):
    prediction_id: str
    model_id: str
    model_type: str
    texts: List[str]
    user_id: str


class PredictionResult(TypedDict, total=False):
    predictions: List[dict]
    model_type: str
    error: str


class ExplanationTaskPayload(TypedDict):
    explanation_id: str
    model_id: str
    text: str
    method: str  # "lime" | "shap"
    user_id: str


class ExplanationResult(TypedDict, total=False):
    explanation: Any
    text: str
    method: str
    model_id: str
    status: str
    error: str


class LLMAnnotationTaskPayload(TypedDict):
    """Carries the credential *mode*, never a key: the worker reads keys from
    the database or its own environment, so none pass through Redis."""

    job_id: str
    dataset_id: str
    owner_id: str
    mode: str  # "shared" | "byok"
    task_ids: List[str]


class LLMJobStatus(TypedDict, total=False):
    owner_id: str
    status: str  # "queued" | "running" | "completed" | "failed" | "budget_exhausted"
    total: int
    done: int
    created: int
    invalid: int
    error: str

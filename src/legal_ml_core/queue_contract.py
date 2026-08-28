"""Typed documentation of the Redis queue message shapes shared between the
Flask API process (producer) and the worker process (consumer). These are
`TypedDict`s for editor/type-checker support, not runtime-validated schemas —
the existing task-processing code already works and this package's job is to
make the previously-implicit dict shapes explicit, not to add a new
validation layer that could reject payloads the working code already handles.

Queue names (Redis lists, `LPUSH`/`BRPOP`):
  training_queue, prediction_queue, explanation_queue

Result keys (Redis strings/lists, `SETEX`/`LPUSH` + `GET`/`BLPOP`):
  job_status_{job_id}, prediction_result_{prediction_id}, explanation_{explanation_id}
"""
from typing import Any, List, Optional, TypedDict


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


class JobStatusUpdate(TypedDict, total=False):
    status: str  # "running" | "completed" | "failed"
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

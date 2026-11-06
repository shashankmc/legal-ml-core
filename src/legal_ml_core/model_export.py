"""Ask the worker to export a Model for publishing, and wait for the answer.

The API process cannot load a saved Model (its preprocessor needs nltk, and a
transformer needs torch), so the worker exports it, as it predicts. The answer
names the export directory on the shared model storage, its files and the
round-trip check (BAIGLE #69, ADR 0006).
"""
import json
import uuid

from flask import current_app, has_app_context

from .queue_contract import model_export_result_key

# Below a 120 s request timeout, so a slow export fails with a reason.
EXPORT_TIMEOUT_SECONDS = 110


class ExportError(Exception):
    """The worker could not export the Model, or did not answer in time."""


def _queue():
    if has_app_context():
        return current_app.config.get("LEGAL_ML_PREDICTION_QUEUE", "prediction_queue")
    return "prediction_queue"


def queue_model_export(redis_client, model, texts):
    """Queue the export of ``model``, checked on ``texts``. Returns its id."""
    export_id = str(uuid.uuid4())
    redis_client.lpush(
        _queue(),
        json.dumps(
            {
                "action": "export",
                "export_id": export_id,
                "model_id": str(model.id),
                "model_type": model.model_type,
                "texts": list(texts),
            }
        ),
    )
    return export_id


def wait_for_model_export(redis_client, export_id, timeout=EXPORT_TIMEOUT_SECONDS):
    """The worker's answer: {"directory", "files", "round_trip": {"ok", "detail", "compared"}}."""
    answer = redis_client.blpop(model_export_result_key(export_id), timeout=timeout)
    if not answer:
        raise ExportError("The training service did not answer in time. Try again later.")
    result = json.loads(answer[1])
    if result.get("error"):
        raise ExportError(result["error"])
    return result


def request_model_export(redis_client, model, texts, timeout=EXPORT_TIMEOUT_SECONDS):
    return wait_for_model_export(redis_client, queue_model_export(redis_client, model, texts), timeout)

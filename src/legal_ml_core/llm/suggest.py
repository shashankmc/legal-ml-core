"""Generating suggestions, for the API (one batch) and the worker (a whole job).

Suggestions are stored apart from labels, so training only ever sees what a human
confirmed or bulk-accepted.
"""

import json
import logging
import uuid

from ..db import db
from ..models import AnnotationTask, Dataset, LLMPrompt, LLMSuggestion
from ..queue_contract import LLM_JOB_STATUS_TTL_SECONDS, llm_job_status_key
from .access import CredentialError, credential_for_mode
from .client import LLMError, estimate_tokens
from .config import int_setting
from .metering import usage_meter
from .prompts import (
    DEFAULT_TEMPLATE,
    MAX_EXAMPLES,
    InvalidOutput,
    build_messages,
    parse_suggestion,
)

logger = logging.getLogger(__name__)

CONFIRMED_SOURCES = ("human", "llm_confirmed")
METER_DOWN_MESSAGE = (
    "The shared LLM key is unavailable right now. Your own API key would still work."
)


class BudgetExhausted(Exception):
    pass


def current_prompt(dataset):
    """``(version, template, definitions)``. Version 0 is the unedited default."""
    row = (
        LLMPrompt.query.filter_by(dataset_id=dataset.id)
        .order_by(LLMPrompt.version.desc())
        .first()
    )
    if row is None:
        return 0, DEFAULT_TEMPLATE, {}
    return row.version, row.template, row.definitions or {}


def confirmed_examples(dataset):
    tasks = (
        AnnotationTask.query.filter(
            AnnotationTask.dataset_id == dataset.id,
            AnnotationTask.status == "annotated",
            AnnotationTask.label_source.in_(CONFIRMED_SOURCES),
        )
        .order_by(AnnotationTask.item_index)
        .limit(MAX_EXAMPLES)
        .all()
    )
    return [(task.text, task.annotation_label) for task in tasks]


def remaining_tokens(owner_id):
    """Fails closed: an unreachable meter blocks the shared key, never BYOK."""
    try:
        return usage_meter().remaining(owner_id)
    except Exception:
        logger.exception("usage meter check failed")
        raise LLMError(METER_DOWN_MESSAGE)


def call_metered(credential, owner_id, messages):
    """One LLM call. The shared key is checked before and metered after."""
    if credential.mode != "shared":
        return credential.client().chat(messages)[0]

    remaining = remaining_tokens(owner_id)
    if remaining is not None and remaining <= 0:
        raise BudgetExhausted()
    content, tokens = credential.client().chat(messages)
    try:
        usage_meter().record(owner_id, tokens, str(uuid.uuid4()))
    except Exception:
        logger.exception("usage meter record failed")
        raise LLMError(METER_DOWN_MESSAGE)
    return content


def estimate_job_tokens(dataset, tasks):
    version, template, definitions = current_prompt(dataset)
    examples = confirmed_examples(dataset)
    per_call = sum(
        estimate_tokens(m["content"])
        for m in build_messages(template, definitions, dataset.labels, examples, "")
    )
    output = int_setting("LLM_MAX_OUTPUT_TOKENS")
    limit = int_setting("LLM_MAX_TEXT_CHARS")
    return sum(
        per_call + output + estimate_tokens(task.text[:limit]) for task in tasks
    )


def suggest_tasks(dataset, tasks, credential, owner_id, on_progress=None):
    """Create suggestions for ``tasks``, one call each, committing as it goes.

    Work already done survives ``BudgetExhausted`` and ``LLMError``, and rerunning
    skips it. Output failing the constraint is discarded and counted as invalid.
    """
    version, template, definitions = current_prompt(dataset)
    examples = confirmed_examples(dataset)
    existing = {
        row.task_id
        for row in LLMSuggestion.query.filter(
            LLMSuggestion.task_id.in_([t.id for t in tasks]),
            LLMSuggestion.model == credential.model,
            LLMSuggestion.prompt_version == version,
        )
    }
    result = {"created": [], "skipped": 0, "invalid": 0}

    for task in tasks:
        if task.id in existing:
            result["skipped"] += 1
        else:
            messages = build_messages(
                template, definitions, dataset.labels, examples, task.text
            )
            content = call_metered(credential, owner_id, messages)
            try:
                label, rationale = parse_suggestion(content, dataset.labels)
            except InvalidOutput as exc:
                logger.info("LLM output discarded for task %s: %s", task.id, exc)
                result["invalid"] += 1
            else:
                suggestion = LLMSuggestion(
                    task_id=task.id,
                    model=credential.model,
                    prompt_version=version,
                    label=label,
                    rationale=rationale,
                )
                db.session.add(suggestion)
                db.session.commit()
                result["created"].append(suggestion)
        if on_progress:
            on_progress(result)
    return result


def _set_status(redis_client, job_id, **fields):
    key = llm_job_status_key(job_id)
    raw = redis_client.get(key)
    status = json.loads(raw) if raw else {}
    status.update(fields)
    redis_client.setex(key, LLM_JOB_STATUS_TTL_SECONDS, json.dumps(status))


def process_llm_annotation_task(task_info, redis_client):
    """Worker entry point for a standalone annotation job."""
    job_id = task_info["job_id"]
    owner_id = task_info["owner_id"]
    dataset = Dataset.query.filter_by(
        id=task_info["dataset_id"], owner_id=owner_id
    ).first()
    if dataset is None:
        _set_status(redis_client, job_id, status="failed", error="Dataset not found.")
        return

    tasks = (
        AnnotationTask.query.filter(
            AnnotationTask.dataset_id == dataset.id,
            AnnotationTask.id.in_(task_info["task_ids"]),
            AnnotationTask.status == "pending",
        )
        .order_by(AnnotationTask.item_index)
        .all()
    )
    _set_status(redis_client, job_id, status="running", total=len(tasks), done=0)

    def progress(result):
        _set_status(
            redis_client,
            job_id,
            done=len(result["created"]) + result["skipped"] + result["invalid"],
            created=len(result["created"]),
            invalid=result["invalid"],
        )

    try:
        credential = credential_for_mode(owner_id, task_info["mode"])
        if credential is None:
            raise CredentialError("The LLM key for this job is no longer available.")
        suggest_tasks(dataset, tasks, credential, owner_id, on_progress=progress)
    except BudgetExhausted:
        _set_status(
            redis_client,
            job_id,
            status="budget_exhausted",
            error="The monthly limit was reached. Add your own API key to continue.",
        )
    except (LLMError, CredentialError) as exc:
        _set_status(redis_client, job_id, status="failed", error=str(exc))
    else:
        _set_status(redis_client, job_id, status="completed")


def fail_llm_job(raw, redis_client):
    try:
        job_id = json.loads(raw).get("job_id")
    except Exception:
        job_id = None
    if job_id:
        _set_status(
            redis_client, job_id, status="failed", error="The annotation job failed."
        )

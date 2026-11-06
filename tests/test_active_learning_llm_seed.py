"""An LLM-suggested first batch (BAIGLE #67).

The LLM labels a random sample of the pool, then the first batch is picked so
that each suggested label is in it, rarest first. Without it a random first
batch can miss a rare label entirely. The holdout is drawn before, so no
suggestion ever touches it.
"""
import json

import pytest
from cryptography.fernet import Fernet

from legal_ml_core import AnnotationTask, LLMSuggestion, db
from legal_ml_core.llm import client as llm_client
from legal_ml_core.llm.suggest import process_llm_annotation_task
from tests.active_learning_helpers import label_next, link_dataset, start, state
from tests.conftest import TEST_USER_ID

DESTINATION = "llm.test"
LABELS = ["obligation", "permission"]
# 54 obligations and 6 permissions: a random batch of 6 often has no permission.
TEXTS = [f"the tenant shall pay rent {i}" for i in range(54)] + [f"the tenant may sublet {i}" for i in range(6)]


class FakeProvider:
    """Stands in for ``requests.post``: 'may', or a text in ``permissions``,
    reads as a permission."""

    def __init__(self):
        self.calls = []
        self.fail = False
        self.permissions = set()

    def __call__(self, url, json=None, headers=None, **kwargs):
        self.calls.append(json)
        self._text = json["messages"][-1]["content"]
        return self

    @property
    def status_code(self):
        return 500 if self.fail else 200

    def json(self):
        rare = " may " in self._text or any(text in self._text for text in self.permissions)
        label = "permission" if rare else "obligation"
        content = json.dumps({"label": label, "rationale": "the verb"})
        return {"choices": [{"message": {"content": content}}], "usage": {"total_tokens": 50}}


@pytest.fixture
def provider(monkeypatch):
    fake = FakeProvider()
    monkeypatch.setattr(llm_client.requests, "post", fake)
    return fake


@pytest.fixture
def headers(app):
    app.config.update(
        LLM_ENCRYPTION_KEY=Fernet.generate_key().decode(),
        LLM_SHARED_ENDPOINT={
            "base_url": "https://llm.test/v1",
            "api_key": "shared-key",
            "model": "shared-model",
            "destination": DESTINATION,
        },
        FEATURE_GATE=lambda feature, claims: feature == "llm.shared_key",
    )
    claims = {"sub": TEST_USER_ID, "email": "a@um.test", "email_verified": True}
    return {"Authorization": f"Bearer test-token::{json.dumps(claims)}"}


@pytest.fixture
def dataset_id(app, tmp_path):
    return link_dataset(app, tmp_path, TEXTS, dataset_labels=LABELS)


def _ack(client, headers, dataset_id):
    resp = client.post(
        f"/api/llm/datasets/{dataset_id}/acknowledgement", json={"destination": DESTINATION}, headers=headers
    )
    assert resp.status_code == 200, resp.get_json()


def _start_llm_seed(client, headers, dataset_id, **options):
    return start(
        client, headers, dataset_id, seed_source="llm", batch_size=6, holdout_percent=10, confirm_estimate=True, **options
    )


def _queued(fake_redis):
    return [json.loads(raw) for raw in fake_redis.lrange("llm_queue", 0, -1)]


def test_the_llm_seed_is_offered(client, headers):
    resp = client.get("/api/active-learning/settings", headers=headers)

    assert resp.get_json()["seed_sources"] == {"random": True, "llm": True}


def test_starting_with_an_llm_seed_queues_one_job_on_a_pool_sample(client, headers, dataset_id, fake_redis):
    _ack(client, headers, dataset_id)

    resp = _start_llm_seed(client, headers, dataset_id)

    assert resp.status_code == 201, resp.get_json()
    assert resp.get_json()["seed"]["status"] == "queued"
    [job] = _queued(fake_redis)
    assert job["dataset_id"] == dataset_id and job["owner_id"] == TEST_USER_ID
    # Three times the batch, and none of it from the holdout.
    assert len(job["task_ids"]) == 18
    holdout = {task.id for task in AnnotationTask.query.filter_by(dataset_id=dataset_id, is_holdout=True)}
    assert holdout and not holdout & set(job["task_ids"])


def test_an_llm_seed_needs_the_reviewed_estimate(client, headers, dataset_id, fake_redis):
    _ack(client, headers, dataset_id)

    resp = start(client, headers, dataset_id, seed_source="llm", batch_size=6)

    assert resp.status_code == 400
    assert state(client, headers, dataset_id)["active"] is False
    assert _queued(fake_redis) == []


def test_an_llm_seed_needs_the_acknowledgement(client, headers, dataset_id, fake_redis):
    resp = _start_llm_seed(client, headers, dataset_id)

    assert resp.status_code == 409
    assert resp.get_json()["code"] == "acknowledgement_required"
    assert state(client, headers, dataset_id)["active"] is False
    assert _queued(fake_redis) == []


def _run_seed_job(fake_redis):
    [raw] = fake_redis.lrange("llm_queue", 0, -1)
    fake_redis.delete("llm_queue")
    process_llm_annotation_task(json.loads(raw), fake_redis)


def _first_batch(client, headers, dataset_id, rare):
    """The next six items the queue serves, labelled as a user would."""
    return label_next(client, headers, dataset_id, lambda text: "permission" if text in rare else "obligation", 6)


def test_the_first_batch_has_every_suggested_label_with_its_suggestion(
    client, headers, app, tmp_path, fake_redis, provider
):
    dataset_id = link_dataset(app, tmp_path, [f"the tenant shall pay rent {i}." for i in range(60)], dataset_labels=LABELS)
    _ack(client, headers, dataset_id)
    assert _start_llm_seed(client, headers, dataset_id).status_code == 201
    # The LLM finds the rare label only in the two sampled items the random
    # order would serve last, so a random first batch never has it.
    [job] = _queued(fake_redis)
    sample = AnnotationTask.query.filter(AnnotationTask.id.in_(job["task_ids"])).order_by(AnnotationTask.score)
    rare = [task.text for task in sample.limit(2)]
    provider.permissions = set(rare)

    _run_seed_job(fake_redis)

    assert state(client, headers, dataset_id)["seed"]["status"] == "completed"
    batch = _first_batch(client, headers, dataset_id, rare)
    assert set(rare) <= {task["text"] for task in batch}
    suggested = {s.task_id for s in LLMSuggestion.query}
    assert all(task["id"] in suggested for task in batch)


def test_a_seed_over_the_monthly_limit_starts_nothing(client, app, headers, dataset_id, fake_redis):
    app.config["LLM_MONTHLY_TOKEN_CAP"] = 10
    _ack(client, headers, dataset_id)

    resp = _start_llm_seed(client, headers, dataset_id)

    assert resp.status_code == 429
    assert resp.get_json()["code"] == "budget_exhausted"
    assert resp.get_json()["byok_available"] is True
    assert state(client, headers, dataset_id)["active"] is False
    assert _queued(fake_redis) == []


def test_a_failed_seed_leaves_the_random_order_to_label(client, headers, dataset_id, fake_redis, provider):
    _ack(client, headers, dataset_id)
    assert _start_llm_seed(client, headers, dataset_id).status_code == 201
    provider.fail = True

    _run_seed_job(fake_redis)

    seed = state(client, headers, dataset_id)["seed"]
    assert seed["status"] == "failed" and seed["error"]
    assert len(_first_batch(client, headers, dataset_id, set())) == 6


def _estimate(client, headers, dataset_id, **options):
    return client.post(
        f"/api/active-learning/datasets/{dataset_id}/seed-estimate",
        json={"batch_size": 6, "holdout_percent": 10, **options},
        headers=headers,
    )


def test_the_seed_is_priced_before_anything_starts(client, headers, dataset_id, fake_redis):
    resp = _estimate(client, headers, dataset_id)

    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["tasks"] == 18
    assert body["estimated_tokens"] > 0
    assert body["destination"] == DESTINATION
    assert body["within_budget"] is True
    assert state(client, headers, dataset_id)["active"] is False
    assert AnnotationTask.query.count() == 0


def test_the_estimate_says_when_the_seed_is_over_the_limit(client, app, headers, dataset_id):
    app.config["LLM_MONTHLY_TOKEN_CAP"] = 10

    assert _estimate(client, headers, dataset_id).get_json()["within_budget"] is False


def test_a_dataset_without_named_labels_cannot_have_an_llm_seed(client, app, headers, tmp_path):
    typed = link_dataset(app, tmp_path, TEXTS)

    resp = _estimate(client, headers, typed)

    assert resp.status_code == 400
    assert "labels" in resp.get_json()["error"]

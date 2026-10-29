"""Stop really stops training (eng D4, A2, OV-E3, OV-E4).

Pause/release tests use a real MLModelTrainer subclass whose stage methods run
a hook first, so the test can land a Stop request at a precise point in the
worker's run (during fit, or after the worker's final write) and then let the
worker continue.
"""
import json
import os

import pandas as pd
import pytest

from legal_ml_core import Dataset, Model, TrainingJob, db
from legal_ml_core.compute.tasks import process_training_task
from legal_ml_core.compute.trainer import MLModelTrainer
from legal_ml_core.storage import get_model_path
from tests.conftest import TEST_USER_ID


def _csv(tmp_path):
    df = pd.DataFrame(
        {
            "text": [
                "the court ruled for the plaintiff",
                "the defendant was found liable",
                "judgment entered against appellant",
                "the parties settled the dispute",
            ]
            * 3,
            "label": ["civil", "criminal"] * 6,
        }
    )
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def queued_job(app, tmp_path):
    """A pending job as /training/start leaves it, plus its queue payload."""
    dataset_path = _csv(tmp_path)
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="D1",
            file_path=dataset_path,
            row_count=12,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.flush()
        model = Model(
            owner_id=TEST_USER_ID,
            dataset_id=dataset.id,
            name="M1",
            model_type="naive_bayes",
            parameters={},
            status="pending",
        )
        db.session.add(model)
        db.session.flush()
        job = TrainingJob(owner_id=TEST_USER_ID, model_id=model.id, status="pending")
        db.session.add(job)
        db.session.commit()
        task_info = {
            "job_id": job.id,
            "model_id": model.id,
            "dataset_id": dataset.id,
            "model_type": "naive_bayes",
            "parameters": {},
            "dataset_path": dataset_path,
            "task_type": "classification",
            "preprocessing_options": {},
        }
    return task_info


def _paused_trainer(on_fit=None, on_save=None):
    """Trainer factory whose fit/save stages pause on a hook, then continue."""

    class PausingTrainer(MLModelTrainer):
        def train_classification_model(self, *a, **k):
            if on_fit:
                on_fit()
            return super().train_classification_model(*a, **k)

        def save_model(self, path):
            if on_save:
                on_save()
            return super().save_model(path)

    return PausingTrainer


def _status(client, auth_headers, job_id):
    resp = client.get(f"/api/training/status/{job_id}", headers=auth_headers)
    assert resp.status_code == 200
    return resp.get_json()["job"]


def _model_file(app, task_info):
    return get_model_path(app.config["MODEL_FOLDER"], task_info["model_id"])


def test_job_is_running_in_db_from_pickup_with_start_time(
    app, client, auth_headers, fake_redis, queued_job
):
    seen = {}

    def during_fit():
        # Redis says nothing useful here: the DB alone must say running.
        fake_redis.delete(f"job_status_{queued_job['job_id']}")
        seen["job"] = _status(client, auth_headers, queued_job["job_id"])

    process_training_task(
        queued_job,
        fake_redis,
        app.config["MODEL_FOLDER"],
        trainer_factory=_paused_trainer(on_fit=during_fit),
    )

    assert seen["job"]["status"] == "running"
    assert seen["job"]["started_at"] is not None
    assert _status(client, auth_headers, queued_job["job_id"])["status"] == "completed"


def test_stop_on_queued_job_means_it_is_never_trained(
    app, client, auth_headers, fake_redis, queued_job
):
    job_id = queued_job["job_id"]
    assert client.post(f"/api/training/stop/{job_id}", headers=auth_headers).status_code == 200

    def must_not_train():
        raise AssertionError("a stopped queued job must never train")

    process_training_task(
        queued_job,
        fake_redis,
        app.config["MODEL_FOLDER"],
        trainer_factory=_paused_trainer(on_fit=must_not_train),
    )

    job = _status(client, auth_headers, job_id)
    assert job["status"] == "cancelled"
    assert job["started_at"] is None
    assert not os.path.exists(_model_file(app, queued_job))


def test_cancel_flag_alone_stops_a_queued_job_at_pickup(app, client, auth_headers, fake_redis, queued_job):
    """The worker trusts the Redis flag even if the DB write raced behind it."""
    from legal_ml_core.queue_contract import job_cancel_key

    job_id = queued_job["job_id"]
    fake_redis.set(job_cancel_key(job_id), "1")

    process_training_task(queued_job, fake_redis, app.config["MODEL_FOLDER"])

    assert _status(client, auth_headers, job_id)["status"] == "cancelled"
    assert not os.path.exists(_model_file(app, queued_job))


def _stop(client, auth_headers, job_id):
    return client.post(f"/api/training/stop/{job_id}", headers=auth_headers)


def test_stop_during_fit_stays_cancelled_with_no_model_file(
    app, client, auth_headers, fake_redis, queued_job
):
    """Order A: Stop lands before the worker's final write."""
    job_id = queued_job["job_id"]
    saves = []

    process_training_task(
        queued_job,
        fake_redis,
        app.config["MODEL_FOLDER"],
        trainer_factory=_paused_trainer(
            on_fit=lambda: _stop(client, auth_headers, job_id),
            on_save=lambda: saves.append(1),
        ),
    )

    job = _status(client, auth_headers, job_id)
    assert job["status"] == "cancelled"
    assert job["metrics"] is None
    assert saves == []
    assert not os.path.exists(_model_file(app, queued_job))
    assert json.loads(fake_redis.get(f"job_status_{job_id}"))["status"] == "cancelled"


def test_stop_landing_during_save_still_wins_and_leaves_no_model_file(
    app, client, auth_headers, fake_redis, queued_job
):
    """Order A, latest possible point: Stop lands after the last stage check
    but before the guarded final write. The write must miss and the file go."""
    job_id = queued_job["job_id"]

    process_training_task(
        queued_job,
        fake_redis,
        app.config["MODEL_FOLDER"],
        trainer_factory=_paused_trainer(on_save=lambda: _stop(client, auth_headers, job_id)),
    )

    job = _status(client, auth_headers, job_id)
    assert job["status"] == "cancelled"
    assert not os.path.exists(_model_file(app, queued_job))
    with app.app_context():
        assert Model.query.get(queued_job["model_id"]).status != "completed"


def test_stop_after_final_write_is_rejected_and_job_stays_completed(
    app, client, auth_headers, fake_redis, queued_job
):
    """Order B: the worker's final write lands first; a late Stop changes nothing."""
    job_id = queued_job["job_id"]
    process_training_task(queued_job, fake_redis, app.config["MODEL_FOLDER"])

    assert _stop(client, auth_headers, job_id).status_code == 400

    job = _status(client, auth_headers, job_id)
    assert job["status"] == "completed"
    assert os.path.exists(_model_file(app, queued_job))
    with app.app_context():
        assert Model.query.get(queued_job["model_id"]).status == "completed"


def test_error_while_cancelling_never_turns_cancelled_into_failed(
    app, client, auth_headers, fake_redis, queued_job
):
    job_id = queued_job["job_id"]

    def stop_then_crash():
        _stop(client, auth_headers, job_id)
        raise RuntimeError("CUDA out of memory while tearing down")

    process_training_task(
        queued_job,
        fake_redis,
        app.config["MODEL_FOLDER"],
        trainer_factory=_paused_trainer(on_fit=stop_then_crash),
    )

    job = _status(client, auth_headers, job_id)
    assert job["status"] == "cancelled"
    assert job["error_message"] is None


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_status_shows_db_terminal_state_over_live_redis_running(
    app, client, auth_headers, fake_redis, queued_job, terminal
):
    job_id = queued_job["job_id"]
    with app.app_context():
        TrainingJob.query.get(job_id).status = terminal
        db.session.commit()
    fake_redis.setex(
        f"job_status_{job_id}", 3600, json.dumps({"status": "running", "progress": 40})
    )

    assert _status(client, auth_headers, job_id)["status"] == terminal


class _FakeHFTrainer:
    """Stands in for transformers.Trainer: drives the registered callbacks
    through a step loop exactly like Trainer.train does, without a model."""

    instances = []

    def __init__(self, model, args, callbacks=None, **kwargs):
        from transformers import TrainerControl, TrainerState

        self.args = args
        self.callbacks = callbacks or []
        self.state = TrainerState()
        self.control = TrainerControl()
        self.steps_run = 0
        self.evaluated = False
        _FakeHFTrainer.instances.append(self)

    def train(self):
        for step in range(1, 1001):
            self.state.global_step = step
            self.steps_run = step
            for cb in self.callbacks:
                cb.on_step_end(self.args, self.state, self.control)
            if self.control.should_training_stop:
                return

    def evaluate(self):
        self.evaluated = True
        return {}

    def predict(self, dataset):
        self.evaluated = True
        raise AssertionError("predict must be skipped on cancel")


@pytest.fixture
def stub_hf(monkeypatch):
    import sys

    import transformers

    # transformers' lazy loader replaces sys.modules["transformers"] on first
    # attribute access; patch the object that `from transformers import` sees.
    transformers.Trainer, transformers.AutoTokenizer  # force the lazy load
    transformers = sys.modules["transformers"]

    class FakeTokenizer:
        def __call__(self, texts, **kw):
            return {"input_ids": [[1, 2] for _ in texts]}

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: FakeTokenizer())
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        classmethod(lambda cls, *a, **k: object()),
    )
    monkeypatch.setattr(transformers, "Trainer", _FakeHFTrainer)
    _FakeHFTrainer.instances = []
    return _FakeHFTrainer


def test_stop_ends_transformer_training_within_one_logging_step_and_skips_eval(
    fake_redis, tmp_path, stub_hf
):
    from legal_ml_core.compute.trainer import TrainingCancelled

    stop_at_step = 23
    trainer = MLModelTrainer(
        redis_client=fake_redis,
        cancel_check=lambda: stub_hf.instances[0].state.global_step >= stop_at_step,
    )
    texts = ["the court ruled", "the defendant was liable"] * 10
    labels = [0, 1] * 10

    with pytest.raises(TrainingCancelled):
        trainer.train_transformer_model(
            texts, labels, "legalbert", {"model_id": "m", "output_dir": str(tmp_path)}
        )

    hf = stub_hf.instances[0]
    logging_steps = hf.args.logging_steps
    assert hf.steps_run - stop_at_step <= logging_steps
    assert hf.evaluated is False
    assert trainer.model is None

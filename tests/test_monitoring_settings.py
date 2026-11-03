"""A run's monitoring settings (BAIGLE #89): the host's defaults, a run's own
choices under `parameters.monitoring`, the refusals, and which points are kept."""
import json
from types import SimpleNamespace

import pytest

from legal_ml_core import Dataset, Model, TrainingJob, TrainingMetricPoint, db
from legal_ml_core.compute.tasks import MetricLogger, process_training_task
from legal_ml_core.compute.trainer import MLModelTrainer
from legal_ml_core.monitoring import effective_monitoring
from tests.conftest import TEST_USER_ID
from tests.monitoring_helpers import create_model, run_classical, start_run
from tests.test_adjust_and_retrain import _name_groups, _retrain
from tests.test_training_cancel import stub_hf  # noqa: F401
from tests.test_training_parameters import _run, clustering  # noqa: F401


@pytest.fixture
def dataset_id(app, tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("text,label\n" + "the court ruled,civil\nthe jury convicted,criminal\n" * 10)
    dataset = Dataset(
        owner_id=TEST_USER_ID, name="D1", file_path=str(path), row_count=20, columns=[], task_type="classification"
    )
    db.session.add(dataset)
    db.session.commit()
    return dataset.id


@pytest.mark.parametrize(
    "monitoring, field, says",
    [
        ("often", "monitoring", "Monitoring settings must be a set of named settings."),
        ({"colour": "red"}, "monitoring.colour", "There is no monitoring setting called colour."),
        ({"log_every_n_steps": 0}, "monitoring.log_every_n_steps", "Log every N steps must be a whole number above 0."),
        ({"log_every_n_steps": 2.5}, "monitoring.log_every_n_steps", "Log every N steps must be a whole number above 0."),
        ({"log_every_n_steps": True}, "monitoring.log_every_n_steps", "Log every N steps must be a whole number above 0."),
        ({"eval_steps": -5}, "monitoring.eval_steps", "Evaluate every N steps must be a whole number above 0."),
        ({"eval_strategy": "hourly"}, "monitoring.eval_strategy", "Evaluation must be one of: epoch, steps."),
        ({"include": "eval_*"}, "monitoring.include", "Metrics to keep must be a list of names."),
        ({"exclude": [3]}, "monitoring.exclude", "Metrics to drop must be a list of names."),
        ({"learning_curve": "yes"}, "monitoring.learning_curve", "Learning curve must be true or false."),
        ({"system_metrics": 1}, "monitoring.system_metrics", "Record resource use must be true or false."),
        ({"flush_every_points": 0}, "monitoring.flush_every_points", "Points per write must be a whole number above 0."),
        ({"flush_every_points": 1.5}, "monitoring.flush_every_points", "Points per write must be a whole number above 0."),
        ({"flush_every_seconds": 0}, "monitoring.flush_every_seconds", "Seconds between writes must be a number above 0."),
        ({"flush_every_seconds": -1.5}, "monitoring.flush_every_seconds", "Seconds between writes must be a number above 0."),
        ({"flush_every_seconds": "2"}, "monitoring.flush_every_seconds", "Seconds between writes must be a number above 0."),
        ({"flush_every_seconds": True}, "monitoring.flush_every_seconds", "Seconds between writes must be a number above 0."),
    ],
)
def test_bad_monitoring_settings_are_refused_naming_the_field(client, auth_headers, dataset_id, monitoring, field, says):
    response = create_model(client, auth_headers, dataset_id, monitoring)

    assert response.status_code == 400
    assert response.get_json() == {"error": says, "field": field}


def test_an_edit_to_bad_monitoring_settings_is_refused_too(client, auth_headers, dataset_id):
    model_id = create_model(client, auth_headers, dataset_id, {}).get_json()["model"]["id"]

    response = client.put(
        f"/api/models/{model_id}", json={"parameters": {"monitoring": {"eval_strategy": "never"}}}, headers=auth_headers
    )

    assert response.status_code == 400
    assert response.get_json()["field"] == "monitoring.eval_strategy"


DEFAULT_MONITORING = {
    "log_every_n_steps": 10,
    "eval_strategy": "epoch",
    "eval_steps": 100,
    "include": ["*"],
    "exclude": [],
    "learning_curve": False,
    "system_metrics": False,
    "flush_every_points": 50,
    "flush_every_seconds": 2.0,
}


def _recorded(client, auth_headers, model_id):
    runs = client.get(f"/api/models/{model_id}/runs", headers=auth_headers).get_json()["runs"]
    return runs[0]["parameters"]["monitoring"]


def test_a_run_with_no_monitoring_settings_records_the_defaults(client, auth_headers, dataset_id):
    model_id = start_run(client, auth_headers, dataset_id, {})

    assert _recorded(client, auth_headers, model_id) == DEFAULT_MONITORING


def test_the_host_sets_the_defaults_and_a_run_overrides_them(app, client, auth_headers, dataset_id):
    app.config["LEGAL_ML_MONITORING"] = {"log_every_n_steps": 50, "exclude": ["learning_rate"]}

    model_id = start_run(client, auth_headers, dataset_id, {"log_every_n_steps": 5, "eval_strategy": "steps"})

    assert _recorded(client, auth_headers, model_id) == DEFAULT_MONITORING | {
        "log_every_n_steps": 5,
        "eval_strategy": "steps",
        "exclude": ["learning_rate"],
    }


def test_a_bad_host_default_stops_the_app_at_start_not_on_a_run():
    from flask import Flask

    from legal_ml_core import init_ml_feature

    host = Flask(__name__)
    host.config.update(SQLALCHEMY_DATABASE_URI="sqlite:///:memory:", LEGAL_ML_MONITORING={"eval_strategy": "daily"})
    db.init_app(host)

    with pytest.raises(ValueError, match="LEGAL_ML_MONITORING: Evaluation must be one of: epoch, steps."):
        init_ml_feature(host, None, "/tmp/models", db=db)


def test_group_names_are_kept_when_a_retrain_changes_only_the_monitoring(
    client, auth_headers, app, fake_redis, clustering
):
    """Monitoring changes what is watched, not what is learned, so the groups are the same groups."""
    model_id, _ = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})
    _name_groups(client, auth_headers, model_id, "clusters")
    changed = {"n_clusters": 2, "monitoring": {"exclude": ["silhouette_score"]}}
    assert client.put(f"/api/models/{model_id}", json={"parameters": changed}, headers=auth_headers).status_code == 200

    model = _retrain(client, auth_headers, app, fake_redis, model_id)

    assert model.metrics["display_labels"] == {"clusters": {"0": "Contracts"}}


STAGES = ["preprocessing_started", "preprocessing_done", "training_started", "training_done", "model_saved"]


def _reporting_trainer(*names):
    """A classical trainer that also reports these points while it fits, as a
    learning curve will."""

    class ReportingTrainer(MLModelTrainer):
        def train_classification_model(self, *a, **k):
            for name in names:
                self.metric_logger.log(name, 0.5)
            return super().train_classification_model(*a, **k)

    return ReportingTrainer


def _stored(job_id):
    points = TrainingMetricPoint.query.filter_by(training_job_id=job_id).order_by(TrainingMetricPoint.step)
    return [p.metric_name for p in points]


def _names_stored(*args):
    return [p["metric_name"] for p in run_classical(*args)]


def test_a_classical_run_stores_only_the_points_its_globs_keep(client, auth_headers, app, fake_redis, dataset_id):
    trainer = _reporting_trainer("loss", "learning_rate", "eval_accuracy", "eval_loss")

    stored = _names_stored(
        client, auth_headers, app, fake_redis, dataset_id, {"include": ["loss", "eval_*"], "exclude": ["eval_loss"]}, trainer
    )

    assert stored == STAGES[:3] + ["loss", "eval_accuracy"] + STAGES[3:]


def test_the_stages_are_kept_even_when_every_metric_is_dropped(client, auth_headers, app, fake_redis, dataset_id):
    stored = _names_stored(
        client, auth_headers, app, fake_redis, dataset_id, {"exclude": ["*"]}, _reporting_trainer("loss")
    )

    assert stored == STAGES


def test_a_run_keeps_every_point_by_default(client, auth_headers, app, fake_redis, dataset_id):
    stored = _names_stored(client, auth_headers, app, fake_redis, dataset_id, {}, _reporting_trainer("loss", "eval_f1"))

    assert stored == STAGES[:3] + ["loss", "eval_f1"] + STAGES[3:]


def test_a_fine_tuning_run_stores_only_the_points_its_globs_keep(app, dataset_id):
    from legal_ml_core.compute.trainer import TrainingMetricCallback
    from legal_ml_core.monitoring import FilteredMetricLogger

    model = Model(owner_id=TEST_USER_ID, dataset_id=dataset_id, name="M", model_type="legalbert", parameters={})
    db.session.add(model)
    db.session.flush()
    job = TrainingJob(owner_id=TEST_USER_ID, model_id=model.id, status="running")
    db.session.add(job)
    db.session.commit()
    monitoring = effective_monitoring(run={"exclude": ["learning_rate", "grad_norm"]})
    kept = FilteredMetricLogger(MetricLogger(job.id), monitoring)
    callback = TrainingMetricCallback(kept)

    state = SimpleNamespace(epoch=0.5, global_step=10)
    callback.on_log(None, state, None, logs={"loss": 0.7, "learning_rate": 2e-5, "grad_norm": 1.2})
    callback.on_log(None, state, None, logs={"eval_loss": 0.6, "eval_accuracy": 0.8})
    kept.flush()  # reaches the batched logger underneath

    assert _stored(job.id) == ["loss", "eval_loss", "eval_accuracy"]


def _training_arguments(stub_hf, tmp_path, monitoring):
    """The arguments fine-tuning hands the trainer. Stopped at its first step,
    so the stand-in needs no model."""
    from legal_ml_core.compute.trainer import TrainingCancelled

    trainer = MLModelTrainer(redis_client=None, cancel_check=lambda: bool(stub_hf.instances))
    parameters = {"model_id": "m", "output_dir": str(tmp_path), "monitoring": monitoring}
    with pytest.raises(TrainingCancelled):
        trainer.train_transformer_model(["the court ruled", "the jury convicted"] * 10, [0, 1] * 10, "legalbert", parameters)
    return stub_hf.instances[0].args


def test_fine_tuning_logs_and_evaluates_on_the_run_settings(stub_hf, tmp_path):
    monitoring = effective_monitoring(run={"log_every_n_steps": 25, "eval_strategy": "steps", "eval_steps": 40})

    args = _training_arguments(stub_hf, tmp_path, monitoring)

    assert (args.logging_steps, args.eval_strategy, args.eval_steps) == (25, "steps", 40)


def test_fine_tuning_evaluates_each_epoch_by_default(stub_hf, tmp_path):
    args = _training_arguments(stub_hf, tmp_path, effective_monitoring())

    assert (args.logging_steps, args.eval_strategy) == (10, "epoch")


def test_the_trainer_gets_the_monitoring_the_run_recorded(app, client, auth_headers, fake_redis, dataset_id):
    app.config["LEGAL_ML_MONITORING"] = {"log_every_n_steps": 50}
    model_id = start_run(client, auth_headers, dataset_id, {"eval_strategy": "steps"})
    seen = {}

    class Watching(MLModelTrainer):
        def train_classification_model(self, texts, labels, model_type, parameters):
            seen.update(parameters["monitoring"])
            return super().train_classification_model(texts, labels, model_type, parameters)

    task = json.loads(fake_redis.rpop("training_queue"))
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"], trainer_factory=Watching)

    assert seen == _recorded(client, auth_headers, model_id)
    assert (seen["log_every_n_steps"], seen["eval_strategy"]) == (50, "steps")


def test_a_fractional_number_of_seconds_between_writes_is_accepted(client, auth_headers, dataset_id):
    model_id = start_run(client, auth_headers, dataset_id, {"flush_every_seconds": 0.5})

    assert _recorded(client, auth_headers, model_id)["flush_every_seconds"] == 0.5


def test_the_metric_points_are_written_as_often_as_the_run_says(
    app, client, auth_headers, fake_redis, dataset_id, monkeypatch
):
    from legal_ml_core.compute import tasks

    app.config["LEGAL_ML_MONITORING"] = {"flush_every_seconds": 10}
    start_run(client, auth_headers, dataset_id, {"flush_every_points": 7})
    made = []

    class Recording(MetricLogger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(kwargs)

    monkeypatch.setattr(tasks, "MetricLogger", Recording)
    task = json.loads(fake_redis.rpop("training_queue"))
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"], trainer_factory=MLModelTrainer)

    assert made == [{"flush_every_points": 7, "flush_every_seconds": 10}]
    assert _stored(task["job_id"]) == STAGES  # the end-of-run flush still writes every point


def test_the_api_docs_list_every_monitoring_setting_with_its_default():
    from pathlib import Path

    import yaml

    from legal_ml_core.monitoring import MONITORING_FIELDS

    spec = yaml.safe_load((Path(__file__).parent.parent / "docs" / "openapi.yaml").read_text())
    documented = spec["components"]["schemas"]["Monitoring"]["properties"]

    assert {name: field["default"] for name, field in documented.items()} == DEFAULT_MONITORING
    assert set(documented) == set(MONITORING_FIELDS)

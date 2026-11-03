"""A run's monitoring settings (BAIGLE #89): the host's defaults, a run's own
choices under `parameters.monitoring`, the refusals, and which points are kept."""
import pytest

from legal_ml_core import Dataset, db
from tests.conftest import TEST_USER_ID
from tests.test_adjust_and_retrain import _name_groups, _retrain
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


def _create(client, auth_headers, dataset_id, monitoring, model_type="logistic_regression"):
    return client.post(
        "/api/models/",
        json={"name": "M", "dataset_id": dataset_id, "model_type": model_type, "parameters": {"monitoring": monitoring}},
        headers=auth_headers,
    )


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
        ({"system_metrics": 1}, "monitoring.system_metrics", "Resource readings must be true or false."),
    ],
)
def test_bad_monitoring_settings_are_refused_naming_the_field(client, auth_headers, dataset_id, monitoring, field, says):
    response = _create(client, auth_headers, dataset_id, monitoring)

    assert response.status_code == 400
    assert response.get_json() == {"error": says, "field": field}


def test_an_edit_to_bad_monitoring_settings_is_refused_too(client, auth_headers, dataset_id):
    model_id = _create(client, auth_headers, dataset_id, {}).get_json()["model"]["id"]

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
}


def _start(client, auth_headers, dataset_id, monitoring):
    created = _create(client, auth_headers, dataset_id, monitoring)
    assert created.status_code == 201, created.get_json()
    model_id = created.get_json()["model"]["id"]
    started = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)
    assert started.status_code == 200, started.get_json()
    return model_id


def _recorded(client, auth_headers, model_id):
    runs = client.get(f"/api/models/{model_id}/runs", headers=auth_headers).get_json()["runs"]
    return runs[0]["parameters"]["monitoring"]


def test_a_run_with_no_monitoring_settings_records_the_defaults(client, auth_headers, dataset_id):
    model_id = _start(client, auth_headers, dataset_id, {})

    assert _recorded(client, auth_headers, model_id) == DEFAULT_MONITORING


def test_the_host_sets_the_defaults_and_a_run_overrides_them(app, client, auth_headers, dataset_id):
    app.config["LEGAL_ML_MONITORING"] = {"log_every_n_steps": 50, "exclude": ["learning_rate"]}

    model_id = _start(client, auth_headers, dataset_id, {"log_every_n_steps": 5, "eval_strategy": "steps"})

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

"""A run's monitoring settings (BAIGLE #89): the host's defaults, a run's own
choices under `parameters.monitoring`, the refusals, and which points are kept."""
import pytest

from legal_ml_core import Dataset, db
from tests.conftest import TEST_USER_ID


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

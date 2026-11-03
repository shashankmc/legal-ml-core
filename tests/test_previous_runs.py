"""Each run keeps the settings it used and the scores it got (BAIGLE #83)."""
from datetime import datetime, timedelta

from legal_ml_core import Dataset, Model, TrainingJob, db
from tests.conftest import TEST_USER_ID
from tests.test_monitoring_settings import DEFAULT_MONITORING


def _make_model(app, parameters, owner_id=TEST_USER_ID):
    with app.app_context():
        dataset = Dataset(
            owner_id=owner_id,
            name="D1",
            file_path="/tmp/d1.csv",
            row_count=10,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.flush()
        model = Model(
            owner_id=owner_id,
            dataset_id=dataset.id,
            name="M1",
            model_type="logistic_regression",
            parameters=parameters,
            status="pending",
        )
        db.session.add(model)
        db.session.commit()
        return model.id


def _add_run(app, model_id, created_at, owner_id=TEST_USER_ID, **fields):
    with app.app_context():
        job = TrainingJob(owner_id=owner_id, model_id=model_id, created_at=created_at, **fields)
        db.session.add(job)
        db.session.commit()
        return job.id


def _runs(client, headers, model_id):
    response = client.get(f"/api/models/{model_id}/runs", headers=headers)
    assert response.status_code == 200
    return response.get_json()["runs"]


def test_a_run_records_the_settings_it_started_with_defaults_filled_in(client, auth_headers, app, fake_redis):
    """With the defaults, a run saved with some settings compares equal to the
    same run saved by the form, which sends them all."""
    model_id = _make_model(app, {"C": 2.0})
    client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)

    # Settings edited after the start must not rewrite what the run used.
    client.put(f"/api/models/{model_id}", json={"parameters": {"C": 5.0}}, headers=auth_headers)

    [run] = _runs(client, auth_headers, model_id)
    assert run["parameters"] == {
        "C": 2.0,
        "penalty": "l2",
        "solver": "lbfgs",
        "l1_ratio": 0.5,
        "class_weight": "none",
        "monitoring": DEFAULT_MONITORING,
    }
    assert run["status"] == "pending"


def test_runs_list_newest_first_with_headline_scores_and_dates(client, auth_headers, app):
    model_id = _make_model(app, {"C": 1.0})
    _add_run(
        app,
        model_id,
        datetime(2026, 10, 1),
        status="completed",
        parameters={"C": 1.0},
        metrics={"accuracy": 0.8, "f1_score": 0.75, "confusion_matrix": [[4, 1], [1, 4]]},
        started_at=datetime(2026, 10, 1, 9),
        completed_at=datetime(2026, 10, 1, 10),
    )
    _add_run(app, model_id, datetime(2026, 10, 2), status="failed", parameters={"C": 9.0}, error_message="Out of memory")

    newest, oldest = _runs(client, auth_headers, model_id)

    assert newest["status"] == "failed"
    assert newest["error_message"] == "Out of memory"
    assert newest["scores"] == {}
    assert oldest["scores"] == {"accuracy": 0.8, "f1_score": 0.75}
    assert oldest["created_at"] == "2026-10-01T00:00:00"
    assert oldest["started_at"] == "2026-10-01T09:00:00"
    assert oldest["completed_at"] == "2026-10-01T10:00:00"


def test_a_run_from_before_the_snapshot_lists_with_no_recorded_settings(client, auth_headers, app):
    model_id = _make_model(app, {"C": 1.0})
    _add_run(app, model_id, datetime(2026, 9, 1), status="completed", metrics={"accuracy": 0.7})

    [run] = _runs(client, auth_headers, model_id)

    assert run["parameters"] is None
    assert run["scores"] == {"accuracy": 0.7}


def test_another_owners_runs_are_not_found(client, auth_headers, app):
    model_id = _make_model(app, {}, owner_id="other-user")
    _add_run(app, model_id, datetime(2026, 10, 1), owner_id="other-user", status="completed")

    response = client.get(f"/api/models/{model_id}/runs", headers=auth_headers)

    assert response.status_code == 404


def test_runs_past_the_fiftieth_are_signalled_by_has_more(client, auth_headers, app):
    """Without it, the 50th run would read as the Model's first."""
    model_id = _make_model(app, {"C": 1.0})
    for day in range(1, 52):
        _add_run(app, model_id, datetime(2026, 8, 1) + timedelta(days=day), status="completed")

    body = client.get(f"/api/models/{model_id}/runs", headers=auth_headers).get_json()

    assert len(body["runs"]) == 50
    assert body["has_more"] is True
    assert body["runs"][-1]["created_at"] == "2026-08-03T00:00:00"


def test_a_short_list_of_runs_has_no_more(client, auth_headers, app):
    model_id = _make_model(app, {"C": 1.0})
    _add_run(app, model_id, datetime(2026, 10, 1), status="completed")

    body = client.get(f"/api/models/{model_id}/runs", headers=auth_headers).get_json()

    assert body["has_more"] is False

from legal_ml_core import Dataset, Model, TrainingJob, db
from tests.conftest import TEST_USER_ID


def _make_dataset(app):
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="D1",
            file_path="/tmp/d1.csv",
            row_count=10,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.commit()
        return dataset.id


def test_model_performance_requires_completed_model_with_metrics(client, auth_headers, app):
    dataset_id = _make_dataset(app)
    with app.app_context():
        model = Model(
            owner_id=TEST_USER_ID,
            dataset_id=dataset_id,
            name="M1",
            model_type="svm",
            parameters={},
            status="training",
        )
        db.session.add(model)
        db.session.commit()
        model_id = model.id

    resp = client.get(f"/api/visualization/model-performance/{model_id}", headers=auth_headers)
    assert resp.status_code == 400

    with app.app_context():
        model = Model.query.get(model_id)
        model.status = "completed"
        model.metrics = {"accuracy": 0.9, "precision": 0.8, "recall": 0.85, "f1_score": 0.82}
        db.session.commit()

    resp = client.get(f"/api/visualization/model-performance/{model_id}", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["performance_data"]["accuracy"] == 0.9
    assert "performance_chart" in body


def test_parameter_comparison_across_runs(client, auth_headers, app):
    dataset_id = _make_dataset(app)
    with app.app_context():
        for lr, acc in [(0.01, 0.7), (0.1, 0.85)]:
            model = Model(
                owner_id=TEST_USER_ID,
                dataset_id=dataset_id,
                name=f"svm-{lr}",
                model_type="svm",
                parameters={"C": lr},
                status="completed",
                metrics={"accuracy": acc},
            )
            db.session.add(model)
            db.session.flush()
            job = TrainingJob(
                owner_id=TEST_USER_ID,
                model_id=model.id,
                status="completed",
                metrics={"accuracy": acc},
            )
            db.session.add(job)
        db.session.commit()

    resp = client.get(
        "/api/visualization/parameter-comparison?model_type=svm&metric=accuracy",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    body = resp.get_json()
    assert len(body["points"]) == 2
    values = sorted(p["metric_value"] for p in body["points"])
    assert values == [0.7, 0.85]


def test_parameter_comparison_requires_model_type(client, auth_headers):
    resp = client.get("/api/visualization/parameter-comparison", headers=auth_headers)
    assert resp.status_code == 400

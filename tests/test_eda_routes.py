import json

from legal_ml_core import Dataset, db
from legal_ml_core.routes.eda import perform_eda_analysis
from tests.conftest import TEST_USER_ID
import pandas as pd


def _refuse(constant):
    raise AssertionError(f"{constant} is not valid JSON")


def test_perform_eda_analysis_classification(tmp_path):
    df = pd.DataFrame(
        {
            "text": ["short one", "a somewhat longer piece of legal text here"] * 5,
            "label": ["positive", "negative"] * 5,
        }
    )
    analysis = perform_eda_analysis(df, "classification")

    assert analysis["dataset_overview"]["shape"]["rows"] == 10
    assert "label_distribution" in analysis["task_specific_analysis"]
    assert analysis["task_specific_analysis"]["label_distribution"] == {
        "positive": 5,
        "negative": 5,
    }
    assert isinstance(analysis["insights"], list)


def test_eda_route_end_to_end(client, auth_headers, app, tmp_path):
    csv_path = tmp_path / "eda_sample.csv"
    df = pd.DataFrame(
        {
            "text": ["clause one", "clause two", "clause three", "clause four"],
            "label": ["a", "b", "a", "b"],
        }
    )
    df.to_csv(csv_path, index=False)

    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="EDA dataset",
            file_path=str(csv_path),
            row_count=4,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.commit()
        dataset_id = dataset.id

    resp = client.get(f"/api/datasets/{dataset_id}/eda", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["dataset_overview"]["shape"]["rows"] == 4


def test_eda_route_404_for_missing_dataset(client, auth_headers):
    resp = client.get("/api/datasets/does-not-exist/eda", headers=auth_headers)
    assert resp.status_code == 404


def test_eda_route_answers_valid_json_for_a_column_with_no_values(client, auth_headers, app, tmp_path):
    csv_path = tmp_path / "blank_labels.csv"
    pd.DataFrame({"text": ["clause one", "clause two", "clause three"], "label": [None, None, None]}).to_csv(
        csv_path, index=False
    )

    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="Blank labels",
            file_path=str(csv_path),
            row_count=3,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.commit()
        dataset_id = dataset.id

    resp = client.get(f"/api/datasets/{dataset_id}/eda", headers=auth_headers)

    assert resp.status_code == 200
    body = json.loads(resp.get_data(as_text=True), parse_constant=_refuse)
    assert body["column_analysis"]["label"]["mean"] is None

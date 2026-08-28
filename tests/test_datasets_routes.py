import io

from legal_ml_core.routes.datasets import validate_dataset_file
from tests.conftest import TEST_USER_ID


def _csv_bytes(header, rows):
    lines = [",".join(header)] + [",".join(row) for row in rows]
    return ("\n".join(lines)).encode("utf-8")


def test_validate_dataset_file_classification_requires_text_and_label(tmp_path):
    path = tmp_path / "no_label.csv"
    path.write_bytes(_csv_bytes(["text"], [["hello"], ["world"]]))

    df, error = validate_dataset_file(str(path), "classification", annotation_mode=False)
    assert df is None
    assert "label" in error

    df, error = validate_dataset_file(str(path), "classification", annotation_mode=True)
    assert error is None
    assert len(df) == 2


def test_validate_dataset_file_rejects_missing_text_column(tmp_path):
    path = tmp_path / "no_text.csv"
    path.write_bytes(_csv_bytes(["label"], [["a"], ["b"]]))

    df, error = validate_dataset_file(str(path), "classification")
    assert df is None
    assert "text" in error


def test_upload_list_get_delete_dataset(client, auth_headers):
    csv_content = _csv_bytes(
        ["text", "label"],
        [["contract clause one", "positive"], ["contract clause two", "negative"]],
    )
    data = {
        "file": (io.BytesIO(csv_content), "sample.csv"),
        "name": "Sample Dataset",
        "task_type": "classification",
    }
    resp = client.post(
        "/api/datasets/upload",
        data=data,
        content_type="multipart/form-data",
        headers=auth_headers,
    )
    assert resp.status_code == 201, resp.get_json()
    dataset_id = resp.get_json()["dataset"]["id"]
    assert resp.get_json()["dataset"]["row_count"] == 2

    resp = client.get("/api/datasets/", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.get_json()["datasets"]) == 1

    resp = client.get(f"/api/datasets/{dataset_id}", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.get_json()["dataset"]["id"] == dataset_id

    resp = client.delete(f"/api/datasets/{dataset_id}", headers=auth_headers)
    assert resp.status_code == 200

    resp = client.get(f"/api/datasets/{dataset_id}", headers=auth_headers)
    assert resp.status_code == 404


def test_upload_rejects_disallowed_extension(client, auth_headers):
    data = {
        "file": (io.BytesIO(b"not a dataset"), "sample.txt"),
        "name": "Bad Dataset",
        "task_type": "classification",
    }
    resp = client.post(
        "/api/datasets/upload",
        data=data,
        content_type="multipart/form-data",
        headers=auth_headers,
    )
    assert resp.status_code == 400


def test_datasets_are_scoped_to_owning_user(client, auth_headers, app):
    from flask_jwt_extended import create_access_token

    from legal_ml_core import Dataset, db

    with app.app_context():
        other = Dataset(
            user_id="other-user",
            name="Not mine",
            file_path="/tmp/x.csv",
            row_count=1,
            columns=[],
            task_type="classification",
        )
        db.session.add(other)
        db.session.commit()
        other_id = other.id

    resp = client.get(f"/api/datasets/{other_id}", headers=auth_headers)
    assert resp.status_code == 404

import io

import pytest

from legal_ml_core import Dataset, db
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
    from legal_ml_core import Dataset, db

    with app.app_context():
        other = Dataset(
            owner_id="other-user",
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


def test_paginated_rows_route(client, auth_headers, app, tmp_path):
    """Full-data browsing serves any window (not just the first 10 rows)."""
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "text,label\n" + "\n".join(f"doc {i},A" for i in range(120))
    )
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="Rows",
            file_path=str(csv_path),
            row_count=120,
            columns=["text", "label"],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.commit()
        dataset_id = dataset.id

    first = client.get(
        f"/api/datasets/{dataset_id}/rows?offset=0&limit=50", headers=auth_headers
    )
    assert first.status_code == 200
    body = first.get_json()
    assert len(body["rows"]) == 50
    assert body["total_rows"] == 120
    assert body["has_more"] is True

    last = client.get(
        f"/api/datasets/{dataset_id}/rows?offset=100&limit=50", headers=auth_headers
    )
    assert len(last.get_json()["rows"]) == 20
    assert last.get_json()["has_more"] is False


def _rows_dataset(app, tmp_path, csv_text, owner_id=TEST_USER_ID, name="rows.csv"):
    csv_path = tmp_path / name
    csv_path.write_text(csv_text)
    with app.app_context():
        dataset = Dataset(
            owner_id=owner_id,
            name="Rows",
            file_path=str(csv_path),
            row_count=0,
            columns=["text", "label"],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.commit()
        return dataset.id


# 60 texts with the word "breach" in class A, 10 in class B, and 30 with a
# longer word that merely starts with it, which a term filter must not match.
_FILTER_CSV = "text,label\n" + "\n".join(
    [f"Breach of clause {i},A" for i in range(60)]
    + [f"breach notice {i},B" for i in range(10)]
    + [f"breaches galore {i},A" for i in range(30)]
)


def test_rows_filter_by_term_and_class_page_through_the_filtered_total(
    client, auth_headers, app, tmp_path
):
    dataset_id = _rows_dataset(app, tmp_path, _FILTER_CSV)

    first = client.get(
        f"/api/datasets/{dataset_id}/rows?term=breach&label=A&offset=0&limit=50",
        headers=auth_headers,
    ).get_json()
    assert first["total_rows"] == 60
    assert len(first["rows"]) == 50
    assert first["has_more"] is True
    assert all(row["label"] == "A" for row in first["rows"])

    last = client.get(
        f"/api/datasets/{dataset_id}/rows?term=breach&label=A&offset=50&limit=50",
        headers=auth_headers,
    ).get_json()
    assert [row["text"] for row in last["rows"]] == [
        f"Breach of clause {i}" for i in range(50, 60)
    ]
    assert last["has_more"] is False


def test_rows_filter_by_term_alone_matches_whole_words_in_any_case(
    client, auth_headers, app, tmp_path
):
    dataset_id = _rows_dataset(app, tmp_path, _FILTER_CSV)

    body = client.get(
        f"/api/datasets/{dataset_id}/rows?term=BREACH", headers=auth_headers
    ).get_json()
    assert body["total_rows"] == 70


def test_rows_filter_by_class_alone(client, auth_headers, app, tmp_path):
    dataset_id = _rows_dataset(app, tmp_path, _FILTER_CSV)

    body = client.get(
        f"/api/datasets/{dataset_id}/rows?label=B", headers=auth_headers
    ).get_json()
    assert body["total_rows"] == 10


def test_the_filtered_total_equals_the_eda_count_in_the_class(
    client, auth_headers, app, tmp_path
):
    """The drill-down count on the EDA page and the rows it opens must agree."""
    dataset_id = _rows_dataset(app, tmp_path, _FILTER_CSV)

    eda = client.get(f"/api/datasets/{dataset_id}/eda", headers=auth_headers).get_json()
    stats = eda["task_specific_analysis"]["keyword_stats"]
    entry = next(e for e in stats["B"] if e["term"] == "breach")

    rows = client.get(
        f"/api/datasets/{dataset_id}/rows?term=breach&label=B", headers=auth_headers
    ).get_json()
    assert rows["total_rows"] == entry["in_class"]


def test_a_class_filter_on_a_dataset_without_labels_is_refused(
    client, auth_headers, app, tmp_path
):
    dataset_id = _rows_dataset(app, tmp_path, "text\nbreach one\nbreach two\n")

    resp = client.get(f"/api/datasets/{dataset_id}/rows?label=A", headers=auth_headers)
    assert resp.status_code == 400
    assert "label" in resp.get_json()["error"].lower()


def test_filtered_rows_of_another_owners_dataset_are_not_found(
    client, auth_headers, app, tmp_path
):
    dataset_id = _rows_dataset(app, tmp_path, _FILTER_CSV, owner_id="other-user")

    resp = client.get(
        f"/api/datasets/{dataset_id}/rows?term=breach&label=A", headers=auth_headers
    )
    assert resp.status_code == 404


def _multipart(rows):
    boundary = "b0undary"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\nbig\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"task_type\"\r\n\r\nclassification\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"big.csv\"\r\n"
        "Content-Type: text/csv\r\n\r\n"
        "text,label\n" + "the court ruled for the plaintiff,civil\n" * rows + f"\r\n--{boundary}--\r\n"
    ).encode()
    return body, f"multipart/form-data; boundary={boundary}"


@pytest.mark.parametrize("declared", [True, False], ids=["declared size", "chunked"])
def test_an_upload_over_the_size_limit_is_refused_as_too_large(client, auth_headers, app, declared):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    body, content_type = _multipart(rows=200)
    headers = dict(auth_headers, **({} if declared else {"Transfer-Encoding": "chunked"}))

    response = client.post(
        "/api/datasets/upload",
        input_stream=io.BytesIO(body),
        content_length=len(body) if declared else None,
        content_type=content_type,
        headers=headers,
        environ_overrides={} if declared else {"wsgi.input_terminated": True},
    )

    assert response.status_code == 413
    assert "too large" in response.get_json()["error"]
    with app.app_context():
        assert Dataset.query.count() == 0

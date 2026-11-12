import pytest

from legal_ml_core import Dataset, DatasetInvalid, create_dataset
from tests.conftest import TEST_USER_ID


def _write_csv(tmp_path, header, rows, name="data.csv"):
    path = tmp_path / name
    lines = [",".join(header)] + [",".join(row) for row in rows]
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _create(path, **overrides):
    args = dict(
        owner_id=TEST_USER_ID,
        file_path=path,
        name="Clauses",
        description="Two clauses",
        task_type="classification",
        annotation_mode=False,
        labels=None,
    )
    args.update(overrides)
    return create_dataset(**args)


def test_creates_an_owned_dataset_with_row_and_column_info(app, tmp_path):
    path = _write_csv(
        tmp_path,
        ["text", "label"],
        [["the tenant shall pay", "obligation"], ["the tenant may leave", "permission"]],
    )

    with app.app_context():
        dataset = _create(path, labels=["obligation", "permission"])

        stored = Dataset.query.get(dataset.id)
        assert stored.owner_id == TEST_USER_ID
        assert stored.name == "Clauses"
        assert stored.description == "Two clauses"
        assert stored.task_type == "classification"
        assert stored.file_path == path
        assert stored.row_count == 2
        assert stored.labels == ["obligation", "permission"]
        assert [c["name"] for c in stored.columns] == ["text", "label"]
        assert stored.columns[1]["unique_count"] == 2


@pytest.mark.parametrize(
    "header, overrides, message",
    [
        (["text"], {}, "Classification datasets for training must have a 'label' column"),
        (["label"], {}, "Classification datasets must have a 'text' column"),
        (["text", "label"], {"task_type": "poetry"}, "Invalid task type. Valid types:"),
        (["text", "label"], {"name": "  "}, "Dataset name is required"),
    ],
)
def test_refuses_with_a_student_readable_message_and_stores_nothing(
    app, tmp_path, header, overrides, message
):
    path = _write_csv(tmp_path, header, [["a"] * len(header), ["b"] * len(header)])

    with app.app_context():
        with pytest.raises(DatasetInvalid) as refused:
            _create(path, **overrides)

        assert str(refused.value).startswith(message)
        assert Dataset.query.count() == 0


def test_annotation_mode_accepts_a_file_without_labels(app, tmp_path):
    path = _write_csv(tmp_path, ["text"], [["the tenant shall pay"], ["the tenant may leave"]])

    with app.app_context():
        dataset = _create(path, annotation_mode=True)

        assert dataset.row_count == 2


def _rows(n):
    return [[f"clause {i}", "a" if i % 2 else "b"] for i in range(n)]


def test_the_host_sets_the_row_cap(app, tmp_path):
    app.config["LEGAL_ML_MAX_ROWS"] = 3
    at_cap = _write_csv(tmp_path, ["text", "label"], _rows(3), "three.csv")
    over_cap = _write_csv(tmp_path, ["text", "label"], _rows(4), "four.csv")

    with app.app_context():
        assert _create(at_cap).row_count == 3
        with pytest.raises(DatasetInvalid, match=r"^Dataset exceeds maximum allowed rows \(3\)$"):
            _create(over_cap)


def test_a_callable_cap_is_read_on_every_call(app, tmp_path):
    cap = {"value": 5}
    app.config["LEGAL_ML_MAX_ROWS"] = lambda: cap["value"]
    path = _write_csv(tmp_path, ["text", "label"], _rows(4))

    with app.app_context():
        assert _create(path).row_count == 4
        cap["value"] = 2
        with pytest.raises(DatasetInvalid, match=r"\(2\)"):
            _create(path)


@pytest.mark.parametrize("cap", [0, -5, lambda: 0])
def test_a_cap_of_zero_or_less_is_a_host_error_not_a_student_one(app, tmp_path, cap):
    app.config["LEGAL_ML_MAX_ROWS"] = cap
    path = _write_csv(tmp_path, ["text", "label"], _rows(1))

    with app.app_context():
        with pytest.raises(ValueError, match="LEGAL_ML_MAX_ROWS") as refused:
            _create(path)
        assert not isinstance(refused.value, DatasetInvalid)

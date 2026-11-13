"""The holdout keeps the shape of the labels that already exist, on any shape."""
from collections import Counter

import pytest

from legal_ml_core import ALConfig, AnnotationTask, db
from tests.active_learning_helpers import label, link_dataset, next_item, start, state


def _rows(shape, unlabelled=0):
    labels = [name for name, n in shape.items() for _ in range(n)] + [None] * unlabelled
    return [f"row number {i}" for i in range(len(labels))], labels


def _held_back_labels(dataset_id, labels):
    drawn = ALConfig.query.filter_by(dataset_id=dataset_id).one().holdout_indices
    return Counter(labels[i] for i in drawn)


def _started(client, auth_headers, app, tmp_path, shape, unlabelled=0, **options):
    texts, labels = _rows(shape, unlabelled)
    dataset_id = link_dataset(app, tmp_path, texts, labels=labels)
    resp = start(client, auth_headers, dataset_id, **options)
    assert resp.status_code == 201
    return dataset_id, labels


@pytest.mark.parametrize(
    "shape, unlabelled, expected",
    [
        (
            {"Verbot": 60, "Gebot": 25, "Recht": 10, "Frist": 5},
            100,
            {"Verbot": 12, "Gebot": 5, "Recht": 2, "Frist": 1, None: 20},
        ),
        ({"upheld": 150, "dismissed": 50}, 50, {"upheld": 30, "dismissed": 10, None: 10}),
    ],
    ids=["four_labels_steep", "two_labels"],
)
def test_the_holdout_keeps_each_class_share(client, auth_headers, app, tmp_path, shape, unlabelled, expected):
    dataset_id, labels = _started(client, auth_headers, app, tmp_path, shape, unlabelled)

    assert _held_back_labels(dataset_id, labels) == expected


def test_unlabelled_rows_keep_their_share_at_random(client, auth_headers, app, tmp_path):
    dataset_id, labels = _started(client, auth_headers, app, tmp_path, {"a": 30, "b": 10}, unlabelled=60)

    assert _held_back_labels(dataset_id, labels) == {"a": 6, "b": 2, None: 12}


def test_a_class_of_one_stays_out_of_the_holdout(client, auth_headers, app, tmp_path):
    dataset_id, labels = _started(client, auth_headers, app, tmp_path, {"big": 28, "other": 11, "lone": 1}, unlabelled=10)

    held = _held_back_labels(dataset_id, labels)
    assert held["lone"] == 0
    assert sum(held.values()) == 10


def test_every_class_of_one_still_gives_a_holdout_of_the_right_size(client, auth_headers, app, tmp_path):
    shape = {f"label{i}": 1 for i in range(10)}

    dataset_id, labels = _started(client, auth_headers, app, tmp_path, shape, unlabelled=10)

    assert sum(_held_back_labels(dataset_id, labels).values()) == 4


def test_a_fully_unlabelled_dataset_draws_as_before(client, auth_headers, app, tmp_path):
    dataset_id, labels = _started(client, auth_headers, app, tmp_path, {}, unlabelled=50)

    assert state(client, auth_headers, dataset_id)["holdout"]["size"] == 10
    assert _held_back_labels(dataset_id, labels) == {None: 10}


def test_labels_saved_before_the_loop_stratify_too(client, auth_headers, app, tmp_path):
    texts = [f"row number {i}" for i in range(100)]
    dataset_id = link_dataset(app, tmp_path, texts)
    for i in range(50):
        task = next_item(client, auth_headers, dataset_id)["task"]
        label(client, auth_headers, task["id"], "rare" if i < 10 else "common")

    start(client, auth_headers, dataset_id)

    held = AnnotationTask.query.filter_by(dataset_id=dataset_id, is_holdout=True, status="annotated").all()
    assert Counter(task.annotation_label for task in held) == {"common": 8, "rare": 2}


def test_the_state_counts_the_holdout_by_class(client, auth_headers, app, tmp_path):
    dataset_id, _ = _started(client, auth_headers, app, tmp_path, {"Verbot": 40, "Gebot": 10}, unlabelled=50)

    holdout = state(client, auth_headers, dataset_id)["holdout"]

    assert holdout["label_counts"] == {"Verbot": 8, "Gebot": 2}
    assert holdout["labelled"] == 10
    assert holdout["size"] == 20


def test_the_state_counts_held_back_items_as_they_are_labelled(client, auth_headers, app, tmp_path):
    dataset_id, _ = _started(client, auth_headers, app, tmp_path, {"Verbot": 40, "Gebot": 10}, unlabelled=50)

    task = next_item(client, auth_headers, dataset_id, holdout=1)["task"]
    label(client, auth_headers, task["id"], "Gebot")

    assert state(client, auth_headers, dataset_id)["holdout"]["label_counts"] == {"Verbot": 8, "Gebot": 3}


def _as_started_before_147(dataset_id):
    """Put a loop back as a loop started before #147 left it: no tasks for rows the
    file labels, their counts stored on the loop, the Dataset not marked queued."""
    from legal_ml_core import ALConfig, AnnotationTask
    from legal_ml_core.annotation_queue import ALL_ROWS_QUEUED, file_labels, read_rows
    from legal_ml_core.models import Dataset

    dataset = db.session.get(Dataset, dataset_id)
    config = ALConfig.query.filter_by(dataset_id=dataset_id).one()
    df = read_rows(dataset.file_path)
    labels = file_labels(df, dataset)
    held = set(config.holdout_indices)
    counts = lambda rows: {k: int(v) for k, v in rows.astype(str).value_counts().items()}
    config.file_label_counts = counts(labels[~labels.index.isin(held)])
    config.holdout_file_counts = counts(labels[labels.index.isin(held)])
    AnnotationTask.query.filter_by(dataset_id=dataset_id, label_source="file").delete()
    dataset.columns = [{k: v for k, v in c.items() if k != ALL_ROWS_QUEUED} for c in dataset.columns]
    db.session.commit()


def test_a_loop_started_before_147_counts_each_label_once_after_its_rows_get_tasks(client, auth_headers, app, tmp_path):
    dataset_id, _ = _started(client, auth_headers, app, tmp_path, {"Verbot": 40, "Gebot": 10}, unlabelled=50)
    before = state(client, auth_headers, dataset_id)
    _as_started_before_147(dataset_id)
    assert state(client, auth_headers, dataset_id)["label_counts"] == before["label_counts"]

    next_item(client, auth_headers, dataset_id)  # reading the queue adds the file-labelled tasks

    after = state(client, auth_headers, dataset_id)
    assert after["label_counts"] == before["label_counts"] == {"Verbot": 32, "Gebot": 8}
    assert after["holdout"]["label_counts"] == before["holdout"]["label_counts"] == {"Verbot": 8, "Gebot": 2}

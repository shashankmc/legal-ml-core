"""The cap on labels: training opens with classes still short, on any Dataset shape.

Nothing here may rest on the names or the number of the labels, so each rule runs
on Datasets of different shapes.
"""
import pytest

from legal_ml_core import Model
from legal_ml_core.active_learning.loop import validate_host_config
from tests.active_learning_helpers import (
    balanced_texts,
    label_next,
    link_dataset,
    loop_model,
    retrain,
    run_scoring,
    run_training,
    start,
    state,
)

# Each adds up to 40 labels, which is the cap at a batch size of 8.
SHAPES = {
    "five_labels_steep": {"Verbot": 30, "Gebot": 5, "Recht": 3, "Frist": 1, "Pflicht": 1},
    "three_labels_even_tail": {"upheld": 20, "dismissed": 18, "remanded": 2},
}


def _started(client, auth_headers, app, tmp_path, rows=60, batch_size=8, dataset_labels=None):
    dataset_id = link_dataset(app, tmp_path, balanced_texts(rows), dataset_labels=dataset_labels)
    resp = start(client, auth_headers, dataset_id, batch_size=batch_size, holdout_percent=0)
    assert resp.status_code == 201
    return dataset_id


def _label_shape(client, auth_headers, dataset_id, shape, limit=None):
    """Label ``shape``'s counts, the largest class last so a ``limit`` stops short."""
    values = [name for name, n in sorted(shape.items(), key=lambda kv: kv[1]) for _ in range(n)]
    values = values[:limit] if limit else values
    answers = iter(values)
    label_next(client, auth_headers, dataset_id, lambda text: next(answers), len(values))


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_short_classes_hold_training_back_until_the_cap(client, auth_headers, app, tmp_path, shape):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    _label_shape(client, auth_headers, dataset_id, shape, limit=39)

    body = state(client, auth_headers, dataset_id)["retrain"]
    assert body["allowed"] is False
    assert body["label_cap"] == 40
    assert body["labels_total"] == 39
    assert retrain(client, auth_headers, dataset_id).status_code == 409


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_the_cap_opens_training_and_names_the_short_classes(client, auth_headers, app, tmp_path, fake_redis, shape):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    _label_shape(client, auth_headers, dataset_id, shape)

    body = state(client, auth_headers, dataset_id)["retrain"]
    assert body["allowed"] is True
    assert body["classes_short"] == {name: 5 - n for name, n in shape.items() if n < 5}
    assert retrain(client, auth_headers, dataset_id).status_code == 202
    run_training(app, fake_redis)
    run_scoring(app, fake_redis)
    assert (loop_model(dataset_id).status, loop_model(dataset_id).error_message if hasattr(Model,"error_message") else None) == ("completed", None)


def test_a_class_with_one_example_is_named_and_left_out_of_training(client, auth_headers, app, tmp_path, fake_redis):
    shape = SHAPES["five_labels_steep"]
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _label_shape(client, auth_headers, dataset_id, shape)

    assert state(client, auth_headers, dataset_id)["retrain"]["classes_left_out"] == ["Frist", "Pflicht"]

    retrain(client, auth_headers, dataset_id)
    run_training(app, fake_redis)

    report = loop_model(dataset_id).metrics["classification_report"]
    assert "Verbot" in report and "Recht" in report
    assert "Frist" not in report and "Pflicht" not in report


def test_no_class_is_left_out_with_two_examples(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _label_shape(client, auth_headers, dataset_id, SHAPES["three_labels_even_tail"])

    assert state(client, auth_headers, dataset_id)["retrain"]["classes_left_out"] == []


def test_the_cap_cannot_open_training_on_one_trainable_class(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _label_shape(client, auth_headers, dataset_id, {"main": 39, "lone": 1})

    body = state(client, auth_headers, dataset_id)["retrain"]
    assert body["allowed"] is False
    assert body["needs_second_label"] is True
    resp = retrain(client, auth_headers, dataset_id)
    assert resp.status_code == 409
    assert resp.get_json()["error"] == "Training needs at least 2 examples each of two labels."
    assert fake_redis.llen("training_queue") == 0


def test_the_cap_waits_for_enough_examples_to_split_every_class(client, auth_headers, app, tmp_path):
    """Twelve classes of two pass the cap of 20, but a fifth of 24 cannot show
    each of them once. That takes 56."""
    dataset_id = _started(client, auth_headers, app, tmp_path, batch_size=4)
    _label_shape(client, auth_headers, dataset_id, {f"class{i}": 2 for i in range(12)})

    body = state(client, auth_headers, dataset_id)["retrain"]
    assert body["allowed"] is False
    assert body["labels_needed"] == 32
    resp = retrain(client, auth_headers, dataset_id)
    assert resp.status_code == 409
    assert resp.get_json()["error"] == "With 12 labels, training needs at least 56 examples in all."


def test_the_smallest_shape_the_cap_allows_trains(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path, batch_size=2)
    _label_shape(client, auth_headers, dataset_id, {"a": 7, "b": 2, "c": 2})

    body = state(client, auth_headers, dataset_id)["retrain"]
    assert body["label_cap"] == 10
    assert body["allowed"] is True
    retrain(client, auth_headers, dataset_id)
    run_training(app, fake_redis)
    assert (loop_model(dataset_id).status, loop_model(dataset_id).error_message if hasattr(Model,"error_message") else None) == ("completed", None)


def test_a_host_sets_the_cap_in_batches(client, auth_headers, app, tmp_path):
    app.config["LEGAL_ML_AL_CAP_BATCHES"] = 2
    dataset_id = _started(client, auth_headers, app, tmp_path)

    _label_shape(client, auth_headers, dataset_id, {"main": 13, "tail": 3})

    body = state(client, auth_headers, dataset_id)["retrain"]
    assert body["label_cap"] == 16
    assert body["allowed"] is True


def test_the_settings_give_the_cap_in_batches(client, auth_headers):
    body = client.get("/api/active-learning/settings", headers=auth_headers).get_json()

    assert body["cap_batches"] == 5


@pytest.mark.parametrize("value", [0, -3, "five"])
def test_a_bad_cap_setting_fails_at_start(value):
    with pytest.raises(ValueError):
        validate_host_config({"LEGAL_ML_AL_CAP_BATCHES": value})


def test_a_declared_label_with_no_examples_does_not_hold_training_back(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path, dataset_labels=["one", "two", "three"])

    _label_shape(client, auth_headers, dataset_id, {"one": 5, "two": 5})

    body = state(client, auth_headers, dataset_id)["retrain"]
    assert body["allowed"] is True
    assert body["classes_short"] == {}
    assert body["classes_unseen"] == ["three"]


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_every_way_to_train_opens_at_the_cap(client, auth_headers, app, tmp_path, fake_redis, shape):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _label_shape(client, auth_headers, dataset_id, shape)

    created = client.post(
        "/api/models/",
        json={"name": "Mine", "dataset_id": dataset_id, "model_type": "logistic_regression", "parameters": {}},
        headers=auth_headers,
    )
    assert created.status_code in (200, 201)
    model_id = Model.query.filter_by(dataset_id=dataset_id).one().id

    by_model = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)
    by_start = client.post("/api/training/start", json={"model_id": model_id}, headers=auth_headers)

    assert by_model.status_code == 200
    assert by_start.status_code == 200
    assert fake_redis.llen("training_queue") == 1


def test_every_way_to_train_stays_shut_below_the_cap(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _label_shape(client, auth_headers, dataset_id, SHAPES["three_labels_even_tail"], limit=20)

    created = client.post(
        "/api/models/",
        json={"name": "Mine", "dataset_id": dataset_id, "model_type": "logistic_regression", "parameters": {}},
        headers=auth_headers,
    )

    assert created.status_code == 409
    assert created.get_json()["error"] == "Each label needs at least 5 examples before training, or 40 labels in all."
    assert fake_redis.llen("training_queue") == 0

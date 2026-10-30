"""The holdout: drawn once, never queued or trained on, the source of every metric."""
from legal_ml_core import ALConfig, ALRound, AnnotationTask, Model, db
from tests.active_learning_helpers import (
    PROJECT_ID,
    balanced_texts,
    by_topic,
    label,
    label_next,
    link_dataset,
    loop_model,
    next_item,
    retrain,
    run_scoring,
    run_training,
    start,
    state,
)


def _started(client, auth_headers, app, tmp_path, pairs=25, **options):
    dataset_id = link_dataset(app, tmp_path, balanced_texts(pairs))
    resp = start(client, auth_headers, dataset_id, **options)
    assert resp.status_code == 201
    return dataset_id


def _holdout_tasks(dataset_id):
    return AnnotationTask.query.filter_by(dataset_id=dataset_id, is_holdout=True).all()


def _train_round(client, auth_headers, app, fake_redis, dataset_id, labels=16):
    label_next(client, auth_headers, dataset_id, by_topic, labels)
    assert retrain(client, auth_headers, dataset_id).status_code == 202
    run_training(app, fake_redis)
    run_scoring(app, fake_redis)


def _label_holdout(client, auth_headers, dataset_id, count):
    label_next(client, auth_headers, dataset_id, by_topic, count, holdout=1)


def test_a_fifth_of_the_rows_are_held_out_by_default(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    holdout = state(client, auth_headers, dataset_id)["holdout"]

    assert holdout["percent"] == 20
    assert holdout["size"] == 10
    assert len(_holdout_tasks(dataset_id)) == 10


def test_the_holdout_size_can_be_changed_at_start(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path, holdout_percent=40)

    assert state(client, auth_headers, dataset_id)["holdout"]["size"] == 20


def test_a_holdout_above_half_is_refused(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, ["a", "b"])

    resp = start(client, auth_headers, dataset_id, holdout_percent=51)

    assert resp.status_code == 400


def test_the_holdout_is_drawn_once(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    drawn = ALConfig.query.filter_by(dataset_id=dataset_id).one().holdout_indices

    assert start(client, auth_headers, dataset_id, holdout_percent=40).status_code == 409

    assert ALConfig.query.filter_by(dataset_id=dataset_id).one().holdout_indices == drawn


def test_a_reset_keeps_the_holdout(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    drawn = {t.item_index for t in _holdout_tasks(dataset_id)}

    client.post(f"/api/annotation/projects/{PROJECT_ID}/tasks/reset", headers=auth_headers)

    assert {t.item_index for t in _holdout_tasks(dataset_id)} == drawn


def test_the_queue_never_serves_holdout_items(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    holdout = {t.item_index for t in _holdout_tasks(dataset_id)}

    served = label_next(client, auth_headers, dataset_id, by_topic, 40)

    assert not {t["item_index"] for t in served} & holdout
    body = next_item(client, auth_headers, dataset_id)
    assert body["task"] is None
    assert body["stats"]["holdout_remaining"] == 10


def test_stats_count_holdout_rows_as_still_unlabelled(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    stats = next_item(client, auth_headers, dataset_id)["stats"]

    assert stats["total"] == 50
    assert stats["annotated"] == 0


def test_holdout_items_are_labelled_only_on_request(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    holdout = {t.item_index for t in _holdout_tasks(dataset_id)}

    assert next_item(client, auth_headers, dataset_id, holdout=1)["task"]["item_index"] in holdout


def test_holdout_labels_do_not_count_as_new_training_labels(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    _label_holdout(client, auth_headers, dataset_id, 3)

    body = state(client, auth_headers, dataset_id)
    assert body["retrain"]["new_labels"] == 0
    assert body["holdout"]["labelled"] == 3


def test_training_never_reads_a_holdout_row(client, auth_headers, app, tmp_path, fake_redis):
    from legal_ml_core.annotation_queue import apply_saved_annotations
    from legal_ml_core.models import Dataset
    import pandas as pd

    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, by_topic, 12)
    _label_holdout(client, auth_headers, dataset_id, 10)
    dataset = db.session.get(Dataset, dataset_id)

    frame = apply_saved_annotations(pd.read_csv(dataset.file_path), dataset)

    held_out = {t.text for t in _holdout_tasks(dataset_id)}
    assert len(frame) == 12
    assert not set(frame["text"]) & held_out


def test_rows_labelled_in_the_file_can_be_held_out_and_are_not_trained_on(client, auth_headers, app, tmp_path):
    texts = balanced_texts(30)
    labels = [by_topic(t) for t in texts[:40]] + [None] * 20
    dataset_id = link_dataset(app, tmp_path, texts, labels=labels)
    start(client, auth_headers, dataset_id)

    body = state(client, auth_headers, dataset_id)

    config = ALConfig.query.filter_by(dataset_id=dataset_id).one()
    from_file_held_out = [i for i in config.holdout_indices if i < 40]
    assert sum(body["label_counts"].values()) == 40 - len(from_file_held_out)
    assert body["holdout"]["labelled"] == len(from_file_held_out)


def test_a_labelled_holdout_grades_the_model_on_held_out_rows(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _label_holdout(client, auth_headers, dataset_id, 10)

    _train_round(client, auth_headers, app, fake_redis, dataset_id)

    metrics = loop_model(dataset_id).metrics
    assert metrics["verification"] == "holdout"
    assert metrics["holdout_labels"] == 10
    assert metrics["accuracy"] == 1.0
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 10
    assert metrics["confusion_matrix_labels"] == ["civil", "criminal"]
    assert ALRound.query.filter_by(dataset_id=dataset_id, number=0).one().metrics["accuracy"] == 1.0


def test_an_unlabelled_holdout_leaves_the_model_unverified(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    _train_round(client, auth_headers, app, fake_redis, dataset_id)

    body = state(client, auth_headers, dataset_id)
    assert loop_model(dataset_id).metrics["verification"] == "not_independently_verified"
    assert body["holdout"]["verification"] == "not_independently_verified"
    assert body["holdout"]["needed"] == 10


def test_turning_the_holdout_off_leaves_the_model_unverified(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path, holdout_percent=0)

    _train_round(client, auth_headers, app, fake_redis, dataset_id)

    assert _holdout_tasks(dataset_id) == []
    assert loop_model(dataset_id).metrics["verification"] == "not_independently_verified"
    assert state(client, auth_headers, dataset_id)["holdout"]["size"] == 0


def test_checking_after_labelling_the_holdout_grades_the_saved_model(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _train_round(client, auth_headers, app, fake_redis, dataset_id)
    _label_holdout(client, auth_headers, dataset_id, 10)

    resp = client.post(f"/api/active-learning/datasets/{dataset_id}/verify", headers=auth_headers)
    assert resp.status_code == 202
    run_scoring(app, fake_redis)

    model = db.session.get(Model, loop_model(dataset_id).id)
    assert model.metrics["verification"] == "holdout"
    assert state(client, auth_headers, dataset_id)["holdout"]["verification"] == "holdout"


def test_checking_with_too_few_holdout_labels_is_refused(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _train_round(client, auth_headers, app, fake_redis, dataset_id)
    _label_holdout(client, auth_headers, dataset_id, 9)

    resp = client.post(f"/api/active-learning/datasets/{dataset_id}/verify", headers=auth_headers)

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "Label 1 more held-back item before checking the score."
    assert fake_redis.llen("scoring_queue") == 0


def test_checking_with_no_model_is_refused(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    resp = client.post(f"/api/active-learning/datasets/{dataset_id}/verify", headers=auth_headers)

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "Train a Model before checking its score."


def test_llm_batches_leave_holdout_items_out(client, auth_headers, app, tmp_path):
    from legal_ml_core.routes.llm import _pending_tasks
    from legal_ml_core.models import Dataset

    dataset_id = _started(client, auth_headers, app, tmp_path)
    dataset = db.session.get(Dataset, dataset_id)

    assert not [t for t in _pending_tasks(dataset, None) if t.is_holdout]


def test_labelling_a_holdout_item_keeps_it_out_of_the_pool(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    task = _holdout_tasks(dataset_id)[0]

    label(client, auth_headers, task.id, "civil")

    db.session.expire_all()
    assert db.session.get(AnnotationTask, task.id).is_holdout is True

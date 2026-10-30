"""The loop: label a batch, retrain on request, and the queue reorders."""
import pytest

from legal_ml_core import ALRound, TrainingJob, db
from legal_ml_core.active_learning.loop import validate_host_config
from tests.active_learning_helpers import (
    PROJECT_ID,
    balanced_texts,
    by_topic,
    label_next,
    link_dataset,
    loop_model,
    next_item,
    retrain,
    run_scoring,
    run_training,
    start,
    state,
    task_for,
)


def _all(label_value):
    return lambda text: label_value


def _started(client, auth_headers, app, tmp_path, pairs=20, **options):
    dataset_id = link_dataset(app, tmp_path, balanced_texts(pairs))
    resp = start(client, auth_headers, dataset_id, **options)
    assert resp.status_code == 201
    return dataset_id


def test_starting_a_dataset_opens_the_seed_round(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    body = state(client, auth_headers, dataset_id)

    assert body["active"] is True
    assert body["round"]["number"] == 0
    assert body["round"]["status"] == "labelling"
    assert body["strategy"] == "prediction_entropy"


def test_a_dataset_not_started_reads_as_inactive(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, ["a", "b"])

    assert state(client, auth_headers, dataset_id) == {"success": True, "active": False}


def test_a_dataset_not_started_keeps_row_order(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, ["a", "b", "c"])

    assert next_item(client, auth_headers, dataset_id)["task"]["item_index"] == 0


def test_someone_elses_dataset_cannot_be_started(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, ["a"])
    other = {"Authorization": auth_headers["Authorization"].replace("11111111-1111", "22222222-2222")}

    assert start(client, other, dataset_id).status_code == 404


def test_only_classification_can_start(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, ["a"], task_type="ner")

    resp = start(client, auth_headers, dataset_id)

    assert resp.status_code == 400
    assert "text classification" in resp.get_json()["error"].lower()


def test_starting_twice_is_refused(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    assert start(client, auth_headers, dataset_id).status_code == 409


def test_a_dataset_with_nothing_to_label_does_not_start(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, ["a", "b"], labels=["x", "y"])

    resp = start(client, auth_headers, dataset_id)

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "This Dataset has no rows to label."
    assert state(client, auth_headers, dataset_id)["active"] is False


def test_a_strategy_outside_the_hosts_list_is_refused(client, auth_headers, app, tmp_path):
    app.config["LEGAL_ML_AL_STRATEGIES"] = ["random", "prediction_entropy"]
    dataset_id = link_dataset(app, tmp_path, ["a"])

    assert start(client, auth_headers, dataset_id, strategy="breaking_ties").status_code == 400


def test_settings_list_the_hosts_strategies_and_default(client, auth_headers, app):
    app.config["LEGAL_ML_AL_STRATEGIES"] = ["random", "breaking_ties"]
    app.config["LEGAL_ML_AL_DEFAULT_STRATEGY"] = "breaking_ties"

    body = client.get("/api/active-learning/settings", headers=auth_headers).get_json()

    assert body["strategies"] == ["random", "breaking_ties"]
    assert body["default_strategy"] == "breaking_ties"


def test_a_bad_host_strategy_list_fails_at_import(app):
    with pytest.raises(ValueError):
        validate_host_config({"LEGAL_ML_AL_STRATEGIES": ["sampling_mode"]})
    with pytest.raises(ValueError):
        validate_host_config({"LEGAL_ML_AL_STRATEGIES": ["random"], "LEGAL_ML_AL_DEFAULT_STRATEGY": "breaking_ties"})


def test_retrain_with_no_labels_is_refused(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    resp = retrain(client, auth_headers, dataset_id)

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "There are no new labels to train on."


def test_retrain_prompt_appears_once_a_batch_is_labelled(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path, batch_size=16)

    label_next(client, auth_headers, dataset_id, by_topic, 15)
    assert state(client, auth_headers, dataset_id)["retrain"]["suggested"] is False

    label_next(client, auth_headers, dataset_id, by_topic, 1)
    retrain_state = state(client, auth_headers, dataset_id)["retrain"]
    assert retrain_state["suggested"] is True
    assert retrain_state["new_labels"] == 16


def test_nothing_retrains_until_the_user_asks(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path, batch_size=2)

    label_next(client, auth_headers, dataset_id, by_topic, 3)

    assert fake_redis.llen("training_queue") == 0


def _trained_round(client, auth_headers, app, fake_redis, dataset_id, labels=16):
    label_next(client, auth_headers, dataset_id, by_topic, labels)
    resp = retrain(client, auth_headers, dataset_id)
    assert resp.status_code == 202
    run_training(app, fake_redis)
    run_scoring(app, fake_redis)


def test_retrain_trains_rescoring_and_moves_to_the_next_round(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    _trained_round(client, auth_headers, app, fake_redis, dataset_id)

    body = state(client, auth_headers, dataset_id)
    assert body["round"]["number"] == 1
    assert body["round"]["status"] == "labelling"
    assert body["retrain"]["new_labels"] == 0
    assert loop_model(dataset_id).status == "completed"
    rounds = ALRound.query.filter_by(dataset_id=dataset_id).order_by(ALRound.number).all()
    assert rounds[0].labels_used == 16


def test_the_queue_serves_the_highest_score_first(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _trained_round(client, auth_headers, app, fake_redis, dataset_id)

    served = next_item(client, auth_headers, dataset_id)["task"]

    remaining = [task_for(dataset_id, i) for i in range(40)]
    top = max(t.score for t in remaining if t.status == "pending" and not t.is_holdout)
    assert task_for(dataset_id, served["item_index"]).score == top


def test_a_second_round_keeps_one_model_and_counts_only_new_labels(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _trained_round(client, auth_headers, app, fake_redis, dataset_id)

    label_next(client, auth_headers, dataset_id, by_topic, 3)

    assert state(client, auth_headers, dataset_id)["retrain"]["new_labels"] == 3
    model_id = loop_model(dataset_id).id
    assert retrain(client, auth_headers, dataset_id).status_code == 202
    run_training(app, fake_redis)
    run_scoring(app, fake_redis)
    assert loop_model(dataset_id).id == model_id
    assert state(client, auth_headers, dataset_id)["round"]["number"] == 2


def test_a_double_retrain_starts_one_job(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, by_topic, 16)

    first = retrain(client, auth_headers, dataset_id)
    second = retrain(client, auth_headers, dataset_id)

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.get_json()["message"] == "Training already started"
    assert fake_redis.llen("training_queue") == 1


def test_a_failed_training_returns_to_labelling_with_the_reason(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, by_topic, 16)
    retrain(client, auth_headers, dataset_id)
    job = TrainingJob.query.one()
    job.status = "failed"
    job.error_message = "Ran out of memory"
    db.session.commit()

    body = state(client, auth_headers, dataset_id)

    assert body["round"]["status"] == "labelling"
    assert body["round"]["error"] == "Ran out of memory"
    assert body["retrain"]["allowed"] is True


def test_a_training_that_could_not_queue_reports_it_and_keeps_labelling(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, by_topic, 16)

    def boom(*args):
        raise ConnectionError("redis down")

    fake_redis.lpush = boom
    resp = retrain(client, auth_headers, dataset_id)

    assert resp.status_code == 503
    assert state(client, auth_headers, dataset_id)["round"]["status"] == "labelling"


def test_the_strategy_can_change_for_the_next_rescore(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    resp = client.put(
        f"/api/active-learning/datasets/{dataset_id}/strategy",
        json={"strategy": "breaking_ties"},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    assert state(client, auth_headers, dataset_id)["strategy"] == "breaking_ties"


def test_reset_clears_scores_and_starts_the_rounds_again(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    _trained_round(client, auth_headers, app, fake_redis, dataset_id)

    resp = client.post(f"/api/annotation/projects/{PROJECT_ID}/tasks/reset", headers=auth_headers)

    assert resp.status_code == 200
    body = state(client, auth_headers, dataset_id)
    assert body["round"]["number"] == 0
    assert body["retrain"]["new_labels"] == 0


def test_the_llm_seed_is_not_available_yet(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, ["a", "b"])

    resp = start(client, auth_headers, dataset_id, seed_source="llm")

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "LLM-suggested seed batches are not available yet."
    assert state(client, auth_headers, dataset_id)["active"] is False


def test_settings_say_the_llm_seed_is_off(client, auth_headers):
    body = client.get("/api/active-learning/settings", headers=auth_headers).get_json()

    assert body["seed_sources"] == {"random": True, "llm": False}
    assert body["min_per_class"] == 5


def test_a_random_seed_shuffles_the_queue(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    served = label_next(client, auth_headers, dataset_id, by_topic, 6)

    assert [t["item_index"] for t in served] != [0, 1, 2, 3, 4, 5]


def test_retrain_is_refused_until_every_class_has_its_minimum(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, balanced_texts(10), dataset_labels=["civil", "criminal"])
    start(client, auth_headers, dataset_id, batch_size=8)

    label_next(client, auth_headers, dataset_id, _all("civil"), 8)

    body = state(client, auth_headers, dataset_id)["retrain"]
    assert body["allowed"] is False
    assert body["suggested"] is False
    assert body["classes_short"] == {"criminal": 5}
    resp = retrain(client, auth_headers, dataset_id)
    assert resp.status_code == 409
    assert resp.get_json()["error"] == "Each label needs at least 5 examples before training."


def test_retrain_opens_once_every_class_has_its_minimum(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, balanced_texts(10), dataset_labels=["civil", "criminal"])
    start(client, auth_headers, dataset_id)

    label_next(client, auth_headers, dataset_id, _all("civil"), 5)
    label_next(client, auth_headers, dataset_id, _all("criminal"), 5)

    body = state(client, auth_headers, dataset_id)["retrain"]
    assert body["allowed"] is True
    assert body["classes_short"] == {}


def test_a_higher_minimum_can_be_chosen(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, balanced_texts(10), dataset_labels=["civil", "criminal"])
    start(client, auth_headers, dataset_id, min_per_class=7)

    label_next(client, auth_headers, dataset_id, _all("civil"), 5)
    label_next(client, auth_headers, dataset_id, _all("criminal"), 5)

    assert state(client, auth_headers, dataset_id)["retrain"]["classes_short"] == {"civil": 2, "criminal": 2}


def test_a_minimum_too_small_to_train_on_is_refused(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, ["a", "b"])

    resp = start(client, auth_headers, dataset_id, min_per_class=2)

    assert resp.status_code == 400
    assert resp.get_json()["error"] == "Each label needs at least 5 examples to train on."


def test_one_label_is_not_enough_to_train(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)

    label_next(client, auth_headers, dataset_id, _all("civil"), 8)

    body = state(client, auth_headers, dataset_id)["retrain"]
    assert body["allowed"] is False
    assert body["needs_second_label"] is True


def test_labels_already_in_the_file_count_towards_the_minimum(client, auth_headers, app, tmp_path):
    texts = balanced_texts(10)
    labels = ["civil"] * 5 + ["criminal"] * 5 + [None] * 10
    dataset_id = link_dataset(app, tmp_path, texts, labels=labels)

    start(client, auth_headers, dataset_id, holdout_percent=0)

    body = state(client, auth_headers, dataset_id)
    assert body["label_counts"] == {"civil": 5, "criminal": 5}
    assert body["retrain"]["allowed"] is True
    assert body["retrain"]["new_labels"] == 10

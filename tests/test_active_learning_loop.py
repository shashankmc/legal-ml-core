"""The loop: label a batch, retrain on request, and the queue reorders."""
import json
from datetime import datetime, timedelta

import pytest

from legal_ml_core import ALRound, Model, TrainingJob, db
from legal_ml_core.active_learning.loop import validate_host_config
from legal_ml_core.compute.tasks import process_training_task
from tests.active_learning_helpers import (
    PROJECT_ID,
    alternating,
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
from tests.conftest import TEST_USER_ID


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

    labeller = alternating()
    label_next(client, auth_headers, dataset_id, labeller, 15)
    assert state(client, auth_headers, dataset_id)["retrain"]["suggested"] is False

    label_next(client, auth_headers, dataset_id, labeller, 1)
    retrain_state = state(client, auth_headers, dataset_id)["retrain"]
    assert retrain_state["suggested"] is True
    assert retrain_state["new_labels"] == 16


def test_nothing_retrains_until_the_user_asks(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path, batch_size=2)

    label_next(client, auth_headers, dataset_id, by_topic, 3)

    assert fake_redis.llen("training_queue") == 0


def _trained_round(client, auth_headers, app, fake_redis, dataset_id, labels=16):
    label_next(client, auth_headers, dataset_id, alternating(), labels)
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
    label_next(client, auth_headers, dataset_id, alternating(), 16)

    first = retrain(client, auth_headers, dataset_id)
    second = retrain(client, auth_headers, dataset_id)

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.get_json()["message"] == "Training already started"
    assert fake_redis.llen("training_queue") == 1


def test_a_failed_training_returns_to_labelling_with_the_reason(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, alternating(), 16)
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
    label_next(client, auth_headers, dataset_id, alternating(), 16)

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


def _train_without_rescoring(app, fake_redis):
    """The worker finishes the job but dies before it hands over to scoring."""
    task = json.loads(fake_redis.rpop("training_queue"))
    task.pop("al_round_id")
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"])
    db.session.expire_all()


def test_a_finished_job_with_no_scoring_moves_the_round_on(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, alternating(), 16)
    retrain(client, auth_headers, dataset_id)
    _train_without_rescoring(app, fake_redis)

    assert state(client, auth_headers, dataset_id)["round"]["status"] == "scoring"

    state(client, auth_headers, dataset_id)
    assert fake_redis.llen("scoring_queue") == 1
    run_scoring(app, fake_redis)
    body = state(client, auth_headers, dataset_id)
    assert body["round"]["number"] == 1
    assert ALRound.query.filter_by(dataset_id=dataset_id, number=0).one().metrics is not None


def test_a_finished_job_keeps_the_round_when_scoring_cannot_queue(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, alternating(), 16)
    retrain(client, auth_headers, dataset_id)
    _train_without_rescoring(app, fake_redis)

    def boom(*args):
        raise ConnectionError("redis down")

    fake_redis.lpush = boom
    body = state(client, auth_headers, dataset_id)

    assert body["round"]["status"] == "labelling"
    assert body["round"]["error"] == "The queue could not be reordered. Try retraining."
    assert body["retrain"]["allowed"] is True


def _round_with_no_job(dataset_id, age_seconds):
    round_row = ALRound.query.filter_by(dataset_id=dataset_id).one()
    ALRound.query.filter_by(id=round_row.id).update(
        {"status": "training", "updated_at": datetime.utcnow() - timedelta(seconds=age_seconds)},
        synchronize_session=False,
    )
    db.session.commit()
    db.session.expire_all()


def test_a_round_that_never_got_a_job_returns_to_labelling(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, alternating(), 16)
    _round_with_no_job(dataset_id, age_seconds=600)

    body = state(client, auth_headers, dataset_id)

    assert body["round"]["status"] == "labelling"
    assert body["round"]["error"] == "Training could not start. Try again."
    assert body["retrain"]["allowed"] is True


def test_a_round_just_claimed_is_given_time_to_get_its_job(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, alternating(), 16)
    _round_with_no_job(dataset_id, age_seconds=1)

    assert state(client, auth_headers, dataset_id)["round"]["status"] == "training"


def test_a_round_records_its_job_with_the_job(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, alternating(), 16)

    job_id = retrain(client, auth_headers, dataset_id).get_json()["job_id"]

    assert ALRound.query.filter_by(dataset_id=dataset_id).one().job_id == job_id


def test_retrain_is_refused_while_the_model_already_trains(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, alternating(), 16)
    retrain(client, auth_headers, dataset_id)
    run_training(app, fake_redis)
    run_scoring(app, fake_redis)
    label_next(client, auth_headers, dataset_id, alternating(), 2)
    outside = client.post(f"/api/models/{loop_model(dataset_id).id}/train", json={}, headers=auth_headers)
    assert outside.status_code == 200

    resp = retrain(client, auth_headers, dataset_id)

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "This Model is already training. Retrain when it finishes."
    assert fake_redis.llen("training_queue") == 1
    body = state(client, auth_headers, dataset_id)
    assert body["round"]["status"] == "labelling"
    assert body["retrain"]["new_labels"] == 2


def _named_label_loop(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, balanced_texts(10), dataset_labels=["civil", "criminal"])
    start(client, auth_headers, dataset_id)
    return dataset_id


def _outside_model(dataset_id):
    model = Model(
        owner_id=TEST_USER_ID, dataset_id=dataset_id, name="Mine", model_type="logistic_regression", status="pending"
    )
    db.session.add(model)
    db.session.commit()
    return model.id


def test_train_routes_keep_the_minimum_for_a_dataset_in_a_loop(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _named_label_loop(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, _all("civil"), 5)
    model_id = _outside_model(dataset_id)

    by_model = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)
    by_start = client.post("/api/training/start", json={"model_id": model_id}, headers=auth_headers)

    for resp in (by_model, by_start):
        assert resp.status_code == 409
        assert resp.get_json()["error"] == "Each label needs at least 5 examples before training."
    assert fake_redis.llen("training_queue") == 0


def test_a_model_cannot_be_created_below_the_minimum(client, auth_headers, app, tmp_path):
    dataset_id = _named_label_loop(client, auth_headers, app, tmp_path)

    resp = client.post(
        "/api/models/",
        json={"name": "Mine", "dataset_id": dataset_id, "model_type": "logistic_regression", "parameters": {}},
        headers=auth_headers,
    )

    assert resp.status_code == 409
    assert resp.get_json()["error"] == "Each label needs at least 5 examples before training."
    assert Model.query.filter_by(dataset_id=dataset_id).count() == 0


def test_train_routes_open_once_the_minimum_is_met(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _named_label_loop(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, _all("civil"), 5)
    label_next(client, auth_headers, dataset_id, _all("criminal"), 5)

    resp = client.post(f"/api/models/{_outside_model(dataset_id)}/train", json={}, headers=auth_headers)

    assert resp.status_code == 200
    assert fake_redis.llen("training_queue") == 1


def test_a_dataset_with_no_loop_trains_as_before(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = link_dataset(app, tmp_path, balanced_texts(10), labels=[by_topic(t) for t in balanced_texts(10)])

    resp = client.post(f"/api/models/{_outside_model(dataset_id)}/train", json={}, headers=auth_headers)

    assert resp.status_code == 200


def _rename(client, auth_headers, dataset_id, old, new):
    return client.put(
        f"/api/active-learning/datasets/{dataset_id}/labels",
        json={"from": old, "to": new},
        headers=auth_headers,
    )


def _typo_loop(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, alternating(), 16)
    label_next(client, auth_headers, dataset_id, _all("crimnal"), 1)
    return dataset_id


def test_a_mistyped_label_blocks_retrain_until_it_is_corrected(client, auth_headers, app, tmp_path, fake_redis):
    dataset_id = _typo_loop(client, auth_headers, app, tmp_path)
    before = state(client, auth_headers, dataset_id)
    assert before["retrain"]["classes_short"] == {"crimnal": 4}
    assert before["typed_labels"] is True

    resp = _rename(client, auth_headers, dataset_id, "crimnal", " criminal ")

    assert resp.status_code == 200
    body = resp.get_json()
    assert body["label_counts"] == {"civil": 8, "criminal": 9}
    assert body["retrain"]["allowed"] is True
    assert retrain(client, auth_headers, dataset_id).status_code == 202


def test_correcting_a_label_reaches_held_back_items_too(client, auth_headers, app, tmp_path):
    dataset_id = _started(client, auth_headers, app, tmp_path)
    task = next_item(client, auth_headers, dataset_id, holdout=1)["task"]
    client.post(f"/api/annotation/tasks/{task['id']}/save", json={"annotation_label": "crimnal"}, headers=auth_headers)

    assert _rename(client, auth_headers, dataset_id, "crimnal", "criminal").status_code == 200

    db.session.expire_all()
    assert task_for(dataset_id, task["item_index"]).annotation_label == "criminal"


def test_a_label_nothing_uses_cannot_be_corrected(client, auth_headers, app, tmp_path):
    dataset_id = _typo_loop(client, auth_headers, app, tmp_path)

    resp = _rename(client, auth_headers, dataset_id, "nothing", "civil")

    assert resp.status_code == 404
    assert resp.get_json()["error"] == "No saved label has that name."


@pytest.mark.parametrize("new", ["", "   ", None, 3])
def test_a_label_cannot_be_corrected_to_nothing(client, auth_headers, app, tmp_path, new):
    dataset_id = _typo_loop(client, auth_headers, app, tmp_path)

    assert _rename(client, auth_headers, dataset_id, "crimnal", new).status_code == 400


def test_a_dataset_that_names_its_labels_keeps_them(client, auth_headers, app, tmp_path):
    dataset_id = _named_label_loop(client, auth_headers, app, tmp_path)
    label_next(client, auth_headers, dataset_id, _all("civil"), 1)

    assert state(client, auth_headers, dataset_id)["typed_labels"] is False
    assert _rename(client, auth_headers, dataset_id, "civil", "criminal").status_code == 400


def test_correcting_a_label_needs_a_loop(client, auth_headers, app, tmp_path):
    dataset_id = link_dataset(app, tmp_path, ["a", "b"])

    assert _rename(client, auth_headers, dataset_id, "x", "y").status_code == 409

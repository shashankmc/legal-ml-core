"""Changing a trained Model's settings and retraining it (BAIGLE #86): the
settings are frozen while a run is active, and group names survive a retrain
only when the settings did not change, because group numbers do not carry over."""
import json

from legal_ml_core import Model, TrainingJob, db
from legal_ml_core.compute.tasks import process_training_task
from tests.test_training_parameters import _run, clustering, topics  # noqa: F401


def _start(client, auth_headers, model_id):
    response = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)
    assert response.status_code == 200, response.get_json()


def _retrain(client, auth_headers, app, fake_redis, model_id):
    _start(client, auth_headers, model_id)
    task = json.loads(fake_redis.rpop("training_queue"))
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"])
    db.session.expire_all()
    return db.session.get(Model, model_id)


def _name_groups(client, auth_headers, model_id, key):
    response = client.put(
        f"/api/models/{model_id}/labels", json={key: {"0": "Contracts"}}, headers=auth_headers
    )
    assert response.status_code == 200


def _put(client, auth_headers, model_id, body):
    return client.put(f"/api/models/{model_id}", json=body, headers=auth_headers)


def test_settings_cannot_change_while_the_model_is_training(client, auth_headers, app, fake_redis, clustering):
    model_id, _ = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})
    _start(client, auth_headers, model_id)

    response = _put(client, auth_headers, model_id, {"parameters": {"n_clusters": 3}})

    assert response.status_code == 409
    assert "training" in response.get_json()["error"]
    db.session.expire_all()
    assert db.session.get(Model, model_id).parameters == {"n_clusters": 2}


def test_a_training_model_can_still_be_renamed(client, auth_headers, app, fake_redis, clustering):
    model_id, _ = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})
    _start(client, auth_headers, model_id)

    response = _put(
        client, auth_headers, model_id, {"name": "Renamed", "parameters": {"n_clusters": 2}}
    )

    assert response.status_code == 200
    assert response.get_json()["model"]["name"] == "Renamed"


def test_settings_can_change_once_the_run_has_finished(client, auth_headers, app, fake_redis, clustering):
    model_id, _ = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})

    response = _put(client, auth_headers, model_id, {"parameters": {"n_clusters": 3}})

    assert response.status_code == 200
    assert response.get_json()["model"]["parameters"] == {"n_clusters": 3}


def test_group_names_are_kept_when_a_model_retrains_on_the_same_settings(
    client, auth_headers, app, fake_redis, clustering
):
    model_id, _ = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})
    _name_groups(client, auth_headers, model_id, "clusters")

    model = _retrain(client, auth_headers, app, fake_redis, model_id)

    assert model.status == "completed"
    assert model.metrics["display_labels"] == {"clusters": {"0": "Contracts"}}


def test_group_names_are_cleared_when_a_clustering_retrains_on_changed_settings(
    client, auth_headers, app, fake_redis, clustering
):
    model_id, _ = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})
    _name_groups(client, auth_headers, model_id, "clusters")
    assert _put(client, auth_headers, model_id, {"parameters": {"n_clusters": 3}}).status_code == 200

    model = _retrain(client, auth_headers, app, fake_redis, model_id)

    assert model.status == "completed"
    assert "display_labels" not in model.metrics


def test_topic_names_are_cleared_when_a_topic_model_retrains_on_changed_settings(
    client, auth_headers, app, fake_redis, topics
):
    model_id, _ = _run(client, auth_headers, app, fake_redis, topics, "lda", {"n_components": 2})
    _name_groups(client, auth_headers, model_id, "topics")
    assert _put(client, auth_headers, model_id, {"parameters": {"n_components": 3}}).status_code == 200

    model = _retrain(client, auth_headers, app, fake_redis, model_id)

    assert model.status == "completed"
    assert "display_labels" not in model.metrics


def test_group_names_stay_until_the_changed_run_succeeds(client, auth_headers, app, fake_redis, clustering):
    """Saving new settings alone keeps the names: the results on show are still the old run's."""
    model_id, _ = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})
    _name_groups(client, auth_headers, model_id, "clusters")

    response = _put(client, auth_headers, model_id, {"parameters": {"n_clusters": 3}})

    assert response.get_json()["model"]["metrics"]["display_labels"] == {"clusters": {"0": "Contracts"}}


# What the settings form sends: every setting, with the defaults filled in.
KMEANS_FORM = {"n_clusters": 2, "init": "k-means++", "max_iter": 300, "random_state": 42}


def test_settings_cannot_change_while_a_run_is_running(client, auth_headers, app, fake_redis, clustering):
    model_id, _ = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})
    _start(client, auth_headers, model_id)
    job = TrainingJob.query.filter_by(model_id=model_id, status="pending").one()
    job.status = "running"
    db.session.commit()

    response = _put(client, auth_headers, model_id, {"parameters": {"n_clusters": 3}})

    assert response.status_code == 409
    assert response.get_json()["job_id"] == str(job.id)


def test_the_form_saving_the_same_settings_with_defaults_filled_in_is_not_a_change(
    client, auth_headers, app, fake_redis, clustering
):
    model_id, _ = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})
    _start(client, auth_headers, model_id)

    response = _put(client, auth_headers, model_id, {"parameters": KMEANS_FORM})

    assert response.status_code == 200


def test_group_names_are_kept_when_save_and_retrain_sends_the_same_settings_in_full(
    client, auth_headers, app, fake_redis, clustering
):
    model_id, _ = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})
    _name_groups(client, auth_headers, model_id, "clusters")
    assert _put(client, auth_headers, model_id, {"parameters": KMEANS_FORM}).status_code == 200

    model = _retrain(client, auth_headers, app, fake_redis, model_id)

    assert model.status == "completed"
    assert model.metrics["display_labels"] == {"clusters": {"0": "Contracts"}}


def test_group_names_are_cleared_when_the_last_run_has_no_recorded_settings(
    client, auth_headers, app, fake_redis, clustering
):
    """Runs from before settings were recorded could have used any settings, and
    names on the wrong groups are worse than lost names."""
    model_id, job = _run(client, auth_headers, app, fake_redis, clustering, "kmeans", {"n_clusters": 2})
    job.parameters = None
    db.session.commit()
    _name_groups(client, auth_headers, model_id, "clusters")

    model = _retrain(client, auth_headers, app, fake_redis, model_id)

    assert model.status == "completed"
    assert "display_labels" not in model.metrics

"""Learning curves for the classical algorithms (BAIGLE #93). With
`monitoring.learning_curve` on, a run also scores the algorithm on growing
parts of its training split. The points carry the training-set size as their
`training_step`, under `learning_curve_*` names, so a chart can draw them
against size."""
import json

import pandas as pd
import pytest

from legal_ml_core import TrainingJob, db
from legal_ml_core.compute.tasks import process_training_task
from legal_ml_core.compute.trainer import MLModelTrainer
from legal_ml_core.monitoring import effective_monitoring
from tests.test_monitoring_settings import _start, dataset_id  # noqa: F401

CLASSIFIERS = ["svm", "decision_tree", "random_forest", "naive_bayes", "logistic_regression"]


class RecordingLogger:
    def __init__(self):
        self.points = []

    def log(self, metric_name, value, epoch=None, training_step=None, group=None):
        self.points.append(
            {"name": metric_name, "value": value, "training_step": training_step, "group": group}
        )


@pytest.fixture
def classification_df():
    texts = [
        "the court ruled in favour of the plaintiff",
        "the defendant was found liable for damages",
        "judgment was entered against the appellant",
        "the parties reached a settlement agreement",
        "the tribunal dismissed the case entirely",
        "the appeal was rejected by the higher court",
        "the jury returned a verdict of guilty",
        "the contract was declared void by the judge",
    ] * 5
    return pd.DataFrame({"text": texts, "label": ["civil", "criminal"] * 20})


def _train(fake_redis, df, model_type, monitoring, cancel_check=None):
    logger = RecordingLogger()
    trainer = MLModelTrainer(redis_client=fake_redis, metric_logger=logger, cancel_check=cancel_check)
    texts, labels, _ = trainer.preprocess_data(df, "classification", {})
    parameters = {"model_id": "lc", "monitoring": effective_monitoring(run=monitoring)}
    metrics = trainer.train_classification_model(texts, labels, model_type, parameters)
    return logger.points, metrics


def _curve(points, name):
    return [(p["training_step"], p["group"]) for p in points if p["name"] == name]


@pytest.mark.parametrize("model_type", CLASSIFIERS)
def test_every_classifier_scores_growing_parts_of_its_training_data(fake_redis, classification_df, model_type):
    points, _ = _train(fake_redis, classification_df, model_type, {"learning_curve": True})

    train = _curve(points, "learning_curve_train_accuracy")
    evaluation = _curve(points, "learning_curve_eval_accuracy")
    sizes = [size for size, _ in train]
    assert len(sizes) >= 3
    assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)
    assert sizes[-1] == 32  # the whole training split: 80% of 40 rows
    assert {group for _, group in train} == {"train"}
    assert evaluation == [(size, "eval") for size in sizes]
    assert all(0.0 <= p["value"] <= 1.0 for p in points if p["name"].startswith("learning_curve_"))


@pytest.mark.parametrize("model_type", CLASSIFIERS)
def test_learning_curves_are_off_by_default_and_leave_the_run_unchanged(fake_redis, classification_df, model_type):
    off, metrics_off = _train(fake_redis, classification_df, model_type, {})
    _, metrics_on = _train(fake_redis, classification_df, model_type, {"learning_curve": True})

    assert off == []
    assert metrics_on == metrics_off


def test_logistic_regression_also_records_its_loss_iteration_by_iteration(fake_redis, classification_df):
    points, _ = _train(fake_redis, classification_df, "logistic_regression", {"learning_curve": True})

    train = [(p["training_step"], p["value"]) for p in points if p["name"] == "loss"]
    evaluation = _curve(points, "eval_loss")
    iterations = [step for step, _ in train]
    assert len(iterations) >= 3 and iterations[0] == 1
    assert iterations == sorted(iterations) and len(set(iterations)) == len(iterations)
    assert {p["group"] for p in points if p["name"] == "loss"} == {"train"}
    assert evaluation == [(step, "eval") for step in iterations]
    assert train[-1][1] < train[0][1]  # it learns


@pytest.mark.parametrize("model_type", ["svm", "decision_tree", "random_forest", "naive_bayes"])
def test_algorithms_without_iterations_record_no_loss(fake_redis, classification_df, model_type):
    points, _ = _train(fake_redis, classification_df, model_type, {"learning_curve": True})

    assert not [p for p in points if p["name"] in ("loss", "eval_loss")]


@pytest.mark.parametrize("model_type", ["naive_bayes", "logistic_regression"])
def test_a_stop_is_honoured_between_learning_curve_fits(fake_redis, classification_df, model_type):
    from legal_ml_core.compute.trainer import TrainingCancelled

    checks = []

    def stop_after_two_fits():
        checks.append(1)
        return len(checks) > 2

    logger = RecordingLogger()
    trainer = MLModelTrainer(redis_client=fake_redis, metric_logger=logger, cancel_check=stop_after_two_fits)
    texts, labels, _ = trainer.preprocess_data(classification_df, "classification", {})
    parameters = {"model_id": "lc", "monitoring": effective_monitoring(run={"learning_curve": True})}

    with pytest.raises(TrainingCancelled):
        trainer.train_classification_model(texts, labels, model_type, parameters)

    assert [p["training_step"] for p in logger.points if p["group"] == "eval"] == [3, 8]
    assert trainer.model is None or not hasattr(trainer.model, "classes_")  # the real fit never ran


def _points(client, auth_headers, job_id):
    return client.get(f"/api/training/{job_id}/metrics", headers=auth_headers).get_json()["points"]


def test_a_run_stores_its_learning_curve_against_training_set_size(app, client, auth_headers, fake_redis, dataset_id):
    _start(client, auth_headers, dataset_id, {"learning_curve": True})
    task = json.loads(fake_redis.rpop("training_queue"))

    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"], trainer_factory=MLModelTrainer)

    assert db.session.get(TrainingJob, task["job_id"]).status == "completed"
    curve = [
        (p["metric_name"], p["group"], p["training_step"])
        for p in _points(client, auth_headers, task["job_id"])
        if p["metric_name"].startswith("learning_curve_")
    ]
    assert ("learning_curve_train_accuracy", "train", 16) in curve
    assert ("learning_curve_eval_accuracy", "eval", 16) in curve
    assert len({size for _, _, size in curve}) >= 3


def test_a_stop_during_the_learning_curve_cancels_the_run(app, client, auth_headers, fake_redis, dataset_id):
    _start(client, auth_headers, dataset_id, {"learning_curve": True})
    task = json.loads(fake_redis.rpop("training_queue"))

    def stopping_on_the_first_curve_point(redis_client, metric_logger, cancel_check):
        class Stopping:
            def log(self, metric_name, *args, **kwargs):
                if metric_name.startswith("learning_curve_"):
                    client.post(f"/api/training/stop/{task['job_id']}", headers=auth_headers)
                return metric_logger.log(metric_name, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(metric_logger, name)

        return MLModelTrainer(redis_client, metric_logger=Stopping(), cancel_check=cancel_check)

    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"], trainer_factory=stopping_on_the_first_curve_point)

    assert db.session.get(TrainingJob, task["job_id"]).status == "cancelled"
    names = [p["metric_name"] for p in _points(client, auth_headers, task["job_id"])]
    assert names[-1] == "training_cancelled"
    assert names.count("learning_curve_eval_accuracy") == 1  # no fit after the Stop


@pytest.mark.parametrize(
    "task_type, model_type",
    [("clustering", "kmeans"), ("clustering", "hierarchical"), ("clustering", "dbscan"), ("topic_modeling", "lda"), ("topic_modeling", "nmf")],
)
def test_groupings_have_no_labels_to_score_so_they_train_as_before(fake_redis, classification_df, task_type, model_type):
    """A learning curve scores against labels; finding groups has none."""
    logger = RecordingLogger()
    trainer = MLModelTrainer(redis_client=fake_redis, metric_logger=logger)
    texts, _, _ = trainer.preprocess_data(classification_df, task_type, {})
    parameters = {"model_id": "lc", "n_clusters": 2, "n_components": 2, "monitoring": effective_monitoring(run={"learning_curve": True})}

    if task_type == "clustering":
        trainer.train_clustering_model(texts, model_type, parameters)
    else:
        trainer.train_topic_modeling(texts, model_type, parameters)

    assert logger.points == []


def test_a_part_the_algorithm_cannot_fit_is_left_out_of_the_curve(fake_redis, classification_df, monkeypatch):
    from sklearn.naive_bayes import MultinomialNB

    real_fit = MultinomialNB.fit

    def refuse_small_parts(self, X, y, *args, **kwargs):
        if X.shape[0] < 10:
            raise ValueError("too few rows")
        return real_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(MultinomialNB, "fit", refuse_small_parts)

    points, metrics = _train(fake_redis, classification_df, "naive_bayes", {"learning_curve": True})

    assert [size for size, _ in _curve(points, "learning_curve_eval_accuracy")] == [16, 24, 32]
    assert 0.0 <= metrics["accuracy"] <= 1.0

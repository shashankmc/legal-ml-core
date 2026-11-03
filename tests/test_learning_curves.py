"""Learning curves for the classical algorithms (BAIGLE #93). With
`monitoring.learning_curve` on, a run also scores the algorithm on growing
parts of its training split. The points carry the training-set size as their
`training_step`, under `learning_curve_*` names, so a chart can draw them
against size."""
import pandas as pd
import pytest

from legal_ml_core.compute.trainer import MLModelTrainer
from legal_ml_core.monitoring import effective_monitoring

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

"""The worker rescores the pool from a queue message once a round's model is saved."""
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from legal_ml_core import ALConfig, ALRound, AnnotationTask, Dataset, Model, db
from legal_ml_core.active_learning.scoring import requeue_unfinished_scoring
from legal_ml_core.storage import get_model_path
from legal_ml_core.worker import _scoring_loop
from tests.conftest import TEST_USER_ID

TRAINING_TEXTS = ["contract breach damages"] * 4 + ["murder theft assault"] * 4
TRAINING_LABELS = ["civil"] * 4 + ["criminal"] * 4


def _saved_model(app, dataset_id):
    model = Model(
        owner_id=TEST_USER_ID, dataset_id=dataset_id, name="AL", model_type="logistic_regression", status="completed"
    )
    db.session.add(model)
    db.session.commit()
    vectorizer = TfidfVectorizer()
    encoder = LabelEncoder()
    clf = LogisticRegression().fit(vectorizer.fit_transform(TRAINING_TEXTS), encoder.fit_transform(TRAINING_LABELS))
    joblib.dump(
        {"model": clf, "vectorizer": vectorizer, "label_encoder": encoder, "preprocessor": None},
        get_model_path(app.config["MODEL_FOLDER"], model.id),
    )
    return model


def _scoring_setup(app, strategy="prediction_entropy", texts=("contract breach damages", "murder contract", "zebra")):
    dataset = Dataset(owner_id=TEST_USER_ID, name="D", file_path="x.csv", row_count=9, task_type="classification")
    db.session.add(dataset)
    db.session.commit()
    model = _saved_model(app, dataset.id)
    db.session.add(ALConfig(dataset_id=dataset.id, strategy=strategy, model_id=model.id))
    rnd = ALRound(dataset_id=dataset.id, number=0, strategy="random", status="scoring", model_id=model.id)
    db.session.add(rnd)
    for index, text in enumerate(texts):
        db.session.add(AnnotationTask(dataset_id=dataset.id, item_index=index, text=text))
    db.session.commit()
    return dataset, model, rnd


def _run(app, fake_redis, rnd, model, strategy, queue="scoring_queue"):
    fake_redis.lpush(
        queue,
        json.dumps(
            {
                "round_id": rnd.id,
                "dataset_id": rnd.dataset_id,
                "model_id": model.id,
                "strategy": strategy,
            }
        ),
    )
    _scoring_loop(
        fake_redis,
        app.config["MODEL_FOLDER"],
        app,
        queue,
        stop=lambda: fake_redis.llen(queue) == 0,
        sleep=lambda s: None,
    )
    db.session.expire_all()


def _scores(dataset):
    return {t.text: t.score for t in AnnotationTask.query.filter_by(dataset_id=dataset.id)}


def test_scoring_ranks_the_uncertain_item_above_the_clear_one(app, fake_redis):
    dataset, model, rnd = _scoring_setup(app)

    _run(app, fake_redis, rnd, model, "prediction_entropy")

    scores = _scores(dataset)
    assert scores["murder contract"] > scores["contract breach damages"]


def test_scoring_closes_the_round_and_opens_the_next(app, fake_redis):
    dataset, model, rnd = _scoring_setup(app)

    _run(app, fake_redis, rnd, model, "prediction_entropy")

    rounds = ALRound.query.filter_by(dataset_id=dataset.id).order_by(ALRound.number).all()
    assert [(r.number, r.status) for r in rounds] == [(0, "completed"), (1, "labelling")]
    assert rounds[0].completed_at is not None
    assert rounds[1].strategy == "prediction_entropy"


def test_every_strategy_scores_the_whole_pool(app, fake_redis):
    for strategy in ("least_confidence", "breaking_ties", "embedding_kmeans", "random"):
        dataset, model, rnd = _scoring_setup(app, strategy=strategy)

        _run(app, fake_redis, rnd, model, strategy)

        assert all(score is not None for score in _scores(dataset).values()), strategy


def test_labelled_and_holdout_items_are_not_scored(app, fake_redis):
    dataset, model, rnd = _scoring_setup(app)
    tasks = AnnotationTask.query.filter_by(dataset_id=dataset.id).order_by(AnnotationTask.item_index).all()
    tasks[0].status = "annotated"
    tasks[1].is_holdout = True
    db.session.commit()

    _run(app, fake_redis, rnd, model, "prediction_entropy")

    scores = _scores(dataset)
    assert scores["contract breach damages"] is None
    assert scores["murder contract"] is None
    assert scores["zebra"] is not None


def test_an_empty_pool_still_closes_the_round(app, fake_redis):
    dataset, model, rnd = _scoring_setup(app, texts=())

    _run(app, fake_redis, rnd, model, "prediction_entropy")

    assert db.session.get(ALRound, rnd.id).status == "completed"


def test_a_scoring_failure_is_recorded_and_the_loop_survives(app, fake_redis):
    dataset, model, rnd = _scoring_setup(app)
    good_dataset, good_model, good_round = _scoring_setup(app)
    fake_redis.lpush(
        "scoring_queue",
        json.dumps(
            {
                "round_id": good_round.id,
                "dataset_id": good_dataset.id,
                "model_id": good_model.id,
                "strategy": "random",
            }
        ),
    )
    fake_redis.lpush("scoring_queue", "not json at all")
    # This model file does not exist, so scoring it fails.
    _run(app, fake_redis, rnd, type("Missing", (), {"id": "gone"}), "random")

    failed = db.session.get(ALRound, rnd.id)
    assert failed.status == "labelling"
    assert failed.error == "The queue could not be reordered. Try retraining."
    assert db.session.get(ALRound, good_round.id).status == "completed"


def test_a_round_no_longer_scoring_is_left_alone(app, fake_redis):
    dataset, model, rnd = _scoring_setup(app)
    rnd.status = "labelling"
    db.session.commit()

    _run(app, fake_redis, rnd, model, "prediction_entropy")

    assert all(score is None for score in _scores(dataset).values())


def test_a_round_records_the_strategy_that_ordered_it(app, fake_redis):
    """The user can change strategy while a rescore runs: the next round was still
    ordered by the one in the message."""
    dataset, model, rnd = _scoring_setup(app, strategy="breaking_ties")

    _run(app, fake_redis, rnd, model, "prediction_entropy")

    assert ALRound.query.filter_by(dataset_id=dataset.id, number=1).one().strategy == "prediction_entropy"


def test_a_worker_restart_resumes_a_pass_the_old_worker_took_with_it(app, fake_redis):
    dataset, _, rnd = _scoring_setup(app)
    assert fake_redis.llen("scoring_queue") == 0

    requeue_unfinished_scoring(fake_redis, "scoring_queue")
    _scoring_loop(
        fake_redis,
        app.config["MODEL_FOLDER"],
        app,
        stop=lambda: fake_redis.llen("scoring_queue") == 0,
        sleep=lambda s: None,
    )
    db.session.expire_all()

    assert db.session.get(ALRound, rnd.id).status == "completed"
    assert all(score is not None for score in _scores(dataset).values())


def test_a_worker_restart_leaves_finished_rounds_out_of_the_queue(app, fake_redis):
    _, _, rnd = _scoring_setup(app)
    rnd.status = "completed"
    db.session.commit()

    requeue_unfinished_scoring(fake_redis, "scoring_queue")

    assert fake_redis.llen("scoring_queue") == 0

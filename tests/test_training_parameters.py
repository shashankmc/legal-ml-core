"""What a user chooses reaches the estimator that is saved, through the real
route, queue message and worker. Everything is read off the fitted estimator,
because the stored parameters always showed the choice, even when it was ignored."""
import json

import joblib
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from legal_ml_core import Dataset, Model, TrainingJob, TrainingMetricPoint, db
from legal_ml_core.compute.tasks import process_training_task
from legal_ml_core.storage import get_model_path
from tests.conftest import TEST_USER_ID

TOPICS = [
    "the court ruled for the plaintiff on the contract claim",
    "the defendant was found liable for negligent damages",
    "the appellant lodged a notice against the sentence",
    "the parties settled the employment dispute by agreement",
    "the tribunal dismissed the asylum application entirely",
    "the jury returned a guilty verdict after the trial",
]


def _dataset(app, tmp_path, task_type, labels):
    rows = pd.DataFrame({"text": TOPICS * 5, "label": labels})
    path = tmp_path / "train.csv"
    rows.to_csv(path, index=False)
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="D1",
            file_path=str(path),
            row_count=len(rows),
            columns=[],
            task_type=task_type,
        )
        db.session.add(dataset)
        db.session.commit()
        return dataset.id


def _run(client, auth_headers, app, fake_redis, dataset_id, model_type, parameters):
    """Create, start and run a model as the wizard and the worker do."""
    created = client.post(
        "/api/models/",
        json={"name": "M", "dataset_id": dataset_id, "model_type": model_type, "parameters": parameters},
        headers=auth_headers,
    )
    assert created.status_code == 201, created.get_json()
    model_id = created.get_json()["model"]["id"]
    started = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)
    assert started.status_code == 200, started.get_json()

    task = json.loads(fake_redis.rpop("training_queue"))
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"])

    return model_id, db.session.get(TrainingJob, task["job_id"])


def _train(client, auth_headers, app, fake_redis, dataset_id, model_type, parameters):
    model_id, job = _run(client, auth_headers, app, fake_redis, dataset_id, model_type, parameters)
    assert job.status == "completed", job.error_message
    return model_id, joblib.load(get_model_path(app.config["MODEL_FOLDER"], model_id))["model"]


def _fitted(model):
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


def _params(estimator, *names):
    return {name: getattr(estimator, name) for name in names}


@pytest.fixture
def classifying(app, tmp_path):
    return _dataset(app, tmp_path, "classification", ["civil", "criminal"] * 15)


@pytest.fixture
def clustering(app, tmp_path):
    return _dataset(app, tmp_path, "clustering", ["x"] * 30)


@pytest.fixture
def topics(app, tmp_path):
    return _dataset(app, tmp_path, "topic_modeling", ["x"] * 30)


@pytest.mark.parametrize(
    "model_type, chosen, expected",
    [
        (
            "svm",
            {"kernel": "poly", "C": 2.5, "gamma": "auto", "degree": 4, "class_weight": "none"},
            {"kernel": "poly", "C": 2.5, "gamma": "auto", "degree": 4},
        ),
        (
            "decision_tree",
            {"max_depth": 2, "min_samples_split": 4, "min_samples_leaf": 3, "criterion": "entropy"},
            {"max_depth": 2, "min_samples_split": 4, "min_samples_leaf": 3, "criterion": "entropy"},
        ),
        (
            "random_forest",
            {"n_estimators": 7, "max_depth": 2, "min_samples_split": 5, "min_samples_leaf": 2},
            {"n_estimators": 7, "max_depth": 2, "min_samples_split": 5, "min_samples_leaf": 2},
        ),
        ("naive_bayes", {"alpha": 0.5, "fit_prior": False}, {"alpha": 0.5, "fit_prior": False}),
        (
            "logistic_regression",
            {"C": 0.5, "penalty": "l1", "solver": "liblinear"},
            {"C": 0.5, "penalty": "l1", "solver": "liblinear"},
        ),
        (
            "logistic_regression",
            {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.3},
            {"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.3},
        ),
    ],
)
def test_a_saved_classifier_has_the_chosen_parameters(
    client, auth_headers, app, fake_redis, classifying, model_type, chosen, expected
):
    _, model = _train(client, auth_headers, app, fake_redis, classifying, model_type, chosen)

    assert _params(_fitted(model), *expected) == expected


def test_the_forest_really_grows_the_chosen_trees(client, auth_headers, app, fake_redis, classifying):
    _, model = _train(
        client, auth_headers, app, fake_redis, classifying, "random_forest", {"n_estimators": 7, "max_depth": 2}
    )

    forest = _fitted(model)
    assert len(forest.estimators_) == 7
    assert max(tree.get_depth() for tree in forest.estimators_) <= 2


def test_elasticnet_without_a_ratio_mixes_both_penalties(client, auth_headers, app, fake_redis, classifying):
    _, model = _train(
        client, auth_headers, app, fake_redis, classifying, "logistic_regression",
        {"penalty": "elasticnet", "solver": "saga"},
    )

    assert _fitted(model).l1_ratio == 0.5


def test_unlimited_depth_is_the_wizards_untouched_default(client, auth_headers, app, fake_redis, classifying):
    _, model = _train(client, auth_headers, app, fake_redis, classifying, "decision_tree", {"max_depth": None})

    assert _fitted(model).max_depth is None


@pytest.mark.parametrize(
    "model_type, chosen, expected",
    [
        (
            "kmeans",
            {"n_clusters": 3, "init": "random", "max_iter": 150, "random_state": 7},
            {"n_clusters": 3, "init": "random", "max_iter": 150, "random_state": 7},
        ),
        (
            "hierarchical",
            {"n_clusters": 3, "linkage": "complete", "affinity": "manhattan"},
            {"n_clusters": 3, "linkage": "complete", "metric": "manhattan"},
        ),
        (
            "dbscan",
            {"eps": 0.7, "min_samples": 2, "metric": "cosine"},
            {"eps": 0.7, "min_samples": 2, "metric": "cosine"},
        ),
    ],
)
def test_a_saved_clustering_has_the_chosen_parameters(
    client, auth_headers, app, fake_redis, clustering, model_type, chosen, expected
):
    _, model = _train(client, auth_headers, app, fake_redis, clustering, model_type, chosen)

    assert _params(model, *expected) == expected


@pytest.mark.parametrize(
    "model_type, chosen, expected",
    [
        (
            "lda",
            {"n_components": 3, "max_iter": 12, "learning_decay": 0.8, "random_state": 7},
            {"n_components": 3, "max_iter": 12, "learning_decay": 0.8, "random_state": 7},
        ),
        (
            "nmf",
            {"n_components": 3, "max_iter": 200, "alpha": 0.001, "l1_ratio": 0.5},
            {"n_components": 3, "max_iter": 200, "alpha_W": 0.001, "alpha_H": 0.001, "l1_ratio": 0.5},
        ),
    ],
)
def test_a_saved_topic_model_has_the_chosen_parameters(
    client, auth_headers, app, fake_redis, topics, model_type, chosen, expected
):
    _, model = _train(client, auth_headers, app, fake_redis, topics, model_type, chosen)

    assert _params(model, *expected) == expected
    assert model.components_.shape[0] == 3


def test_regularisation_that_empties_every_topic_says_so(client, auth_headers, app, fake_redis, topics):
    _, job = _run(client, auth_headers, app, fake_redis, topics, "nmf", {"alpha": 0.1, "l1_ratio": 0.5})

    assert job.status == "failed"
    assert "alpha" in job.error_message


def _tiny_bert(path):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

    vocab = path / "vocab.txt"
    words = sorted({word for text in TOPICS for word in text.split()})
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(path)
    config = BertConfig(
        vocab_size=5 + len(words),
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=8,
    )
    BertForSequenceClassification(config).save_pretrained(path)
    return str(path)


def test_a_transformer_trains_for_the_chosen_epochs(client, auth_headers, app, fake_redis, classifying, tmp_path):
    created = client.post(
        "/api/models/",
        json={"name": "M", "dataset_id": classifying, "model_type": "legalbert", "parameters": {"epochs": 2, "batch_size": 8}},
        headers=auth_headers,
    )
    client.post(f"/api/models/{created.get_json()['model']['id']}/train", json={}, headers=auth_headers)
    task = json.loads(fake_redis.rpop("training_queue"))
    task["parameters"]["model_name"] = _tiny_bert(tmp_path)  # stands in for a download

    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"])

    job = db.session.get(TrainingJob, task["job_id"])
    assert job.status == "completed", job.error_message
    points = TrainingMetricPoint.query.filter_by(training_job_id=task["job_id"]).all()
    assert {point.epoch for point in points if point.metric_name == "eval_accuracy"} == {1.0, 2.0}


def test_chosen_parameters_cannot_replace_what_the_server_decides(
    client, auth_headers, app, fake_redis, classifying
):
    model_id, _ = _train(
        client, auth_headers, app, fake_redis, classifying, "naive_bayes", {"model_id": "someone-elses"}
    )

    assert fake_redis.get(f"confusion_matrix_{model_id}") is not None
    assert fake_redis.get("confusion_matrix_someone-elses") is None


@pytest.mark.parametrize(
    "model_type, chosen, named",
    [
        ("logistic_regression", {"penalty": "l1", "solver": "lbfgs"}, "liblinear or saga"),
        ("logistic_regression", {"penalty": "l1"}, "liblinear or saga"),
        ("logistic_regression", {"penalty": "elasticnet", "solver": "lbfgs"}, "saga"),
        ("logistic_regression", {"penalty": "elasticnet", "solver": "liblinear"}, "saga"),
    ],
)
def test_a_combination_sklearn_refuses_is_refused_at_creation(
    client, auth_headers, classifying, model_type, chosen, named
):
    response = client.post(
        "/api/models/",
        json={"name": "M", "dataset_id": classifying, "model_type": model_type, "parameters": chosen},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert named in response.get_json()["error"]


@pytest.mark.parametrize("affinity", ["l1", "manhattan"])
def test_ward_linkage_needs_euclidean_distance(client, auth_headers, clustering, affinity):
    response = client.post(
        "/api/models/",
        json={"name": "M", "dataset_id": clustering, "model_type": "hierarchical", "parameters": {"linkage": "ward", "affinity": affinity}},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "euclidean" in response.get_json()["error"]


def test_a_legalbert_variant_that_does_not_exist_is_not_offered(client, auth_headers, classifying):
    catalogue = client.get("/api/models/available", headers=auth_headers).get_json()["models"]
    assert catalogue["transformer"]["legalbert"]["parameters"]["model_name"]["options"] == [
        "nlpaueb/legal-bert-base-uncased"
    ]

    response = client.post(
        "/api/models/",
        json={"name": "M", "dataset_id": classifying, "model_type": "legalbert", "parameters": {"model_name": "lex_glue"}},
        headers=auth_headers,
    )
    assert response.status_code == 400


def test_a_model_cannot_be_edited_into_a_refused_combination(client, auth_headers, classifying):
    created = client.post(
        "/api/models/",
        json={"name": "M", "dataset_id": classifying, "model_type": "logistic_regression", "parameters": {}},
        headers=auth_headers,
    )

    response = client.put(
        f"/api/models/{created.get_json()['model']['id']}",
        json={"parameters": {"penalty": "l1", "solver": "lbfgs"}},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "liblinear or saga" in response.get_json()["error"]


def test_a_model_saved_with_a_refused_combination_does_not_queue(client, auth_headers, app, fake_redis, classifying):
    """Models from before the combination was refused, trained on defaults then."""
    with app.app_context():
        model = Model(
            owner_id=TEST_USER_ID,
            dataset_id=classifying,
            name="Old",
            model_type="logistic_regression",
            parameters={"penalty": "elasticnet", "solver": "lbfgs"},
            status="completed",
        )
        db.session.add(model)
        db.session.commit()
        model_id = model.id

    response = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)

    assert response.status_code == 400
    assert "saga" in response.get_json()["error"]
    assert fake_redis.llen("training_queue") == 0
    assert db.session.get(Model, model_id).status == "completed"

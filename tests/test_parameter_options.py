"""Every value the catalogue offers either trains or is refused with a reason
when the Model is created, and nothing that cannot train reaches the worker."""
import json
from itertools import product

import pandas as pd
import pytest
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from legal_ml_core import Dataset, Model, TrainingJob, db
from legal_ml_core.compute.tasks import process_training_task
from legal_ml_core.routes.models import AVAILABLE_MODELS, parameter_refusal
from tests.conftest import TEST_USER_ID

DOCUMENTS = [
    "the court ruled for the plaintiff on the contract claim",
    "the defendant was found liable for negligent damages",
    "the appellant lodged a notice against the sentence",
    "the parties settled the employment dispute by agreement",
    "the tribunal dismissed the asylum application entirely",
    "the jury returned a guilty verdict after the trial",
]
TWO_LABELS = ["civil", "criminal"] * 15
THREE_LABELS = ["civil", "criminal", "family"] * 10


def _dataset(app, tmp_path, task_type, labels, documents=None):
    documents = documents or DOCUMENTS * 5
    path = tmp_path / f"{task_type}-{len(set(labels))}-{len(documents)}.csv"
    pd.DataFrame({"text": documents, "label": labels}).to_csv(path, index=False)
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="D1",
            file_path=str(path),
            row_count=len(documents),
            columns=[],
            task_type=task_type,
        )
        db.session.add(dataset)
        db.session.commit()
        return dataset.id


@pytest.fixture
def datasets(app, tmp_path):
    return {
        "classification": _dataset(app, tmp_path, "classification", TWO_LABELS),
        "transformer": _dataset(app, tmp_path, "classification", TWO_LABELS),
        "clustering": _dataset(app, tmp_path, "clustering", ["x"] * 30),
        "topic_modeling": _dataset(app, tmp_path, "topic_modeling", ["x"] * 30),
    }


@pytest.fixture
def multiclass(app, tmp_path):
    return _dataset(app, tmp_path, "classification", THREE_LABELS)


def _create(client, auth_headers, dataset_id, model_type, parameters):
    return client.post(
        "/api/models/",
        json={"name": "M", "dataset_id": dataset_id, "model_type": model_type, "parameters": parameters},
        headers=auth_headers,
    )


def _train_or_refuse(client, auth_headers, app, fake_redis, dataset_id, model_type, parameters, swap_download=None):
    """Create a Model and run it as the wizard and worker do. Returns the refusal
    at creation, or None once training has completed."""
    created = _create(client, auth_headers, dataset_id, model_type, parameters)
    if created.status_code == 400:
        return created.get_json()["error"]
    assert created.status_code == 201, created.get_json()
    model_id = created.get_json()["model"]["id"]
    started = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)
    assert started.status_code == 200, started.get_json()

    task = json.loads(fake_redis.rpop("training_queue"))
    if swap_download:
        task["parameters"]["model_name"] = swap_download
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"])

    job = db.session.get(TrainingJob, task["job_id"])
    assert job.status == "completed", f"{model_type} {parameters}: {job.error_message}"
    return None


def _models():
    for category, models in AVAILABLE_MODELS.items():
        for model_type, info in models.items():
            yield category, model_type, info


def _select_combinations():
    """Every combination of every model's select options."""
    for category, model_type, info in _models():
        selects = {k: v["options"] for k, v in info["parameters"].items() if v["type"] == "select"}
        for values in product(*selects.values()):
            yield category, model_type, dict(zip(selects, values))


def _single_values():
    """Each boolean both ways, and each number at both ends of its range."""
    for category, model_type, info in _models():
        for name, field in info["parameters"].items():
            if field["type"] == "boolean":
                values = [True, False]
            elif field["type"] in ("int", "float"):
                values = [field["min"], field["max"]]
                if field["default"] is None:
                    values.append(None)
            else:
                continue
            for value in values:
                yield category, model_type, {name: value}


SWEEP = sorted({
    json.dumps(case) for case in [*_select_combinations(), *_single_values()]
})


@pytest.fixture
def tiny_bert(tmp_path):
    """Stands in for a download: the sweep trains with the chosen settings, and
    only the weights are replaced."""
    words = sorted({word for document in DOCUMENTS for word in document.split()})
    path = tmp_path / "bert"
    path.mkdir()
    vocab = path / "vocab.txt"
    vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    BertTokenizerFast(vocab_file=str(vocab)).save_pretrained(path)
    config = BertConfig(
        vocab_size=5 + len(words), hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=8
    )
    BertForSequenceClassification(config).save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("case", SWEEP)
def test_every_offered_value_trains_or_is_refused_at_creation(
    client, auth_headers, app, fake_redis, datasets, tiny_bert, case
):
    category, model_type, chosen = json.loads(case)
    download = tiny_bert if category == "transformer" else None

    refusal = _train_or_refuse(
        client, auth_headers, app, fake_redis, datasets[category], model_type, chosen, swap_download=download
    )

    if refusal:
        rules = AVAILABLE_MODELS[category][model_type]["rules"]
        assert refusal in [rule["reason"] for rule in rules]


@pytest.mark.parametrize(
    "penalty, solver", list(product(["l1", "l2", "elasticnet", "none"], ["liblinear", "lbfgs", "newton-cg", "sag", "saga"]))
)
def test_every_solver_and_penalty_trains_or_is_refused_on_three_labels(
    client, auth_headers, app, fake_redis, multiclass, penalty, solver
):
    refusal = _train_or_refuse(
        client, auth_headers, app, fake_redis, multiclass, "logistic_regression", {"penalty": penalty, "solver": solver}
    )

    rules = AVAILABLE_MODELS["classification"]["logistic_regression"]["rules"]
    assert refusal is None or refusal in [rule["reason"] for rule in rules]


@pytest.mark.parametrize(
    "category, model_type, name, value",
    [
        (category, model_type, name, value)
        for category, model_type, info in _models()
        for name, field in info["parameters"].items()
        if field["type"] == "select"
        for value in field["options"]
    ],
)
def test_no_option_is_offered_that_every_combination_refuses(category, model_type, name, value):
    info = AVAILABLE_MODELS[category][model_type]
    options = {k: v["options"] for k, v in info["parameters"].items() if v["type"] == "select"}
    options[name] = [value]

    combinations = [dict(zip(options, values)) for values in product(*options.values())]

    assert any(parameter_refusal(model_type, chosen) is None for chosen in combinations)


@pytest.mark.parametrize("category, model_type, info", list(_models()))
def test_the_defaults_are_accepted(category, model_type, info):
    defaults = {name: field["default"] for name, field in info["parameters"].items()}

    assert parameter_refusal(model_type, defaults) is None


@pytest.mark.parametrize("ratio", [0.0, 1.0])
def test_the_ends_of_the_elasticnet_mix_train(client, auth_headers, app, fake_redis, datasets, ratio):
    chosen = {"penalty": "elasticnet", "solver": "saga", "l1_ratio": ratio}

    assert _train_or_refuse(
        client, auth_headers, app, fake_redis, datasets["classification"], "logistic_regression", chosen
    ) is None


def test_the_catalogue_says_which_options_depend_on_another(client, auth_headers):
    models = client.get("/api/models/available", headers=auth_headers).get_json()["models"]

    rules = models["classification"]["logistic_regression"]["rules"]
    assert {
        "when": {"penalty": "elasticnet"},
        "parameter": "solver",
        "options": ["saga"],
        "reason": "The elasticnet penalty needs the saga solver.",
    } in rules


def test_a_document_with_no_usable_words_does_not_stop_cosine_clustering(
    client, auth_headers, app, fake_redis, tmp_path
):
    documents = DOCUMENTS * 4 + ["the and of", "of the"] + DOCUMENTS[:4]
    dataset_id = _dataset(app, tmp_path, "clustering", ["x"] * len(documents), documents)

    refusal = _train_or_refuse(
        client, auth_headers, app, fake_redis, dataset_id, "hierarchical", {"linkage": "average", "affinity": "cosine"}
    )

    assert refusal is None


# ---------------------------------------------------------------------- ranges


@pytest.mark.parametrize(
    "model_type, chosen, says",
    [
        ("logistic_regression", {"C": 0}, "between 0.001 and 100"),
        ("logistic_regression", {"C": -1}, "between 0.001 and 100"),
        ("svm", {"C": 0}, "between 0.001 and 100"),
        ("svm", {"degree": 0}, "between 1 and 10"),
        ("random_forest", {"n_estimators": 0}, "Number of trees must be between 1 and 1000"),
        ("random_forest", {"max_depth": 0}, "Maximum tree depth must be between 1 and 50"),
        ("decision_tree", {"max_depth": 51}, "Maximum tree depth must be between 1 and 50"),
        ("decision_tree", {"min_samples_split": 1}, "between 2 and 100"),
        ("naive_bayes", {"alpha": 0}, "between 0.01 and 10"),
        ("logistic_regression", {"l1_ratio": 1.5}, "between 0 and 1"),
    ],
)
def test_a_value_out_of_range_is_refused_when_created(client, auth_headers, datasets, model_type, chosen, says):
    response = _create(client, auth_headers, datasets["classification"], model_type, chosen)

    assert response.status_code == 400
    assert says in response.get_json()["error"]


@pytest.mark.parametrize(
    "model_type, chosen",
    [
        ("kmeans", {"n_clusters": 1}),
        ("kmeans", {"max_iter": 0}),
        ("hierarchical", {"n_clusters": 1}),
        ("dbscan", {"eps": 0}),
        ("dbscan", {"min_samples": 0}),
    ],
)
def test_clustering_ranges_are_checked_as_well(client, auth_headers, datasets, model_type, chosen):
    assert _create(client, auth_headers, datasets["clustering"], model_type, chosen).status_code == 400


@pytest.mark.parametrize(
    "model_type, chosen",
    [
        ("lda", {"n_components": 1}),
        ("lda", {"learning_decay": 0.4}),
        ("nmf", {"n_components": 51}),
        ("nmf", {"alpha": -0.1}),
    ],
)
def test_topic_model_ranges_are_checked_as_well(client, auth_headers, datasets, model_type, chosen):
    assert _create(client, auth_headers, datasets["topic_modeling"], model_type, chosen).status_code == 400


@pytest.mark.parametrize("chosen", [{"epochs": 0}, {"batch_size": 2}, {"learning_rate": 0}, {"max_length": 2048}])
def test_transformer_ranges_are_checked_as_well(client, auth_headers, datasets, chosen):
    assert _create(client, auth_headers, datasets["classification"], "legalbert", chosen).status_code == 400


@pytest.mark.parametrize("value", [True, "2", float("nan"), float("inf")])
def test_something_other_than_a_number_is_refused(client, auth_headers, datasets, value):
    response = client.post(
        "/api/models/",
        data=json.dumps({"name": "M", "dataset_id": datasets["classification"], "model_type": "svm", "parameters": {"C": value}}),
        content_type="application/json",
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "Regularisation parameter" in response.get_json()["error"]


def test_a_fraction_is_refused_where_a_whole_number_is_needed(client, auth_headers, datasets):
    response = _create(client, auth_headers, datasets["classification"], "random_forest", {"n_estimators": 2.5})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Number of trees must be a whole number."


def test_a_value_out_of_range_is_refused_on_edit(client, auth_headers, datasets):
    created = _create(client, auth_headers, datasets["classification"], "svm", {"C": 1})
    model_id = created.get_json()["model"]["id"]

    response = client.put(f"/api/models/{model_id}", json={"parameters": {"C": 0}}, headers=auth_headers)

    assert response.status_code == 400
    assert "between 0.001 and 100" in response.get_json()["error"]


def _saved(dataset_id, model_type, parameters):
    model = Model(owner_id=TEST_USER_ID, dataset_id=dataset_id, name="M", model_type=model_type, parameters=parameters)
    db.session.add(model)
    db.session.commit()
    return model.id


def test_a_saved_value_out_of_range_is_refused_at_start(client, auth_headers, fake_redis, datasets):
    model_id = _saved(datasets["classification"], "svm", {"C": 0})

    response = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)

    assert response.status_code == 400
    assert "between 0.001 and 100" in response.get_json()["error"]
    assert fake_redis.llen("training_queue") == 0


# ------------------------------------------------------------- the Dataset's say


LIBLINEAR_TWO_LABELS = "The liblinear solver handles two labels only. Choose another solver for this Dataset."


def test_liblinear_is_refused_on_a_dataset_with_more_than_two_labels(client, auth_headers, multiclass):
    response = _create(client, auth_headers, multiclass, "logistic_regression", {"solver": "liblinear"})

    assert response.status_code == 400
    assert response.get_json()["error"] == LIBLINEAR_TWO_LABELS


def test_liblinear_is_accepted_on_two_labels(client, auth_headers, datasets):
    response = _create(client, auth_headers, datasets["classification"], "logistic_regression", {"solver": "liblinear"})

    assert response.status_code == 201


def test_liblinear_is_refused_on_edit_to_a_multiclass_dataset(client, auth_headers, multiclass):
    model_id = _create(client, auth_headers, multiclass, "logistic_regression", {}).get_json()["model"]["id"]

    response = client.put(f"/api/models/{model_id}", json={"parameters": {"solver": "liblinear"}}, headers=auth_headers)

    assert response.status_code == 400
    assert response.get_json()["error"] == LIBLINEAR_TWO_LABELS


def test_a_saved_liblinear_model_is_refused_at_start_on_a_multiclass_dataset(client, auth_headers, multiclass):
    model_id = _saved(multiclass, "logistic_regression", {"solver": "liblinear"})

    response = client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)

    assert response.status_code == 400
    assert response.get_json()["error"] == LIBLINEAR_TWO_LABELS


def test_liblinear_on_labels_only_the_annotations_give_is_refused(client, auth_headers, app, tmp_path):
    from legal_ml_core import AnnotationTask

    dataset_id = _dataset(app, tmp_path, "classification", TWO_LABELS)
    db.session.add(
        AnnotationTask(
            dataset_id=dataset_id, item_index=0, text=DOCUMENTS[0], status="annotated",
            annotation_label="family",
        )
    )
    db.session.commit()

    response = _create(client, auth_headers, dataset_id, "logistic_regression", {"solver": "liblinear"})

    assert response.status_code == 400
    assert "liblinear" in response.get_json()["error"]


def test_a_dataset_that_cannot_be_read_gets_a_plain_failure_in_training(
    client, auth_headers, app, fake_redis, multiclass
):
    dataset = db.session.get(Dataset, multiclass)
    readable, dataset.file_path = dataset.file_path, "/nowhere.csv"
    db.session.commit()

    created = _create(client, auth_headers, multiclass, "logistic_regression", {"solver": "liblinear"})
    assert created.status_code == 201
    model_id = created.get_json()["model"]["id"]
    assert client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers).status_code == 200

    task = json.loads(fake_redis.rpop("training_queue"))
    task["dataset_path"] = readable
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"])

    job = db.session.get(TrainingJob, task["job_id"])
    assert job.status == "failed"
    assert job.error_message == LIBLINEAR_TWO_LABELS


@pytest.mark.parametrize("model_type", ["kmeans", "hierarchical"])
def test_more_clusters_than_documents_is_refused(client, auth_headers, app, tmp_path, model_type):
    dataset_id = _dataset(app, tmp_path, "clustering", ["x"] * 6, DOCUMENTS)

    response = _create(client, auth_headers, dataset_id, model_type, {"n_clusters": 7})

    assert response.status_code == 400
    assert response.get_json()["error"] == "Number of clusters cannot be more than the 6 rows in this Dataset."


def test_the_default_cluster_count_is_refused_on_a_smaller_dataset(client, auth_headers, app, tmp_path):
    dataset_id = _dataset(app, tmp_path, "clustering", ["x"] * 6, DOCUMENTS)

    response = _create(client, auth_headers, dataset_id, "kmeans", {})

    assert response.status_code == 400
    assert "6 rows" in response.get_json()["error"]

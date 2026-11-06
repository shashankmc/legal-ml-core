"""Exporting a Model for publishing (BAIGLE #69, ADR 0006).

The worker writes skops (scikit-learn) or safetensors (transformers) files, a
preprocessing.json and a predict.py reference script, then runs that very
script on the check texts and compares it with the platform's own predictions.
"""
import json
import os
import runpy

import pytest

from legal_ml_core import Model, db
from legal_ml_core.compute import export as export_module
from legal_ml_core.compute.tasks import process_prediction_task
from legal_ml_core.model_export import ExportError, queue_model_export, wait_for_model_export
from legal_ml_core.worker import _prediction_loop
from tests.test_training_parameters import TOPICS, _tiny_bert, _train, classifying  # noqa: F401

CHECK_TEXTS = TOPICS[:4]


def _export(app, fake_redis, model_id, texts=CHECK_TEXTS):
    model = db.session.get(Model, model_id)
    export_id = queue_model_export(fake_redis, model, texts)
    _prediction_loop(
        fake_redis,
        app.config["MODEL_FOLDER"],
        app,
        stop=lambda: fake_redis.llen("prediction_queue") == 0,
        sleep=lambda s: None,
    )
    return wait_for_model_export(fake_redis, export_id, timeout=1)


def _platform_labels(app, model_id, model_type, texts):
    result = process_prediction_task(
        {"prediction_id": "p", "model_id": model_id, "model_type": model_type, "texts": texts}, app.config["MODEL_FOLDER"]
    )
    return [p["prediction"] for p in result["predictions"]]


def test_a_scikit_learn_model_exports_to_skops_with_its_preprocessing_and_script(
    client, auth_headers, app, fake_redis, classifying
):
    model_id, _ = _train(client, auth_headers, app, fake_redis, classifying, "logistic_regression", {})

    result = _export(app, fake_redis, model_id)

    assert result["round_trip"]["ok"] is True, result
    assert result["round_trip"]["compared"] == len(CHECK_TEXTS)
    directory = result["directory"]
    assert sorted(result["files"]) == ["model.skops", "predict.py", "preprocessing.json"]
    preprocessing = json.loads(open(os.path.join(directory, "preprocessing.json")).read())
    assert preprocessing["format"] == "skops"
    assert sorted(preprocessing["labels"]) == ["civil", "criminal"]
    assert preprocessing["text_cleaning"]["lowercase"] is True
    assert "scikit-learn" in preprocessing["versions"]
    # What a reuser runs gives what the platform gives.
    script = runpy.run_path(os.path.join(directory, "predict.py"))
    assert script["predict"](directory, TOPICS) == _platform_labels(app, model_id, "logistic_regression", TOPICS)


def test_an_export_that_predicts_differently_is_refused(client, auth_headers, app, fake_redis, classifying, monkeypatch):
    model_id, _ = _train(client, auth_headers, app, fake_redis, classifying, "logistic_regression", {})
    # A script that maps the Model's output to the wrong labels.
    monkeypatch.setattr(
        export_module, "SKLEARN_SCRIPT", export_module.SKLEARN_SCRIPT.replace('config["labels"][int(index)]', 'config["labels"][::-1][int(index)]')
    )

    result = _export(app, fake_redis, model_id)

    assert result["round_trip"]["ok"] is False
    assert "differ" in result["round_trip"]["detail"]


def test_a_transformer_exports_to_safetensors_with_its_tokenizer(client, auth_headers, app, fake_redis, classifying, tmp_path):
    import json as json_module

    from legal_ml_core.compute.tasks import process_training_task

    created = client.post(
        "/api/models/",
        json={"name": "M", "dataset_id": classifying, "model_type": "legalbert", "parameters": {"epochs": 1, "batch_size": 8}},
        headers=auth_headers,
    )
    model_id = created.get_json()["model"]["id"]
    client.post(f"/api/models/{model_id}/train", json={}, headers=auth_headers)
    task = json_module.loads(fake_redis.rpop("training_queue"))
    task["parameters"]["model_name"] = _tiny_bert(tmp_path)
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"])

    result = _export(app, fake_redis, model_id)

    assert result["round_trip"]["ok"] is True, result
    assert {"model.safetensors", "config.json", "tokenizer.json", "preprocessing.json", "predict.py"} <= set(result["files"])
    script = runpy.run_path(os.path.join(result["directory"], "predict.py"))
    assert script["predict"](result["directory"], CHECK_TEXTS) == _platform_labels(app, model_id, "legalbert", CHECK_TEXTS)


def test_waiting_gives_up_when_the_worker_does_not_answer(fake_redis):
    with pytest.raises(ExportError, match="training service"):
        wait_for_model_export(fake_redis, "nobody-answers", timeout=1)


def test_a_worker_error_is_raised(fake_redis):
    fake_redis.lpush("model_export_result_e1", json.dumps({"error": "Model file not found"}))

    with pytest.raises(ExportError, match="Model file not found"):
        wait_for_model_export(fake_redis, "e1", timeout=1)

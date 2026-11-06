"""Export a classification Model in a format safe to publish (ADR 0006).

scikit-learn Models become one skops file, transformers become safetensors plus
their tokenizer. Both come with preprocessing.json and a predict.py reference
script, and the round trip runs that very script on the check texts against the
platform's own predictions, so what is published is what was checked.
"""
import json
import os
import platform
import runpy
import shutil
from importlib.metadata import PackageNotFoundError, version

import joblib

from ..storage import get_storage_backend
from .tasks import process_prediction_task

SKLEARN_TYPES = ("svm", "decision_tree", "random_forest", "naive_bayes", "logistic_regression")
TRANSFORMER_TYPES = ("legalbert", "roberta")
# The defaults of TextPreprocessor.clean_text.
CLEANING_DEFAULTS = {
    "remove_html": True,
    "remove_urls": True,
    "remove_emails": True,
    "remove_numbers": False,
    "remove_special_chars": False,
    "lowercase": True,
}

CLEAN_TEXT = '''
def clean_text(text, options):
    """The platform's text cleaning, step by step, as preprocessing.json sets it."""
    if not isinstance(text, str):
        return ""
    if options["remove_html"]:
        text = re.sub(r"<[^>]+>", "", text)
    if options["remove_urls"]:
        text = re.sub(r"http\\S+|www\\S+|https\\S+", "", text, flags=re.MULTILINE)
    if options["remove_emails"]:
        text = re.sub(r"\\S+@\\S+", "", text)
    if options["remove_numbers"]:
        text = re.sub(r"\\d+", "", text)
    if options["remove_special_chars"]:
        text = re.sub(r"[^a-zA-Z\\s]", "", text)
    if options["lowercase"]:
        text = text.lower()
    return text.strip()
'''

SKLEARN_SCRIPT = '''"""Predict labels with this Model.

Install the versions listed in preprocessing.json, then:

    python predict.py "The tenant shall pay the rent monthly."
"""
import json
import os
import re
import sys

import skops.io as sio

# The only types model.skops may contain. Review them before loading the file.
TRUSTED = __TRUSTED__
''' + CLEAN_TEXT + '''

def predict(directory, texts):
    with open(os.path.join(directory, "preprocessing.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    bundle = sio.load(os.path.join(directory, "model.skops"), trusted=TRUSTED)
    features = bundle["vectorizer"].transform([clean_text(text, config["text_cleaning"]) for text in texts])
    return [config["labels"][int(index)] for index in bundle["model"].predict(features)]


if __name__ == "__main__":
    for label in predict(os.path.dirname(os.path.abspath(__file__)), sys.argv[1:]):
        print(label)
'''

TRANSFORMER_SCRIPT = '''"""Predict labels with this Model.

Install the versions listed in preprocessing.json, then:

    python predict.py "The tenant shall pay the rent monthly."
"""
import json
import os
import re
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
''' + CLEAN_TEXT + '''

def predict(directory, texts):
    with open(os.path.join(directory, "preprocessing.json"), encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(directory)
    model = AutoModelForSequenceClassification.from_pretrained(directory)
    model.eval()
    labels = []
    for text in texts:
        inputs = tokenizer(
            clean_text(text, config["text_cleaning"]),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config["max_length"],
        )
        with torch.no_grad():
            labels.append(config["labels"][int(model(**inputs).logits.argmax(dim=-1).item())])
    return labels


if __name__ == "__main__":
    for label in predict(os.path.dirname(os.path.abspath(__file__)), sys.argv[1:]):
        print(label)
'''


def _versions(*packages):
    found = {"python": platform.python_version()}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            pass
    return found


def _cleaning(bundle):
    preprocessor = bundle.get("preprocessor")
    options = getattr(preprocessor, "options", None) or {}
    return {name: bool(options.get(name, default)) for name, default in CLEANING_DEFAULTS.items()}


def _labels(bundle):
    encoder = bundle.get("label_encoder")
    return [str(label) for label in encoder.classes_] if encoder is not None else []


def _write(directory, name, text):
    with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
        handle.write(text)


def _export_sklearn(bundle, directory):
    import skops.io as sio

    path = os.path.join(directory, "model.skops")
    sio.dump({"vectorizer": bundle["vectorizer"], "model": bundle["model"]}, path)
    trusted = sorted(sio.get_untrusted_types(file=path))
    _write(directory, "predict.py", SKLEARN_SCRIPT.replace("__TRUSTED__", json.dumps(trusted, indent=4)))
    return {
        "format": "skops",
        "model_file": "model.skops",
        "trusted_types": trusted,
        "versions": _versions("scikit-learn", "skops", "numpy", "scipy", "legal-ml-core"),
    }


def _export_transformer(bundle, directory):
    tokenizer = bundle.get("tokenizer")
    if tokenizer is None:
        raise ValueError("This Model was saved without its tokenizer, so it cannot be exported.")
    bundle["model"].save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    _write(directory, "predict.py", TRANSFORMER_SCRIPT)
    return {
        "format": "safetensors",
        "max_length": 256,
        "versions": _versions("transformers", "torch", "tokenizers", "safetensors", "legal-ml-core"),
    }


def process_export_task(task_info, model_storage_path):
    """Export the Model and check it; returns the answer for the API."""
    model_id, model_type, texts = task_info["model_id"], task_info["model_type"], task_info["texts"]
    if model_type not in SKLEARN_TYPES + TRANSFORMER_TYPES:
        raise ValueError("Only classification Models can be published.")
    model_path = get_storage_backend().get_model_path(model_storage_path, model_id)
    if not os.path.exists(model_path):
        raise FileNotFoundError("The Model file is missing. Train the Model again.")
    bundle = joblib.load(model_path)

    directory = os.path.join(model_storage_path, f"export_{model_id}")
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)
    described = (_export_sklearn if model_type in SKLEARN_TYPES else _export_transformer)(bundle, directory)
    preprocessing = {"text_cleaning": _cleaning(bundle), "labels": _labels(bundle), **described}
    _write(directory, "preprocessing.json", json.dumps(preprocessing, indent=2))

    exported = runpy.run_path(os.path.join(directory, "predict.py"))["predict"](directory, texts)
    platform_labels = [
        entry["prediction"]
        for entry in process_prediction_task(
            {"prediction_id": "export", "model_id": model_id, "model_type": model_type, "texts": texts},
            model_storage_path,
        )["predictions"]
    ]
    differing = sum(ours != theirs for ours, theirs in zip(exported, platform_labels))
    ok = bool(texts) and differing == 0 and len(exported) == len(platform_labels)
    detail = (
        f"Predictions match on {len(texts)} items."
        if ok
        else f"{differing} of {len(texts)} predictions differ between the exported Model and the platform's."
    )
    return {
        "directory": directory,
        "files": sorted(os.listdir(directory)),
        "round_trip": {"ok": ok, "detail": detail, "compared": len(texts)},
    }

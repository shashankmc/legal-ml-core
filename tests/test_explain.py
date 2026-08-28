import pandas as pd

from legal_ml_core.compute.explain import generate_lime_explanation
from legal_ml_core.compute.trainer import MLModelTrainer


def _train_multiclass_model(fake_redis):
    """Five roughly-balanced classes so the predicted class for a given text
    is unlikely to land on label index 1 by chance — this is exactly the
    scenario that used to raise KeyError before explain_instance() was told
    which label to compute weights for.
    """
    texts = [
        "the licensee shall submit the application",
        "the party is permitted to terminate the agreement",
        "the authority has the power to issue licences",
        "the export of these goods is prohibited",
        "the applicant has the right to appeal",
    ] * 6
    labels = ["Obligation", "Permission", "Power", "Prohibition", "Right"] * 6
    df = pd.DataFrame({"text": texts, "label": labels})

    trainer = MLModelTrainer(redis_client=fake_redis)
    text_series, encoded_labels, _ = trainer.preprocess_data(df, "classification", {})
    trainer.train_classification_model(text_series, encoded_labels, "naive_bayes", {"model_id": "explain-test"})
    return trainer


def test_lime_explanation_works_regardless_of_predicted_label_index(fake_redis):
    trainer = _train_multiclass_model(fake_redis)
    model_data = {
        "preprocessor": trainer.preprocessor,
        "vectorizer": trainer.vectorizer,
        "model": trainer.model,
        "label_encoder": trainer.label_encoder,
    }

    # Try every class's example text — regardless of which label index the
    # model predicts, LIME must be asked to explain that specific label.
    for text in [
        "the licensee shall submit the application",
        "the authority has the power to issue licences",
        "the export of these goods is prohibited",
    ]:
        explanation = generate_lime_explanation(model_data, text, num_features=5)
        assert "error" not in explanation, explanation
        assert explanation["type"] == "lime_text"
        assert explanation["predicted_class"] in ["Obligation", "Permission", "Power", "Prohibition", "Right"]
        assert isinstance(explanation["features"], list)

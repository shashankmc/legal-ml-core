import os


def get_model_path(model_storage_root, model_id):
    """Path convention for a trained model's pickle file.

    Both the API process (download/explain) and the worker process (save
    after training) must agree on this path — previously each side
    hardcoded the string "/app/models/{model_id}.pkl" independently.
    """
    return os.path.join(model_storage_root, f"{model_id}.pkl")


def get_transformer_output_dir(model_storage_root, model_id):
    """Output dir HuggingFace `Trainer` writes checkpoints/logs to during
    LegalBERT/RoBERTa fine-tuning."""
    return os.path.join(model_storage_root, f"transformer_{model_id}")


def get_dataset_path(data_storage_root, filename):
    """Path convention for an uploaded dataset file."""
    return os.path.join(data_storage_root, filename)

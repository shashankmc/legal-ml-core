"""Storage path conventions (and a pluggable storage backend).

Both the API process (download/explain) and the worker process (save after
training) must agree on where artifacts live. The path *conventions* are
centralised here so they can never drift between the two processes.

The *backend* (local filesystem by default) is pluggable so a host with object
storage can swap it without touching route/compute code: set
``LEGAL_ML_STORAGE_BACKEND`` in the app config to an instance (or a zero-arg
callable returning one) of a type implementing :class:`StorageBackend`. The
default :class:`LocalStorageBackend` preserves the historical on-disk layout.
"""

import os
from typing import Protocol

from flask import current_app, has_app_context


class StorageBackend(Protocol):
    """Abstraction over where model/dataset artifacts are stored."""

    def get_model_path(self, storage_root, model_id):
        ...

    def get_transformer_output_dir(self, storage_root, model_id):
        ...

    def get_dataset_path(self, storage_root, filename):
        ...


class LocalStorageBackend:
    """Default on-disk storage layout (unchanged from the original code)."""

    def get_model_path(self, storage_root, model_id):
        return os.path.join(storage_root, f"{model_id}.pkl")

    def get_transformer_output_dir(self, storage_root, model_id):
        return os.path.join(storage_root, f"transformer_{model_id}")

    def get_dataset_path(self, storage_root, filename):
        return os.path.join(storage_root, filename)


_default_storage = LocalStorageBackend()


def get_storage_backend():
    """Return the configured storage backend, falling back to the local one."""
    if has_app_context():
        configured = current_app.config.get("LEGAL_ML_STORAGE_BACKEND")
        if configured is not None:
            if callable(configured) and not hasattr(configured, "get_model_path"):
                configured = configured()
            if configured is not None:
                return configured
    return _default_storage


# Module-level helpers kept for backwards compatibility; they use the default
# backend so existing callers (and tests) keep working unchanged.
def get_model_path(model_storage_root, model_id):
    return _default_storage.get_model_path(model_storage_root, model_id)


def get_transformer_output_dir(model_storage_root, model_id):
    return _default_storage.get_transformer_output_dir(model_storage_root, model_id)


def get_dataset_path(data_storage_root, filename):
    return _default_storage.get_dataset_path(data_storage_root, filename)

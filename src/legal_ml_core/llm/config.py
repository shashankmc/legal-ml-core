"""Settings for the LLM feature.

Read from the Flask config first, then the environment, because the worker
process builds a minimal app and only has the environment.

    LLM_SHARED_ENDPOINT    dict with base_url, api_key, model, destination and
                           optional extra_body. Environment equivalents:
                           LLM_SHARED_BASE_URL, LLM_SHARED_API_KEY,
                           LLM_SHARED_MODEL, LLM_SHARED_DESTINATION and
                           LLM_SHARED_EXTRA_BODY (JSON).
                           api_key is required: for a server without auth, use
                           any non-empty value.
    LLM_ENCRYPTION_KEY     Fernet key protecting stored BYOK keys.
    LLM_AUDIT_HOOK         Optional callable (event, owner_id, **fields), Flask
                           config only.
    LLM_BYOK_ALLOWED_HOSTS Comma-separated hosts BYOK may call. When unset, any
                           public https host is allowed.
"""

import json
import os

from flask import current_app, has_app_context

DEFAULTS = {
    "LLM_MONTHLY_TOKEN_CAP": 200_000,
    "LLM_AGREEMENT_MIN_SAMPLE": 30,
    "LLM_AGREEMENT_THRESHOLD": 0.8,
    "LLM_MAX_SYNC_TASKS": 10,
    "LLM_MAX_OUTPUT_TOKENS": 300,
    "LLM_MAX_TEXT_CHARS": 12_000,
    "LLM_TIMEOUT_SECONDS": 30,
}


def setting(name, default=None):
    if has_app_context() and name in current_app.config:
        return current_app.config[name]
    value = os.environ.get(name)
    if value is not None:
        return value
    return DEFAULTS.get(name, default)


def int_setting(name):
    return int(setting(name))


def float_setting(name):
    return float(setting(name))


def shared_endpoint():
    """The shared endpoint, or ``None`` when this deployment has none."""
    endpoint = setting("LLM_SHARED_ENDPOINT")
    if endpoint is None:
        endpoint = {
            "base_url": os.environ.get("LLM_SHARED_BASE_URL"),
            "api_key": os.environ.get("LLM_SHARED_API_KEY"),
            "model": os.environ.get("LLM_SHARED_MODEL"),
            "destination": os.environ.get("LLM_SHARED_DESTINATION"),
            "extra_body": json.loads(os.environ.get("LLM_SHARED_EXTRA_BODY") or "{}"),
        }
    # A deployment can ship defaults for the rest; only the key turns it on.
    if not all(endpoint.get(k) for k in ("base_url", "api_key", "model", "destination")):
        return None
    return endpoint

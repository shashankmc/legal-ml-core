"""Helpers that used to bridge the package to host-owned `users`/`projects`.

Ownership is now an opaque ``owner_id`` (see ownership.py), so the package no
longer needs foreign keys into — or stub tables for — host tables. What remains
is project reference validation, which is a *host policy* the package asks about
rather than queries directly.
"""

from flask import current_app


def ensure_external_fk_stub_tables(db):
    """Deprecated no-op.

    Kept so existing hosts/workers importing it don't break. It previously
    fabricated `users`/`projects` Table objects so the package's foreign keys
    could resolve in the worker process; those foreign keys have been removed in
    favour of an opaque owner id, so there is nothing to stub.
    """
    return None


def project_belongs_to_owner(session, project_id, owner_id):
    """True if `project_id` is a valid project reference for this owner.

    Projects are a host concern. A host that wants project validation installs a
    ``PROJECT_VALIDATOR`` callable on the app config with the signature
    ``(session, project_id, owner_id) -> bool``. Without one, the package cannot
    know about a host's project table, so it accepts any non-empty project_id
    (the reference is opaque and only meaningful to the host).
    """
    if not project_id:
        return True
    validator = current_app.config.get("PROJECT_VALIDATOR")
    if validator is None:
        return True
    return bool(validator(session, project_id, owner_id))

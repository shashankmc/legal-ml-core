"""Helpers that used to bridge the package to host-owned `users`/`projects`.

Ownership is now an opaque ``owner_id`` (see ownership.py), so the package no
longer needs foreign keys into — or stub tables for — host tables. What remains
is project reference validation, which is a *host policy* the package asks about
rather than queries directly.
"""

from flask import current_app


def project_belongs_to_owner(session, project_id, owner_id):
    """True if `project_id` is a valid project reference for this owner.

    Three scopes, selected by the ``PROJECT_SCOPE`` app config:

    - ``"host_table"`` (or any host that sets a ``PROJECT_VALIDATOR`` callable
      with signature ``(session, project_id, owner_id) -> bool``): the host owns
      projects and decides. This is BAIGLE's path.
    - ``"package_table"``: validate against the package's own ``ml_projects``
      table, so a self-contained ML-only host gets grouping without bringing a
      project model.
    - ``"none"`` (default): the package knows nothing about projects; any
      non-empty reference is accepted (it is opaque and only meaningful to the
      host).
    """
    if not project_id:
        return True
    validator = current_app.config.get("PROJECT_VALIDATOR")
    if validator is not None:
        return bool(validator(session, project_id, owner_id))

    scope = current_app.config.get("PROJECT_SCOPE", "none")
    if scope == "package_table":
        from .models import MLProject

        return (
            session.query(MLProject)
            .filter_by(id=project_id, owner_id=owner_id)
            .first()
            is not None
        )

    return True

"""Owner scoping, decoupled from any particular identity store.

The ML feature's rows (datasets, models, training jobs, comparisons) belong to
*an owner*. Historically that owner was a local `users` row reached through a
hard foreign key, which forced every host to use this repo's `users` table. This
module replaces that with an opaque ``owner_id`` (a UUID string) plus a pluggable
:class:`OwnerRef` strategy, so a host can scope rows by:

- a Keycloak subject claim (no local user table at all — the reuse case),
- its own user/model resolver (BAIGLE today), or
- nothing (single-tenant / ML-only deployments).
"""

from __future__ import annotations

import uuid
from typing import Any, Optional, Protocol

from flask import current_app

# Reserved owner id used by NoOwnerRef. A valid UUID so it fits the owner_id
# column type; stable so repeated calls agree.
NULL_OWNER_ID = "00000000-0000-0000-0000-000000000000"


class OwnerRef(Protocol):
    """Resolve the owner id for the current request/context."""

    def resolve_owner_id(self, payload: Optional[dict]) -> str:
        """Return the owner id (a UUID string) for this request.

        ``payload`` is the verified token claims, or ``None`` when auth is
        disabled. Implementations must return a non-empty string.
        """
        ...


class KeycloakOwnerRef:
    """Owner id is the Keycloak subject claim — no local user table needed.

    This is the recommended strategy for hosts that only need the ML feature and
    already issue Keycloak tokens. Optionally a ``claim`` other than ``sub`` can
    be used (e.g. a stable username).
    """

    def __init__(self, claim: str = "sub"):
        self.claim = claim

    def resolve_owner_id(self, payload: Optional[dict]) -> str:
        if not payload:
            raise ValueError("KeycloakOwnerRef requires verified token claims.")
        value = payload.get(self.claim)
        if not value:
            raise ValueError(f"Token has no '{self.claim}' claim to use as owner id.")
        return str(value)


class HostModelOwnerRef:
    """Owner id comes from a host-supplied callable.

    The callable receives the verified claims and returns an owner id (or a
    dict/object with an ``id``). This preserves BAIGLE's current behaviour, where
    a ``users`` row is JIT-provisioned and its id is the owner.
    """

    def __init__(self, resolver):
        self.resolver = resolver

    def resolve_owner_id(self, payload: Optional[dict]) -> str:
        resolved = self.resolver(payload)
        if resolved is None:
            raise ValueError("Owner resolver returned no identity.")
        if isinstance(resolved, dict):
            owner_id = resolved.get("id")
        else:
            owner_id = getattr(resolved, "id", None)
        if not owner_id:
            raise ValueError("Owner resolver returned an identity without an id.")
        return str(owner_id)


class NoOwnerRef:
    """Single-tenant: every row shares one constant owner id.

    For ML-only deployments with no user management. Using a constant (rather
    than None) keeps owner_id NOT NULL and avoids a schema special case.
    """

    def __init__(self, owner_id: str = NULL_OWNER_ID):
        self.owner_id = owner_id

    def resolve_owner_id(self, payload: Optional[dict]) -> str:
        return self.owner_id


def owner_ref() -> OwnerRef:
    """Return the configured OwnerRef, defaulting to Keycloak-subject scoping.

    Reads ``OWNER_REF`` from the app config; ``None`` falls back to
    ``KeycloakOwnerRef`` because that is the safe, table-free default for a
    mounted package. Hosts that need JIT provisioning set ``OWNER_REF`` to a
    ``HostModelOwnerRef`` (or ``IDENTITY_RESOLVER``, handled by auth.py).
    """
    configured = current_app.config.get("OWNER_REF")
    if configured is not None:
        return configured
    return KeycloakOwnerRef()


def new_owner_uuid() -> str:
    """A fresh owner UUID (used by tests and by NoOwnerRef-style seeding)."""
    return str(uuid.uuid4())

"""The package's declarative base and database handle.

Ownership model (increment 4 of the reusability plan):

- ``MLBase`` is a plain SQLAlchemy 2.0 ``DeclarativeBase`` owned by this
  package. Every package model subclasses it, so all the package's tables share
  one registry and one ``MetaData``.
- A host that has its own models subclasses ``MLBase`` too, and builds its
  Flask-SQLAlchemy handle as ``SQLAlchemy(model_class=MLBase)``. Host and
  package models then live on the *same* registry, so string relationships
  across the boundary resolve, and there is a single ``metadata`` for Alembic.
- ``db`` (the module global) is a proxy, not a concrete handle. Package code
  uses ``db.session`` exactly as before; the proxy forwards every attribute to
  the *active* Flask-SQLAlchemy handle — the host's when one was provided to
  ``init_ml_feature``, or this package's default handle otherwise. This keeps
  route/compute code unchanged while letting the host own the engine/session.

A ``DeclarativeBase`` owns exactly one registry/metadata, so a host cannot mix
``MLBase`` with an unrelated base — adopting ``MLBase`` is the contract. Two
separate ``SQLAlchemy()`` objects never share a registry; the base, not the
handle, is the shared thing.
"""

from flask import current_app, has_app_context
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class _QueryProperty:
    """Legacy ``Model.query`` accessor for models on a plain ``MLBase``.

    Flask-SQLAlchemy injects this only on handles it builds (``db.Model``), so
    models subclassing ``MLBase`` directly would otherwise lose ``Model.query``.
    Resolve the session lazily from the active handle at access time.
    """

    def __get__(self, obj, cls):
        if obj is not None:
            raise AttributeError("query is a class-level accessor, not instance")
        return get_db().session.query(cls)


class MLBase(DeclarativeBase):
    """Shared declarative base for the ML feature (and adopting hosts)."""

    query = _QueryProperty()


# Default handle for hosts that don't bring their own (ML-only deployments, the
# worker). Hosts with existing models build their own SQLAlchemy(model_class=
# MLBase) and pass it to init_ml_feature.
_default_db = SQLAlchemy(model_class=MLBase)

# The handle package code should use right now: the value set by
# init_ml_feature (host-provided or default). Set at app-init time.
_active_db = None


def set_active_db(handle):
    """Bind the Flask-SQLAlchemy handle package code should use."""
    global _active_db
    _active_db = handle


def get_db():
    """Return the active Flask-SQLAlchemy handle.

    Falls back to ``current_app``'s handle when one has been init_app'd but
    ``set_active_db`` was not called (e.g. a host that passes nothing to
    ``init_ml_feature`` but still does ``legal_ml_core.db.init_app(app)``).
    """
    if _active_db is not None:
        return _active_db
    if has_app_context():
        ext = current_app.extensions.get("sqlalchemy")
        if ext is not None:
            return ext
    return _default_db


class _DBProxy:
    """Forwards attribute access to the active handle.

    Models subclass ``MLBase`` directly, so they never touch this proxy. Only
    the imperative ``db.session``/``db.func``/``db.select`` style calls do, and
    those must resolve to whichever handle the app actually uses.
    """

    def __getattr__(self, name):
        return getattr(get_db(), name)


# Public handle used by route/compute code (`from ..db import db`).
db = _DBProxy()

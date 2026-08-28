from flask_sqlalchemy import SQLAlchemy

# This package owns the SQLAlchemy() instance rather than accepting one from the
# host app. Flask-SQLAlchemy's `model_class=` parameter does not share a registry
# with an already-defined SQLAlchemy 2.0 DeclarativeBase (verified: it wraps the
# base via `declarative_base(cls=...)`, producing either a second registry or an
# `ArgumentError: already has a primary mapper defined`). Owning `db` here and
# having the host app re-export it (`from legal_ml_core import db`) guarantees
# every model — package or host — is mapped through the exact same registry, so
# string-based relationships (`User.datasets`, `Project.datasets`, etc. defined
# on host models) resolve correctly regardless of which side defines which class.
db = SQLAlchemy()

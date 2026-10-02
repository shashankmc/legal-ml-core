def init_ml_feature(app, redis_client, model_storage_path, db=None):
    """Registers the ML feature's six blueprints onto a host Flask app.

    ``db`` is the host's Flask-SQLAlchemy handle. It must have been built on
    ``legal_ml_core.db.MLBase`` (``SQLAlchemy(model_class=MLBase)``) so host and
    package models share one registry/metadata. Pass it here and the package
    uses the host's engine and ``db.session``; omit it to use the package's own
    default handle (ML-only hosts and the worker).

    The package models subclass ``MLBase`` directly; route/compute code uses the
    package's ``db`` proxy, which forwards to this handle.
    """
    from .db import MLBase, _DBProxy, get_db, set_active_db

    # Resolve the package's own `db` proxy (passed by default-handle callers) to
    # the concrete handle, so it is not stored as active (which would recurse).
    handle = get_db() if db is None else db
    if isinstance(handle, _DBProxy):
        handle = get_db()

    model_class = getattr(handle, "Model", None)
    # Flask-SQLAlchemy derives its own base class from model_class, so
    # handle.Model is not MLBase itself even when built correctly. What must
    # match is the metadata the package models are attached to.
    metadata = getattr(model_class, "metadata", None)
    if metadata is None or metadata is not MLBase.metadata:
        raise RuntimeError(
            "legal_ml_core.init_ml_feature needs a Flask-SQLAlchemy `db` "
            "built on legal_ml_core.db.MLBase "
            "(SQLAlchemy(model_class=MLBase)), so host and package models "
            "share one registry. Build your db with that base and pass it here."
        )
    set_active_db(handle)

    app.redis_client = redis_client
    app.config.setdefault("LEGAL_ML_MODEL_STORAGE_PATH", model_storage_path)

    from .routes.annotation import annotation_bp
    from .routes.datasets import datasets_bp
    from .routes.eda import eda_bp
    from .routes.models import models_bp
    from .routes.training import training_bp
    from .routes.visualization import visualization_bp

    app.register_blueprint(datasets_bp, url_prefix="/api/datasets")
    app.register_blueprint(models_bp, url_prefix="/api/models")
    app.register_blueprint(training_bp, url_prefix="/api/training")
    app.register_blueprint(visualization_bp, url_prefix="/api/visualization")
    app.register_blueprint(eda_bp, url_prefix="/api")
    app.register_blueprint(annotation_bp, url_prefix="/api/annotation")

def init_ml_feature(app, redis_client, model_storage_path, db=None):
    """Registers the ML feature's six blueprints onto a host Flask app.

    The host app's own `db` (Flask-SQLAlchemy instance) must be the exact
    object this package exports as `legal_ml_core.db` — see db.py for why a
    second, separately-constructed SQLAlchemy() instance won't share a
    registry with this package's models. Pass the host's `db` here only as an
    optional sanity check that this wiring is correct; it is not otherwise
    used (routes import `db`/model classes directly from this package).
    """
    from .db import db as package_db

    if db is not None and db is not package_db:
        raise RuntimeError(
            "legal_ml_core.init_ml_feature received a `db` that is not "
            "legal_ml_core.db. Import db from this package in your app's "
            "database module (`from legal_ml_core import db`) instead of "
            "constructing a separate SQLAlchemy() instance — otherwise "
            "cross-package relationships (User.datasets, Project.datasets, "
            "etc.) will not resolve."
        )

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

# legal-ml-core

Core ML feature for BAIGLE-style legal-ML apps — dataset management, model
training/prediction/explanation, visualization, EDA, annotation, and
experiment monitoring — as an installable Flask package, extracted from the
`reducing_digital_divide` monorepo.

This package is the single source of truth for the ML feature's SQLAlchemy
models and compute engine, shared between the Flask API process and a
separate worker process (previously two independent, drifting
representations of the same database schema).

## What's in scope

- Dataset upload/validation/preview/EDA
- Model catalog, configuration, training, prediction, LIME/SHAP explanation
- Visualization (performance metrics, confusion matrix, clustering, topic modeling, model comparison)
- Annotation task queue
- Experiment monitoring: per-training-job metric time series (dense for the
  HuggingFace Trainer path, coarse phase markers for classical sklearn
  algorithms) and cross-run hyperparameter-vs-performance comparison

Out of scope (stays in the host app): auth, generic project CRUD, and the
unrelated "learning platform" tutorials feature (including its sandboxed
code-execution/package-installation queues).

## Installing into a host Flask app

```python
from flask import Flask
from legal_ml_core import db, init_ml_feature
import redis

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://..."
db.init_app(app)  # `db` is this package's SQLAlchemy() instance — see below

redis_client = redis.Redis.from_url("redis://...")
init_ml_feature(app, redis_client, model_storage_path="/app/models", db=db)
```

**Important**: your host app's own models (User, Project, etc.) must
subclass `db.Model` from *this* package (`from legal_ml_core import db`), not
a separately-constructed `SQLAlchemy()` instance. Flask-SQLAlchemy's
`model_class=` parameter does **not** let two independently-constructed
declarative bases share one mapper registry — verified directly against
Flask-SQLAlchemy 3.0.5, where passing a custom SQLAlchemy-2.0-style
`DeclarativeBase` either produces a second, separate registry or raises
`ArgumentError: already has a primary mapper defined`. Sharing this
package's own `db` instance is the only combination that was confirmed to
work: it's what makes host-side string relationships like `User.datasets` or
`Project.datasets` resolve correctly.

## Running the worker

```python
from legal_ml_core import run_ml_worker

run_ml_worker(
    redis_url="redis://...",
    database_url="postgresql://...",
    model_storage_path="/app/models",
)
```

This starts the training/prediction/explanation queue-consumer threads. It
does **not** include the learning-platform's code-execution/package-install
queues — a host app composing its own worker process should run those
alongside this, in the same process, using its own code.

## Experiment monitoring limitation

Per-step training curves (`TrainingMetricPoint`, polled via
`GET /training/<job_id>/metrics`) are only dense/real for the HuggingFace
`Trainer` path (LegalBERT/RoBERTa fine-tuning), which has a genuine
epoch/step loop. Classical sklearn algorithms (SVM, decision tree, random
forest, naive Bayes, logistic regression, KMeans/agglomerative/DBSCAN,
LDA/NMF) fit synchronously with no epoch loop, so they only get coarse phase
markers (`preprocessing_started`, `preprocessing_done`, `training_started`,
`training_done`, `model_saved`). Cross-run hyperparameter-vs-performance
comparison (`GET /visualization/parameter-comparison`) works uniformly
across all model types since it's retrospective, reading already-stored
`Model.parameters`/`TrainingJob.metrics`.

## Development

```bash
pip install -e ".[test]"
pytest
```

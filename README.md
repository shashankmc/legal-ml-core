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
- Pluggable ownership: opaque `owner_id` scoped by a Keycloak subject, a host
  resolver, or nothing (single-tenant) — no local user table required

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

**Ownership is pluggable and needs no local user table.** ML rows carry an
opaque `owner_id`, resolved by an `OwnerRef` strategy. The default,
`KeycloakOwnerRef`, uses the token `sub` — so a host that authenticates with
Keycloak needs no `users`/`projects` tables at all:

```python
app.config["OWNER_REF"] = KeycloakOwnerRef()          # default; sub is the owner
# or: HostModelOwnerRef(lambda payload: my_resolver(payload))  # host user model
# or: NoOwnerRef()                                            # single-tenant, no auth
```

Projects are optional: without a host `PROJECT_VALIDATOR` callable, a non-empty
`project_id` is treated as an opaque reference the host owns.

> The earlier hard foreign keys into `users`/`projects` (and the stub tables the
> worker used to fabricate) were removed in the reusability refactor; see
> `docs/legal-ml-core-reusability-plan.md` in the BAIGLE repo.

**Important**: if your host has its own models, they must subclass `db.Model`
from *this* package (`from legal_ml_core import db`), not a
separately-constructed `SQLAlchemy()` instance. Flask-SQLAlchemy's
`model_class=` parameter does **not** let two independently-constructed
declarative bases share one mapper registry — verified directly against
Flask-SQLAlchemy 3.0.5. Sharing this package's own `db` instance is the only
combination confirmed to work. (Increment 4 of the reusability plan inverts
this so the host provides `db`; until then the package owns it.)

## Packaged into host images

Hosts commonly vendor this package into more than one image (e.g. a Flask API
image and a separate worker image). Because the package is copied at build time,
a change here only takes effect after **every** image that vendors it is
rebuilt — rebuilding just the API image leaves a stale worker running the old
models. Rebuild all of them (in BAIGLE, `docker compose build` with no service
argument).

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

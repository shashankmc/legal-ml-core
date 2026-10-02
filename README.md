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

A host that has its own models owns the `SQLAlchemy` handle, built on this
package's shared `MLBase`:

```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from legal_ml_core import MLBase, init_ml_feature
import redis

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://..."

db = SQLAlchemy(model_class=MLBase)   # host owns the handle, on the shared base
db.init_app(app)

# Host models subclass MLBase too, so they share one registry with the package:
#   from legal_ml_core import MLBase
#   class Project(MLBase): ...

redis_client = redis.Redis.from_url("redis://...")
init_ml_feature(app, redis_client, model_storage_path="/app/models", db=db)
```

An **ML-only host** with no models of its own can skip the handle and let the
package use its default one (pass `db=None`, or just call
`from legal_ml_core import db; db.init_app(app)` — `db` is a proxy onto the
package's default handle):

```python
from legal_ml_core import db, init_ml_feature
db.init_app(app)
init_ml_feature(app, redis_client, model_storage_path="/app/models")  # db omitted
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

Projects are optional, selected by `PROJECT_SCOPE`:

- `"none"` (default) — a non-empty `project_id` is an opaque reference the host
  owns; the package never validates it.
- `"host_table"` — the host installs a `PROJECT_VALIDATOR(session, project_id,
  owner_id) -> bool` callable (BAIGLE's path; it has its own `projects` table).
- `"package_table"` — the package validates against its own optional
  `ml_projects` table (`MLProject`), for a self-contained host that wants
  grouping without bringing a model.

> The earlier hard foreign keys into `users`/`projects` (and the stub tables the
> worker used to fabricate) were removed in the reusability refactor; see
> `docs/legal-ml-core-reusability-plan.md` in the BAIGLE repo.

**Registry sharing contract.** All models — the package's and the host's — must
be attached to **one** declarative base: `legal_ml_core.db.MLBase`. Build your
handle as `SQLAlchemy(model_class=MLBase)` and subclass `MLBase` for your own
models; do **not** construct a second independent `SQLAlchemy()` on a different
base — two such handles never share a registry. `init_ml_feature` enforces this
by checking that the handle's metadata is `MLBase`'s. (Flask-SQLAlchemy derives
its own base class from `model_class`, so `db.Model is not MLBase`; the thing
that must match is the metadata, which is what the check compares.)

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

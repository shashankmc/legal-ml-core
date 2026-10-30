# legal-ml-core

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/shashankmc/legal-ml-core/actions/workflows/ci.yml/badge.svg)](https://github.com/shashankmc/legal-ml-core/actions/workflows/ci.yml)

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
- LLM annotation: suggestions for a batch or a whole dataset, through one
  OpenAI-compatible adapter, with a host-gated shared key or a user's own key
- Pluggable ownership: opaque `owner_id` scoped by a Keycloak subject, a host
  resolver, or nothing (single-tenant) — no local user table required

Out of scope (stays in the host app): auth, generic project CRUD, and the
unrelated "learning platform" tutorials feature (including its sandboxed
code-execution/package-installation queues).

## LLM annotation

Everything under `/api/llm` is described in `docs/openapi.yaml`. Suggestions never
become labels by themselves: a human confirms them, or bulk-accepts after an
agreement check. Single-label classification only.

The package holds no policy. A host configures:

```python
app.config["FEATURE_GATE"] = lambda feature, claims: my_policy(feature, claims)
app.config["LLM_SHARED_ENDPOINT"] = {
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": os.environ["OPENROUTER_API_KEY"],
    "model": "...",
    "destination": "OpenRouter",  # shown to users before their text is sent
    "extra_body": {"provider": {"zdr": True, "data_collection": "deny"}},
}
app.config["LLM_ENCRYPTION_KEY"] = os.environ["LLM_ENCRYPTION_KEY"]  # Fernet key
```

- The shared key needs a token whose `email_verified` is `true` **and** a gate that
  says yes. Without a `FEATURE_GATE`, nobody gets it. In `AUTH_MODE="platform"` the
  host must put `email_verified` in the tokens it signs.
- For llama.cpp or vLLM, point `base_url` at the server and drop `extra_body`. The
  endpoint counts as configured only with an `api_key`, so use any non-empty value
  for a server without auth.
- The worker reads the same settings from the environment: `LLM_SHARED_BASE_URL`,
  `LLM_SHARED_API_KEY`, `LLM_SHARED_MODEL`, `LLM_SHARED_DESTINATION`,
  `LLM_SHARED_EXTRA_BODY` (JSON) and `LLM_ENCRYPTION_KEY`.
- `USAGE_METER` replaces the default DB counter (`LLM_MONTHLY_TOKEN_CAP`, default
  200000). Pass the same kind to `run_ml_worker(usage_meter=...)`.
- `LLM_BYOK_ALLOWED_HOSTS` limits where users' keys may go. Unset, any public https
  host is allowed.
- Call `purge_owner_llm_data(owner_id)` from account erasure.
- `LLM_AUDIT_HOOK`, a callable `(event, owner_id, **fields)`, receives each new
  acknowledgement for the host's audit log.

Other settings and defaults are in `llm/config.py`.

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

**Authentication is selected by `AUTH_MODE`:**

- `"keycloak"` (default) — verifies an RS/ES-signed Keycloak access token
  against `AUTH_ISSUER` / `AUTH_AUDIENCE` (JWKS from `AUTH_JWKS_URI` or the
  issuer). Never accepts HS256.
- `"platform"` — the host (e.g. Legal Blocks) signs one short-lived HS256 token
  per forwarded request; its `sub` is the owner, so each platform user's data
  is isolated without Keycloak. Tokens must carry a string `sub`, `iat` and
  `exp`, with `exp - iat` at most 60 seconds; RS256 and other algorithms are
  rejected. Bad, expired or missing tokens get a 401 and a log line.
  `OWNER_REF = NoOwnerRef()` is refused in this mode (no shared owner).
- `"none"` — no token; requires `LEGAL_ML_ALLOW_NO_AUTH = True`.

The platform secret's location is the host's concern; the package reads it at
request time (so the host may create it after this service starts):

```python
app.config["AUTH_MODE"] = "platform"
app.config["LEGAL_ML_PLATFORM_SECRET"] = "..."                 # the value, or
app.config["LEGAL_ML_PLATFORM_SECRET_FILE"] = "/run/ml-secret/secret"  # a path
```

The value wins over the file. If neither yields a secret the request gets a 503
naming where the secret was expected (e.g. `platform secret not found at
/run/ml-secret/secret`), so a deployment problem never looks like a sign-in
problem.

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

**Liveness.** While a job trains, the worker beats a per-job heartbeat key
every 15s, and a process-wide liveness key while it is up. The API treats a
`running` job with no beat for `LEGAL_ML_HEARTBEAT_STALE_SECONDS` (Flask config,
default 120) as dead and marks it failed, and tells queued jobs that the
training service isn't running when the worker key is that stale. The worker's
own window is `run_ml_worker(liveness=legal_ml_core.liveness.LivenessTimings(stale_after=...))`, also
120s by default; keep the two equal.

```python
app.config["LEGAL_ML_HEARTBEAT_STALE_SECONDS"] = 120  # default
```

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

## API and data formats

- [`docs/openapi.yaml`](docs/openapi.yaml): OpenAPI 3.1 description of every
  HTTP route `init_ml_feature` registers (kept in sync by
  `tests/test_openapi.py`).
- [`docs/data-formats.md`](docs/data-formats.md): dataset upload columns per
  task type, and how trained models are stored and exported.

## Development

```bash
pip install -e ".[test]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for tests, releases and citation
metadata, and [CHANGELOG.md](CHANGELOG.md) for what changed in each version.

## How to cite

If you use legal-ml-core in research or teaching, please cite it. The
citation metadata is in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this
repository" button reads it); [`codemeta.json`](codemeta.json) carries the
same metadata in CodeMeta form.

> Chakravarthy, S. (2026). *legal-ml-core* (Version 0.3.0) [Computer
> software]. Maastricht University. https://github.com/shashankmc/legal-ml-core

**DOI:** not yet assigned. Once the repository is connected to Zenodo, each
GitHub release will be archived there and its DOI (and a DOI badge) will be
added to this section and to `CITATION.cff`.

The user interface for this backend is
[vue-legal-ml-builder](https://github.com/shashankmc/vue-legal-ml-builder);
cite both if you use both.

## License

[MIT](LICENSE) © 2026 Shashank Chakravarthy, Maastricht University.

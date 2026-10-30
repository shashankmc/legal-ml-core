# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is 0.x, minor releases may contain breaking changes.

## [Unreleased]

## [0.3.0] - 2026-10-30

### Added

- LLM annotation under `/api/llm`: suggestions for a batch, standalone annotation
  jobs on the worker (new `llm_queue`), agreement check and bulk accept. One
  OpenAI-compatible adapter. Suggestions become labels only through a human or a
  passed agreement check.
- Access: host-injected `FEATURE_GATE` (fails closed without `email_verified`),
  encrypted user keys (BYOK), per-project acknowledgement of the LLM destination,
  and `LLM_AUDIT_HOOK`.
- `USAGE_METER` seam with a default per-month DB counter, and
  `run_ml_worker(usage_meter=...)`.
- `purge_owner_llm_data` for hosts' account erasure.
- `label_source` on annotation tasks. Migration `lmc_0002_llm`.
- FAIR4RS metadata: `LICENSE` (MIT), `CITATION.cff`, `codemeta.json` and
  `.zenodo.json`, plus a "How to cite" section in the README.
- `CHANGELOG.md` and `CONTRIBUTING.md`.
- GitHub Actions CI workflow running the pytest suite on push and pull request.
- `docs/openapi.yaml`: OpenAPI 3.1 description of every HTTP route the
  blueprints register, with a test that fails if a registered route is missing
  from the spec.
- `docs/data-formats.md`: dataset upload format per task type and how trained
  models are stored and exported.
- `pyproject.toml`: authors, keywords, classifiers and project URLs; `PyYAML`
  in the `test` extra.

### Fixed

- `tests/test_datasets_routes.py` no longer imports the removed
  `flask_jwt_extended`, so the suite passes in a clean environment.

## [0.2.0] - 2026-10-29

### Added

- `AUTH_MODE="platform"`: the host signs one short-lived HS256 token per
  forwarded request (string `sub`, `iat`, `exp`, `exp - iat` at most 60 s);
  the token `sub` is the owner. The secret comes from
  `LEGAL_ML_PLATFORM_SECRET` or `LEGAL_ML_PLATFORM_SECRET_FILE` and is read per
  request; a missing secret returns 503 naming where it was expected. Bad
  tokens return 401. `NoOwnerRef` is refused in this mode (#31).
- Worker liveness: a per-job heartbeat thread (every 15 s) and a process-wide
  `legal_ml_worker_alive` key. `/training/status` and `/training/jobs` mark a
  running job with no beat for `LEGAL_ML_HEARTBEAT_STALE_SECONDS` (default
  120) as failed with a plain-language message, and queued jobs report
  "Waiting: the training service isn't running" when the worker is stale.
  Timings are configurable with `LivenessTimings` (#29).
- `job_state.transition`: every guarded job-status write is logged with job id
  and owner (#29).

### Changed

- Stop really stops training: `/training/stop/<job_id>` sets a Redis cancel
  flag and cancels through a guarded update. The worker checks the flag at
  pickup, between preprocessing, fit and save, and on every HuggingFace
  Trainer step; a cancelled run skips evaluation and save and removes any
  written model file. A cancelled job never becomes completed or failed (#28).
- `/training/status`: job status and error message always come from the
  database; Redis only refines progress and live metrics of active jobs.
- `/training/start` returns 409 "This model is already training" when the
  model already has a pending or running job, and locks the model row
  (`SELECT ... FOR UPDATE`) so concurrent starts serialise (#30).
- Logging replaces `print` in the worker.

### Fixed

- Annotation queue, counts, listing and reset are scoped to the dataset
  owner; a project id alone is no longer treated as an ownership check, and a
  dataset filter the caller does not own returns 404 (#27).
- Worker loops survive errors: a malformed or unloadable task fails its job
  with "Training could not start. Try again." and the loop keeps serving.
  Redis connection errors back off from 1 s to 30 s with one log line when
  Redis goes down and one when it returns (#30).
- A failed cancel-flag write during Stop is logged and the stop still
  succeeds; Redis errors during a job end it as failed instead of leaving it
  running.

## [0.1.0] - 2026-10-21

Initial extraction from the `reducing_digital_divide` (BAIGLE) monorepo. This
version was not tagged; it corresponds to commit `02c5b61`.

### Added

- Flask blueprints for datasets, models, training, visualization, EDA and
  annotation, registered by `init_ml_feature`, and `run_ml_worker` for the
  training, prediction and explanation queue consumers.
- Keycloak bearer-token validation and `AUTH_MODE` with a guarded `"none"`.
- Pluggable ownership through an opaque `owner_id` (`KeycloakOwnerRef`,
  `HostModelOwnerRef`, `NoOwnerRef`) and optional projects (`PROJECT_SCOPE`).
- Shared declarative base `MLBase` so the host owns the SQLAlchemy handle.
- Package-owned migrations, pluggable storage backend and configurable queue
  names.
- Probability calibration, paginated dataset rows, per-class keywords, the
  upload task-type catalogue, and experiment-monitoring metric points.

### Fixed

- LIME explanation `KeyError` for predicted classes other than index 1.
- Worker foreign-key resolution and the missing `psycopg2` dependency.
- Model/task compatibility checks, live progress and failure reasons.

[Unreleased]: https://github.com/shashankmc/legal-ml-core/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/shashankmc/legal-ml-core/compare/02c5b61...v0.2.0
[0.1.0]: https://github.com/shashankmc/legal-ml-core/commits/02c5b61

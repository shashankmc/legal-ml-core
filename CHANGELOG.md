# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the version is 0.x, minor releases may contain breaking changes.

## [Unreleased]

## [0.5.1] - 2026-10-31

### Fixed

- **Every earlier release ignored the parameters chosen for a Model.** The worker
  handed the trainer the queue message, where the choices sit one level down under
  `parameters`, so classifiers, clustering, topic models and transformers all
  trained on defaults while `Model.parameters` showed the choice. This predates
  v0.2.0 and affects every release up to and including v0.5.0. Models trained so far
  used defaults and keep them until retrained: retraining applies their saved
  choices, so scores, cluster counts and training time may change.
- A balanced model keeps its benefit. Sigmoid calibration refitted its scores to the
  unweighted label frequencies and undid it, so `class_weight: balanced` now skips
  calibration and reports `probabilities_calibrated: false`.
- `GET /api/training/batch-prediction/<id>` answers `status: pending` from the moment
  a batch is queued, not 404 until the result arrives.
- `penalty: none`, the `newton-cg` and `sag` solvers and `cosine` hierarchical distance
  train, which the wizard always offered and the catalogue refused. `cosine` keeps a
  document with no usable words, which scikit-learn's own cosine option refuses.
- `max_depth: null`, the wizard's default for unlimited depth, is accepted.
- `penalty: elasticnet` mixes both penalties, through a new `l1_ratio` parameter
  (default 0.5). scikit-learn 1.8 and later otherwise fit it as plain l2.
- An NMF `alpha` strong enough to remove every topic fails with a reason.

### Changed

- Parameters that cannot train are refused with a 400 when a Model is created, when
  it is edited and when training starts, including Models already saved with them:
  `l1` without `liblinear` or `saga`, `elasticnet` without `saga`, and `ward` linkage
  without `euclidean` or `l2`.
- `lex_glue` is no longer a LegalBERT variant: no such model exists to download.
- Numbers are checked against their catalogue range on create, edit and start, in
  words: "Number of trees must be between 1 and 1000." Saved Models outside a range
  must be edited before they retrain. Some ranges widen to what the wizard allowed:
  `C` from 0.001, `degree` from 1, `n_estimators` from 1, `eps` from 0.01, `alpha`
  (Naive Bayes) from 0.01, K-Means `max_iter` from 10, `batch_size` from 4.
- `liblinear` is refused on a Dataset of more than two labels, read from the file
  only when `liblinear` is chosen. If the file cannot be read then, training fails
  with the same words. `n_clusters` is refused above the Dataset's rows.
- `GET /api/models/available` gives each model's `rules`, so a client can apply the
  dependent options, such as the solvers a penalty allows, without its own copy.
- NMF `l1_ratio` is no longer offered: an offered mix could remove every topic and
  fail in training. Pure L2 cannot. A saved value still applies. NMF `random_state`
  is offered instead.
- UK spelling across the model catalogue, including its descriptions.

## [0.5.0] - 2026-10-30

### Added

- `class_weight` parameter (`none` or `balanced`) for SVM, decision tree, random
  forest and logistic regression, in the trainer and the model catalogue.

### Fixed

- `/api/training/batch-predict` is served: the worker now consumes
  `batch_prediction_queue`, through `run_ml_worker(batch_prediction_queue=...)`.
  Before this, tasks stayed queued and the polling route answered 404.

### Security

- Requires PyJWT 2.15, whose JWKS client limits refreshes, so tokens with unknown
  key ids cannot force a fetch each. A test pins it.

## [0.4.0] - 2026-10-30

### Added

- Active learning for single-label text classification, under
  `/api/active-learning`: a holdout drawn once per dataset, a random seed batch,
  five query strategies on `predict_proba` output with NumPy and scikit-learn, a
  minimum per label up to a cap on labels, and a user-triggered retrain with a
  prompt. The worker rescores the pool on a new `scoring_queue`.
- The holdout draw is stratified on labels that already exist, with unlabelled
  rows random, and `holdout.label_counts` reports it by class.
- Label cap: a class short of its minimum holds training back only until
  `LEGAL_ML_AL_CAP_BATCHES` batches of labels. Past that the state names the short
  classes in `classes_short`, `classes_left_out` and `classes_unseen`. A class with
  one example is left out of training, and training still needs two classes with
  2 examples and enough rows for the stratified split.
- Migrations `lmc_0003_active_learning` (`al_configs`, `al_rounds`, and
  `annotation_tasks.score` and `is_holdout`) and `lmc_0004_holdout_file_counts`.
  Hosts with their own Alembic graph need a merge revision.
- `metrics.verification`: a model trained on a dataset with a loop is graded on
  its holdout, or reads `not_independently_verified`.
- Host config: `LEGAL_ML_AL_STRATEGIES`, `LEGAL_ML_AL_DEFAULT_STRATEGY`,
  `LEGAL_ML_AL_BATCH_SIZE`, `LEGAL_ML_AL_MIN_PER_CLASS`, `LEGAL_ML_AL_CAP_BATCHES`,
  `LEGAL_ML_AL_HOLDOUT_PERCENT`, `LEGAL_ML_AL_SCORING_STALE_SECONDS`,
  `LEGAL_ML_SCORING_QUEUE`.
- `PUT /api/active-learning/datasets/<id>/labels` corrects a typed label on every
  saved item.

### Changed

- Held-out rows are left out of training, LLM batches, LLM prompt examples,
  `GET /projects/<id>/tasks`, suggestion lists, confirm and bulk accept. The
  holdout is not drawn from rows an LLM touched, and only human or file labels
  grade a model.
- With a loop started, creating or training any model on the dataset keeps the
  minimum per label, up to the cap.
- The trainer no longer fails when a class has no row on the test side, and the
  confusion matrix keeps a row for it.
- The stopped-training message says what to do next, in plain words.

## [0.3.0] - 2026-10-30

### Fixed

- Annotation queue: tasks are created for rows with text and a blank label, so
  the queue is no longer always empty. Summarization is not queued, and saving
  an empty label is rejected. Saved annotations reach training for
  classification-family tasks.
- Training start: one start path. `/training/start` no longer returns a 500 on
  Postgres, duplicate starts return the active job, a failed start marks its job
  failed, and stale pending jobs are cleared on read.

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

[Unreleased]: https://github.com/shashankmc/legal-ml-core/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/shashankmc/legal-ml-core/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/shashankmc/legal-ml-core/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/shashankmc/legal-ml-core/compare/02c5b61...v0.2.0
[0.1.0]: https://github.com/shashankmc/legal-ml-core/commits/02c5b61

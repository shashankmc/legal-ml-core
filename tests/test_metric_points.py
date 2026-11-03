from legal_ml_core import Dataset, Model, TrainingJob, TrainingMetricPoint, db
from legal_ml_core.compute.tasks import MetricLogger
from legal_ml_core.compute.trainer import TrainingMetricCallback
from tests.conftest import TEST_USER_ID


def _make_job(app):
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="D1",
            file_path="/tmp/d.csv",
            row_count=1,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.flush()
        model = Model(
            owner_id=TEST_USER_ID,
            dataset_id=dataset.id,
            name="M1",
            model_type="legalbert",
            parameters={},
            status="training",
        )
        db.session.add(model)
        db.session.flush()
        job = TrainingJob(owner_id=TEST_USER_ID, model_id=model.id, status="running")
        db.session.add(job)
        db.session.commit()
        return job.id


def test_metric_logger_assigns_monotonic_cursor(app):
    job_id = _make_job(app)
    with app.app_context():
        logger = MetricLogger(job_id)
        step0 = logger.log("loss", 0.9, epoch=0.1)
        step1 = logger.log("loss", 0.7, epoch=0.2)
        step2 = logger.log("training_started", 1.0)  # coarse marker, no epoch

        assert (step0, step1, step2) == (0, 1, 2)
        logger.flush()

        points = TrainingMetricPoint.query.filter_by(training_job_id=job_id).order_by(
            TrainingMetricPoint.step
        ).all()
        assert [p.metric_name for p in points] == ["loss", "loss", "training_started"]
        assert points[0].epoch == 0.1
        assert points[2].epoch is None


def test_training_metric_callback_logs_numeric_log_entries_only(app):
    job_id = _make_job(app)

    class FakeState:
        epoch = 1.5
        global_step = 42

    with app.app_context():
        logger = MetricLogger(job_id)
        callback = TrainingMetricCallback(logger)

        callback.on_log(
            args=None,
            state=FakeState(),
            control=None,
            logs={"loss": 0.42, "eval_accuracy": 0.81, "epoch": 1.5, "learning_rate_schedule": "cosine"},
        )
        logger.flush()

        points = TrainingMetricPoint.query.filter_by(training_job_id=job_id).all()
        metric_names = {p.metric_name for p in points}
        # numeric entries are logged; the non-numeric "learning_rate_schedule"
        # string value is skipped rather than crashing on float(value), and
        # "epoch" is each point's x value, not a curve of its own
        assert metric_names == {"loss", "eval_accuracy"}
        assert all(p.epoch == 1.5 for p in points)


def test_training_metric_callback_noop_without_logger():
    callback = TrainingMetricCallback(metric_logger=None)
    # Should not raise even though there's no logger and no db/app context.
    callback.on_log(args=None, state=None, control=None, logs={"loss": 0.1})


class _TrainerState:
    def __init__(self, epoch, global_step):
        self.epoch = epoch
        self.global_step = global_step


def _points(job_id):
    return TrainingMetricPoint.query.filter_by(training_job_id=job_id).order_by(
        TrainingMetricPoint.step
    ).all()


def test_fine_tuning_points_record_the_trainers_step_and_their_group(app):
    job_id = _make_job(app)
    with app.app_context():
        logger = MetricLogger(job_id)
        callback = TrainingMetricCallback(logger)

        callback.on_log(None, _TrainerState(0.5, 10), None, logs={"loss": 0.7, "learning_rate": 2e-5})
        callback.on_log(None, _TrainerState(1.0, 20), None, logs={"eval_loss": 0.6, "eval_accuracy": 0.8})
        logger.flush()

        points = _points(job_id)
        assert [(p.metric_name, p.metric_group, p.training_step) for p in points] == [
            ("loss", "train", 10),
            ("learning_rate", "train", 10),
            ("eval_loss", "eval", 20),
            ("eval_accuracy", "eval", 20),
        ]
        # The write counter stays the polling cursor, apart from the trainer's step.
        assert [p.step for p in points] == [0, 1, 2, 3]


def test_the_trainers_end_of_run_summary_is_not_stored_as_curve_points(app):
    """HF's last on_log carries run totals (about 1e14 for total_flos) that
    would flatten every curve drawn beside them."""
    job_id = _make_job(app)
    with app.app_context():
        logger = MetricLogger(job_id)
        callback = TrainingMetricCallback(logger)

        callback.on_log(None, _TrainerState(1.0, 20), None, logs={
            "eval_loss": 0.6,
            "eval_runtime": 1.2,
            "eval_samples_per_second": 80.0,
            "eval_steps_per_second": 10.0,
        })
        callback.on_log(None, _TrainerState(1.0, 20), None, logs={
            "train_runtime": 30.5,
            "train_samples_per_second": 52.4,
            "train_steps_per_second": 6.5,
            "total_flos": 1.2e14,
            "train_loss": 0.55,
            "epoch": 1.0,
        })
        logger.flush()

        assert [p.metric_name for p in _points(job_id)] == ["eval_loss", "eval_runtime"]


def test_stage_markers_are_stored_in_the_stage_group(app):
    job_id = _make_job(app)
    with app.app_context():
        logger = MetricLogger(job_id)
        logger.log("training_started", 1.0)
        logger.log("model_saved", 1.0)
        logger.flush()

        assert [(p.metric_group, p.training_step) for p in _points(job_id)] == [
            ("stage", None),
            ("stage", None),
        ]


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_points_are_written_once_the_batch_is_full(app):
    job_id = _make_job(app)
    with app.app_context():
        logger = MetricLogger(job_id, flush_every_points=3, flush_every_seconds=60, clock=_Clock())
        logger.log("loss", 0.9)
        logger.log("loss", 0.8)
        assert _points(job_id) == []

        logger.log("loss", 0.7)
        assert [p.step for p in _points(job_id)] == [0, 1, 2]

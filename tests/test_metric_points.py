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

        points = TrainingMetricPoint.query.filter_by(training_job_id=job_id).all()
        metric_names = {p.metric_name for p in points}
        # numeric entries are logged; the non-numeric "learning_rate_schedule"
        # string value is skipped rather than crashing on float(value)
        assert metric_names == {"loss", "eval_accuracy", "epoch"}
        assert all(p.epoch == 1.5 for p in points)


def test_training_metric_callback_noop_without_logger():
    callback = TrainingMetricCallback(metric_logger=None)
    # Should not raise even though there's no logger and no db/app context.
    callback.on_log(args=None, state=None, control=None, logs={"loss": 0.1})

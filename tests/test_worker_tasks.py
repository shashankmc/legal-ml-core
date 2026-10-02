import pandas as pd
import pytest

from legal_ml_core import Dataset, Model, TrainingJob, TrainingMetricPoint, db
from legal_ml_core.compute.tasks import process_prediction_task, process_training_task
from legal_ml_core.compute.trainer import MLModelTrainer
from legal_ml_core.storage import get_model_path
from tests.conftest import TEST_USER_ID


def _classification_csv(tmp_path):
    df = pd.DataFrame(
        {
            "text": [
                "the court ruled for the plaintiff",
                "the defendant was found liable",
                "judgment entered against appellant",
                "the parties settled the dispute",
            ]
            * 3,
            "label": ["civil", "criminal"] * 6,
        }
    )
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_process_training_task_end_to_end(app, fake_redis, tmp_path):
    dataset_path = _classification_csv(tmp_path)

    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="D1",
            file_path=dataset_path,
            row_count=12,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.flush()
        model = Model(
            owner_id=TEST_USER_ID,
            dataset_id=dataset.id,
            name="M1",
            model_type="naive_bayes",
            parameters={},
            status="training",
        )
        db.session.add(model)
        db.session.flush()
        job = TrainingJob(owner_id=TEST_USER_ID, model_id=model.id, status="pending")
        db.session.add(job)
        db.session.commit()
        model_id, job_id = model.id, job.id

        task_info = {
            "job_id": job_id,
            "model_id": model_id,
            "dataset_id": dataset.id,
            "model_type": "naive_bayes",
            "parameters": {},
            "dataset_path": dataset_path,
            "task_type": "classification",
            "preprocessing_options": {},
        }

        process_training_task(task_info, fake_redis, app.config["MODEL_FOLDER"])

        updated_model = Model.query.get(model_id)
        updated_job = TrainingJob.query.get(job_id)
        assert updated_model.status == "completed"
        assert updated_model.file_path == get_model_path(app.config["MODEL_FOLDER"], model_id)
        assert updated_job.status == "completed"
        assert "accuracy" in updated_job.metrics

        points = TrainingMetricPoint.query.filter_by(training_job_id=job_id).order_by(
            TrainingMetricPoint.step
        ).all()
        metric_names = [p.metric_name for p in points]
        assert metric_names == [
            "preprocessing_started",
            "preprocessing_done",
            "training_started",
            "training_done",
            "model_saved",
        ]

    status_raw = fake_redis.get(f"job_status_{job_id}")
    assert status_raw is not None
    assert '"status": "completed"' in status_raw or '"status":"completed"' in status_raw


def test_process_training_task_marks_failure_on_bad_dataset_path(app, fake_redis):
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="D1",
            file_path="/nonexistent/path.csv",
            row_count=0,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.flush()
        model = Model(
            owner_id=TEST_USER_ID,
            dataset_id=dataset.id,
            name="M1",
            model_type="naive_bayes",
            parameters={},
            status="training",
        )
        db.session.add(model)
        db.session.flush()
        job = TrainingJob(owner_id=TEST_USER_ID, model_id=model.id, status="pending")
        db.session.add(job)
        db.session.commit()
        model_id, job_id = model.id, job.id

        task_info = {
            "job_id": job_id,
            "model_id": model_id,
            "dataset_id": dataset.id,
            "model_type": "naive_bayes",
            "parameters": {},
            "dataset_path": "/nonexistent/path.csv",
            "task_type": "classification",
            "preprocessing_options": {},
        }

        process_training_task(task_info, fake_redis, app.config["MODEL_FOLDER"])

        assert Model.query.get(model_id).status == "failed"
        assert TrainingJob.query.get(job_id).status == "failed"


def test_process_prediction_task_uses_saved_model(app, tmp_path):
    with app.app_context():
        trainer = MLModelTrainer(redis_client=_NullRedis())
        df = pd.DataFrame(
            {
                "text": ["the court ruled for the plaintiff", "the defendant was liable"] * 4,
                "label": ["civil", "criminal"] * 4,
            }
        )
        texts, labels, _ = trainer.preprocess_data(df, "classification", {})
        trainer.train_classification_model(texts, labels, "naive_bayes", {"model_id": "predict-model"})

        model_path = get_model_path(str(tmp_path), "predict-model")
        trainer.save_model(model_path)

        result = process_prediction_task(
            {
                "prediction_id": "p1",
                "model_id": "predict-model",
                "model_type": "naive_bayes",
                "texts": ["a new legal document about the plaintiff"],
                "owner_id": TEST_USER_ID,
            },
            str(tmp_path),
        )

        assert result["model_type"] == "naive_bayes"
        assert len(result["predictions"]) == 1
        assert result["predictions"][0]["prediction"] in ("civil", "criminal")


class _NullRedis:
    def setex(self, *a, **k):
        pass


def test_failed_training_persists_error_message(app, fake_redis):
    """A failure reason must reach the DB so the UI can explain it."""
    with app.app_context():
        dataset = Dataset(
            owner_id=TEST_USER_ID,
            name="D1",
            file_path="/tmp/d1.csv",
            row_count=10,
            columns=[],
            task_type="classification",
        )
        db.session.add(dataset)
        db.session.flush()
        model = Model(
            owner_id=TEST_USER_ID,
            dataset_id=dataset.id,
            name="M1",
            model_type="naive_bayes",
            parameters={},
            status="training",
        )
        db.session.add(model)
        db.session.flush()
        job = TrainingJob(owner_id=TEST_USER_ID, model_id=model.id, status="pending")
        db.session.add(job)
        db.session.commit()
        job_id = job.id

        task_info = {
            "job_id": job_id,
            "model_id": model.id,
            "dataset_id": dataset.id,
            "model_type": "naive_bayes",
            "parameters": {},
            "dataset_path": "/nonexistent/path.csv",
            "task_type": "classification",
            "preprocessing_options": {},
        }
        process_training_task(task_info, fake_redis, app.config["MODEL_FOLDER"])

        saved = TrainingJob.query.get(job_id)
        assert saved.status == "failed"
        assert saved.error_message

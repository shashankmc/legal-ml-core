"""Resource and throughput readings (BAIGLE #94): with `monitoring.system_metrics`
on, a run records its memory, its GPU when there is one, its speed and the time
it has left, as `system` points."""
import json
import sys
from types import SimpleNamespace

import pytest

from legal_ml_core.system_metrics import system_readings


class _FakeCuda:
    """torch.cuda on a machine with a GPU, 512 MB in use and 40% busy."""

    def __init__(self, utilisation=40):
        self._utilisation = utilisation

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def memory_allocated():
        return 512 * 1024 * 1024

    def utilization(self):
        if isinstance(self._utilisation, Exception):
            raise self._utilisation
        return self._utilisation


@pytest.fixture
def fake_gpu(monkeypatch):
    def install(**kwargs):
        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=_FakeCuda(**kwargs)))

    return install


def test_without_a_gpu_only_the_memory_is_read():
    """The test machine's torch is the CPU build, so CUDA is not available."""
    readings = system_readings()

    assert set(readings) == {"memory_mb"}
    assert readings["memory_mb"] > 0


def test_with_a_gpu_its_memory_and_use_are_read(fake_gpu):
    fake_gpu()

    readings = system_readings()

    assert readings["gpu_memory_mb"] == 512
    assert readings["gpu_use_percent"] == 40
    assert readings["memory_mb"] > 0


def test_a_gpu_whose_use_cannot_be_read_still_reports_its_memory(fake_gpu):
    """torch.cuda.utilization needs pynvml, which is not always installed."""
    fake_gpu(utilisation=ModuleNotFoundError("pynvml"))

    assert set(system_readings()) == {"memory_mb", "gpu_memory_mb"}


def test_a_process_without_torch_reads_only_the_memory(monkeypatch):
    """The API process has no torch."""
    monkeypatch.setitem(sys.modules, "torch", None)

    assert set(system_readings()) == {"memory_mb"}


def test_the_memory_is_read_without_psutil(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)

    assert system_readings()["memory_mb"] > 0


# A classical run, through the worker.

from legal_ml_core import TrainingJob, TrainingMetricPoint, db  # noqa: E402
from legal_ml_core.compute.tasks import process_training_task  # noqa: E402
from tests.test_monitoring_settings import STAGES, _start, dataset_id  # noqa: E402,F401


def _run_classical(client, auth_headers, app, fake_redis, dataset_id, monitoring):
    _start(client, auth_headers, dataset_id, monitoring)
    task = json.loads(fake_redis.rpop("training_queue"))
    process_training_task(task, fake_redis, app.config["MODEL_FOLDER"])
    assert db.session.get(TrainingJob, task["job_id"]).status == "completed"
    return TrainingMetricPoint.query.filter_by(training_job_id=task["job_id"]).order_by(TrainingMetricPoint.step).all()


def test_a_classical_run_records_readings_at_each_stage(client, auth_headers, app, fake_redis, dataset_id):
    points = _run_classical(client, auth_headers, app, fake_redis, dataset_id, {"system_metrics": True})

    # After each stage marker, the readings taken there.
    stages = [p.metric_name for p in points if p.metric_group == "stage"]
    assert stages == STAGES
    memory = [p for p in points if p.metric_name == "memory_mb"]
    assert len(memory) == len(STAGES)
    assert all(p.metric_group == "system" and p.value > 0 for p in memory)
    assert not any(p.metric_name.startswith("gpu_") for p in points)


def test_a_classical_run_records_how_fast_it_trained(client, auth_headers, app, fake_redis, dataset_id):
    points = _run_classical(client, auth_headers, app, fake_redis, dataset_id, {"system_metrics": True})

    speed = [p for p in points if p.metric_name == "samples_per_second"]
    assert len(speed) == 1
    assert speed[0].metric_group == "system" and speed[0].value > 0


def test_a_classical_run_records_its_gpu_when_there_is_one(client, auth_headers, app, fake_redis, dataset_id, fake_gpu):
    fake_gpu()

    points = _run_classical(client, auth_headers, app, fake_redis, dataset_id, {"system_metrics": True})

    assert [p.value for p in points if p.metric_name == "gpu_memory_mb"] == [512] * len(STAGES)


def test_readings_are_off_by_default(client, auth_headers, app, fake_redis, dataset_id):
    points = _run_classical(client, auth_headers, app, fake_redis, dataset_id, {})

    assert [p.metric_name for p in points] == STAGES

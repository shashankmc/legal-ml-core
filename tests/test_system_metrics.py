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


# Fine-tuning, through the Hugging Face trainer's callbacks.

from legal_ml_core.compute.trainer import MLModelTrainer, SystemMetricsCallback  # noqa: E402
from legal_ml_core.monitoring import effective_monitoring  # noqa: E402
from tests.test_training_cancel import stub_hf  # noqa: E402,F401


class _Logged:
    """Collects what a callback logs, as (name, value, group, training_step)."""

    def __init__(self):
        self.points = []

    def log(self, metric_name, value, epoch=None, training_step=None, group=None):
        self.points.append((metric_name, value, group, training_step))

    def named(self, name):
        return [(value, step) for metric, value, _, step in self.points if metric == name]


def _fine_tune(callback, steps, seconds_per_step=0.5, logging_steps=10, max_steps=100):
    """Drive the callback as the trainer does: 8 samples a step, `seconds_per_step` each."""
    clock = SimpleNamespace(now=0.0)
    callback._clock = lambda: clock.now
    args = SimpleNamespace(logging_steps=logging_steps, train_batch_size=8, gradient_accumulation_steps=1)
    state = SimpleNamespace(global_step=0, epoch=0.0, max_steps=max_steps)
    callback.on_train_begin(args, state, None)
    for step in range(1, steps + 1):
        clock.now += seconds_per_step
        state.global_step, state.epoch = step, step / max_steps
        callback.on_step_end(args, state, None)


def test_fine_tuning_records_readings_at_the_logging_interval():
    logged = _Logged()

    _fine_tune(SystemMetricsCallback(logged), steps=25)

    assert [step for _, step in logged.named("memory_mb")] == [10, 20]
    assert {group for _, _, group, _ in logged.points} == {"system"}


def test_fine_tuning_records_its_speed_and_the_time_it_has_left():
    """8 samples every half second is 16 a second; at step 10 of 100, the 90
    steps left take 45 seconds."""
    logged = _Logged()

    _fine_tune(SystemMetricsCallback(logged), steps=20)

    assert logged.named("samples_per_second") == [(16, 10), (16, 20)]
    assert logged.named("seconds_remaining") == [(45, 10), (40, 20)]


def test_fine_tuning_records_its_gpu_when_there_is_one(fake_gpu):
    fake_gpu(utilisation=75)
    logged = _Logged()

    _fine_tune(SystemMetricsCallback(logged), steps=10)

    assert (logged.named("gpu_memory_mb"), logged.named("gpu_use_percent")) == ([(512, 10)], [(75, 10)])


def _fine_tuning_callbacks(stub_hf, tmp_path, monitoring):
    """The callbacks fine-tuning hands the trainer. Stopped at its first step,
    so the stand-in needs no model."""
    from legal_ml_core.compute.trainer import TrainingCancelled

    trainer = MLModelTrainer(redis_client=None, metric_logger=_Logged(), cancel_check=lambda: bool(stub_hf.instances))
    parameters = {"model_id": "m", "output_dir": str(tmp_path), "monitoring": monitoring}
    with pytest.raises(TrainingCancelled):
        trainer.train_transformer_model(["the court ruled", "the jury convicted"] * 10, [0, 1] * 10, "legalbert", parameters)
    return [type(callback) for callback in stub_hf.instances[0].callbacks]


def test_fine_tuning_takes_readings_only_when_the_run_asks(stub_hf, tmp_path):
    assert SystemMetricsCallback in _fine_tuning_callbacks(stub_hf, tmp_path, effective_monitoring(run={"system_metrics": True}))
    stub_hf.instances.clear()
    assert SystemMetricsCallback not in _fine_tuning_callbacks(stub_hf, tmp_path, effective_monitoring())

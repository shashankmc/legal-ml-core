"""Resource and throughput readings (BAIGLE #94): with `monitoring.system_metrics`
on, a run records its memory, its GPU when there is one, its speed and the time
it has left, as `system` points."""
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

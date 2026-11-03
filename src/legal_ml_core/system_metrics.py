"""Resource readings for a training run (BAIGLE #94), recorded as `system`
points when the run's `monitoring.system_metrics` is on.

Nothing here is a hard dependency. Memory comes from psutil when it is
installed, else from the standard library. The GPU is read only when torch is
importable and CUDA is available: the API process has no torch."""
import os
import resource
import sys

_MB = 1024 * 1024


def _memory_bytes():
    """This process's resident memory."""
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except ImportError:
        pass
    try:
        with open("/proc/self/statm") as statm:
            return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    # The peak, not the current size: the best left without /proc. Kilobytes
    # on Linux, bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def _gpu_readings():
    try:
        import torch
    except ImportError:
        return {}
    if not torch.cuda.is_available():
        return {}
    readings = {"gpu_memory_mb": torch.cuda.memory_allocated() / _MB}
    try:
        readings["gpu_use_percent"] = float(torch.cuda.utilization())
    except Exception:
        pass  # needs pynvml, which is optional
    return readings


def system_readings():
    """The readings taken now, by point name."""
    return {"memory_mb": _memory_bytes() / _MB, **_gpu_readings()}

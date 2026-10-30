"""Worker liveness signals (CEO R5, D10, D12).

Two Redis keys, both written from their own daemon threads so they keep
beating while a training call blocks the training thread, and both stop the
moment the worker process dies (the OOM case):

- a per-job heartbeat (`job_heartbeat_{id}`), which also refreshes the job's
  `job_status_{id}` TTL so runs longer than that TTL keep their status;
- a per-process liveness key (`legal_ml_worker_alive`).

The API side (job_state.sweep_stale_job) reads them: a `running` job without a
heartbeat becomes `failed`; a `pending` job without a live worker gets a
"waiting" message. Accepted limit: a hang inside a live process keeps beating.
"""
import logging
import threading
import time
from dataclasses import dataclass

from .queue_contract import (
    JOB_STATUS_TTL_SECONDS,
    WORKER_LIVENESS_KEY,
    job_heartbeat_key,
    job_status_key,
)

logger = logging.getLogger(__name__)

STALE_JOB_MESSAGE = (
    "Training stopped before it finished, usually because the computer ran out of "
    "memory. Your Dataset and labels are safe. Train again. If it stops a second "
    "time, choose Logistic Regression, which needs the least memory, or use a "
    "Dataset with fewer rows."
)
WORKER_DOWN_MESSAGE = "Waiting: the training service isn't running"

# No beat for this long => the worker is gone. The API reads its own value
# from LEGAL_ML_HEARTBEAT_STALE_SECONDS (README); both default to this.
DEFAULT_STALE_SECONDS = 120


@dataclass(frozen=True)
class LivenessTimings:
    """Seconds. Defaults are the spec values; tests pass tiny ones."""

    heartbeat_interval: float = 15
    stale_after: float = DEFAULT_STALE_SECONDS
    status_ttl: float = JOB_STATUS_TTL_SECONDS


DEFAULT_TIMINGS = LivenessTimings()


def _ms(seconds):
    return max(1, int(seconds * 1000))


def _key_ttl_ms(timings):
    """Heartbeat/liveness keys carry their beat time; staleness is judged from
    that (see is_beat_stale), so the TTL is only a backstop that clears keys of
    dead workers. It is deliberately longer than `stale_after`."""
    return _ms(max(timings.stale_after * 2, 2))


def is_beat_stale(raw_value, stale_after, now=None):
    """True if a heartbeat/liveness value is missing or older than stale_after."""
    if raw_value is None:
        return True
    try:
        beat_at = float(raw_value)
    except (TypeError, ValueError):
        return True
    return ((now if now is not None else time.time()) - beat_at) > stale_after


class _Beater:
    """Calls `beat()` now and then every `interval` seconds on a daemon thread
    until `stop()`. A failing beat is logged and retried next interval."""

    def __init__(self, interval, name):
        self._interval = interval
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def beat(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def start(self):
        self._safe_beat()  # synchronous first beat: no gap after start()
        self._thread.start()
        return self

    def stop(self):
        self._stopped.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=self._interval + 1)

    def _run(self):
        while not self._stopped.wait(self._interval):
            self._safe_beat()

    def _safe_beat(self):
        try:
            self.beat()
        except Exception as e:
            logger.warning("%s beat failed: %s", self._thread.name, e)


class Heartbeat(_Beater):
    """Per-job heartbeat; `stop()` also removes the key (job no longer runs)."""

    def __init__(self, redis_client, job_id, timings=DEFAULT_TIMINGS):
        super().__init__(timings.heartbeat_interval, f"heartbeat-{job_id}")
        self._redis = redis_client
        self._job_id = job_id
        self._timings = timings

    def beat(self):
        self._redis.psetex(
            job_heartbeat_key(self._job_id), _key_ttl_ms(self._timings), repr(time.time())
        )
        self._redis.pexpire(job_status_key(self._job_id), _ms(self._timings.status_ttl))

    def stop(self):
        super().stop()
        try:
            self._redis.delete(job_heartbeat_key(self._job_id))
        except Exception as e:
            logger.warning("could not clear heartbeat for job %s: %s", self._job_id, e)


class WorkerLiveness(_Beater):
    def __init__(self, redis_client, timings=DEFAULT_TIMINGS):
        super().__init__(timings.heartbeat_interval, "legal-ml-worker-liveness")
        self._redis = redis_client
        self._timings = timings

    def beat(self):
        self._redis.psetex(WORKER_LIVENESS_KEY, _key_ttl_ms(self._timings), repr(time.time()))


def start_worker_liveness(redis_client, timings=DEFAULT_TIMINGS):
    """Start the process-wide liveness beat. Returns it (call `.stop()` to end)."""
    return WorkerLiveness(redis_client, timings).start()

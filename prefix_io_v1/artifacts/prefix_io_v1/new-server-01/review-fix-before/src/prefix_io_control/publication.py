"""Optional CPU-side snapshot publication; no policy, queue, I/O or GPU calls."""
from threading import Lock, get_ident
from time import monotonic_ns
from weakref import ref
from .observation import capture_reactor


class SnapshotPublisher:
    """One reactor owner, one replaceable immutable snapshot, arbitrary readers.

    The caller explicitly chooses the sampling interval. This is a CPU preparation
    API, not a vLLM option. Readers MUST supply a run and finite freshness limit.
    A busy reader causes a dropped observation, never a blocked reactor.
    No completed-parent history is retained; absence is not completion evidence.
    """

    def __init__(self, *, run_id: str, interval_ns: int, clock=monotonic_ns):
        if type(run_id) is not str or not run_id.strip():
            raise ValueError("run_id must be a nonempty string")
        if type(interval_ns) is not int or interval_ns <= 0:
            raise ValueError("interval_ns must be a positive integer")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.run_id = run_id
        self.interval_ns = interval_ns
        self._clock = clock
        self._lock = Lock()
        self._latest = None
        self._owner = None
        self._next_due = None
        self._last_clock = None
        self._epoch = 0
        self._faulted = False
        # Owner-only diagnostics; read after the reactor stops, not a live API.
        self.published_count = 0
        self.contention_drops = 0

    def __call__(self, reactor):
        # This check must precede throttling, including calls within one interval.
        if reactor._worker.ident != get_ident():
            raise RuntimeError("publisher must run on the reactor owner")
        if self._owner is None:
            self._owner = ref(reactor)
        elif self._owner() is not reactor:
            raise RuntimeError("publisher cannot be reused across reactors")
        if self._faulted:
            return
        try:
            now = self._clock()
            if type(now) is not int or now < 0:
                raise ValueError("clock must return nonnegative integer nanoseconds")
            if self._last_clock is not None and now < self._last_clock:
                raise ValueError("monotonic clock moved backwards")
            self._last_clock = now
            if self._next_due is not None and now < self._next_due:
                return
            self._next_due = now + self.interval_ns  # No catch-up bursts.
            self._epoch += 1
            snapshot = capture_reactor(reactor, run_id=self.run_id, epoch=self._epoch)
            if not self._lock.acquire(blocking=False):
                self.contention_drops += 1
                return
            try:
                self._latest = snapshot  # At most one retained observation.
                self.published_count += 1
            finally:
                self._lock.release()
        except Exception:
            self._faulted = True  # Readers reject the old snapshot on collector failure.
            raise

    def latest(self, *, run_id: str, now_ns: int, max_age_ns: int):
        if type(max_age_ns) is not int or max_age_ns <= 0:
            raise ValueError("max_age_ns must be a positive integer")
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("now_ns must be nonnegative integer nanoseconds")
        with self._lock:
            snapshot = self._latest
        if self._faulted or snapshot is None:
            return None
        return snapshot if snapshot.fresh(
            run_id=run_id, now_ns=now_ns, max_age_ns=max_age_ns) else None

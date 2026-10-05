import gc
from dataclasses import FrozenInstanceError, replace
from threading import Event, Thread, get_ident
from time import monotonic_ns
from types import SimpleNamespace as NS
from weakref import ref
import pytest
from prefix_io_control.publication import SnapshotPublisher
from prefix_io_control.dependencies import Parent, Resource, ResourceId, analyze
from py_kvcache.reactor import IoReactor
from tests.prefix_io_v1_progress._cpu_backend import running_rig


class CpuReactor(NS):
    pass


def fixture(count=2):
    return CpuReactor(_worker=NS(ident=get_ident()), actual_staging_bytes=65535,
        staging_pool=NS(free_count=2),
        _active=[NS(job_id=i, total_files=9, done_files=1, inflight_files=2,
                    future_set=False, failed=None) for i in range(count)],
        _inflight={1: NS(op_kind="read"), 2: NS(op_kind="write")},
        _pending_copies=[NS(nbytes=4096, is_store=False)])


def test_rate_limit_skips_capture_and_never_catches_up(monkeypatch):
    import prefix_io_control.publication as mod
    now = [100]
    pub = SnapshotPublisher(run_id="r", interval_ns=10, clock=lambda: now[0])
    native = mod.capture_reactor
    calls = []
    def capture(*args, **kwargs):
        calls.append(kwargs["epoch"])
        return native(*args, **kwargs)
    monkeypatch.setattr(mod, "capture_reactor", capture)
    r = fixture(100)
    pub(r)
    now[0] = 109
    pub(r)
    assert calls == [1]
    now[0] = 10000
    pub(r)
    pub(r)
    assert calls == [1, 2] and pub.published_count == 2
    snap = pub.latest(run_id="r", now_ns=monotonic_ns(), max_age_ns=1_000_000_000)
    assert snap.epoch == 2 and len(snap.parents) == 32 and snap.parents_truncated
    assert "gpu_immediately_reusable_bytes" not in snap.available_fields
    assert snap.parents[0].lifecycle_state == "ACTIVE"
    with pytest.raises(FrozenInstanceError):
        snap.epoch = 0


def test_only_latest_snapshot_is_retained():
    tick = iter(range(100))
    pub = SnapshotPublisher(run_id="r", interval_ns=1, clock=lambda: next(tick))
    r = fixture()
    pub(r)
    first = ref(pub.latest(run_id="r", now_ns=monotonic_ns(), max_age_ns=1_000_000_000))
    assert first() is not None
    for _ in range(20):
        pub(r)
    gc.collect()
    assert first() is None
    assert pub.published_count == 21 and pub._latest.epoch == 21


def test_reader_contention_drops_observation_without_blocking_owner():
    pub = SnapshotPublisher(run_id="r", interval_ns=1)
    r = fixture()
    finished = Event()
    errors = []
    def owner():
        r._worker.ident = get_ident()
        try:
            pub(r)
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()
    pub._lock.acquire()
    thread = Thread(target=owner, daemon=True)
    thread.start()
    try:
        progressed = finished.wait(2)
    finally:
        pub._lock.release()
        thread.join(2)
    assert progressed and not errors
    assert pub.contention_drops == 1 and pub.published_count == 0
    assert pub.latest(run_id="r", now_ns=monotonic_ns(), max_age_ns=1_000_000) is None


def test_stale_wrong_run_and_future_timestamp_are_rejected():
    pub = SnapshotPublisher(run_id="r", interval_ns=1)
    pub(fixture())
    snap = pub._latest
    assert pub.latest(run_id="r", now_ns=snap.monotonic_ns+5, max_age_ns=10) is snap
    for run_id, now in [("other", snap.monotonic_ns), ("r", snap.monotonic_ns-1),
                        ("r", snap.monotonic_ns+11)]:
        assert pub.latest(run_id=run_id, now_ns=now, max_age_ns=10) is None


def test_cross_owner_call_is_rejected_even_during_sampling_interval():
    pub = SnapshotPublisher(run_id="r", interval_ns=1_000_000_000)
    r = fixture()
    pub(r)
    r._worker.ident = -1
    with pytest.raises(RuntimeError, match="owner"):
        pub(r)


def test_publisher_cannot_be_reused_for_another_reactor():
    pub = SnapshotPublisher(run_id="r", interval_ns=1)
    r = fixture()
    pub(r)
    with pytest.raises(RuntimeError, match="reused"):
        pub(fixture())


def test_capture_failure_invalidates_previous_snapshot():
    now = [1]
    pub = SnapshotPublisher(run_id="r", interval_ns=1, clock=lambda: now[0])
    r = fixture()
    pub(r)
    del r._active[0].inflight_files
    now[0] = 2
    with pytest.raises(AttributeError):
        pub(r)
    assert pub.latest(run_id="r", now_ns=monotonic_ns(), max_age_ns=1_000_000_000) is None


def test_clock_regression_invalidates_snapshot():
    now = [10]
    pub = SnapshotPublisher(run_id="r", interval_ns=100, clock=lambda: now[0])
    r = fixture()
    pub(r)
    now[0] = 9
    with pytest.raises(ValueError, match="backwards"):
        pub(r)
    assert pub.latest(run_id="r", now_ns=monotonic_ns(), max_age_ns=1_000_000_000) is None


@pytest.mark.parametrize("run_id,interval", [("", 1), (" ", 1), (None, 1),
    (1, 1), ("r", 0), ("r", -1), ("r", True), ("r", 1.5)])
def test_invalid_publication_configuration(run_id, interval):
    with pytest.raises(ValueError):
        SnapshotPublisher(run_id=run_id, interval_ns=interval)


@pytest.mark.parametrize("value", [0, -1, True, None])
def test_reader_requires_finite_positive_age(value):
    pub = SnapshotPublisher(run_id="r", interval_ns=1)
    with pytest.raises(ValueError):
        pub.latest(run_id="r", now_ns=1, max_age_ns=value)


def test_native_reactor_hook_publishes_and_preserves_parent_progress(monkeypatch):
    with running_rig(monkeypatch) as rig:
        pub = SnapshotPublisher(run_id="cpu-run", interval_ns=1_000_000)
        seen = Event()
        def sink(reactor):
            pub(reactor)
            if pub.published_count:
                seen.set()
        rig.r._observation_sink = sink
        rig.r._observation_failures = 0
        rig.copy_ready.clear()
        future = rig.submit(files=2)
        rig.start()
        waiter = rig.waiter({7})
        assert seen.wait(2)
        snap = pub.latest(run_id="cpu-run", now_ns=monotonic_ns(), max_age_ns=1_000_000_000)
        assert snap is not None and snap.active_parent_count == 1
        assert snap.gpu_immediately_reusable_bytes is None
        rig.copy_ready.set()
        waiter.join(2)
        assert not waiter.is_alive() and future.result() == 8192
        assert rig.r._observation_failures == 0 and rig.ordinary_credits == 0


def test_native_progress_survives_observer_exception_once(monkeypatch):
    with running_rig(monkeypatch) as rig:
        calls = []
        def broken(reactor):
            calls.append(1)
            raise ValueError("CPU observer failure")
        rig.r._observation_sink = broken
        rig.r._observation_failures = 0
        future = rig.submit(files=3)
        rig.start()
        waiter = rig.waiter({7})
        waiter.join(2)
        assert not waiter.is_alive() and future.result() == 12288
        assert calls == [1] and rig.r._observation_failures == 1
        assert rig.r._observation_sink is None


def test_disabled_hook_does_not_call_clock_or_capture(monkeypatch):
    r = IoReactor.__new__(IoReactor)
    r._observation_sink = None
    def forbidden(*args, **kwargs):
        raise AssertionError("off observer executed")
    monkeypatch.setattr("prefix_io_control.publication.monotonic_ns", forbidden)
    monkeypatch.setattr("prefix_io_control.publication.capture_reactor", forbidden)
    r._observe_once()


def test_invalid_native_sink_fails_before_backend_resources(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("allocated backend resources")
    monkeypatch.setattr("py_kvcache.reactor.LiburingRing", forbidden)
    with pytest.raises(TypeError, match="observation_sink"):
        IoReactor(config=None, file_mapper=None, layout=None, observation_sink=1)


@pytest.mark.parametrize("failed,done,inflight,future,state", [
    (True, 2, 1, True, "FAILED_DRAINING"), (True, 2, 0, True, "FAILED_FINAL"),
    (False, 3, 0, True, "COMPLETED"), (False, 2, 1, True, "INCONSISTENT"),
    (False, 0, 0, False, "ACCEPTED"), (False, 1, 1, False, "ACTIVE")])
def test_lifecycle_is_observation_not_a_replacement_state_machine(failed, done, inflight, future, state):
    p = Parent("r", 1, 3, done, inflight, future, failed, None)
    assert p.lifecycle_state == state


@pytest.mark.parametrize("confirmation,expected", [(None, 0), (False, 0), (True, 4096)])
def test_parent_success_requires_native_release_confirmation(confirmation, expected):
    p = Parent("r", 1, 3, 3, 0, True, False, None)
    resource = Resource(ResourceId("r", "gpu", 0, 1, 1), 4096, 0,
                        frozenset({1}), "observed_blocking", native_reusable=confirmation)
    result = analyze(resource, (p,), run_id="r", current_generation=1)
    assert result.reusable_now_bytes == expected
    assert result.potential_bytes == (4096 if not confirmation else 0)


def test_release_confirmation_cannot_override_active_refs_or_stale_identity():
    p = Parent("r", 1, 3, 3, 0, True, False, None)
    resource = Resource(ResourceId("r", "gpu", 0, 1, 1), 4096, 1,
                        frozenset({1}), "observed_blocking", native_reusable=True)
    assert analyze(resource, (p,), run_id="r", current_generation=1).reusable_now_bytes == 0
    assert analyze(replace(resource, active_refs=0), (p,), run_id="r", current_generation=2).reusable_now_bytes == 0


@pytest.mark.parametrize("guard", ["wrong_thread", "different_reactor", "missing_worker"])
def test_failed_owner_guard_invalidates_previous_snapshot(guard):
    # Constant sampling time proves the checks also apply inside an interval.
    pub = SnapshotPublisher(run_id="r", interval_ns=100, clock=lambda: 10)
    reactor = fixture()
    pub(reactor)
    snapshot = pub._latest
    reader_args = dict(run_id="r", now_ns=snapshot.monotonic_ns + 1, max_age_ns=2)
    assert pub.latest(**reader_args) is snapshot
    worker = reactor._worker
    target = reactor
    if guard == "wrong_thread":
        worker.ident = -1
        error = RuntimeError
    elif guard == "different_reactor":
        target = fixture()
        error = RuntimeError
    else:
        del reactor._worker
        error = AttributeError
    with pytest.raises(error):
        pub(target)
    # A fresh timestamp cannot make an invalidated observation usable.
    assert pub.latest(**reader_args) is None
    reactor._worker = worker
    worker.ident = get_ident()
    pub(reactor)
    assert pub.latest(**reader_args) is None
    assert pub.published_count == 1

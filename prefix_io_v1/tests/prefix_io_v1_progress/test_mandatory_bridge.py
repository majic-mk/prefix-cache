"""Real py-kvcache methods on fake CUDA/io_uring; GPU qualification remains false."""
from concurrent.futures import Future
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from py_kvcache.reactor import _MandatoryWait, _ProgressState, IoReactor
from ._cpu_backend import Rig, running_rig

def done(thread):
    thread.join(timeout=2)
    assert not thread.is_alive(), "blocking wait did not progress"

def test_wait_publishes_before_blocking_with_zero_ordinary_credits(monkeypatch):
    with running_rig(monkeypatch) as rig:
        future = rig.submit(files=3)
        rig.start()
        waiter = rig.waiter({7})
        done(waiter)  # No scheduler epochs or credit refreshes are supplied.
        assert future.result() == 3*4096
        assert rig.marked.is_set() and rig.ordinary_credits == 0
        assert rig.issued == rig.writes == 3
        rig.r.shutdown()
        assert not rig.r._prefix_progress.required
        assert not rig.r._active

def test_mandatory_does_not_skip_compute_or_copy_events(monkeypatch):
    with running_rig(monkeypatch) as rig:
        rig.compute_ready.clear()
        future = rig.submit(files=1)
        rig.start()
        waiter = rig.waiter({7})
        assert rig.copy_issued.wait(2) and rig.copy_polled.wait(2)
        assert not rig.write_queued.is_set() and not future.done()
        rig.copy_ready.clear()
        rig.compute_ready.set()
        rig.copy_polled.clear()
        assert rig.copy_polled.wait(2)
        assert not future.done() and rig.writes == 0
        rig.copy_ready.set()
        done(waiter)
        assert future.result() == 4096

def test_mandatory_cannot_exceed_slot_capacity(monkeypatch):
    with running_rig(monkeypatch, slots=1, iodepth=16) as rig:
        rig.copy_ready.clear()
        future = rig.submit(files=3)
        rig.start()
        waiter = rig.waiter({7})
        assert rig.copy_issued.wait(2) and rig.copy_polled.wait(2)
        assert rig.issued == 1 and rig.r.staging_pool.free_count == 0
        assert not future.done()
        rig.copy_ready.set()
        done(waiter)
        assert rig.writes == 3 and len(rig.r.staging_pool.releases) == 3

def test_d2h_continues_to_write_without_ordinary_credit(monkeypatch):
    with running_rig(monkeypatch) as rig:
        rig.r.ring.auto = False
        future = rig.submit(files=1)
        rig.start()
        waiter = rig.waiter({7})
        assert rig.write_queued.wait(2)
        assert not future.done() and rig.ordinary_credits == 0
        assert future in rig.r._prefix_progress.required
        rig.r.ring.permits.release()
        done(waiter)
        assert future.result() == 4096
        assert len(rig.r.file_store.finished) == 1

def test_failed_future_keeps_signal_until_other_io_drains(monkeypatch):
    with running_rig(monkeypatch) as rig:
        rig.r.ring.auto = False
        rig.r.ring.results[1] = 4095  # Real native short-write failure handling.
        future = rig.submit(files=2)
        rig.start()
        waiter = rig.waiter({7})
        assert rig.write_queued.wait(2)
        rig.r.ring.permits.release()
        done(waiter)
        with pytest.raises(IOError, match="expected"):
            future.result()
        # Second D2H/write is accepted already; failure completion is not drain.
        assert future in rig.r._prefix_progress.required
        rig.r.ring.auto = True
        rig.r.shutdown()
        assert not rig.r._active and not rig.r._prefix_progress.required
        assert len(rig.r.file_store.finished) == 1
        assert len(rig.r.file_store.cleaned) == 1
        assert len(rig.r.staging_pool.releases) == 2

def test_duplicate_cqe_does_not_double_release_or_complete(monkeypatch):
    with running_rig(monkeypatch) as rig:
        rig.r.ring.duplicate = True
        future = rig.submit(files=2)
        callbacks = []
        future.add_done_callback(lambda f: callbacks.append(f))
        rig.start()
        done(rig.waiter({7}))
        rig.r.shutdown()
        assert len(callbacks) == 1
        assert len(rig.r.staging_pool.releases) == 2

def test_shutdown_drains_without_a_handler_wait_or_new_epochs(monkeypatch):
    with running_rig(monkeypatch) as rig:
        future = rig.submit(files=3)
        rig.start()
        shutdown = threading.Thread(target=rig.r.shutdown, daemon=True)
        shutdown.start()
        done(shutdown)
        assert future.result() == 3*4096 and rig.ordinary_credits == 0
        assert not rig.r._active and not rig.r._prefix_progress.required

def test_empty_and_unknown_handler_waits_do_not_publish(monkeypatch):
    rig = Rig(monkeypatch)
    rig.handler.wait(set())
    rig.handler.wait({900})
    assert rig.r._incoming.empty()
    assert not rig.r.request_mandatory([])

def test_signal_is_run_and_future_scoped_and_idempotent(monkeypatch):
    rig = Rig(monkeypatch)
    future = rig.submit(files=1)
    rig.r._drain_incoming(block=False)
    state = rig.r._prefix_progress
    rig.r._intake(_MandatoryWait(_ProgressState("old-run").token, frozenset({future})))
    assert not state.required
    rig.r._intake(_MandatoryWait(state.token, frozenset({Future()})))
    assert not state.required
    message = _MandatoryWait(state.token, frozenset({future}))
    rig.r._intake(message)
    rig.r._intake(message)
    assert state.required == {future}
    # Recycled integer job ID, distinct Future: stale notification cannot match.
    rig.r._active.clear()
    state.required.clear()
    rig.handler._active.clear()
    new_future = rig.submit(files=1, job_id=7)
    rig.r._drain_incoming(block=False)
    rig.r._intake(message)
    assert not state.required and new_future is not future

def test_completed_unknown_future_is_ignored(monkeypatch):
    rig = Rig(monkeypatch)
    future = Future()
    future.set_result(0)
    rig.r._intake(_MandatoryWait(rig.r._prefix_progress.token, frozenset({future})))
    assert not rig.r._prefix_progress.required

def test_after_shutdown_no_signal_is_enqueued(monkeypatch):
    rig = Rig(monkeypatch)
    rig.r._closed = True
    assert not rig.r.request_mandatory([Future()])
    assert rig.r._incoming.empty()

def test_wrong_thread_cannot_read_mutable_membership(monkeypatch):
    rig = Rig(monkeypatch)
    with pytest.raises(RuntimeError, match="owner"):
        rig.r.is_mandatory(Future())

def test_disabled_bridge_has_no_signal_state_or_queue_items(monkeypatch):
    rig = Rig(monkeypatch, enabled=False)
    assert rig.r._prefix_progress is None
    assert not rig.r.request_mandatory([Future()])
    assert rig.r._incoming.empty()
    assert not rig.r.is_mandatory(Future())

def test_disabled_handler_wait_does_not_call_bridge(monkeypatch):
    rig = Rig(monkeypatch, enabled=False)
    future = Future()
    future.set_result(0)
    rig.handler._active[7] = (future, 0.0, ("GPU", "SHARED_STORAGE"))
    rig.coordinator.request_mandatory = Mock(side_effect=AssertionError("off"))
    rig.handler.wait({7})
    rig.coordinator.request_mandatory.assert_not_called()

def test_running_native_future_cannot_be_cancelled_mid_transfer(monkeypatch):
    with running_rig(monkeypatch) as rig:
        rig.copy_ready.clear()
        future = rig.submit(files=1)
        rig.start()
        waiter = rig.waiter({7})
        assert rig.copy_issued.wait(2)
        assert future.cancel() is False
        rig.copy_ready.set()
        done(waiter)
        assert future.result() == 4096

@pytest.mark.parametrize("run_id", ["", " ", 1, False])
def test_invalid_run_id_rejected_before_resources(monkeypatch, run_id):
    ring = Mock(side_effect=AssertionError("must not allocate"))
    monkeypatch.setattr("py_kvcache.reactor.LiburingRing", ring)
    with pytest.raises(ValueError, match="run_id"):
        IoReactor(config=None, file_mapper=None, layout=None, progress_run_id=run_id)
    ring.assert_not_called()

def test_multiple_parent_wait_drains_every_accepted_parent(monkeypatch):
    with running_rig(monkeypatch, slots=1) as rig:
        first = rig.submit(files=2, job_id=7)
        second = rig.submit(files=3, job_id=8)
        rig.start()
        done(rig.waiter({7, 8}))
        assert first.result() == 8192 and second.result() == 12288
        assert rig.writes == 5 and rig.ordinary_credits == 0

def test_off_runs_native_work_without_mandatory_signals(monkeypatch):
    with running_rig(monkeypatch, enabled=False) as rig:
        rig.ordinary_credits = 1  # Off has no production quota; the test shim is open.
        future = rig.submit(files=3)
        rig.start()
        done(rig.waiter({7}))
        assert future.result() == 12288
        assert rig.writes == rig.issued == 3
        assert rig.r._prefix_progress is None and not rig.marked.is_set()

def test_fatal_native_failure_drops_signal_refs_and_never_reports_success(monkeypatch):
    rig = Rig(monkeypatch)
    future = rig.submit(files=1)
    rig.r._drain_incoming(block=False)
    state = rig.r._prefix_progress
    rig.r._intake(_MandatoryWait(state.token, frozenset({future})))
    assert state.required == {future}
    cause = RuntimeError("CPU injected fatal error")
    rig.r._fail_everything(cause)
    with pytest.raises(RuntimeError, match="reactor thread crashed") as error:
        future.result()
    assert error.value.__cause__ is cause
    assert not state.required and rig.r._closed
    assert not rig.r.request_mandatory([future])
    assert not rig.r.file_store.finished
    assert not rig.r.staging_pool.releases  # No new release/recovery behavior.

def test_owner_observation_during_native_cpu_event_loop(monkeypatch):
    from dataclasses import FrozenInstanceError
    from prefix_io_control.observation import capture_reactor
    observed = threading.Event()
    captured = []
    def capture(reactor, **kwargs):
        snapshot = capture_reactor(reactor, **kwargs)
        if snapshot.d2h_inflight_bytes and not observed.is_set():
            captured.append(snapshot)
            observed.set()
        return snapshot
    with running_rig(monkeypatch) as rig:
        rig.copy_ready.clear()
        rig.observe = capture
        future = rig.submit(files=1)
        rig.start()
        waiter = rig.waiter({7})
        assert observed.wait(2)
        snapshot = captured[0]
        assert snapshot.active_parent_count == 1
        assert snapshot.d2h_inflight_bytes == 4096
        assert snapshot.gpu_immediately_reusable_bytes is None
        assert snapshot.parents[0].remaining_stages is None
        with pytest.raises(FrozenInstanceError):
            snapshot.epoch = 0
        rig.copy_ready.set()
        done(waiter)
        assert future.result() == 4096
        assert snapshot.d2h_inflight_bytes == 4096  # Immutable prior event, not live state.

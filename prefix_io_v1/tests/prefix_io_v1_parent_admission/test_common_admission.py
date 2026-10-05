"""Common protection count/backpressure contracts, independent of P3 policy."""
import threading
from unittest.mock import Mock
import pytest
from py_kvcache import reactor as mod
from py_kvcache.reactor import IoReactor,TransferCoordinator,NativeDrainUnknown
from tests.prefix_io_v1_start_budget._fixture import rig,eventually
from ._fixture import common,physical_streams

def test_incoming_parents_are_counted_before_native_intake(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        r=common(x,None)
        first=x.submit(files=1,job_id=7);second=x.submit(files=1,job_id=8)
        s=r.parent_admission_snapshot()
        assert s["accepted_parents"]==2 and s["peak_accepted_parents"]==2
        assert not r._active and not first.done() and not second.done()
        x.start();assert first.result(timeout=2)==second.result(timeout=2)==4096
        x.stop();assert r.parent_admission_snapshot()["accepted_parents"]==0

def test_bounded_submit_waits_and_original_incoming_marks_drain(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        r=common(x,1);x.copy_ready.clear()
        first=x.submit(files=1,job_id=7);admitted=[]
        worker=threading.Thread(target=lambda:admitted.append(x.submit(files=1,job_id=8)),daemon=True)
        worker.start();eventually(lambda:r._parent_waiters==1)
        assert r._accepted_parent_count==1 and not admitted
        x.start();eventually(lambda:first in r._prefix_progress.required)
        x.copy_ready.set();assert first.result(timeout=2)==4096
        worker.join(timeout=2);assert not worker.is_alive()
        assert admitted[0].result(timeout=2)==4096
        x.stop();s=r.parent_admission_snapshot()
        assert s["peak_accepted_parents"]==1 and s["retired_parents"]==2
        assert s["backpressure_waits"]>=1 and s["accepted_parents"]==0

def test_shutdown_notifies_unaccepted_candidate_without_dropping_old_parent(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        r=common(x,1);x.copy_ready.clear()
        first=x.submit(files=1,job_id=7);returned=[]
        thread=threading.Thread(target=lambda:returned.append(x.submit(files=1,job_id=8)),daemon=True)
        thread.start();eventually(lambda:r._parent_waiters==1)
        r.shutdown(wait=False);thread.join(timeout=2)
        assert not thread.is_alive() and len(returned)==1
        with pytest.raises(RuntimeError):returned[0].result()
        assert not first.done() and r._accepted_parent_count==1
        x.copy_ready.set();x.start();assert first.result(timeout=2)==4096
        x.stop();s=r.parent_admission_snapshot()
        assert s["retired_parents"]==1 and s["peak_accepted_parents"]==1

def test_failure_future_waits_for_other_accepted_file_cqe(monkeypatch):
    with rig(monkeypatch,enabled=False,slots=4,depth=2) as x:
        r=common(x,2);physical_streams(x);r.ring.auto=False
        future=x.submit(files=2);calls=[]
        future.add_done_callback(lambda f:calls.append(f.exception()))
        x.start();eventually(lambda:x.writes==2)
        first=r.ring.pending[0];r.ring.results[first]=4095
        r.ring.permits.release()
        eventually(lambda:r._active[0].failed is not None)
        assert not future.done() and x.handler.get_finished()==[]
        assert r._active[0].inflight_files==1 and r._accepted_parent_count==1
        assert len(r._inflight)==1 and len(r.staging_pool.releases)==1
        r.ring.auto=True
        with pytest.raises(IOError):future.result(timeout=2)
        eventually(lambda:r._accepted_parent_count==0)
        assert len(calls)==1 and len(r.staging_pool.releases)==2
        x.stop()

def test_owner_snapshot_uses_original_incoming_and_contains_real_parent_count(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        r=common(x,2);physical_streams(x);x.copy_ready.clear()
        f=x.submit(files=1);x.start()
        eventually(lambda:len(r._pending_copies)==1)
        with pytest.raises(RuntimeError):r.parent_admission_snapshot()
        s=r.inspect_snapshot(timeout=1)
        assert s["owner_capture"] is True and s["admission"]["accepted_parents"]==1
        assert s["native"]["pending_copies"]==1 and not f.done()
        assert s["stage_accounting"]["stages"]["d2h"]["inflight_ops"]==1
        x.copy_ready.set();assert f.result(timeout=2)==4096
        x.stop();s=r.inspect_snapshot(timeout=1)
        assert s["native_shutdown_read"] is True and s["admission"]["accepted_parents"]==0

@pytest.mark.parametrize("value",[0,-1,True,65,1.5])
def test_invalid_common_limit_rejected_before_resources(monkeypatch,value):
    resources=Mock(side_effect=AssertionError("must validate before resources"))
    monkeypatch.setattr(mod,"DirectIoFileStore",resources)
    with pytest.raises(ValueError):
        IoReactor(config=None,file_mapper=None,layout=None,progress_run_id="r",
                  max_accepted_parents=value)
    resources.assert_not_called()

def test_bound_requires_native_progress_identity(monkeypatch):
    resources=Mock(side_effect=AssertionError("must validate before resources"))
    monkeypatch.setattr(mod,"DirectIoFileStore",resources)
    with pytest.raises(ValueError):
        IoReactor(config=None,file_mapper=None,layout=None,max_accepted_parents=1)
    resources.assert_not_called()

@pytest.mark.parametrize("timeout",[0,-1,True,float("nan"),float("inf")])
def test_snapshot_timeout_is_finite_positive(monkeypatch,timeout):
    with rig(monkeypatch,enabled=False) as x:
        common(x,1)
        with pytest.raises(ValueError):x.r.inspect_snapshot(timeout)

def test_coordinator_forwards_common_cap_without_allocating_new_queue(monkeypatch):
    constructor=Mock(return_value=object())
    monkeypatch.setattr(mod,"IoReactor",constructor)
    c=TransferCoordinator(config=None,file_mapper=None,layout=None,
        progress_run_id="r",max_accepted_parents=3)
    assert c.reactor is constructor.return_value
    assert constructor.call_args.kwargs["max_accepted_parents"]==3

@pytest.mark.parametrize("fail",[False,True])
def test_whole_parent_drain_retires_before_callback_submits_successor(monkeypatch,fail):
    with rig(monkeypatch,enabled=False) as x:
        r=common(x,1);physical_streams(x)
        if fail:r.ring.results[1]=4095
        first=x.submit(files=1,job_id=7);successors=[]
        first.add_done_callback(lambda _:successors.append(x.submit(files=1,job_id=8)))
        x.start()
        if fail:
            with pytest.raises(IOError):first.result(timeout=2)
        else:assert first.result(timeout=2)==4096
        eventually(lambda:len(successors)==1)
        assert successors[0].result(timeout=2)==4096
        x.stop();s=r.parent_admission_snapshot()
        assert s["accepted_parents"]==0 and s["retired_parents"]==2
        assert s["peak_accepted_parents"]==1 and s["backpressure_waits"]==0

def test_duplicate_parent_retirement_is_detected_not_double_capacity(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        r=common(x,1);f=x.submit(files=1);job=r._incoming.queue[0]
        x.start();assert f.result(timeout=2)==4096;x.stop()
        assert job.accepted_parent_retired and r._accepted_parent_count==0
        with pytest.raises(RuntimeError):r._retire_accepted_parent(job)
        assert r._accepted_parent_count==0

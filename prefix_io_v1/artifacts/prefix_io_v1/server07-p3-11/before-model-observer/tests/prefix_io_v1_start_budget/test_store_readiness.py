import threading
from concurrent.futures import Future
from types import SimpleNamespace as NS
import pytest
from prefix_io_control.store_readiness import StoreReadinessProbe
from tests.prefix_io_v1_start_budget._fixture import rig,eventually

def native():
    r=NS(_worker=threading.current_thread(),_prefix_progress=NS(run_id="r"),
        _active=[],_pending_copies=[],_inflight={},_data_inflight=1,iodepth=1,
        staging_pool=NS(free_count=0),is_mandatory=lambda f:True)
    return r

# Weak references need a normal instance (SimpleNamespace is not weak-referenceable).
class Reactor: pass

def fake():
    r=Reactor();r.__dict__.update(native().__dict__);return r

def job(jid=7):
    class NeverQuery:
        def query(self):raise AssertionError("observer must not query CUDA")
    return NS(is_store=True,failed=None,future_set=False,job_id=jid,total_files=3,
        next_file_index=1,inflight_files=1,done_files=0,future=Future(),compute_event=NeverQuery())

def test_bounded_samples_and_no_false_causal_or_release_claims():
    r=fake();j=job();r._active=[j]
    r._pending_copies=[NS(is_store=True,job=j,end_event=j.compute_event)]
    p=StoreReadinessProbe(run_id="r",max_samples=2,interval_ns=1,clock=iter(range(5)).__next__)
    for _ in range(5):p(r)
    assert not p.faulted and p.samples==5 and p.overwritten==3
    row=p.records[-1]["stores"][0]
    assert row["queued_files"]==2 and row["mandatory"]
    assert row["d2h_pending_observed"]==1
    assert row["compute_event_state"]=="not_queried"
    assert row["immediate_reusable_gpu_bytes"] is None
    assert p.counts["device_capacity_full"]==5
    with pytest.raises(RuntimeError):p.export()

def test_truncation_is_visible_and_not_unbounded_scan():
    r=fake();r._active=[job(i) for i in range(80)]
    r._inflight={i:NS(op_kind="write",job=r._active[-1]) for i in range(100)}
    p=StoreReadinessProbe(run_id="r",clock=lambda:0);p(r)
    assert not p.faulted and p.truncated_samples==1
    assert p.records[0]["parents_observed"]==32
    assert len(p.records[0]["stores"])==32
    assert all(x["pending_counts_are_lower_bounds"] for x in p.records[0]["stores"])

def test_sampling_interval_and_clock_regression_invalidate():
    now=[10];r=fake()
    p=StoreReadinessProbe(run_id="r",interval_ns=10,clock=lambda:now[0])
    p(r);now[0]=15;p(r);assert p.samples==1
    now[0]=9;p(r);assert p.faulted

def test_bad_owner_run_and_reuse_disable_only_observation():
    r=fake();p=StoreReadinessProbe(run_id="wrong");p(r);assert p.faulted
    p=StoreReadinessProbe(run_id="r",clock=lambda:0);p(r);p(fake());assert p.faulted
    p=StoreReadinessProbe(run_id="r");r._worker=NS(ident=-1);p(r);assert p.faulted

def test_native_mock_pipeline_reports_pending_then_drains_without_policy(monkeypatch):
    with rig(monkeypatch,enabled=False,slots=2,depth=1) as x:
        p=StoreReadinessProbe(run_id="cpu-run",interval_ns=1)
        x.r._observation_sink=p
        x.copy_ready.clear()
        f=x.submit(files=3)
        x.start()
        eventually(lambda:p.counts["queued_stores"]>0)
        assert not f.done() and x.writes==0
        waiter=x.waiter({7})
        eventually(lambda:p.counts["mandatory_stores"]>0)
        x.copy_ready.set();waiter.join(2)
        assert not waiter.is_alive() and f.result()==3*4096
        x.stop()
        report=p.export()
        assert not report["faulted"] and report["native_scheduling_modified"] is False
        assert report["inferred_gpu_release_bytes"] is None
        assert len(x.r.file_store.finished)==3

def test_probe_failure_does_not_fail_native_completion(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        p=StoreReadinessProbe(run_id="wrong")
        x.r._observation_sink=p
        f=x.submit(files=1);x.start()
        assert f.result(timeout=2)==4096
        x.stop()
        assert p.faulted and x.r.file_store.finished

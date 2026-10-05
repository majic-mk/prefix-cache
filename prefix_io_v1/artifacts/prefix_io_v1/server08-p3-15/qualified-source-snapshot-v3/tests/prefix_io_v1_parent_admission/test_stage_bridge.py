"""Four physical stage controller hooks, fake CUDA/AIO; no performance result."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from py_kvcache import reactor as mod
from py_kvcache.reactor import IoReactor
from prefix_io_control.dispatch_budget import STAGES
from tests.prefix_io_v1_start_budget._fixture import rig,eventually
from ._fixture import controlled

@pytest.mark.parametrize("mode",["shadow","fixed","pressure"])
def test_zero_stage_quota_preserves_mandatory_and_continuations(monkeypatch,mode):
    with rig(monkeypatch,enabled=False) as x:
        a,c=controlled(x,mode=mode)
        first=x.submit(files=1);x.coordinator.request_mandatory([first]);x.start()
        assert first.result(timeout=2)==4096
        second=x.load();x.coordinator.request_mandatory([second])
        assert second.result(timeout=2)==4096
        x.stop();s=c.snapshot(native_shutdown=True)
        assert not s["faulted"] and not s["pending_attempt"]
        for stage in STAGES:
            assert s["observed_api_accepted"][stage]==dict(ops=1,bytes=4096)
            assert a.stats[stage]["accepted_ops"]==1
            assert a.stats[stage]["accepted_bytes"]==4096
        assert a.valid and not a.records
        assert x.r.parent_admission_snapshot()["accepted_parents"]==0
        if mode!="shadow":assert s["performance_override_ops"]>=4

@pytest.mark.parametrize("kind",["store","load"])
def test_ordinary_defer_keeps_native_candidate_indices_fd_and_parent(monkeypatch,kind):
    with rig(monkeypatch,enabled=False) as x:
        a,c=controlled(x,age=1000000)
        f=x.submit(files=1) if kind=="store" else x.load()
        x.start();eventually(lambda:c.denied_performance>0)
        s=x.r.inspect_snapshot(timeout=1)
        assert s["admission"]["accepted_parents"]==1 and not f.done()
        assert not c.pending and not a.records and not x.h2d_launches
        if kind=="store":
            assert x.r._active[0].next_file_index==0 and x.r._active[0].inflight_files==0
            assert x.r.staging_pool.free_count==x.r.staging_pool.slot_count
        else:
            assert len(x.r._ready_fds_load)==1 and x.reads==0
            ready=x.r._ready_fds_load[0];assert ready.fd==10 and ready.sequence>0
        x.coordinator.request_mandatory([f]);assert f.result(timeout=2)==4096
        x.stop();assert not c.waiting

def test_age_store_needs_no_worker_wait_or_external_epoch_grant(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        a,c=controlled(x,age=100,epoch=1_000_000)
        f=x.submit(files=1);x.start();eventually(lambda:c.denied_performance>0)
        assert not f.done() and not x.r._prefix_progress.required
        x.clock[0]=100
        assert f.result(timeout=2)==4096
        x.stop();s=c.snapshot(native_shutdown=True)
        assert s["progress_totals"]["age"]==1
        assert s["epoch_refreshes"]==1 and a.valid

def test_full_parent_limit_requests_progress_even_with_zero_quota(monkeypatch):
    import threading
    with rig(monkeypatch,enabled=False) as x:
        a,c=controlled(x,limit=1,age=1000000)
        one=x.submit(files=1,job_id=7);two=[]
        thread=threading.Thread(target=lambda:two.append(x.submit(files=1,job_id=8)),daemon=True)
        thread.start();eventually(lambda:x.r._parent_waiters==1);x.start()
        assert one.result(timeout=2)==4096;thread.join(timeout=2)
        assert not thread.is_alive() and two
        x.coordinator.request_mandatory(two);assert two[0].result(timeout=2)==4096
        x.stop();s=x.r.parent_admission_snapshot()
        assert s["peak_accepted_parents"]==1 and s["retired_parents"]==2
        assert a.stats["d2h"]["accepted_ops"]==2

def test_shared_ready_read_mandatory_support_and_actual_fusion_bytes(monkeypatch):
    with rig(monkeypatch,enabled=False,share=True) as x:
        a,c=controlled(x,age=1000000);x.r.ring.auto=False
        x.preload([b"shared"]);x.start();eventually(lambda:x.opens==1)
        one=x.load(job_id=20,hashes=[b"shared"]);two=x.load(job_id=21,hashes=[b"shared"])
        x.coordinator.request_mandatory([one,two])
        eventually(lambda:len(x.r._preload_waiters.get(b"shared",()))==2)
        x.r.ring.auto=True
        assert one.result(timeout=2)==two.result(timeout=2)==4096
        x.stop();s=c.snapshot(native_shutdown=True)
        assert x.reads==1 and x.h2d_launches==[dict(entries=2,bytes=8192)]
        assert s["observed_api_accepted"]["ssd_read"]==dict(ops=1,bytes=4096)
        assert s["observed_api_accepted"]["h2d"]==dict(ops=1,bytes=8192)
        assert s["progress_totals"]["mandatory_support"]==1
        assert a.valid and not a.records

def test_copy_quota_charges_actual_mapping_bytes(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        a,c=controlled(x,quota=8)
        original=x.r._fill_swap_ptrs
        def partial(*args):
            count=original(*args);args[5][:count]=1024;return count
        x.r._fill_swap_ptrs=partial
        f=x.load();x.start();assert f.result(timeout=2)==4096
        x.stop();s=c.snapshot(native_shutdown=True)
        assert s["observed_api_accepted"]["h2d"]==dict(ops=1,bytes=1024)
        assert a.stats["h2d"]["accepted_bytes"]==1024

def test_large_clean_registry_is_unknown_without_iterating_it(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        controlled(x)
        class TooLarge(dict):
            def __len__(self):return 65
            def values(self):raise AssertionError("unbounded registry scan")
        x.r._preload_slots=TooLarge()
        assert x.r._prefix_clean_reclaimable_bytes() is None
        # Restore original native registry before fixture cleanup.
        x.r._preload_slots={}

def test_off_controller_hook_does_not_read_clock_or_resources(monkeypatch):
    r=IoReactor.__new__(IoReactor)
    clock=Mock(side_effect=AssertionError("off clock"))
    monkeypatch.setattr(mod.time,"monotonic_ns",clock)
    assert r._prefix_stage_decide("ssd_read",4096,1) is None
    clock.assert_not_called()

@pytest.mark.parametrize("fault",["mapping","start_event"])
def test_pre_backend_d2h_failure_is_per_parent_and_not_partial_dma(monkeypatch,fault):
    with rig(monkeypatch,enabled=False) as x:
        a,c=controlled(x,quota=8)
        error=RuntimeError("before backend")
        if fault=="mapping":x.r._build_mapping=Mock(side_effect=error)
        else:
            original=mod.torch.cuda.Event;calls=[0]
            def event(**kw):
                calls[0]+=1;obj=original(**kw)
                if calls[0]==2:obj.record=Mock(side_effect=error)
                return obj
            monkeypatch.setattr(mod.torch.cuda,"Event",event)
        f=x.submit(files=1);x.start()
        with pytest.raises(RuntimeError) as caught:f.result(timeout=2)
        assert caught.value is error and not x.h2d_launches
        assert len(x.r.staging_pool.releases)==1
        x.stop();s=c.snapshot(native_shutdown=True)
        assert s["uncertain_ops"]==0 and s["observed_api_accepted"]["d2h"]["ops"]==0
        assert not x.r._native_drain_unknown

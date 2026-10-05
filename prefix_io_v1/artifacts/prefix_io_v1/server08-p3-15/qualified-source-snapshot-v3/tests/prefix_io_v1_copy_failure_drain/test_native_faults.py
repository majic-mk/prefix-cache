"""T08/T29 physical-failure contracts with fake CUDA/AIO, no GPU claims."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from py_kvcache import reactor as mod
from py_kvcache.reactor import NativeDrainUnknown
from tests.prefix_io_v1_start_budget._fixture import rig,eventually
from tests.prefix_io_v1_parent_admission._fixture import common,physical_streams,controlled

def inject_copy_fault(monkeypatch,x,kind,is_store,error):
    if kind=="backend":
        monkeypatch.setattr(mod,"ops",NS(swap_blocks_batch=Mock(side_effect=error)))
        return
    original=mod.torch.cuda.Event;count=[0]
    def event(**kw):
        count[0]+=1;obj=original(**kw)
        if count[0]==(3 if is_store else 2):obj.record=Mock(side_effect=error)
        return obj
    monkeypatch.setattr(mod.torch.cuda,"Event",event)

@pytest.mark.parametrize("is_store",[False,True])
@pytest.mark.parametrize("kind",["backend","end_event"])
def test_stream_sync_proof_precedes_slot_release_and_original_error(monkeypatch,is_store,kind):
    with rig(monkeypatch,enabled=False) as x:
        a,c=controlled(x,quota=8)
        r=x.r;error=RuntimeError("ambiguous copy failure")
        inject_copy_fault(monkeypatch,x,kind,is_store,error)
        seen=[]
        def sync():
            seen.append(True)
            assert not future.done()
            assert r._accepted_parent_count==1 and not r.staging_pool.releases
            assert r.staging_pool.free_count==r.staging_pool.slot_count-1
            if not is_store:assert len(r._copy_ready)==1
        if is_store:r._streams[0].synchronize=sync
        else:r._copy_stream.synchronize=sync
        future=x.submit(files=1) if is_store else x.load()
        x.start()
        with pytest.raises(RuntimeError) as caught:future.result(timeout=2)
        assert caught.value is error
        assert seen==[True] and len(r.staging_pool.releases)==1
        eventually(lambda:r._accepted_parent_count==0)
        x.stop();s=c.snapshot(native_shutdown=True);stage="d2h" if is_store else "h2d"
        if kind=="backend":
            assert s["uncertain_ops"]==1 and s["observed_api_accepted"][stage]["ops"]==0
        else:
            assert s["observed_api_accepted"][stage]==dict(ops=1,bytes=4096)
            assert s["completion_unknown_ops"]==1
            assert not s["completion_accounting_complete"]
        assert not s["physical_drain_inferred"]
        assert not a.valid  # Stream proof does not invent transfer-size completion.
        if kind=="end_event":
            assert a.stats[stage]["accepted_ops"]==1
            assert a.stats[stage]["accepted_bytes"]==4096
            assert a.stats[stage]["inflight_ops"]==1
            assert (stage,next(key[1] for key in a.records if key[0]==stage)) in a.records
        else:
            assert a.stats[stage]["accepted_ops"]==0
        assert r.parent_admission_snapshot()["failure_sync_by_stage"][stage]==1

@pytest.mark.parametrize("is_store",[False,True])
@pytest.mark.parametrize("kind",["backend","end_event"])
def test_failed_stream_sync_freezes_all_original_owners_and_futures(monkeypatch,is_store,kind):
    with rig(monkeypatch,enabled=False) as x:
        a,c=controlled(x,quota=8)
        r=x.r;error=RuntimeError("original copy error")
        inject_copy_fault(monkeypatch,x,kind,is_store,error)
        bad=Mock(side_effect=RuntimeError("synchronize did not prove drain"))
        if is_store:r._streams[0].synchronize=bad
        else:r._copy_stream.synchronize=bad
        future=x.submit(files=1) if is_store else x.load();x.start()
        eventually(lambda:r._native_drain_unknown);r._worker.join(timeout=2)
        assert not r._worker.is_alive() and not future.done()
        assert len(r._active)==1 and r._active[0].inflight_files==1
        assert len(x.handler._active)==1 and not r.staging_pool.releases
        assert r.staging_pool.free_count==r.staging_pool.slot_count-1
        if not is_store:assert len(r._copy_ready)==1
        s=r.parent_admission_snapshot()
        assert s["accepted_parents"] is None and s["accepted_count_lower_bound"]==1
        assert not s["count_valid"] and not s["fatal_drain_verified"]
        stage="d2h" if is_store else "h2d"
        assert not a.valid
        if kind=="end_event":
            assert a.stats[stage]["accepted_ops"]==1 and a.stats[stage]["accepted_bytes"]==4096
            assert a.stats[stage]["inflight_ops"]==1
            assert any(key[0]==stage for key in a.records)
        else:assert a.stats[stage]["accepted_ops"]==0
        with pytest.raises(NativeDrainUnknown):x.handler.get_finished()
        with pytest.raises(NativeDrainUnknown):x.handler.wait({7 if is_store else 20})
        with pytest.raises(NativeDrainUnknown):x.handler.shutdown()
        assert x.handler._active and not r.staging_pool._closed and not future.done()

def test_normal_native_async_path_never_synchronizes_stream(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        common(x,2);a,d2h,h2d=physical_streams(x)
        f=x.submit(files=1);x.start();assert f.result(timeout=2)==4096
        g=x.load();assert g.result(timeout=2)==4096
        x.stop();d2h.synchronize.assert_not_called();h2d.synchronize.assert_not_called()
        assert a.valid and not a.records

def test_single_failed_copy_does_not_publish_parent_failure_before_sibling_dma(monkeypatch):
    with rig(monkeypatch,enabled=False,slots=4,depth=2) as x:
        common(x,2);a,stream,_=physical_streams(x);x.copy_ready.clear()
        original=mod.ops.swap_blocks_batch;calls=[0];error=RuntimeError("second copy failed")
        def swap(*args):
            calls[0]+=1
            if calls[0]==2:raise error
            original(*args)
        monkeypatch.setattr(mod.ops,"swap_blocks_batch",swap)
        f=x.submit(files=2);completed=[]
        f.add_done_callback(lambda _:completed.append(True));x.start()
        eventually(lambda:r_failed(x))
        assert not f.done() and len(x.r._pending_copies)==1
        assert x.r._active[0].inflight_files==1 and x.r._accepted_parent_count==1
        assert x.handler.get_finished()==[] and len(x.r.staging_pool.releases)==1
        x.copy_ready.set()
        with pytest.raises(RuntimeError) as caught:f.result(timeout=2)
        assert caught.value is error
        eventually(lambda:x.r._accepted_parent_count==0)
        assert completed==[True] and len(x.r.staging_pool.releases)==2
        x.stop()

def r_failed(x):
    return bool(x.r._active and x.r._active[0].failed is not None)

@pytest.mark.parametrize("prove",[False,True])
def test_partial_aio_enqueue_holds_all_parents_until_original_ring_drained(monkeypatch,prove):
    with rig(monkeypatch,enabled=False,slots=4,depth=2) as x:
        r=common(x,2);physical_streams(x);r.ring.auto=False
        original=r.file_store.queue_write;calls=[0];error=IOError("after queue accepted")
        def write(*args,**kw):
            calls[0]+=1;original(*args,**kw)
            if calls[0]==2:raise error
        r.file_store.queue_write=write
        first=x.submit(files=1,job_id=7);second=x.submit(files=1,job_id=8);seen=[]
        def close():
            seen.append(True)
            assert not first.done() and not second.done()
            assert r._accepted_parent_count==2 and len(r._inflight)==2
            assert not r.staging_pool.releases and not r.file_store.cleaned
            if not prove:raise RuntimeError("kernel teardown failed")
            r.ring.pending.clear()  # Fake close's explicit physical-drain witness.
        r.ring.close=close;x.start()
        eventually(lambda:not r._worker.is_alive())
        assert seen==[True]
        if prove:
            assert first.done() and second.done() and r._native_fatal_drain_verified
            assert r.parent_admission_snapshot()["accepted_parents"]==0
            assert len(r.staging_pool.releases)==2 and not r._inflight
            for f in [first,second]:
                assert isinstance(f.exception(),RuntimeError) and f.exception().__cause__ is error
            r.ring.close=lambda:None
        else:
            assert not first.done() and not second.done() and r._native_drain_unknown
            assert len(r._inflight)==2 and r.ring.pending and not r.staging_pool.releases
            assert len(x.handler._active)==2
            with pytest.raises(NativeDrainUnknown):x.handler.wait({7,8})

def test_partial_read_enqueue_retains_original_fd_and_staging_on_unknown_drain(monkeypatch):
    with rig(monkeypatch,enabled=False) as x:
        r=common(x,2);physical_streams(x)
        original=r.file_store.queue_read;error=IOError("partial read accepted")
        def read(*args,**kw):
            original(*args,**kw);raise error
        r.file_store.queue_read=read
        r.ring.close=Mock(side_effect=RuntimeError("AIO close failed"))
        f=x.load();x.start();eventually(lambda:r._native_drain_unknown)
        r._worker.join(timeout=2)
        assert not f.done() and not r.staging_pool.releases
        assert len(r._inflight)==1
        op=next(iter(r._inflight.values()))
        assert op.fd==10 and op.slot_index>=0 and op.op_kind=="read"
        assert r.ring.pending and r.parent_admission_snapshot()["accepted_count_lower_bound"]==1

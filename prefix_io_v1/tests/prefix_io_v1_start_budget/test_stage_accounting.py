import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
import pytest
from py_kvcache import reactor as mod
from py_kvcache.reactor import IoReactor
from prefix_io_control.stage_accounting import StageAccounting
from ._fixture import rig,eventually

def attach(x):
    a=StageAccounting();a.bind();x.r._prefix_stage_accounting=a
    # Use native D2H launch too; only the CUDA stream/event/kernel are fake.
    stream=NS(wait_event=lambda event:None)
    r=x.r;r._streams=[stream]*r.staging_pool.slot_count
    r._batch_src_ptrs=[np.zeros(4,dtype=np.int64) for _ in r._streams]
    r._batch_dst_ptrs=[np.zeros(4,dtype=np.int64) for _ in r._streams]
    r._batch_sizes=[np.zeros(4,dtype=np.int64) for _ in r._streams]
    r._batch_src_tensors=r._batch_src_ptrs;r._batch_dst_tensors=r._batch_dst_ptrs
    r._batch_size_tensors=r._batch_sizes
    r._launch_swap_blocks=IoReactor._launch_swap_blocks.__get__(r)
    return a

def test_stage_balances_and_short_completion_do_not_infer_release():
    a=StageAccounting();a.bind()
    a.accepted("ssd_read",1,8192);a.accepted("ssd_read",2,4096)
    assert a.snapshot()["stages"]["ssd_read"]["inflight_bytes"]==12288
    a.completed("ssd_read",1,4096);a.completed("ssd_read",2,-5)
    s=a.snapshot();v=s["stages"]["ssd_read"]
    assert v["inflight_bytes"]==0 and v["transferred_bytes"]==4096
    assert v["completed_requested_bytes"]==12288 and v["failed_ops"]==2
    assert s["resource_release_inferred"] is False

def test_native_four_stage_acceptance_and_completion_balances(monkeypatch):
    with rig(monkeypatch,units=4) as x:
        a=attach(x);r=x.r;r.ring.auto=False;x.copy_ready.clear()
        f=x.submit(files=1);x.start()
        eventually(lambda:a.stats["d2h"]["inflight_ops"]==1)
        assert a.stats["ssd_write"]["accepted_ops"]==0 and not f.done()
        x.copy_ready.set();eventually(lambda:a.stats["ssd_write"]["inflight_ops"]==1)
        assert a.stats["d2h"]["inflight_bytes"]==0 and not f.done()
        r.ring.auto=True;assert f.result(timeout=2)==4096
        g=x.load();assert g.result(timeout=2)==4096
        x.stop();s=a.snapshot()
        assert s["valid"] and s["outstanding_records"]==0
        for v in s["stages"].values():
            assert v["accepted_ops"]==v["completed_ops"]==1
            assert v["accepted_bytes"]==v["transferred_bytes"]==4096
            assert v["inflight_bytes"]==v["failed_ops"]==0

def test_shared_preload_one_read_two_consumers_one_actual_fused_copy(monkeypatch):
    with rig(monkeypatch,share=True) as x:
        a=attach(x);x.r.ring.auto=False;x.preload([b"shared"]);x.start()
        eventually(lambda:x.opens==1)
        f=x.load(job_id=20,hashes=[b"shared"]);g=x.load(job_id=21,hashes=[b"shared"])
        eventually(lambda:len(x.r._preload_waiters.get(b"shared",()))==2)
        x.r.ring.auto=True
        assert f.result(timeout=2)==g.result(timeout=2)==4096
        x.stop();s=a.snapshot()
        assert s["valid"] and s["outstanding_records"]==0
        assert s["stages"]["ssd_read"]["accepted_bytes"]==4096
        assert s["stages"]["h2d"]["accepted_ops"]==1
        assert s["stages"]["h2d"]["accepted_bytes"]==8192
        assert s["stages"]["h2d"]["transferred_bytes"]==8192

def test_actual_partial_copy_mapping_not_full_storage_payload(monkeypatch):
    with rig(monkeypatch,units=4) as x:
        a=attach(x)
        # Inject a shorter actual mapping to distinguish bytes from storage payload.
        old=x.r._fill_swap_ptrs
        def partial(*args):
            count=old(*args);args[5][:count]=1024;return count
        x.r._fill_swap_ptrs=partial
        f=x.load();x.start();assert f.result(timeout=2)==4096
        x.stop();s=a.snapshot()
        assert s["valid"] and s["stages"]["ssd_read"]["accepted_bytes"]==4096
        assert s["stages"]["h2d"]["accepted_bytes"]==1024

def test_duplicate_cqe_not_double_accounted(monkeypatch):
    with rig(monkeypatch,units=4) as x:
        a=attach(x);x.r.ring.duplicate=True
        f=x.submit(files=2);x.start();assert f.result(timeout=2)==8192
        x.stop();s=a.snapshot()
        assert s["valid"] and s["stages"]["ssd_write"]["completed_ops"]==2

def test_short_write_is_terminal_io_failure_not_success(monkeypatch):
    with rig(monkeypatch) as x:
        a=attach(x);x.r.ring.results[1]=4095
        f=x.submit(files=1);x.start()
        with pytest.raises(IOError):f.result(timeout=2)
        x.stop();s=a.snapshot();v=s["stages"]["ssd_write"]
        assert s["valid"] and v["failed_ops"]==1 and v["transferred_bytes"]==4095
        assert v["inflight_bytes"]==0 and len(x.r.file_store.cleaned)==1

def test_rejected_write_is_not_counted_as_accepted(monkeypatch):
    with rig(monkeypatch) as x:
        a=attach(x);x.r.file_store.queue_write=Mock(side_effect=IOError("queue rejected"))
        f=x.submit(files=1);x.start()
        with pytest.raises(IOError):f.result(timeout=2)
        x.stop();s=a.snapshot()
        assert s["valid"] and s["stages"]["ssd_write"]["accepted_ops"]==0
        assert s["stages"]["d2h"]["completed_ops"]==1

def test_accounting_fault_does_not_prevent_native_progress(monkeypatch):
    with rig(monkeypatch,units=4) as x:
        a=attach(x);a.accepted=Mock(side_effect=ValueError("injected"))
        f=x.submit(files=2);x.start();assert f.result(timeout=2)==8192
        x.stop();assert not a.valid and a.error.startswith("ValueError")

def test_disabled_copy_accounting_does_not_sum_sizes():
    r=IoReactor.__new__(IoReactor)
    sizes=Mock(side_effect=AssertionError("off must not evaluate copy sizes"))
    r._prefix_stage_copy("h2d",object(),sizes,99)
    sizes.assert_not_called()

@pytest.mark.parametrize("limit",[0,True,4097,-1])
def test_invalid_bound(limit):
    with pytest.raises(ValueError):StageAccounting(limit)

def test_duplicate_unmatched_and_bounded_events_rejected():
    a=StageAccounting(1);a.accepted("h2d",1,3)
    with pytest.raises(ValueError):a.accepted("h2d",1,3)
    with pytest.raises(ValueError):a.accepted("h2d",2,3)
    with pytest.raises(ValueError):a.completed("h2d",2)
    a.completed("h2d",1)
    with pytest.raises(ValueError):a.completed("h2d",1)

def test_owner_and_binding_protection():
    a=StageAccounting();a.bind()
    with pytest.raises(ValueError):a.bind()
    a.accepted("h2d",1,5);a.owner=-1
    with pytest.raises(RuntimeError):a.completed("h2d",1)
    a.invalidate("owner fault")
    s=a.snapshot();assert not s["valid"] and s["stages"]["h2d"]["inflight_bytes"]==5

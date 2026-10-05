from dataclasses import FrozenInstanceError
from threading import get_ident
from types import SimpleNamespace as NS
import pytest
from prefix_io_control.observation import capture_reactor

def fixture(count=2):
    return NS(_worker=NS(ident=get_ident()), actual_staging_bytes=65535,
              staging_pool=NS(free_count=2),
              _active=[NS(job_id=i,total_files=9,done_files=1,inflight_files=2,
                          future_set=False,failed=None) for i in range(count)],
              _inflight={1:NS(op_kind="read"),2:NS(op_kind="write")},
              _pending_copies=[NS(nbytes=4096,is_store=False),NS(nbytes=8192,is_store=True)])

def test_snapshot_bounded_and_unknown_gpu_is_not_zero():
    r = fixture(100)
    s = capture_reactor(r, run_id="r", epoch=2)
    assert len(s.parents) == 32 and s.parents_truncated
    assert s.active_parent_count == 100 and len(r._active) == 100
    assert s.gpu_immediately_reusable_bytes is None
    assert (s.ssd_read_ops,s.ssd_write_ops,s.h2d_inflight_bytes,s.d2h_inflight_bytes)==(1,1,4096,8192)

def test_oversized_io_window_is_unknown():
    r=fixture()
    r._inflight = dict.fromkeys(range(65), NS(op_kind="read"))
    r._pending_copies = [NS(nbytes=1,is_store=False)] * 65
    s=capture_reactor(r,run_id="r",epoch=1)
    assert s.ssd_read_ops is None and s.h2d_inflight_bytes is None

def test_cross_thread_read_is_rejected():
    r=fixture()
    r._worker.ident = -1
    with pytest.raises(RuntimeError, match="owner"):
        capture_reactor(r,run_id="r",epoch=1)

def test_snapshot_is_immutable_and_rejects_stale_or_future_time():
    s=capture_reactor(fixture(),run_id="r",epoch=1)
    with pytest.raises(FrozenInstanceError):
        s.epoch=4
    assert s.fresh(run_id="r",now_ns=s.monotonic_ns+5,max_age_ns=10)
    assert not s.fresh(run_id="other",now_ns=s.monotonic_ns,max_age_ns=10)
    assert not s.fresh(run_id="r",now_ns=s.monotonic_ns+11,max_age_ns=10)
    assert not s.fresh(run_id="r",now_ns=s.monotonic_ns-1,max_age_ns=10)

def test_failure_snapshot_does_not_invent_successful_drain():
    r=fixture(1)
    r._active[0].failed=OSError("fixture")
    r._active[0].future_set=True
    p=capture_reactor(r,run_id="r",epoch=1).parents[0]
    assert p.failed and p.inflight_files==2

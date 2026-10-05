"""Original owner hooks with fake events; no real GPU measurement or approval."""
import threading
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest
import torch
from py_kvcache import reactor as mod
from py_kvcache.reactor import IoReactor
from prefix_io_control.p4_eta import ClosureGeometry
from tests.prefix_io_v1_p4_02_bridge.test_native_events import owner_model,native
from tests.prefix_io_v1_start_budget._fixture import eventually

def test_active_source_is_new_overlay_and_cuda_not_initialized():
    assert "py-kvcache-p4-02-cpu" in mod.__file__
    assert not torch.cuda.is_initialized()

def test_off_terminal_returns_before_clock_or_job_scan(monkeypatch):
    r=IoReactor.__new__(IoReactor)
    class Bomb:
        def __getattribute__(self,name):raise AssertionError("off scanned a job")
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:(_ for _ in ()).throw(AssertionError("off clock")))
    r._prefix_p4_parent_terminal(Bomb(),successful=True)

def test_native_complete_ready_collects_scalar_pending_not_production_eta(monkeypatch):
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:100)
    r=owner_model();view=r._prefix_p4_collect()
    assert r._prefix_p4_bridge.eta_history.snapshot()["pending_count"]==1
    assert all(p.completion_estimate_ns is None for p in view.snapshot.parents)
    assert all(w.estimated_unblock_ns is None for w in view.witnesses)
    assert not r._prefix_p4_bridge.eta_history.production_qualified
    assert not r._active[0].future.done()

def test_repeated_native_collect_does_not_reset_first_ready_time(monkeypatch):
    r=owner_model();clock=iter((100,120,150))
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:next(clock))
    r._prefix_p4_collect();r._prefix_p4_collect()
    j=r._active[0];j.done_files=1;j.inflight_files=0
    r._prefix_p4_parent_terminal(j,successful=True)
    s=r._prefix_p4_bridge.eta_history.snapshot()
    assert s["successful_measurements"]==1
    assert s["recent_measurements"][0]["elapsed_ns"]==50
    assert not j.future.done()  # sampling itself cannot publish callbacks

@pytest.mark.parametrize("condition",["failed","unknown_drain","partial","retired"])
def test_owner_terminal_does_not_count_unproved_success(monkeypatch,condition):
    r=owner_model();clock=iter((100,120))
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:next(clock))
    r._prefix_p4_collect();j=r._active[0];j.done_files=1;j.inflight_files=0
    if condition=="failed":j.failed=RuntimeError("synthetic")
    elif condition=="unknown_drain":r._native_drain_unknown=True
    elif condition=="partial":j.done_files=0;j.inflight_files=1
    else:j.accepted_parent_retired=True
    r._prefix_p4_parent_terminal(j,successful=True)
    assert r._prefix_p4_bridge.eta_history.completed==0
    assert r._prefix_p4_bridge.eta_history.discarded==1
    assert not j.future.done()

def test_wrong_owner_cannot_supply_closure_completion(monkeypatch):
    r=owner_model();monkeypatch.setattr(mod.time,"monotonic_ns",lambda:100)
    r._prefix_p4_collect();r._worker.ident=-1
    r._prefix_p4_parent_terminal(r._active[0],successful=True)
    assert r._prefix_p4_bridge.fault is not None
    assert r._prefix_p4_bridge.eta_history.completed==0
    assert r._prefix_p4_bridge.eta_history.snapshot()["pending_count"]==0
    assert not r._active[0].future.done()

def test_fake_native_pipeline_retains_fusion_and_only_diagnostic_history(monkeypatch):
    with native(monkeypatch,mode="shadow") as rig:
        rig.copy_ready.clear()
        f=rig.load(files=2);rig.start()
        eventually(lambda:bool(rig.h2d_launches))
        # Fixture starts at a frozen zero clock: explicitly advance time before
        # publishing the fake completion. Zero intervals remain rejected.
        assert rig.bridge.eta_history.snapshot()["pending_count"]==1
        rig.clock[0]=1000000
        rig.copy_ready.set()
        f.result(timeout=2);rig.stop()
        assert sum(x["bytes"] for x in rig.h2d_launches)==8192
        snap=rig.bridge.snapshot(native_shutdown=True)
        assert snap["eta_history"]["successful_measurements"]>=1
        assert not snap["production_eta_qualified"]
        assert snap["gpu_release_credit"] is None
        assert not snap["eta_history"]["held_job_or_resource_owners"]
        assert not snap["eta_history"]["new_work_queues"]
        assert len(rig.r.staging_pool.releases)==2

def test_forged_publication_forecast_still_rejected(monkeypatch):
    r=owner_model();monkeypatch.setattr(mod.time,"monotonic_ns",lambda:100)
    view=r._prefix_p4_collect()
    parent=replace(view.snapshot.parents[0],completion_estimate_ns=150)
    forged=replace(view,snapshot=replace(view.snapshot,parents=(parent,)))
    with pytest.raises(ValueError,match="actual owner facts"):
        r._prefix_p4_bridge.publish(forged,view,now_ns=100)

def test_native_pending_metadata_bound_does_not_truncate_64_ready_works(monkeypatch):
    r=owner_model(32,2);monkeypatch.setattr(mod.time,"monotonic_ns",lambda:100)
    view=r._prefix_p4_collect()
    assert len(view.works)==64
    assert r._prefix_p4_bridge.eta_history.snapshot()["pending_count"]==32
    assert len(r._active)==32 and len(r._copy_ready)==64

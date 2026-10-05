"""Real native queues/pump/lifecycle with fake CUDA/file surfaces, CPU only.

These synthetic events prove control/ownership protocols. They neither measure
DMA latency nor establish physical GPU reuse or production interference costs.
"""
import collections,contextlib,threading
from concurrent.futures import Future
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from py_kvcache import reactor as mod
from py_kvcache.reactor import IoReactor,_ReadyCopy,NativeDrainUnknown
from prefix_io_control.p4_bridge import NativeP4Bridge
from prefix_io_control.p4_policy import P4Policy
from prefix_io_control.p4_types import P4Config,IssuePreview
from prefix_io_control.simple_stage_options import parse_simple_options,build_simple_kwargs,PARENT_KEY,POLICY_KEY
from tests.prefix_io_v1_start_budget._fixture import BudgetRig,eventually
from tests.prefix_io_v1_p4_bridge.test_options import fixed_raw

class NativeRig(BudgetRig):
    def __init__(self,monkeypatch,*,mode="dependency_only",units=64,share=False,slots=4):
        super().__init__(monkeypatch,enabled=False,slots=slots,depth=2,share=share)
        r=self.r
        r._init_parent_admission(64)
        r._prefix_init_capacity_observation()
        r._prefix_p4_controls_pending=0
        extra={PARENT_KEY:dict(schema_version=1,run_id="cpu-run",max_accepted_parents=64)}
        if mode=="dependency_only":extra[POLICY_KEY]=fixed_raw(units)
        kwargs=build_simple_kwargs(parse_simple_options(extra))
        r._prefix_stage_accounting=kwargs["stage_accounting"];r._prefix_stage_accounting.bind()
        r._prefix_dispatch_controller=kwargs.get("dispatch_controller")
        if r._prefix_dispatch_controller:r._prefix_dispatch_controller.bind()
        self.bridge=NativeP4Bridge(P4Policy("cpu-run",P4Config(mode,200_000_000,1_000_000_000)))
        self.bridge.bind();r._prefix_p4_bridge=self.bridge
        self.sync_calls=[]
        stream=NS(wait_event=lambda event:None,synchronize=lambda:self.sync_calls.append("sync"))
        r._streams=[stream for _ in range(slots)];r._copy_stream=stream
        r._mapping_tensors=[torch.from_numpy(x) for x in r._mapping_buffers]
        for field in ("src","dst","sizes"):
            arrays=[np.zeros(4,dtype=np.int64) for _ in range(slots)]
            setattr(r,"_batch_"+field+"_ptrs" if field!="sizes" else "_batch_sizes",arrays)
            tensor_name={"src":"_batch_src_tensors","dst":"_batch_dst_tensors","sizes":"_batch_size_tensors"}[field]
            setattr(r,tensor_name,[torch.from_numpy(x) for x in arrays])
        r._launch_swap_blocks=IoReactor._launch_swap_blocks.__get__(r)

@contextlib.contextmanager
def native(monkeypatch,**kwargs):
    rig=NativeRig(monkeypatch,**kwargs)
    try:yield rig
    finally:rig.cleanup()

def assert_accepted_equal(rig):
    r=rig.r;account=r._prefix_stage_accounting.snapshot()
    assert account["valid"] and account["outstanding_records"]==0
    controller=r._prefix_dispatch_controller
    if controller is not None:
        snapshot=controller.snapshot(native_shutdown=True)
        for stage,stats in account["stages"].items():
            assert snapshot["observed_api_accepted"][stage]==dict(ops=stats["accepted_ops"],bytes=stats["accepted_bytes"])
    return account

def test_isolated_source_and_no_cuda_initialized():
    assert "py-kvcache-p4-01-cpu" in mod.__file__
    assert not torch.cuda.is_initialized()

def test_off_returns_before_clock_or_candidate_scan(monkeypatch):
    r=IoReactor.__new__(IoReactor)
    class Bomb:
        def __iter__(self):raise AssertionError("off scanned candidates")
    entries=Bomb()
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:(_ for _ in ()).throw(AssertionError("off clock")))
    assert r._prefix_p4_order(entries,"stores") is entries
    assert r._prefix_p4_collect() is None
    assert r._prefix_stage_decide("d2h",4096,("store",1,0)) is None

@pytest.mark.parametrize("mode",["shadow","interference","joint"])
def test_native_shadow_and_unsupported_cost_modes_complete_with_original_amount(monkeypatch,mode):
    with native(monkeypatch,mode=mode) as rig:
        f=rig.submit(files=2);rig.start();f.result(timeout=2)
        rig.stop()
        account=assert_accepted_equal(rig)
        assert account["stages"]["d2h"]["accepted_bytes"]==8192
        assert account["stages"]["ssd_write"]["accepted_bytes"]==8192
        assert rig.r._prefix_dispatch_controller is None
        snap=rig.bridge.snapshot(native_shutdown=True)
        assert not snap["gpu_qualified"] and snap["gpu_release_credit"] is None
        assert not snap["held_job_or_resource_owners"] and snap["new_work_queues"]==0

def test_zero_quota_mandatory_wait_progresses_without_tick_and_keeps_source_until_event(monkeypatch):
    with native(monkeypatch,units=0) as rig:
        rig.copy_ready.clear();rig.r.ring.auto=False
        f=rig.submit(files=1);rig.start()
        eventually(lambda:bool(rig.r._active))
        assert rig.r._active[0].next_file_index==0 and not rig.h2d_launches
        waiter=rig.waiter([7])
        eventually(lambda:bool(rig.h2d_launches))
        releases_before=list(rig.r.staging_pool.releases)
        held_slot=rig.r._pending_copies[0].slot_index
        assert not f.done() and rig.writes==0 and held_slot not in rig.r.staging_pool._free
        assert rig.clock==[0]
        rig.copy_ready.set()
        eventually(lambda:rig.writes==1)
        assert not f.done() and rig.r.staging_pool.releases==releases_before
        rig.r.ring.auto=True
        waiter.join(timeout=2);assert not waiter.is_alive()
        f.result(timeout=2);rig.stop()
        account=assert_accepted_equal(rig)
        assert account["stages"]["d2h"]["accepted_ops"]==1
        assert account["stages"]["ssd_write"]["accepted_ops"]==1
        assert len(rig.r.staging_pool.releases)==len(releases_before)+1
        assert rig.r._prefix_dispatch_controller.snapshot(native_shutdown=True)["performance_override_ops"]>0

def test_continuation_is_not_split_or_deferred_by_optional_preview(monkeypatch):
    with native(monkeypatch) as rig:
        # Explicit CPU-only seam; production I/J remains closed in its factory.
        def preview(self,work,snapshot,**kwargs):
            return IssuePreview("defer","synthetic_cpu_preview")
        monkeypatch.setattr(P4Policy,"issue_preview",preview)
        f=rig.submit(files=1);rig.start()
        eventually(lambda:bool(rig.r._active))
        assert rig.r._active[0].next_file_index==0 and not rig.h2d_launches
        waiter=rig.waiter([7]);f.result(timeout=2);waiter.join(timeout=2);rig.stop()
        assert rig.writes==1
        assert_accepted_equal(rig)

def test_fused_load_uses_actual_bytes_and_one_common_charge(monkeypatch):
    with native(monkeypatch) as rig:
        rig.copy_ready.clear()
        f=rig.load(files=2);rig.start()
        eventually(lambda:bool(rig.h2d_launches))
        assert sum(x["bytes"] for x in rig.h2d_launches)==8192
        assert not f.done() and not rig.r.staging_pool.releases
        rig.copy_ready.set();f.result(timeout=2);rig.stop()
        account=assert_accepted_equal(rig)
        assert account["stages"]["h2d"]["accepted_bytes"]==8192
        assert len(rig.r.staging_pool.releases)==2

def test_client_future_cancellation_does_not_release_accepted_copy(monkeypatch):
    with native(monkeypatch) as rig:
        rig.copy_ready.clear()
        f=rig.submit(files=1);rig.start()
        eventually(lambda:bool(rig.h2d_launches))
        assert not f.cancel()
        assert not f.done() and not rig.r.staging_pool.releases
        rig.copy_ready.set();f.result(timeout=2);rig.stop()
        assert_accepted_equal(rig)

@pytest.mark.parametrize("known_sync",[True,False])
def test_partial_copy_failure_never_completes_or_reuses_before_sync_proof(monkeypatch,known_sync):
    with native(monkeypatch) as rig:
        real=mod.ops.swap_blocks_batch
        checks=[]
        def partial(src,dst,sizes):
            real(src[:1],dst[:1],sizes[:1])
            raise RuntimeError("synthetic partial API error")
        monkeypatch.setattr(mod.ops,"swap_blocks_batch",partial)
        def sync():
            checks.append((f.done(),list(rig.r.staging_pool.releases)))
            if not known_sync:raise RuntimeError("synthetic missing sync proof")
        for stream in rig.r._streams:stream.synchronize=sync
        f=rig.submit(files=1);rig.start()
        if known_sync:
            with pytest.raises(RuntimeError,match="synthetic partial"):f.result(timeout=2)
            rig.stop()
            assert checks and checks[0]==(False,[])
            assert len(rig.r.staging_pool.releases)==1
            assert not rig.r._prefix_stage_accounting.valid
        else:
            eventually(lambda:getattr(rig.r,"_native_drain_unknown",False))
            assert checks and checks[0]==(False,[])
            assert not f.done() and not rig.r.staging_pool.releases
            assert not rig.r._prefix_stage_accounting.valid
            # Unknown leaves original owners protected through service/context
            # teardown; fixture cleanup must not pretend a physical completion.
            rig.r._worker.join(timeout=2)
            assert not rig.r._worker.is_alive()

def owner_model(count=1,files=1):
    """Synthetic immutable native counters; no GPU tensors/files/worker execute."""
    r=IoReactor.__new__(IoReactor)
    r._worker=NS(ident=threading.get_ident())
    r._prefix_p4_bridge=NativeP4Bridge(P4Policy("cpu-run",P4Config("dependency_only",200,1000)))
    r._prefix_p4_bridge.bind()
    r._active=[];r._ready_fds_load=collections.deque();r._copy_ready=[]
    r._shared_cached={};r._staging_cache=None;r._stop=False
    r._prefix_clean_reclaimable_bytes=lambda:None
    r.is_mandatory=lambda _:False
    r.file_store=NS(io_size=4096);r.layout=NS(storage_block_bytes=4096)
    for pid in range(1,count+1):
        job=NS(accepted_parent_sequence=pid,accepted_parent_retired=False,
            is_store=False,failed=None,future_set=False,total_files=files,done_files=0,
            inflight_files=files,next_file_index=files,future=Future(),
            transfer_size=4096*files,profile=NS(start_ns=0))
        r._active.append(job)
        for fi in range(files):
            slot=(pid-1)*files+fi
            r._prefix_p4_bridge.note_reservation(slot)
            r._copy_ready.append(NS(job=job,file_index=fi,slot_index=slot,
                nbytes=4096,shared=None,cache=None))
    return r

def test_full_window_64_generations_omits_only_optional_restore_witness(monkeypatch):
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:0)
    r=owner_model(32,2)
    view=r._prefix_p4_collect()
    assert len(view.works)==64 and len(view.snapshot.parents)==32
    assert len(view.snapshot.generations)==64 and not view.witnesses
    assert r._prefix_p4_bridge.generation_witness_omissions==32
    assert r._prefix_p4_bridge.fault is None
    assert all(w.minimum_unit_bytes==4096 for w in view.works)

def test_native_work_units_are_stage_specific(monkeypatch):
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:0)
    r=owner_model();job=r._active[0]
    r._copy_ready[0].nbytes=2048
    read=NS(job=job,file_index=1,sequence=1)
    r._ready_fds_load.append(read)
    store=NS(accepted_parent_sequence=2,accepted_parent_retired=False,is_store=True,
        failed=None,future_set=False,total_files=1,done_files=0,inflight_files=0,
        next_file_index=0,future=Future(),profile=NS(start_ns=0),transfer_size=8192)
    r._active.append(store);r.layout.storage_block_bytes=8192
    view=r._prefix_p4_collect()
    units={w.stage:w.minimum_unit_bytes for w in view.works}
    assert units==dict(ssd_read=4096,d2h=8192,h2d=2048)

def test_owner_view_publication_uses_original_incoming_and_is_bounded(monkeypatch):
    with native(monkeypatch) as rig:
        first=rig.r.inspect_p4_view();second=rig.r.inspect_p4_view()
        assert rig.r._prefix_p4_controls_pending==2
        with pytest.raises(RuntimeError,match="bounded"):rig.r.inspect_p4_view()
        rig.start()
        view=first.result(timeout=2);second.result(timeout=2)
        assert view.snapshot.gpu_immediately_reusable_bytes is None
        assert all(w.estimated_unblock_ns is None for w in view.witnesses)
        assert rig.r._prefix_p4_controls_pending==0
        # no costs or jobs yet: exact publication is idempotent, same native owner
        published=rig.r.request_p4_publication(view)
        assert published.result(timeout=2) is True
        assert rig.r.request_p4_publication(view).result(timeout=2) is False
        rig.stop()

def test_preview_uses_time_after_actual_owner_capture(monkeypatch):
    r=owner_model()
    from prefix_io_control.stage_accounting import StageAccounting
    r._prefix_stage_accounting=StageAccounting();r._prefix_stage_accounting.bind()
    r._prefix_dispatch_controller=None
    r.layout.bytes_per_kernel_block=[4096]
    r.accepted_parent_count=lambda:1
    clock=iter(range(100,1000))
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:next(clock))
    seen=[]
    original=P4Policy.issue_preview
    def checked(self,work,snapshot,**kwargs):
        assert snapshot.monotonic_ns<=kwargs["now_ns"]
        assert work.created_ns==r._active[0].profile.start_ns
        seen.append((work.minimum_unit_bytes,kwargs["execution"]))
        return original(self,work,snapshot,**kwargs)
    monkeypatch.setattr(P4Policy,"issue_preview",checked)
    r._prefix_stage_decide("d2h",4096,("store",1,0),r._active[0])
    assert seen==[(4096,"production")] and r._prefix_p4_bridge.fault is None
    assert r._prefix_p4_bridge.last_preview.action=="issue"

@pytest.mark.parametrize("kind",["cache","shared"])
def test_p4_stop_keeps_original_shared_or_cache_consumers_until_event(monkeypatch,kind):
    from py_kvcache.reactor import _SharedPreloadSlot
    from py_kvcache.staging_cache import StagingDataCache
    with native(monkeypatch,share=kind=="shared") as rig:
        r=rig.r
        slot=r.staging_pool.try_reserve()
        if kind=="cache":
            r._staging_cache=StagingDataCache(policy="lru",capacity=4)
            r._staging_cache.put(b"held",slot.index)
            held=r._staging_cache._slots[b"held"]
        else:
            held=_SharedPreloadSlot(b"held",slot.index,None,cached=False)
            r._shared_cache_insert(held)
            owners=("cpu-load-20","cpu-load-21","later-owner")
            r._preload_refcount[b"held"]=len(owners)
            for owner in owners:r._preload_owned[(owner,b"held")]=None
        rig.copy_ready.clear()
        first=rig.load(job_id=20,hashes=[b"held"])
        second=rig.load(job_id=21,hashes=[b"held"])
        rig.start()
        eventually(lambda:held.copies_inflight==2 and bool(r._pending_copies))
        r.shutdown(wait=False);eventually(lambda:r._stop)
        assert not first.done() and not second.done()
        assert held.slot_index not in r.staging_pool._free
        assert r.staging_pool.releases.count(held.slot_index)==0
        rig.copy_ready.set()
        first.result(timeout=2);second.result(timeout=2)
        r._worker.join(timeout=2);assert not r._worker.is_alive()
        assert held.copies_inflight==0 and r.staging_pool.releases.count(held.slot_index)==1
        account=assert_accepted_equal(rig)
        assert account["stages"]["h2d"]["accepted_bytes"]==8192
        assert not rig.bridge.snapshot(native_shutdown=True)["gpu_qualified"]

def test_after_accepted_end_event_failure_preserves_exact_acceptance_and_syncs(monkeypatch):
    with native(monkeypatch) as rig:
        base=mod.torch.cuda.Event
        count=[0]
        class EndFailure:
            def __init__(self,**kwargs):
                self.real=base(**kwargs);self.serial=None
                if kwargs.get("enable_timing"):
                    count[0]+=1;self.serial=count[0]
            def record(self,stream):
                if self.serial==2:raise RuntimeError("synthetic end record")
                return self.real.record(stream)
            def query(self):return self.real.query()
            def elapsed_time(self,other):return self.real.elapsed_time(other)
        monkeypatch.setattr(mod.torch.cuda,"Event",EndFailure)
        syncchecks=[]
        for stream in rig.r._streams:
            stream.synchronize=lambda:syncchecks.append((f.done(),list(rig.r.staging_pool.releases)))
        f=rig.submit(files=1);rig.start()
        with pytest.raises(RuntimeError,match="synthetic end record"):f.result(timeout=2)
        rig.stop()
        assert syncchecks==[(False,[])]
        account=rig.r._prefix_stage_accounting.snapshot()
        assert not account["valid"] and account["stages"]["d2h"]["accepted_bytes"]==4096
        control=rig.r._prefix_dispatch_controller.snapshot(native_shutdown=True)
        assert control["observed_api_accepted"]["d2h"]==dict(ops=1,bytes=4096)
        assert control["completion_unknown_ops"]==1
        assert len(rig.r.staging_pool.releases)==1

def test_native_preview_age_escape_uses_existing_parent_arrival(monkeypatch):
    r=owner_model()
    from prefix_io_control.stage_accounting import StageAccounting
    r._prefix_stage_accounting=StageAccounting();r._prefix_stage_accounting.bind()
    r._prefix_dispatch_controller=None
    r.layout.bytes_per_kernel_block=[4096];r.accepted_parent_count=lambda:1
    clock=iter(range(2000,3000))
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:next(clock))
    r._prefix_stage_decide("d2h",4096,("store",1,0),r._active[0])
    preview=r._prefix_p4_bridge.last_preview
    assert preview.action=="issue" and preview.progress_override
    assert preview.reason=="native_progress_override"
    assert r._active[0].profile.start_ns==0

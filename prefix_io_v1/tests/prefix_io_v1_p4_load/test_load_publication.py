"""Actual scheduler hook functions and native owner state, CPU/fake GPU only."""
import ast, copy, json, threading
from dataclasses import asdict, replace, FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
from py_kvcache import reactor as mod
from py_kvcache.vllm import SharedStorageOffloadingManager, NoopSharedStorageOffloadingHandler
from prefix_io_control.p4_load_observation import (
    SchedulerLoadObservation, SchedulerLoadUnavailable, SchedulerLoadProducer, MAX_REQUESTS,
)
from prefix_io_control.p4_bridge import NativePublication
from prefix_io_control.p4_policy import P4Policy
from prefix_io_control.p4_types import P4Config, BatchChoice
from prefix_io_control.p4_options import P4_KEY,parse_p4_options,build_p4_kwargs
from prefix_io_control.simple_stage_options import parse_simple_options,build_simple_kwargs,PARENT_KEY
from tests.prefix_io_v1_p4_02_bridge.test_native_events import owner_model, native, assert_accepted_equal
from tests.prefix_io_v1_p4_02_bridge.test_options import options, fixed_raw

ROOT=Path(__file__).resolve().parents[2]
AUTHOR=ROOT/"third_party/work/vllm-author-p4-02-cpu/vllm/distributed/kv_transfer/kv_connector/v1/offloading"

def capture(rows=((128,128,1),),*,now=0,producer=None):
    producer=producer or SchedulerLoadProducer("cpu-run")
    output=NS(num_scheduled_tokens={str(i):r[2] for i,r in enumerate(rows)})
    states={str(i):NS(req=NS(num_computed_tokens=r[0],num_prompt_tokens=r[1],is_finished=lambda:False))
            for i,r in enumerate(rows)}
    return producer.capture(output,states,now_ns=now)

def owner(monkeypatch,*,now=0):
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:now)
    r=owner_model()
    r._closed=False;r._submit_lock=threading.Lock()
    r._init_parent_admission(64)
    kw=build_simple_kwargs(parse_simple_options({PARENT_KEY:dict(schema_version=1,run_id="cpu-run",max_accepted_parents=64)}))
    r._prefix_stage_accounting=kw["stage_accounting"];r._prefix_stage_accounting.bind()
    return r

def extracted(filename,cls,name,globals=None):
    """Compile actual frozen author method without importing GPU-backed vLLM."""
    tree=ast.parse((AUTHOR/filename).read_text())
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==cls)
    method=next(x for x in node.body if isinstance(x,ast.FunctionDef) and x.name==name)
    future=ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0)
    module=ast.fix_missing_locations(ast.Module(body=[future,method],type_ignores=[]))
    ns={"logger":NS(warning=lambda *a,**k:None),"OffloadingConnectorMetadata":lambda **k:NS(**k)}
    ns.update(globals or {})
    exec(compile(module,str(AUTHOR/filename),"exec"),ns)
    return ns[name]

def test_actual_scheduled_prefill_decode_and_mixed_work():
    value=capture(((0,32,16),(128,128,1),(30,32,4)))
    assert type(value) is SchedulerLoadObservation
    assert (value.requests,value.prefill_requests,value.decode_requests)==(3,2,2)
    assert (value.prefill_tokens,value.decode_tokens)==(18,3)
    assert not value.production_gpu_state_qualified
    assert len(value.signature)==11

def test_empty_schedule_has_actual_zero_work_but_no_gpu_claim():
    value=capture(())
    assert value.requests==0 and value.total_context_tokens==0
    assert not value.production_gpu_state_qualified

def test_geometry_preserves_distribution_not_just_average():
    a=capture(((100,100,1),(200,200,1)))
    b=capture(((150,150,1),(150,150,1)))
    assert a.total_context_tokens==b.total_context_tokens
    assert a.geometry_sha256!=b.geometry_sha256

def test_source_and_values_are_frozen():
    value=capture()
    with pytest.raises(FrozenInstanceError):value.production_gpu_state_qualified=True
    assert value.source_sha256==__import__("hashlib").sha256(
        Path(__import__("prefix_io_control.p4_load_observation",fromlist=[""]).__file__).read_bytes()).hexdigest()

@pytest.mark.parametrize("rows",[
    ((128,128,True),),((128,128,0),),((-1,128,1),),((128,None,1),),
    ((1_000_001,128,1),),tuple((0,1,1) for _ in range(MAX_REQUESTS+1)),
])
def test_unsupported_scheduler_inputs_are_explicitly_unavailable(rows):
    value=capture(rows)
    assert type(value) is SchedulerLoadUnavailable

def test_missing_owner_request_returns_unknown():
    p=SchedulerLoadProducer("cpu-run")
    value=p.capture(NS(num_scheduled_tokens={"x":1}),{},now_ns=0)
    assert type(value) is SchedulerLoadUnavailable

def test_property_error_is_optional_unknown():
    class Bad:
        @property
        def num_computed_tokens(self):raise RuntimeError("owner unavailable")
    value=SchedulerLoadProducer("cpu-run").capture(
        NS(num_scheduled_tokens={"x":1}),{"x":NS(req=Bad())},now_ns=0)
    assert type(value) is SchedulerLoadUnavailable

def test_off_manager_does_not_read_clock_or_scheduled_values(monkeypatch):
    manager=SharedStorageOffloadingManager(NS())
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:(_ for _ in ()).throw(AssertionError("off clock")))
    class Bomb:
        def __getattr__(self,name):raise AssertionError("off request scan")
    assert manager.capture_prefix_p4_scheduler_load(Bomb(),Bomb()) is None

def test_actual_author_build_metadata_and_worker_relay_use_same_frozen_value(monkeypatch):
    manager=SharedStorageOffloadingManager(NS(),p4_load_run_id="cpu-run")
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:5)
    req=NS(num_computed_tokens=128,num_prompt_tokens=128,is_finished=lambda:False)
    scheduler=NS(manager=manager,_req_status={"r":NS(req=req,transfer_jobs=set())},
        _jobs={},_current_batch_jobs_to_flush=set(),_current_batch_load_jobs={},
        _reqs_to_preload={},_build_store_jobs=lambda out:{})
    fn=extracted("scheduler.py","OffloadingConnectorScheduler","build_connector_meta")
    meta=fn(scheduler,NS(preempted_req_ids=set(),num_scheduled_tokens={"r":1}))
    assert type(meta.prefix_p4_scheduled_load) is SchedulerLoadObservation
    sent=[]
    worker=NS(worker=NS(handle_worker_message=lambda value:sent.append(value)))
    relay=extracted("worker.py","OffloadingConnectorWorker","_publish_prefix_p4_scheduled_load")
    relay(worker,meta);relay(worker,meta)
    assert sent==[meta.prefix_p4_scheduled_load,meta.prefix_p4_scheduled_load]
    tree=ast.parse((AUTHOR/"common.py").read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="OffloadingConnectorMetadata")
    assert any(isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name) and
               x.target.id=="prefix_p4_scheduled_load" for x in cls.body)

@pytest.mark.parametrize("method",["handle_preemptions","start_kv_transfers"])
def test_actual_author_publishes_optional_load_before_transfer_submission(method):
    tree=ast.parse((AUTHOR/"worker.py").read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="OffloadingConnectorWorker")
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name==method)
    assert ast.unparse(fn.body[0]).startswith("self._publish_prefix_p4_scheduled_load(")

def test_native_capture_has_actual_existing_io_and_scheduled_work_only(monkeypatch):
    r=owner(monkeypatch)
    frame=capture()
    assert r.publish_p4_scheduled_load(frame)
    r._prefix_stage("accepted","ssd_read",3,4096)
    view=r._prefix_p4_collect()
    assert view.snapshot.load_signature==frame.signature
    assert view.snapshot.native_state.inflight[0].ops==1
    assert view.snapshot.native_state.inflight[0].nbytes==4096
    assert "scheduled_work_observation" in view.snapshot.capabilities
    assert "production_gpu_load_state" not in view.snapshot.capabilities
    assert all(p.completion_estimate_ns is None for p in view.snapshot.parents)
    assert view.snapshot.gpu_immediately_reusable_bytes is None

def test_load_sequence_and_io_counters_both_invalidate_owner_epoch(monkeypatch):
    r=owner(monkeypatch);p=SchedulerLoadProducer("cpu-run")
    frame=capture(producer=p);r.publish_p4_scheduled_load(frame)
    a=r._prefix_p4_collect()
    r.publish_p4_scheduled_load(capture(producer=p))
    b=r._prefix_p4_collect()
    assert b.snapshot.snapshot_epoch>a.snapshot.snapshot_epoch
    r._prefix_stage("accepted","ssd_read",3,4096)
    c=r._prefix_p4_collect()
    assert c.snapshot.snapshot_epoch>b.snapshot.snapshot_epoch

def test_duplicate_publication_is_idempotent_and_conflict_invalidates(monkeypatch):
    r=owner(monkeypatch);frame=capture()
    assert r.publish_p4_scheduled_load(frame)
    assert not r.publish_p4_scheduled_load(frame)
    with pytest.raises(ValueError):
        r.publish_p4_scheduled_load(replace(frame,geometry_sha256="0"*64))
    assert r._prefix_p4_collect().snapshot.load_signature==()

def test_regressing_foreign_or_mutable_frames_rejected(monkeypatch):
    r=owner(monkeypatch);p=SchedulerLoadProducer("cpu-run")
    old=capture(producer=p);new=capture(producer=p)
    r.publish_p4_scheduled_load(new)
    with pytest.raises(ValueError):r.publish_p4_scheduled_load(old)
    with pytest.raises(ValueError):r.publish_p4_scheduled_load(replace(new,run_id="foreign"))
    with pytest.raises(TypeError):r.publish_p4_scheduled_load(asdict(new))

@pytest.mark.parametrize("now",[201,-1])
def test_stale_or_future_scheduler_load_not_presented_as_current(monkeypatch,now):
    r=owner(monkeypatch,now=0);r.publish_p4_scheduled_load(capture(now=0 if now==201 else 1))
    monkeypatch.setattr(mod.time,"monotonic_ns",lambda:max(0,now))
    view=r._prefix_p4_collect()
    assert view.snapshot.load_signature==()
    assert "scheduled_work_observation" not in view.snapshot.capabilities

def test_unavailable_new_step_clears_previous_valid_load(monkeypatch):
    r=owner(monkeypatch);r.publish_p4_scheduled_load(capture())
    r.publish_p4_scheduled_load(SchedulerLoadUnavailable("cpu-run",2,0,"missing current state"))
    assert r._prefix_p4_collect().snapshot.load_signature==()

def test_external_publication_cannot_forge_actual_load_or_existing_io(monkeypatch):
    r=owner(monkeypatch);r.publish_p4_scheduled_load(capture())
    view=r._prefix_p4_collect()
    forged=NativePublication(replace(view.snapshot,load_signature=("forged",)),view.works,view.witnesses)
    with pytest.raises(ValueError):r._prefix_p4_bridge.publish(forged,view,now_ns=0)

def test_off_load_publish_exits_before_optional_type_validation():
    from py_kvcache.reactor import IoReactor
    r=IoReactor.__new__(IoReactor)
    assert not r.publish_p4_scheduled_load(object())

def test_native_handler_uses_existing_coordinator_without_new_work_queue(monkeypatch):
    r=owner(monkeypatch)
    from py_kvcache.reactor import TransferCoordinator
    c=TransferCoordinator.__new__(TransferCoordinator);c.reactor=r
    h=NoopSharedStorageOffloadingHandler(coordinator=c)
    frame=capture()
    assert h.handle_worker_message(frame)
    assert h.handle_worker_message(frame)
    assert r._prefix_p4_collect().snapshot.load_signature==frame.signature
    assert not h.handle_worker_message({"unrelated":1})
    assert not r._prefix_p4_bridge.snapshot()["held_job_or_resource_owners"]

def test_unqualified_native_batch_cannot_apply_even_a_forged_cpu_choice(monkeypatch):
    r=owner(monkeypatch)
    r._copy_ready=[];job=r._active[0]
    r._ready_fds_load.append(NS(job=job,file_index=0,sequence=1))
    policy=r._prefix_p4_bridge.policy
    calls=[]
    def fake(self,snapshot,works,**kw):
        calls.append(kw)
        return BatchChoice("selected",tuple(w.work_id for w in works),"ssd_read",4096,1,1,
                           "cpu_fixture",True,False)
    monkeypatch.setattr(P4Policy,"choose_batch",fake)
    view=r._prefix_p4_collect()
    assert r._prefix_p4_bridge.batch_prefix(view,view.works,now_ns=0) is None
    assert calls and calls[0]["execution"]=="production"

@pytest.mark.parametrize("mode",["shadow","interference","joint"])
def test_real_native_worker_publication_preserves_original_amount_and_fusion(monkeypatch,mode):
    with native(monkeypatch,mode=mode) as rig:
        p=SchedulerLoadProducer("cpu-run")
        frame=capture(producer=p,now=mod.time.monotonic_ns())
        assert rig.r.publish_p4_scheduled_load(frame)
        f=rig.load(files=2);rig.start();f.result(timeout=2);rig.stop()
        account=assert_accepted_equal(rig)
        assert account["stages"]["ssd_read"]["accepted_bytes"]==8192
        assert account["stages"]["h2d"]["accepted_bytes"]==8192
        assert sum(x["bytes"] for x in rig.h2d_launches)==8192
        assert rig.r._prefix_dispatch_controller is None

def test_same_fixed_common_base_can_be_declared_for_every_factorial_arm():
    for mode in ("shadow","dependency_only","interference","joint"):
        raw=options(mode);raw[P4_KEY]["fixed_stage_policy"]=fixed_raw()
        parsed=parse_p4_options(raw)
        assert parsed.fixed.policy is not None
        kwargs=build_p4_kwargs(parsed)
        assert kwargs["p4_bridge"].declared_common_fixed_base
        assert ("dispatch_controller" in kwargs)==(mode in ("shadow","dependency_only"))
        assert not kwargs["p4_bridge"].policy.production_interference_qualified

@pytest.mark.parametrize("change",[
    dict(prefill_requests=1,prefill_tokens=0),dict(decode_requests=1,decode_tokens=0),
    dict(requests=2,total_context_tokens=256),dict(decode_requests=0,decode_tokens=1),
])
def test_public_values_cannot_claim_inconsistent_request_token_counts(change):
    with pytest.raises(ValueError):replace(capture(),**change)

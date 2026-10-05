"""Fake-event CPU contracts, never real CUDA qualification."""
from dataclasses import replace,asdict
from types import SimpleNamespace as NS
from threading import Thread
import pytest
from prefix_io_control.p4_production_table_contract import TableContext
from prefix_io_control.p4_gpu_step_observation import (
    ForwardObservationProbe,GPUClockReference,capture_prepared_forward,attach_forward_observer)
from prefix_io_control.p4_native_window_journal import NativeWindowJournal
from prefix_io_control.stage_accounting import StageAccounting
from prefix_io_control.dispatch_budget import Amount,ZERO

def context():
    return TableContext("cpu-forward-fixture","a"*64,"GPU-CPU-FIXTURE","b"*64,"c"*64,
        "d"*64,"eager-model-forward","torch-fixture","cuda-fixture","driver-fixture",
        8,128,64,200,"existing_io_plus_delta","paired_residual_margin")

class Event:
    def __init__(self,ms,ready=True,spy=None):self.ms=ms;self.ready=ready;self.spy=spy
    def record(self):
        if self.spy is not None:self.spy.append("event-record")
    def query(self):return self.ready
    def elapsed_time(self,other):return other.ms-self.ms
    def synchronize(self):raise AssertionError("no event wait/sync allowed")

def runner():
    spy=[];r=NS(speculative_config=None,use_async_scheduling=False,_profile_step=1,
        parallel_config=NS(pipeline_parallel_size=1,data_parallel_size=1,tensor_parallel_size=1),
        input_batch=NS(num_reqs=1,req_ids=["real-scalar-id"],num_computed_tokens_cpu=[128],
                       num_prompt_tokens=[128]))
    r._prepare_inputs=lambda *a,**k:spy.append("original-prepare") or "prepare-result"
    r._model_forward=lambda *a,**k:spy.append("original-forward") or object_result
    return r,spy
object_result=object()
def scheduled():
    return NS(num_scheduled_tokens={"real-scalar-id":1},scheduled_spec_decode_tokens={})

def probe(spy=None,*,cross_clock=True,reference_valid=True,fallback=False,max_pending=128):
    events=[Event(1,spy=spy),Event(2,spy=spy)]
    def factory(**kw):
        assert kw=={"enable_timing":True};return events.pop(0)
    reference=GPUClockReference(Event(0),1,reference_valid,fallback,"monotonic_ns_cuda_reference",cross_clock)
    return ForwardObservationProbe(context(),"e"*64,enabled=True,event_factory=factory,
        reference=reference,max_pending=max_pending)

def observed(p=None):
    r,spy=runner();p=p or probe(spy)
    remove=attach_forward_observer(r,p)
    assert r._prepare_inputs(scheduled(),[1])=="prepare-result"
    assert r._model_forward(input_ids="untouched") is object_result
    remove()
    return p,r,spy

def test_original_prepare_forward_return_order_and_exact_source_batch():
    p,r,spy=observed()
    assert spy==["original-prepare","event-record","original-forward","event-record"]
    observations=p.resolve_ready();assert len(observations)==1
    x=observations[0]
    assert x.native_step_ordinal==0 and x.forward_ordinal==1
    assert x.frame.pre_context==128 and x.frame.post_context is None
    assert x.frame.load_signature[4:8]==(1,1,0,128)
    assert x.scope=="model_forward" and not x.full_decode_step
    assert x.output_tokens is None and x.gpu_elapsed_ns==1_000_000
    assert x.clock_domain=="cuda_event_elapsed" and x.clock_domain_valid
    assert not x.gpu_verified and not x.production_qualified
    assert p.last_frame is None and not hasattr(p,"_scheduler")

def test_off_does_not_wrap_or_scan_or_call_events():
    r,spy=runner();oldprepare=r._prepare_inputs;oldforward=r._model_forward
    p=ForwardObservationProbe(context(),"e"*64)
    remove=attach_forward_observer(r,p)
    assert r._prepare_inputs is oldprepare and r._model_forward is oldforward
    r._prepare_inputs(object());assert r._model_forward() is object_result
    assert spy==["original-prepare","original-forward"] and not p.pending
    remove()

@pytest.mark.parametrize("change",["mixed_context","prefill","speculative","async","pipeline","data_parallel",
                                  "tensor_parallel","cohort","bool_count","unknown_prompt","multi_token","no_profile"])
def test_unqualified_prepared_batch_does_not_change_model_calls(change):
    r,spy=runner();s=scheduled()
    if change=="mixed_context":
        r.input_batch=NS(num_reqs=2,req_ids=["real-scalar-id","b"],
            num_computed_tokens_cpu=[128,129],num_prompt_tokens=[128,128])
        s.num_scheduled_tokens["b"]=1
    elif change=="prefill":r.input_batch.num_computed_tokens_cpu=[127]
    elif change=="speculative":r.speculative_config=object()
    elif change=="async":r.use_async_scheduling=True
    elif change in ("pipeline","data_parallel","tensor_parallel"):
        setattr(r.parallel_config,change+"_size",2) if change!="pipeline" else setattr(r.parallel_config,"pipeline_parallel_size",2)
    elif change=="cohort":s.num_scheduled_tokens={"other":1}
    elif change=="bool_count":s.num_scheduled_tokens["real-scalar-id"]=True
    elif change=="unknown_prompt":r.input_batch.num_prompt_tokens=[0]
    elif change=="multi_token":s.num_scheduled_tokens["real-scalar-id"]=2
    elif change=="no_profile":del r._profile_step
    p=probe();remove=attach_forward_observer(r,p)
    r._prepare_inputs(s);assert r._model_forward() is object_result
    assert spy==["original-prepare","original-forward"] and p.pending==[]
    assert p.last_reason!="prepared";remove()

def test_prepared_change_before_actual_forward_is_unknown():
    r,spy=runner();p=probe();remove=attach_forward_observer(r,p)
    r._prepare_inputs(scheduled());r.input_batch.num_computed_tokens_cpu[0]+=1
    assert r._model_forward() is object_result and not p.pending
    assert "changed" in p.last_reason;remove()

def test_pending_tail_is_nonblocking():
    p,_,_=observed();p.pending[0][4].ready=False
    assert p.resolve_ready()==() and len(p.pending)==1
    p.pending[0][4].ready=True;assert p.resolve_ready()[0].clock_domain_valid

@pytest.mark.parametrize("reference_valid,fallback",[(False,False),(True,True)])
def test_failed_gpu_reference_never_falls_back_to_host_cost(reference_valid,fallback):
    p,_,_=observed(probe(reference_valid=reference_valid,fallback=fallback))
    x=p.resolve_ready()[0]
    assert x.gpu_elapsed_ns is None and not x.clock_domain_valid and x.start_ns is None
    assert not x.production_qualified

def test_gpu_elapsed_does_not_imply_cross_clock_native_attribution():
    p,_,_=observed(probe(cross_clock=False));x=p.resolve_ready()[0]
    assert x.gpu_elapsed_ns==1_000_000 and x.clock_domain_valid
    assert x.start_ns is None and x.mapped_wall_clock_domain=="unknown"

def test_event_failure_and_overflow_do_not_drop_original_work():
    r,spy=runner();p=probe(max_pending=1);remove=attach_forward_observer(r,p)
    r._prepare_inputs(scheduled());assert r._model_forward() is object_result
    r._profile_step+=1;r._prepare_inputs(scheduled());assert r._model_forward() is object_result
    assert p.lost==1 and len(p.pending)==1 and spy.count("original-forward")==2
    assert not p.resolve_ready()[0].journal_complete;remove()

def test_original_forward_exception_preserved_and_not_valid_timing():
    r,spy=runner()
    def fail(*a,**k):raise RuntimeError("original-model-fault")
    r._model_forward=fail;p=probe();remove=attach_forward_observer(r,p);r._prepare_inputs(scheduled())
    with pytest.raises(RuntimeError,match="original-model-fault"):r._model_forward()
    x=p.resolve_ready()[0];assert not x.clock_domain_valid and x.gpu_elapsed_ns is None
    remove();assert r._model_forward is fail

class Clock:
    def __init__(self):self.value=0
    def __call__(self):return self.value
def journal(*,max_events=32,enabled=True):
    clock=Clock();j=NativeWindowJournal("cpu-forward-fixture","f"*64,enabled=enabled,
        clock=clock,max_events=max_events);return j,clock
def diagnostic_window():
    p,_,_=observed();w=p.resolve_ready()[0]
    return replace(w,start_ns=10,end_ns=20)

def test_real_stage_accounting_injection_keeps_accept_complete_interface():
    j,c=journal();assert isinstance(j,StageAccounting)
    j.bind();j.publish_owner_frame();c.value=12;j.accepted("h2d",7,8)
    c.value=18;j.completed("h2d",7,8);c.value=30;j.publish_owner_frame()
    events,frames,valid=j.published();assert valid and len(events)==2
    assert frames[-1].accepted[2]==frames[-1].completed[2]==Amount(1,8)
    assert frames[-1].inflight==ZERO and frames[-1].source_sha256=="f"*64
    assert j.snapshot()["outstanding_records"]==0
    a=j.attribute(diagnostic_window(),"h2d",1,8)
    assert a.status=="DIAGNOSTIC_SINGLE_STAGE_ONLY" and a.added_io[2]==Amount(1,8)
    assert not a.production_qualified and not a.gpu_verified

def test_off_native_observer_is_original_accounting_without_clock():
    j,c=journal(enabled=False)
    j.clock=lambda:(_ for _ in ()).throw(AssertionError("off clock"))
    j.accepted("ssd_read",1,8);j.completed("ssd_read",1,8)
    assert j.snapshot()["valid"] and j.snapshot()["stages"]["ssd_read"]["transferred_bytes"]==8
    assert j.published()==((),(),True)

def test_worker_can_only_consume_immutable_frames_not_native_stats():
    j,c=journal();j.publish_owner_frame()
    result=[]
    def worker():
        result.append(j.published())
        try:j.publish_owner_frame()
        except RuntimeError as e:result.append(str(e))
    t=Thread(target=worker);t.start();t.join()
    assert len(result)==2 and "outside reactor owner" in result[1]

@pytest.mark.parametrize("change",["overflow","coverage","cross_clock","wrong_run","fallback","extra_stage","partial",
                                  "wrong_units","d2h","ssd_write","clock_backwards"])
def test_missing_or_ambiguous_native_window_never_becomes_zero_or_cost(change):
    j,c=journal(max_events=1 if change=="overflow" else 32);w=diagnostic_window()
    j.publish_owner_frame();c.value=12;j.accepted("h2d",7,8)
    if change=="extra_stage":c.value=13;j.accepted("ssd_write",8,8)
    c.value=18;j.completed("h2d",7,4 if change=="partial" else 8)
    if change!="coverage":c.value=30;j.publish_owner_frame()
    stage="h2d";units=1
    if change=="cross_clock":w=replace(w,mapped_wall_clock_domain="unknown",start_ns=None)
    elif change=="wrong_run":w=replace(w,frame=replace(w.frame,run_id="other"))
    elif change=="fallback":w=replace(w,fallback_used=True)
    elif change=="wrong_units":units=2
    elif change in ("d2h","ssd_write"):stage=change
    elif change=="clock_backwards":c.value=1;j.publish_owner_frame()
    a=j.attribute(w,stage,units,8)
    assert a.status=="UNKNOWN" and a.added_io is None and a.existing_io is None
    assert not a.production_qualified

def test_explicit_gpu_elapsed_field_survives_dataclass_export():
    p,_,_=observed();x=p.resolve_ready()[0]
    assert asdict(x)["gpu_elapsed_ns"]==1_000_000
    assert "duration_ns" not in asdict(x) and x.duration_ns==x.gpu_elapsed_ns

def test_invalid_reference_object_stays_unknown_without_raising():
    p=probe();p.reference=object();p,_,_=observed(p)
    x=p.resolve_ready()[0]
    assert not x.clock_domain_valid and x.gpu_elapsed_ns is None

@pytest.mark.parametrize("failure",["setup","end","negative_elapsed"])
def test_event_failures_preserve_original_return_and_reject_cost(failure):
    p=probe();r,spy=runner()
    if failure=="setup":
        def fail(**kwargs):raise RuntimeError("event allocation failure")
        p.event_factory=fail
    elif failure=="end":
        events=[Event(1),Event(2)]
        def end_failure():raise RuntimeError("event record failure")
        events[1].record=end_failure
        p.event_factory=lambda **kwargs:events.pop(0)
    else:
        events=[Event(2),Event(1)]
        p.event_factory=lambda **kwargs:events.pop(0)
    remove=attach_forward_observer(r,p)
    r._prepare_inputs(scheduled());assert r._model_forward() is object_result
    result=p.resolve_ready()
    assert spy.count("original-forward")==1
    assert not result or result[0].gpu_elapsed_ns is None
    assert p.lost or not result[0].clock_domain_valid
    remove()

def test_bool_parallel_size_is_unknown_even_when_equal_to_one():
    r,_=runner();r.parallel_config.tensor_parallel_size=True
    with pytest.raises(ValueError,match="distributed"):
        capture_prepared_forward(r,scheduled(),context(),"e"*64)

def test_non_scalar_resource_key_not_retained_by_observer_after_real_completion():
    import weakref,gc
    class Resource:pass
    resource=Resource();ref=weakref.ref(resource);j,c=journal()
    c.value=1;j.accepted("h2d",resource,8)
    c.value=2;j.completed("h2d",resource,8)
    del resource;gc.collect()
    assert ref() is None and not j._active_scalar
    assert j.snapshot()["outstanding_records"]==0
    assert not j.published()[2]

def test_native_gpu_boundary_tie_is_unknown_not_existing_and_added_double_count():
    j,c=journal();j.publish_owner_frame()
    c.value=10;j.accepted("h2d",7,8)
    c.value=18;j.completed("h2d",7,8)
    c.value=30;j.publish_owner_frame()
    a=j.attribute(diagnostic_window(),"h2d",1,8)
    assert a.status=="UNKNOWN" and a.existing_io is None and a.added_io is None
    assert "boundary" in a.reason
    assert j.snapshot()["stages"]["h2d"]["accepted_bytes"]==8

def test_original_invalidate_immediately_invalidates_published_window_preserving_counters():
    j,c=journal();j.publish_owner_frame()
    c.value=12;j.accepted("h2d",7,8)
    c.value=18;j.completed("h2d",7,8)
    c.value=30;j.publish_owner_frame()
    assert j.attribute(diagnostic_window(),"h2d",1,8).status=="DIAGNOSTIC_SINGLE_STAGE_ONLY"
    c.value=40;j.accepted("h2d",8,8)
    j.invalidate("actual native fault")
    assert not j.valid and not j.published()[2]
    assert j.snapshot()["stages"]["h2d"]["inflight_bytes"]==8
    assert j.attribute(diagnostic_window(),"h2d",1,8).status=="UNKNOWN"

def test_host_clock_backwards_keeps_gpu_elapsed_diagnostic_but_refuses_native_mapping(monkeypatch):
    from prefix_io_control import p4_gpu_step_observation as step
    times=iter([100,50]);monkeypatch.setattr(step.time,"monotonic_ns",lambda:next(times))
    p,_,spy=observed();x=p.resolve_ready()[0]
    assert "original-forward" in spy and x.gpu_elapsed_ns==1_000_000
    assert x.clock_domain_valid and not x.host_clock_valid
    assert x.start_ns is None and x.end_ns is None and x.mapped_wall_clock_domain=="unknown"
    j,_=journal()
    assert j.attribute(x,"h2d",1,8).status=="UNKNOWN"
    assert not x.full_decode_step and not x.production_qualified

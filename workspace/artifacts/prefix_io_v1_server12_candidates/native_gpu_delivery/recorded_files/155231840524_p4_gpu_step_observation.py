"""Optional prepared-forward diagnostics, never a model executor or GPU proof.

The adapter reads the actual batch after original _prepare_inputs and brackets
only original _model_forward. It does not claim a full decode step or emitted
tokens. CPU fixtures and native-looking labels remain production-unqualified.
"""
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
import json,time
from .p4_production_table_contract import TableContext

MAX_BATCH=256
def require(ok,msg):
    if not ok:raise ValueError(msg)
def integer(v,name,minimum=0):
    if type(v) is not int and type(v).__module__.split(".")[0]=="numpy" and hasattr(v,"item"):v=v.item()
    require(type(v) is int and v>=minimum,name+" must be explicit CPU integer")
    return v
def digest(v):
    require(type(v) is str and len(v)==64 and all(c in "0123456789abcdef" for c in v),"source SHA required")
def milliseconds(v):
    require(type(v) in (int,float) and isfinite(v) and v>=0,"finite event milliseconds required")
    return round(v*1_000_000)

@dataclass(frozen=True)
class PreparedForward:
    run_id:str
    source_sha256:str
    input_digest:str
    load_signature:tuple
    batch:int
    active_decode:int
    prefill_tokens:int
    pre_context:int
    native_step_ordinal:int
    input_rows:tuple
    post_context:object=None
    output_tokens:object=None
    scope:str="model_forward"
    @property
    def gpu_verified(self):return False
    @property
    def production_qualified(self):return False

def capture_prepared_forward(runner,scheduler_output,context,source_sha256):
    require(type(context) is TableContext,"frozen actual context required");digest(source_sha256)
    require(getattr(runner,"speculative_config",object()) is None,"speculative inputs unknown")
    require(getattr(runner,"use_async_scheduling",None) is False,"async corrections not qualified")
    parallel=runner.parallel_config
    require(all(type(getattr(parallel,name,None)) is int and getattr(parallel,name)==1
                for name in ("pipeline_parallel_size","data_parallel_size","tensor_parallel_size")),
            "distributed forward scope unknown")
    batch=runner.input_batch;count=integer(batch.num_reqs,"actual prepared requests",1)
    require(count<=MAX_BATCH,"bounded actual batch")
    ids=list(batch.req_ids[:count])
    require(len(ids)==count and len(set(ids))==count and all(type(v) is str and 0<len(v)<=1024 for v in ids),
            "actual prepared request identities")
    scheduled=scheduler_output.num_scheduled_tokens
    require(type(scheduled) is dict and set(scheduled)==set(ids),"scheduled/prepared cohort differs")
    require(not getattr(scheduler_output,"scheduled_spec_decode_tokens",None),"speculative token work unknown")
    rows=[]
    for i,rid in enumerate(ids):
        tokens=integer(scheduled[rid],"actual scheduled input tokens",1)
        computed=integer(batch.num_computed_tokens_cpu[i],"actual prepared context")
        prompt=integer(batch.num_prompt_tokens[i],"actual prompt tokens",1)
        require(tokens==1 and computed>=prompt,"only actual prepared pure decode is known")
        rows.append((rid,computed,prompt,tokens))
    require(len({r[1] for r in rows})==1,"mixed context has no exact9 cell")
    pre=rows[0][1]
    signature=(context.model_sha256,context.gpu_uuid,context.kv_layout_sha256,context.kernel_mode,
               count,count,0,pre,context.transfer_quantum_bytes)
    geometry=sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest()
    return PreparedForward(context.run_id,source_sha256,geometry,signature,count,count,0,pre,
        integer(runner._profile_step,"actual profile step after execute increment",1)-1,tuple(rows))

@dataclass(frozen=True)
class GPUClockReference:
    event:object
    anchor_wall_ns:int
    reference_valid:bool
    fallback_used:bool
    clock_domain:str
    cross_clock_qualified:bool=False
    def __post_init__(self):
        integer(self.anchor_wall_ns,"explicit reference anchor",1)
        require(type(self.reference_valid) is bool and type(self.fallback_used) is bool and
                type(self.cross_clock_qualified) is bool,"explicit reference qualification")
        require(self.clock_domain=="monotonic_ns_cuda_reference","explicit CUDA/CPU clock mapping")
    @property
    def production_qualified(self):return False

@dataclass(frozen=True)
class ForwardObservation:
    frame:object
    forward_ordinal:int
    decode_ordinal:object
    start_ns:object
    end_ns:object
    gpu_elapsed_ns:object
    reason:str
    clock_domain_valid:bool
    reference_valid:bool
    fallback_used:bool
    clock_domain:str
    journal_complete:bool
    mapped_wall_clock_domain:str="unknown"
    host_start_ns:object=None
    host_end_ns:object=None
    host_clock_valid:bool=False
    scope:str="model_forward"
    full_decode_step:bool=False
    output_tokens:object=None
    @property
    def duration_ns(self):return self.gpu_elapsed_ns
    @property
    def native_step_ordinal(self):return self.frame.native_step_ordinal if self.frame is not None else None
    @property
    def run_id(self):return self.frame.run_id if self.frame is not None else None
    @property
    def gpu_verified(self):return False
    @property
    def production_qualified(self):return False

class ForwardObservationProbe:
    """Bounded diagnostic event pairs; no synchronization, tensor or work queue."""
    def __init__(self,context,source_sha256,*,enabled=False,event_factory=None,reference=None,
                 max_pending=128,origin="cpu_fixture"):
        require(type(context) is TableContext,"frozen context");digest(source_sha256)
        require(type(enabled) is bool and type(max_pending) is int and 1<=max_pending<=128,"bounded explicit probe")
        require(origin in ("cpu_fixture","native_gpu_diagnostic"),"explicit diagnostic origin")
        self.context=context;self.source_sha256=source_sha256;self.enabled=enabled
        self.event_factory=event_factory;self.reference=reference;self.max_pending=max_pending
        self.origin=origin;self.pending=[];self.forward_ordinal=0;self.decode_ordinal=0
        self.lost=0;self.last_frame=None;self.last_reason="not_prepared"
    @property
    def production_qualified(self):return False
    def prepared(self,runner,scheduler):
        if not self.enabled:return
        try:self.last_frame=capture_prepared_forward(runner,scheduler,self.context,self.source_sha256);self.last_reason="prepared"
        except Exception as exc:self.last_frame=None;self.last_reason=type(exc).__name__+": "+str(exc)

    def call(self,original,runner,*args,**kwargs):
        if not self.enabled:return original(*args,**kwargs)
        self.forward_ordinal+=1;ordinal=self.forward_ordinal;frame=self.last_frame
        # Do not keep the SchedulerOutput (or any request owner) beyond this call.
        self.last_frame=None
        try:
            require(frame is not None,"no known prepared batch")
            batch=runner.input_batch
            require(integer(batch.num_reqs,"actual forward batch",1)==frame.batch and
                    tuple(batch.req_ids[:frame.batch])==tuple(r[0] for r in frame.input_rows),
                    "prepared batch identity changed before actual forward")
            require(tuple((integer(batch.num_computed_tokens_cpu[i],"current context"),
                           integer(batch.num_prompt_tokens[i],"current prompt",1))
                          for i in range(frame.batch))==tuple((r[1],r[2]) for r in frame.input_rows),
                    "prepared context changed before actual forward")
            require(integer(runner._profile_step,"actual forward profile step",1)==frame.native_step_ordinal+1,
                    "original forward step changed")
            ordinal=frame.native_step_ordinal+1
            self.decode_ordinal+=1;decode=self.decode_ordinal
        except Exception as exc:
            self.last_reason=type(exc).__name__+": "+str(exc)
            return original(*args,**kwargs)
        if len(self.pending)>=self.max_pending:
            self.lost+=1;self.last_reason="event observation bound exceeded"
            return original(*args,**kwargs)
        try:
            start=self.event_factory(enable_timing=True);end=self.event_factory(enable_timing=True)
            host_start=time.monotonic_ns();start.record()
        except Exception as exc:
            self.lost+=1;self.last_reason="event setup unknown: "+str(exc)
            return original(*args,**kwargs)
        succeeded=False
        try:
            value=original(*args,**kwargs);succeeded=True;return value
        finally:
            try:
                end.record()
                self.pending.append((frame,ordinal,decode,start,end,succeeded,host_start,time.monotonic_ns()))
            except Exception as exc:self.lost+=1;self.last_reason="event end unknown: "+str(exc)
    def resolve_ready(self):
        """Tail-only, query-based resolution; never synchronize or wait."""
        out=[];remaining=[]
        for frame,ordinal,decode,start,end,succeeded,host_start,host_end in self.pending:
            host_known=(type(host_start) is int and type(host_end) is int and 0<host_start<host_end)
            try:
                if not (start.query() and end.query()):remaining.append((frame,ordinal,decode,start,end,succeeded,host_start,host_end));continue
                duration=milliseconds(start.elapsed_time(end));require(duration>0,"positive GPU event scope")
                reference=self.reference
                require(type(reference) is GPUClockReference and reference.reference_valid and
                        not reference.fallback_used and
                        reference.event.query(),"reference/clock mapping not qualified")
                require(succeeded,"original forward failed")
                first=(reference.anchor_wall_ns+milliseconds(reference.event.elapsed_time(start))
                       if reference.cross_clock_qualified and host_known else None)
                out.append(ForwardObservation(frame,ordinal,decode,first,
                    first+duration if first is not None else None,duration,
                    "FORWARD_DIAGNOSTIC_ONLY" if host_known else "FORWARD_GPU_DIAGNOSTIC_HOST_ORDER_UNKNOWN",
                    True,True,False,"cuda_event_elapsed",self.lost==0,
                    reference.clock_domain if first is not None else "unknown",host_start,host_end,
                    host_clock_valid=host_known))
            except Exception as exc:
                ref=self.reference
                out.append(ForwardObservation(frame,ordinal,decode,None,None,None,str(exc),False,
                    getattr(ref,"reference_valid",False) is True,getattr(ref,"fallback_used",True) is not False,
                    "cuda_event_elapsed",False,"unknown",host_start,host_end,host_clock_valid=host_known))
        self.pending=remaining
        return tuple(out)

def attach_forward_observer(runner,probe):
    """Wrap only existing methods; restore exact original lookup on removal."""
    require(type(probe) is ForwardObservationProbe,"strict optional probe")
    if not probe.enabled:return lambda:None
    old_prepare=runner._prepare_inputs;old_forward=runner._model_forward
    had_prepare="_prepare_inputs" in runner.__dict__;had_forward="_model_forward" in runner.__dict__
    def prepare(*args,**kwargs):
        value=old_prepare(*args,**kwargs)
        if probe.enabled:
            scheduler=args[0] if args else kwargs.get("scheduler_output")
            probe.prepared(runner,scheduler)
        return value
    def forward(*args,**kwargs):return probe.call(old_forward,runner,*args,**kwargs)
    runner._prepare_inputs=prepare;runner._model_forward=forward
    def remove():
        if had_prepare:runner._prepare_inputs=old_prepare
        else:del runner._prepare_inputs
        if had_forward:runner._model_forward=old_forward
        else:del runner._model_forward
    return remove

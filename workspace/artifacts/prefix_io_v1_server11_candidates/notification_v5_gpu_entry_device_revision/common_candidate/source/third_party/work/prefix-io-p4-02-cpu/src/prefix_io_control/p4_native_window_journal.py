"""Bounded owner scalar journal using the original native StageAccounting API.

No worker reads accounting stats, no queue/scheduler/resource owner is retained.
Missing events, ambiguous clocks, and multi-stage store actions are unknown.
"""
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
import json,time
from .stage_accounting import StageAccounting,STAGES
from .dispatch_budget import Amount,ZERO

def require(ok,msg):
    if not ok:raise ValueError(msg)
def integer(v,name,minimum=0):
    require(type(v) is int and v>=minimum,name+" must be explicit integer")
def key_digest(stage,key):
    def scalar(v,depth=0):
        require(depth<4,"bounded native identity")
        if type(v) in (str,int):
            require(type(v) is int or len(v)<=256,"bounded native identity");return v
        if type(v) is tuple and len(v)<=8:return [scalar(x,depth+1) for x in v]
        raise ValueError("resource object is not a scalar native key")
    return sha256(json.dumps([stage,scalar(key)],separators=(",",":")).encode()).hexdigest()

@dataclass(frozen=True)
class NativeStageEvent:
    sequence:int
    operation_sequence:int
    at_ns:int
    kind:str
    stage:str
    physical_bytes:int
    result:object=None

@dataclass(frozen=True)
class NativeOwnerFrame:
    sequence:int
    captured_ns:int
    inflight:tuple
    accepted:tuple
    completed:tuple
    failed_ops:tuple
    valid:bool
    journal_complete:bool
    run_id:str
    source_sha256:str
    clock_domain:str="monotonic_ns"
    @property
    def production_qualified(self):return False

@dataclass(frozen=True)
class NativeWindowAttribution:
    status:str
    reason:str
    existing_io:object=None
    added_io:object=None
    completed_added_io:object=None
    @property
    def production_qualified(self):return False
    @property
    def gpu_verified(self):return False

class NativeWindowJournal(StageAccounting):
    """Inject as the existing stage_accounting instance or owner observation sink."""
    def __init__(self,run_id,source_sha256,*,enabled=False,max_events=1024,clock=time.monotonic_ns):
        super().__init__()
        require(type(enabled) is bool and type(max_events) is int and 1<=max_events<=4096,"bounded observer")
        require(type(run_id) is str and run_id and len(run_id)<=128,"scalar run identity")
        require(type(source_sha256) is str and len(source_sha256)==64 and
                all(c in "0123456789abcdef" for c in source_sha256),"native source SHA")
        self.run_id=run_id;self.source_sha256=source_sha256;self.enabled=enabled
        self.max_events=max_events;self.clock=clock;self.events=[];self.frames=[]
        self.event_sequence=0;self.journal_valid=True;self.lost=0;self.last_reason="not_observed"
        self._active_scalar={};self._journal_lock=RLock();self._last_ns=None
    @property
    def production_qualified(self):return False
    def _unknown(self,reason):
        with self._journal_lock:
            self.journal_valid=False;self.last_reason=str(reason)[:160]
    def invalidate(self,reason):
        value=super().invalidate(reason)
        if self.enabled:self._unknown("original native accounting invalidated: "+str(reason))
        return value
    def _timestamp(self):
        now=self.clock();integer(now,"owner monotonic time")
        require(self._last_ns is None or now>=self._last_ns,"native clock moved backwards")
        self._last_ns=now;return now
    def _frame(self,now):
        self._owner()
        values=lambda ops,byte:tuple(Amount(self.stats[s][ops],self.stats[s][byte]) for s in STAGES)
        frame=NativeOwnerFrame(self.event_sequence,now,values("inflight_ops","inflight_bytes"),
            values("accepted_ops","accepted_bytes"),values("completed_ops","completed_requested_bytes"),
            tuple(self.stats[s]["failed_ops"] for s in STAGES),self.valid,
            self.journal_valid and self.lost==0,self.run_id,self.source_sha256)
        with self._journal_lock:
            if len(self.frames)>=self.max_events:self.lost+=1;self._unknown("owner frame bound exceeded")
            else:self.frames.append(frame)
        return frame
    def publish_owner_frame(self,now_ns=None):
        if not self.enabled:return None
        self._owner()
        try:
            now=self._timestamp() if now_ns is None else now_ns
            integer(now,"owner frame timestamp")
            require(self._last_ns is None or now>=self._last_ns,"owner frame clock moved backwards")
            self._last_ns=now
            return self._frame(now)
        except Exception as exc:self._unknown(exc);return None
    def __call__(self,reactor):
        # The existing observation_sink is called by the original reactor owner.
        # No reactor, queue, Future, event, FD or tensor is stored.
        return self.publish_owner_frame()
    def accepted(self,stage,key,nbytes):
        super().accepted(stage,key,nbytes)
        if not self.enabled:return
        try:
            require(self.valid,"original acceptance accounting invalid")
            ident=key_digest(stage,key);now=self._timestamp();self.event_sequence+=1
            with self._journal_lock:
                if len(self.events)>=self.max_events:self.lost+=1;self._unknown("accepted event bound exceeded")
                else:
                    self._active_scalar[ident]=(self.event_sequence,nbytes)
                    self.events.append(NativeStageEvent(self.event_sequence,self.event_sequence,now,"accepted",stage,nbytes))
            self._frame(now)
        except Exception as exc:self._unknown(exc)  # telemetry cannot change accepted native work
    def completed(self,stage,key,result=None):
        super().completed(stage,key,result)
        if not self.enabled:return
        try:
            require(self.valid,"original completion accounting invalid")
            ident=key_digest(stage,key);now=self._timestamp();self.event_sequence+=1
            with self._journal_lock:
                operation,nbytes=self._active_scalar.pop(ident)
                if len(self.events)>=self.max_events:self.lost+=1;self._unknown("completed event bound exceeded")
                else:self.events.append(NativeStageEvent(self.event_sequence,operation,now,"completed",stage,nbytes,
                    nbytes if result is None else result))
            self._frame(now)
        except Exception as exc:self._unknown(exc)
    def published(self):
        """Cross-thread immutable copy; never consult native accounting stats."""
        with self._journal_lock:
            return tuple(self.events),tuple(self.frames),self.journal_valid and self.lost==0
    def attribute(self,window,stage,units,quantum):
        """Only SSD-read/H2D diagnostics, never a qualified cost cell."""
        def unknown(reason):return NativeWindowAttribution("UNKNOWN",reason)
        if stage not in ("ssd_read","h2d"):return unknown("single-stage D2H/store-write window unproved")
        if type(units) is not int or units not in (1,2,4,8) or type(quantum) is not int or quantum<=0:
            return unknown("unknown finite physical geometry")
        if getattr(window,"scope",None)!="model_forward" or getattr(window,"run_id",None)!=self.run_id:
            return unknown("wrong forward/run scope")
        if (not getattr(window,"reference_valid",False) or getattr(window,"fallback_used",True) or
            not getattr(window,"clock_domain_valid",False) or not getattr(window,"journal_complete",False) or
            getattr(window,"mapped_wall_clock_domain",None)!="monotonic_ns_cuda_reference" or
            type(getattr(window,"start_ns",None)) is not int or type(getattr(window,"end_ns",None)) is not int):
            return unknown("cross-clock/complete GPU window unproved")
        begin,end=window.start_ns,window.end_ns
        if not begin<end:return unknown("invalid forward window")
        events,frames,valid=self.published()
        if not valid:return unknown("missing/overflowed native observations")
        # Independently timed GPU/native boundary ties lack a causal ordering.
        # Preserve the raw events but refuse to count them as existing and added.
        if any(e.at_ns in (begin,end) for e in events):
            return unknown("native operation on GPU window boundary is ambiguous")
        prior=[f for f in frames if f.captured_ns<=begin]
        after=[f for f in frames if f.captured_ns>=end]
        if not prior or not after:return unknown("native owner coverage missing around forward")
        first,last=prior[-1],after[0]
        if not first.valid or not first.journal_complete or not last.valid or not last.journal_complete:
            return unknown("native accounting/owner frame invalid")
        added=[e for e in events if e.kind=="accepted" and begin<=e.at_ns<end]
        if any(e.stage!=stage for e in added):return unknown("multiple physical stages in selected window")
        nbytes=sum(e.physical_bytes for e in added)
        if nbytes!=units*quantum or not added:return unknown("actual stage amount does not match frozen action")
        completed={e.operation_sequence:e for e in events if e.kind=="completed"}
        if any(e.operation_sequence not in completed or
               completed[e.operation_sequence].result!=e.physical_bytes for e in added):
            return unknown("actual accepted added IO did not complete")
        amounts=list(ZERO);amounts[STAGES.index(stage)]=Amount(len(added),nbytes)
        return NativeWindowAttribution("DIAGNOSTIC_SINGLE_STAGE_ONLY","cost/GPU qualification still required",
                                       first.inflight,tuple(amounts),tuple(amounts))

"""Bounded scheduler work values, not a claim that a GPU step is active.

The author scheduler owns the inputs. No request, tensor, job or Future is kept.
The worker republishes one immutable value under the original submit lock.
Qualification of scheduled work against actual GPU step state is separate.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from .simple_stage_policy import _run_id
from .dispatch_budget import integer

MAX_REQUESTS = 256

def _digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("actual source SHA-256 required")

@dataclass(frozen=True)
class SchedulerLoadObservation:
    run_id: str
    sequence: int
    captured_ns: int
    source_sha256: str
    requests: int
    prefill_requests: int
    decode_requests: int
    prefill_tokens: int
    decode_tokens: int
    min_context_tokens: int
    max_context_tokens: int
    total_context_tokens: int
    geometry_sha256: str

    def __post_init__(self):
        _run_id(self.run_id)
        for key in ("sequence","captured_ns","requests","prefill_requests","decode_requests",
                    "prefill_tokens","decode_tokens","min_context_tokens","max_context_tokens",
                    "total_context_tokens"):
            integer(getattr(self,key),key,1 if key == "sequence" else 0)
        _digest(self.source_sha256); _digest(self.geometry_sha256)
        if self.requests > MAX_REQUESTS or self.prefill_requests > self.requests or self.decode_requests > self.requests:
            raise ValueError("bounded scheduled requests required")
        if (not self.requests <= self.prefill_requests+self.decode_requests <= 2*self.requests or
                (self.prefill_requests==0)!=(self.prefill_tokens==0) or
                (self.decode_requests==0)!=(self.decode_tokens==0) or
                self.prefill_tokens < self.prefill_requests or self.decode_tokens < self.decode_requests):
            raise ValueError("scheduled request/token counts disagree")
        if self.min_context_tokens > self.max_context_tokens:
            raise ValueError("context range is reversed")
        if not self.requests and any((self.prefill_requests,self.decode_requests,self.prefill_tokens,
                self.decode_tokens,self.min_context_tokens,self.max_context_tokens,self.total_context_tokens)):
            raise ValueError("empty scheduled work must have zero scalar work")
        if self.requests and (not self.prefill_tokens + self.decode_tokens or
                not self.prefill_requests + self.decode_requests or
                not self.min_context_tokens*self.requests <= self.total_context_tokens <= self.max_context_tokens*self.requests):
            raise ValueError("inconsistent nonempty scheduled work")

    @property
    def signature(self):
        return ("scheduler-work-v1",self.source_sha256,self.requests,self.prefill_requests,
                self.decode_requests,self.prefill_tokens,self.decode_tokens,
                self.min_context_tokens,self.max_context_tokens,self.total_context_tokens,
                self.geometry_sha256)

    @property
    def production_gpu_state_qualified(self):
        return False

    def fresh(self,*,run_id,now_ns,max_age_ns):
        integer(now_ns,"now_ns"); integer(max_age_ns,"max_age_ns",1)
        return self.run_id == run_id and 0 <= now_ns-self.captured_ns <= max_age_ns

@dataclass(frozen=True)
class SchedulerLoadUnavailable:
    run_id: str
    sequence: int
    captured_ns: int
    reason: str
    def __post_init__(self):
        _run_id(self.run_id); integer(self.sequence,"sequence",1);integer(self.captured_ns,"captured_ns")
        if type(self.reason) is not str or not self.reason or len(self.reason)>160:
            raise ValueError("bounded missing-state reason required")

class SchedulerLoadProducer:
    """Only consumes current scheduled inputs; there is no second scheduler."""
    def __init__(self,run_id):
        _run_id(run_id)
        self.run_id=run_id;self.sequence=0
        self.source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()
        self.last_reason="not_observed"

    def capture(self,scheduler_output,req_status,*,now_ns):
        integer(now_ns,"now_ns");self.sequence+=1
        try:
            scheduled=scheduler_output.num_scheduled_tokens
            if type(scheduled) is not dict or len(scheduled)>MAX_REQUESTS or type(req_status) is not dict:
                raise ValueError("unsupported bounded scheduler inputs")
            rows=[];prefill_requests=decode_requests=prefill_tokens=decode_tokens=0
            for req_id,count in scheduled.items():
                if type(req_id) is not str or not req_id or len(req_id)>1024:
                    raise ValueError("unsupported request identity")
                integer(count,"actual scheduled tokens",1)
                status=req_status.get(req_id)
                if status is None:raise ValueError("scheduled request has no current owner state")
                req=status.req
                computed=req.num_computed_tokens
                prompt=req.num_prompt_tokens
                integer(computed,"actual computed tokens");integer(prompt,"actual prompt tokens")
                if count>1_000_000 or computed>1_000_000 or prompt>1_000_000:
                    raise ValueError("scheduled scalar token bound exceeded")
                p=min(count,max(0,prompt-computed));d=count-p
                prefill_tokens+=p;decode_tokens+=d
                prefill_requests+=int(p>0);decode_requests+=int(d>0)
                rows.append((computed,prompt,count,p,d))
            rows.sort()
            contexts=[r[0] for r in rows]
            geometry=sha256(json.dumps(rows,separators=(",",":")).encode()).hexdigest()
            self.last_reason="actual_scheduler_work_not_gpu_active_state"
            return SchedulerLoadObservation(self.run_id,self.sequence,now_ns,self.source_sha256,
                len(rows),prefill_requests,decode_requests,prefill_tokens,decode_tokens,
                min(contexts,default=0),max(contexts,default=0),sum(contexts),geometry)
        except Exception as exc:
            self.last_reason=type(exc).__name__+": "+str(exc)
            return SchedulerLoadUnavailable(self.run_id,self.sequence,now_ns,self.last_reason[:160])

"""Bounded, passive accounting of native backend acceptance and completion.
No queue, scheduling, release-credit inference, or resource ownership lives here.
Backend-accepted includes pending submissions; it is not instantaneous bandwidth.
"""
from threading import get_ident

STAGES=("ssd_read","ssd_write","h2d","d2h")

class StageAccounting:
    def __init__(self,max_records=4096):
        if type(max_records) is not int or not 1<=max_records<=4096:
            raise ValueError("accounting record bound must be 1..4096")
        self.max_records=max_records
        self.bound=False
        self.owner=None
        self.valid=True
        self.error=None
        self.records={}
        self.stats={stage:dict(accepted_ops=0,accepted_bytes=0,completed_ops=0,
            completed_requested_bytes=0,transferred_bytes=0,failed_ops=0,
            inflight_ops=0,inflight_bytes=0,peak_inflight_ops=0,peak_inflight_bytes=0)
            for stage in STAGES}

    def bind(self):
        if self.bound:raise ValueError("accounting already bound to a reactor")
        self.bound=True

    def _owner(self):
        ident=get_ident()
        if self.owner is None:self.owner=ident
        if self.owner!=ident:raise RuntimeError("accounting accessed outside reactor owner")

    def invalidate(self,reason):
        self.valid=False
        self.error=str(reason)[:160]
        # Preserve unresolved counters as evidence, never turn them into completion.
        self.records.clear()

    def accepted(self,stage,key,nbytes):
        self._owner()
        if not self.valid:return
        if stage not in STAGES or type(nbytes) is not int or nbytes<0:
            raise ValueError("invalid stage or bytes")
        ident=(stage,key)
        if ident in self.records:raise ValueError("duplicate native acceptance")
        if len(self.records)>=self.max_records:raise ValueError("bounded accounting exhausted")
        self.records[ident]=nbytes
        s=self.stats[stage]
        s["accepted_ops"]+=1;s["accepted_bytes"]+=nbytes
        s["inflight_ops"]+=1;s["inflight_bytes"]+=nbytes
        s["peak_inflight_ops"]=max(s["peak_inflight_ops"],s["inflight_ops"])
        s["peak_inflight_bytes"]=max(s["peak_inflight_bytes"],s["inflight_bytes"])

    def completed(self,stage,key,result=None):
        self._owner()
        if not self.valid:return
        ident=(stage,key)
        if ident not in self.records:raise ValueError("completion without native acceptance")
        expected=self.records[ident]
        if result is None:result=expected
        if type(result) is not int or result>expected:
            raise ValueError("invalid completed byte count")
        del self.records[ident]
        s=self.stats[stage]
        s["completed_ops"]+=1;s["completed_requested_bytes"]+=expected
        s["transferred_bytes"]+=max(0,result)
        s["failed_ops"]+=int(result!=expected)
        s["inflight_ops"]-=1;s["inflight_bytes"]-=expected

    def snapshot(self):
        # Owner thread or only after native reactor termination.
        return dict(valid=self.valid,error=self.error,bound=self.bound,
            outstanding_records=len(self.records),max_records=self.max_records,
            stages={k:dict(v) for k,v in self.stats.items()},
            acceptance_boundary="native backend API returned successfully; may include pending submission",
            copy_bytes="sum of actual copy mapping sizes, including fused consumers",
            disk_bytes="aligned direct-IO request/result bytes",
            resource_release_inferred=False,byte_caps_enforced=False)

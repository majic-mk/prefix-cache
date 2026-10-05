"""Bounded diagnostic observer of the real native allocator/scheduler.
No allocation, free, fence, completion, queue or model operation is replaced.
Callbacks only observe after original methods (and before allocation for a witness).
"""
from collections import OrderedDict,deque
from functools import wraps
from itertools import islice
from threading import get_ident
from weakref import ref

class NativeOwnerProbe:
    def __init__(self,run_id,block_bytes=917504):
        self.run_id=run_id;self.block_bytes=block_bytes
        self.owner=None;self.pool=None;self.scheduler=None
        self.generations=OrderedDict();self.sequence=0;self.watched={}
        self.records=deque(maxlen=96);self.calls=0;self.errors=0;self.faulted=False
        self.max_reusable_bytes=0;self.actual_safe_reallocations=0
        self.protected_reallocations=0;self.active_reference_observations=0
        self.fenced_free_observations=0;self.restores=[]
    def _owner(self):
        ident=get_ident()
        if self.owner is None:self.owner=ident
        if self.owner!=ident:raise RuntimeError("native owner thread changed")
    def _safe(self,fn,*args):
        if self.faulted:return None
        try:
            self._owner();return fn(*args)
        except Exception:
            self.errors+=1;self.faulted=True;return None
    def _state(self,bid):
        p=self.pool() if self.pool else None;s=self.scheduler() if self.scheduler else None
        record=self.watched.get(bid);gen=self.generations.get(bid)
        if p is None or s is None or record is None or gen!=record["generation"]:return None
        b=p.blocks[bid]
        pending=s._block_id_to_pending_jobs.get(bid,set())
        truncated=len(pending)>32
        pending_ids=tuple(islice(pending,32))
        linked=(b.prev_free_block is not None and b.next_free_block is not None and
            b.prev_free_block.next_free_block is b and b.next_free_block.prev_free_block is b)
        all_acked=bool(record["parents"]) and record["parents"]<=record["acked"]
        reusable=(not b.is_null and b.ref_cnt==0 and linked and not pending and
            not truncated and all_acked)
        return dict(run_id=self.run_id,pool="native_gpu",group=0,block=bid,generation=gen,
            bytes=self.block_bytes,active_refs=b.ref_cnt,free_queue_linked=linked,
            protecting_jobs=list(pending_ids),parents_truncated=truncated,
            observed_parent_jobs=sorted(record["parents"]),native_completed_jobs=sorted(record["acked"]),
            native_reusable=reusable)
    def _capture(self,phase):
        self.calls+=1
        states=[x for bid in self.watched if (x:=self._state(bid)) is not None]
        reusable=sum(x["bytes"] for x in states if x["native_reusable"])
        self.max_reusable_bytes=max(self.max_reusable_bytes,reusable)
        self.active_reference_observations+=sum(x["active_refs"]>0 for x in states)
        self.fenced_free_observations+=sum(x["active_refs"]==0 and bool(x["protecting_jobs"]) for x in states)
        self.records.append(dict(phase=phase,owner_thread=self.owner,resources=states))
    def _allocated(self,pool,blocks,before):
        self.pool=ref(pool)
        if len(blocks)>64:raise RuntimeError("allocation observation bound exceeded")
        for b in blocks:
            old=before.get(b.block_id)
            if old and old["native_reusable"]:
                self.actual_safe_reallocations+=1
                self.records.append(dict(phase="actual_native_reallocation",prior=old,new_ref_cnt=b.ref_cnt))
            elif old and old["protecting_jobs"]:self.protected_reallocations+=1
            self.sequence+=1;self.generations[b.block_id]=self.sequence
            self.generations.move_to_end(b.block_id)
            while len(self.generations)>64:
                victim=next(k for k in self.generations if k not in self.watched)
                self.generations.pop(victim)
            if b.block_id in self.watched:
                self.watched[b.block_id]=dict(generation=self.sequence,parents=set(),acked=set())
        self._capture("allocated")
    def _stores(self,sched,jobs):
        self.scheduler=ref(sched)
        for jid in islice(jobs,4):
            status=sched._jobs.get(jid)
            if status is None or not status.is_store:continue
            for bid in islice(status.non_sliding_window_block_ids or (),8):
                gen=self.generations.get(bid)
                if gen is None:continue
                if bid not in self.watched:
                    if len(self.watched)>=8:continue
                    self.watched[bid]=dict(generation=gen,parents=set(),acked=set())
                record=self.watched[bid]
                if len(record["parents"])>=32:
                    raise RuntimeError("parent closure exceeded observation bound")
                record["parents"].add(jid)
        self._capture("store_jobs_built")
    def _completed(self,sched,output):
        self.scheduler=ref(sched)
        meta=getattr(output,"kv_connector_worker_meta",None)
        done=getattr(meta,"completed_jobs",{}) or {}
        for record in self.watched.values():
            for jid in record["parents"]:
                # Original update_connector_output has already consumed all
                # worker acks and removed the parent; no manual complete_store.
                if done.get(jid,0)>0 and jid not in sched._jobs:record["acked"].add(jid)
        self._capture("native_connector_output_applied")
    def install(self):
        from vllm.v1.core.block_pool import BlockPool
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler as Scheduler
        probe=self
        def wrap(cls,name,maker):
            original=getattr(cls,name);wrapped=maker(original)
            setattr(cls,name,wrapped);self.restores.append((cls,name,original,wrapped))
        def allocation(original):
            @wraps(original)
            def method(pool,*args,**kwargs):
                before=probe._safe(lambda:{bid:x for bid in probe.watched if (x:=probe._state(bid)) is not None}) or {}
                result=original(pool,*args,**kwargs)
                probe._safe(probe._allocated,pool,result,before)
                return result
            return method
        def free(original):
            @wraps(original)
            def method(pool,*args,**kwargs):
                result=original(pool,*args,**kwargs)
                probe._safe(probe._capture,"native_free_blocks_applied")
                return result
            return method
        def store(original):
            @wraps(original)
            def method(sched,*args,**kwargs):
                result=original(sched,*args,**kwargs)
                probe._safe(probe._stores,sched,result)
                return result
            return method
        def complete(original):
            @wraps(original)
            def method(sched,output):
                result=original(sched,output)
                probe._safe(probe._completed,sched,output)
                return result
            return method
        def finished(original):
            @wraps(original)
            def method(sched,*args,**kwargs):
                result=original(sched,*args,**kwargs)
                probe._safe(probe._capture,"request_finished_fences_registered")
                return result
            return method
        wrap(BlockPool,"get_new_blocks",allocation);wrap(BlockPool,"free_blocks",free)
        wrap(Scheduler,"_build_store_jobs",store);wrap(Scheduler,"update_connector_output",complete)
        wrap(Scheduler,"request_finished",finished)
    def export(self):
        return dict(run_id=self.run_id,owner_thread=self.owner,calls=self.calls,errors=self.errors,
            faulted=self.faulted,tracked_generations=len(self.generations),watched_resources=len(self.watched),
            max_reusable_bytes=self.max_reusable_bytes,actual_safe_reallocations=self.actual_safe_reallocations,
            protected_reallocations=self.protected_reallocations,
            active_reference_observations=self.active_reference_observations,
            fenced_free_observations=self.fenced_free_observations,records=list(self.records),
            scope="bounded same-owner diagnostic witness; not a general live controller adapter")
    def uninstall(self):
        for cls,name,original,wrapped in reversed(self.restores):
            if getattr(cls,name) is wrapped:setattr(cls,name,original)
        self.restores.clear()

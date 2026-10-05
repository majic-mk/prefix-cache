"""Opt-in, bounded diagnostic of native flush causes and allocation generations.
No dispatch, fence, completion, or model decision is made by this observer.
Installation is explicit in the controlled experiment; ordinary paths import none
of this module. Unknown ownership stays unknown and earns no release credit.
"""
from collections import OrderedDict, deque
from functools import wraps
from itertools import islice
from threading import get_ident
from time import monotonic
from weakref import ref

REASONS = frozenset(("restore_destination", "new_allocation", "preempted_request",
                     "all_requests_finished", "cache_reset"))

class NativeFlushProbe:
    MAX_BLOCKS = 4096
    MAX_JOBS = 64
    SAMPLE_BLOCKS = 8
    MAX_PARENTS = 32
    MAX_RECORDS = 128

    def __init__(self, run_id):
        self.run_id = run_id
        self.owner = None
        self.pool = None
        self.generations = [None] * self.MAX_BLOCKS
        self.sequence = 0
        self.jobs = OrderedDict()
        self.records = deque(maxlen=self.MAX_RECORDS)
        self.causes = []
        self.batch = 0
        self.errors = 0
        self.faulted = False
        self.dropped_records = 0
        self.dropped_jobs = 0
        self.dropped_causes = 0
        self.allocations = 0
        self.retired_jobs = 0
        self.restores = []

    def safe(self, fn, *args):
        if self.faulted:
            return None
        try:
            ident = get_ident()
            if self.owner is None:
                self.owner = ident
            if self.owner != ident:
                raise RuntimeError("native owner thread changed")
            return fn(*args)
        except Exception:
            self.errors += 1
            self.faulted = True
            return None

    def append(self, record):
        if len(self.records) == self.MAX_RECORDS:
            self.dropped_records += 1
        self.records.append(dict(record, monotonic=monotonic()))

    def allocated(self, pool, blocks):
        if len(pool.blocks) > self.MAX_BLOCKS or len(blocks) > self.MAX_BLOCKS:
            raise RuntimeError("native pool exceeds bounded diagnostic domain")
        if self.pool is not None and self.pool() is not pool:
            raise RuntimeError("multiple native pools unsupported")
        self.pool = ref(pool)
        for block in blocks:
            bid = block.block_id
            if not 0 <= bid < self.MAX_BLOCKS:
                raise RuntimeError("invalid native block ID")
            self.sequence += 1
            self.generations[bid] = self.sequence
        self.allocations += len(blocks)

    def state(self, scheduler, bid, source_generation):
        pool = self.pool() if self.pool else None
        if pool is None or not 0 <= bid < min(len(pool.blocks), self.MAX_BLOCKS):
            return dict(block=bid, source_generation=source_generation, owner_known=False)
        block = pool.blocks[bid]
        pending = scheduler._block_id_to_pending_jobs.get(bid, ())
        generation = self.generations[bid]
        linked = (block.prev_free_block is not None and block.next_free_block is not None
                  and block.prev_free_block.next_free_block is block
                  and block.next_free_block.prev_free_block is block)
        return dict(block=bid, owner_known=True, generation=generation,
                    source_generation=source_generation,
                    same_generation=(generation == source_generation
                                     if generation is not None and source_generation is not None else None),
                    active_refs=block.ref_cnt, is_null=block.is_null,
                    free_queue_linked=linked, protecting_jobs=list(islice(pending, self.MAX_PARENTS)),
                    protecting_jobs_count=len(pending),
                    parents_truncated=len(pending) > self.MAX_PARENTS,
                    immediately_reusable_bytes=None)

    def stores(self, scheduler, jobs):
        for jid in islice(jobs, self.MAX_JOBS):
            status = scheduler._jobs[jid]
            ids = status.non_sliding_window_block_ids or ()
            # First/last samples retain tiny suffix writes and both ends of long stores.
            sample = list(ids[:4]) + list(ids[-4:]) if len(ids) > 8 else list(ids)
            self.jobs[jid] = dict(req_id=status.req_id, source_count=len(ids),
                                 samples={bid:self.generations[bid] for bid in sample},
                                 source_truncated=len(ids) > len(sample))
            while len(self.jobs) > self.MAX_JOBS:
                self.jobs.popitem(last=False)
                self.dropped_jobs += 1
        self.dropped_jobs += max(0, len(jobs) - self.MAX_JOBS)

    def flush(self, scheduler, reason, req_id=None, block_ids=None, job_ids=None):
        if reason not in REASONS:
            raise RuntimeError("unrecognized flush reason")
        if block_ids is not None:
            if len(block_ids) > self.MAX_BLOCKS:
                raise RuntimeError("flush allocation exceeds diagnostic domain")
            selected = set()
            parents_truncated = False
            conflicts = []
            for bid in block_ids:
                parents = scheduler._block_id_to_pending_jobs.get(bid, ())
                if parents:
                    if len(conflicts) < self.SAMPLE_BLOCKS:
                        conflicts.append(dict(block=bid, parent_ids=list(islice(parents, self.MAX_PARENTS)),
                                              parents_truncated=len(parents) > self.MAX_PARENTS))
                    parents_truncated |= len(parents) > self.MAX_PARENTS
                    for jid in islice(parents, self.MAX_PARENTS):
                        if jid in selected or len(selected) < self.MAX_PARENTS:
                            selected.add(jid)
                        else:
                            parents_truncated = True
            ids = list(selected)
            # Native complete flush set is separately recorded by seal().
        else:
            ids = list(islice(job_ids, self.MAX_PARENTS))
            parents_truncated = len(job_ids) > self.MAX_PARENTS
            conflicts = []
        parents = []
        for jid in ids:
            status = scheduler._jobs.get(jid)
            sample = self.jobs.get(jid)
            parents.append(dict(job_id=jid, req_id=status.req_id if status else None,
                pending_worker_acks=status.pending_count if status else None,
                is_store=status.is_store if status else None,
                source_sample_known=sample is not None,
                source_count=sample["source_count"] if sample else None,
                source_truncated=sample["source_truncated"] if sample else None,
                source_blocks=[self.state(scheduler,bid,g) for bid,g in sample["samples"].items()] if sample else []))
        event = dict(reason=reason, trigger_req_id=req_id, parent_ids=ids,
                     parents_truncated=parents_truncated, parents=parents,
                     conflict_blocks=conflicts,
                     conflict_blocks_are_sample=block_ids is not None,
                     owner_thread=self.owner)
        if len(self.causes) < 16:
            self.causes.append(event)
        else:
            self.dropped_causes += 1

    def seal(self, metadata):
        if metadata.jobs_to_flush:
            self.batch += 1
            ids = metadata.jobs_to_flush
            self.append(dict(kind="flush_batch", batch=self.batch,
                             parent_ids=list(islice(ids, self.MAX_PARENTS)),
                             parent_count=len(ids), parents_truncated=len(ids) > self.MAX_PARENTS,
                             causes=self.causes))
        elif self.causes:
            raise RuntimeError("causes without native flush metadata")
        self.causes = []

    def completed(self, scheduler, output):
        meta = getattr(output, "kv_connector_worker_meta", None)
        done = getattr(meta, "completed_jobs", {}) or {}
        for jid in list(self.jobs):
            if done.get(jid, 0) > 0 and jid not in scheduler._jobs:
                sample = self.jobs.pop(jid)
                self.retired_jobs += 1
                self.append(dict(kind="native_parent_retired", job_id=jid, req_id=sample["req_id"],
                    source_count=sample["source_count"], source_truncated=sample["source_truncated"],
                    source_blocks=[self.state(scheduler,bid,g) for bid,g in sample["samples"].items()]))

    def install(self):
        from vllm.v1.core.block_pool import BlockPool
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler as Scheduler
        if getattr(Scheduler, "_prefix_flush_probe", None) is not None:
            raise RuntimeError("flush diagnostic already installed")
        probe = self
        def wrap(cls, name, maker):
            old = getattr(cls, name)
            new = maker(old)
            setattr(cls, name, new)
            self.restores.append((cls, name, old, new))
        def allocation(old):
            @wraps(old)
            def fn(pool, *args, **kwargs):
                result = old(pool, *args, **kwargs)
                probe.safe(probe.allocated, pool, result)
                return result
            return fn
        def stores(old):
            @wraps(old)
            def fn(scheduler, *args, **kwargs):
                result = old(scheduler, *args, **kwargs)
                probe.safe(probe.stores, scheduler, result)
                return result
            return fn
        def seal(old):
            @wraps(old)
            def fn(scheduler, *args, **kwargs):
                result = old(scheduler, *args, **kwargs)
                probe.safe(probe.seal, result)
                return result
            return fn
        def complete(old):
            @wraps(old)
            def fn(scheduler, output):
                result = old(scheduler, output)
                probe.safe(probe.completed, scheduler, output)
                return result
            return fn
        wrap(BlockPool, "get_new_blocks", allocation)
        wrap(Scheduler, "_build_store_jobs", stores)
        wrap(Scheduler, "build_connector_meta", seal)
        wrap(Scheduler, "update_connector_output", complete)
        Scheduler._prefix_flush_probe = self
        self.restores.append((Scheduler, "_prefix_flush_probe", None, self))

    def export(self):
        return dict(run_id=self.run_id, owner_thread=self.owner, errors=self.errors,
                    faulted=self.faulted, allocations=self.allocations, retired_jobs=self.retired_jobs,
                    flush_batches=self.batch, dropped_records=self.dropped_records,
                    dropped_jobs=self.dropped_jobs, dropped_causes=self.dropped_causes,
                    max_pool_blocks=self.MAX_BLOCKS, max_jobs=self.MAX_JOBS,
                    source_samples_per_job=self.SAMPLE_BLOCKS, tracked_jobs=len(self.jobs),
                    records=list(self.records), immediately_reusable_bytes=None,
                    scope="Bounded sampled native evidence only; no production release credit or controller")

    def uninstall(self):
        for cls, name, old, new in reversed(self.restores):
            if getattr(cls, name) is new:
                setattr(cls, name, old)
        self.restores.clear()

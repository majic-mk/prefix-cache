"""Bounded passive store readiness samples through the existing observer hook.

States describe a post-pump instant, never a proven scheduling denial, a CUDA
execution interval, or immediately reusable GPU capacity. No CUDA event queries.
"""
from collections import deque
from itertools import islice
from threading import get_ident
from time import monotonic_ns
from weakref import ref

class StoreReadinessProbe:
    def __init__(self, *, run_id, interval_ns=1_000_000, max_samples=96, clock=monotonic_ns):
        if type(run_id) is not str or not run_id.strip() or len(run_id) > 128:
            raise ValueError("bounded nonempty run identity required")
        for name, value, bound in (("interval_ns", interval_ns, None), ("max_samples", max_samples, 96)):
            if type(value) is not int or value < 1 or (bound is not None and value > bound):
                raise ValueError("invalid " + name)
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.run_id = run_id
        self.interval_ns = interval_ns
        self.records = deque(maxlen=max_samples)
        self.clock = clock
        self.owner = None
        self.last_now = None
        self.next_due = None
        self.samples = self.overwritten = self.truncated_samples = 0
        self.faulted = False
        self.error = None
        self.counts = dict(queued_stores=0, mandatory_stores=0, device_capacity_full=0,
                           no_free_staging=0, d2h_pending=0, write_pending=0)

    def __call__(self, reactor):
        if self.faulted:
            return
        try:
            if reactor._worker.ident != get_ident():
                raise RuntimeError("readiness sampling requires the reactor owner")
            if self.owner is None:
                self.owner = ref(reactor)
            elif self.owner() is not reactor:
                raise RuntimeError("probe cannot span reactors")
            if reactor._prefix_progress.run_id != self.run_id:
                raise ValueError("readiness run identity mismatch")
            now = self.clock()
            if type(now) is not int or now < 0 or (self.last_now is not None and now < self.last_now):
                raise ValueError("invalid monotonic observation clock")
            self.last_now = now
            if self.next_due is not None and now < self.next_due:
                return
            self.next_due = now + self.interval_ns
            jobs = list(islice(reactor._active, 32))
            copies = list(islice(reactor._pending_copies, 64))
            io = list(islice(reactor._inflight.values(), 64))
            truncated = (len(reactor._active) > 32 or len(reactor._pending_copies) > 64
                         or len(reactor._inflight) > 64)
            stores = []
            full = reactor._data_inflight >= reactor.iodepth
            free = reactor.staging_pool.free_count
            preceding = 0
            for job in jobs:
                if not job.is_store or job.failed is not None or job.future_set:
                    continue
                queued = max(0, job.total_files - job.next_file_index)
                mandatory = reactor.is_mandatory(job.future)
                copy_count = sum(c.is_store and c.job is job for c in copies)
                write_count = sum(op.op_kind == "write" and op.job is job for op in io)
                stores.append(dict(parent_id=job.job_id, queued_files=queued,
                    next_file_index=job.next_file_index, total_files=job.total_files,
                    inflight_files=job.inflight_files, done_files=job.done_files,
                    mandatory=mandatory, preceding_queued_store_parents=preceding,
                    d2h_pending_observed=copy_count, ssd_write_pending_observed=write_count,
                    pending_counts_are_lower_bounds=truncated,
                    compute_event_state="not_queried", immediate_reusable_gpu_bytes=None))
                preceding += int(queued > 0)
                self.counts["queued_stores"] += int(queued > 0)
                self.counts["mandatory_stores"] += int(mandatory)
                self.counts["device_capacity_full"] += int(queued > 0 and full)
                self.counts["no_free_staging"] += int(queued > 0 and free == 0)
                self.counts["d2h_pending"] += copy_count
                self.counts["write_pending"] += write_count
            self.samples += 1
            self.truncated_samples += int(truncated)
            if len(self.records) == self.records.maxlen:
                self.overwritten += 1
            self.records.append(dict(time_ns=now, parents_observed=len(jobs),
                active_parents=len(reactor._active), truncated=truncated,
                data_inflight=reactor._data_inflight, iodepth=reactor.iodepth,
                free_staging_slots=free, reclaimable_staging_slots=None, stores=stores))
        except Exception as exc:
            self.faulted = True
            self.error = (type(exc).__name__ + ": " + str(exc))[:160]

    def export(self):
        """Read after native shutdown; deliberately no live reader API."""
        reactor = self.owner() if self.owner else None
        if reactor is not None and reactor._worker.is_alive():
            raise RuntimeError("export only after native worker shutdown")
        return dict(run_id=self.run_id, samples=self.samples, overwritten=self.overwritten,
                    truncated_samples=self.truncated_samples, faulted=self.faulted,
                    error=self.error, counts=dict(self.counts), records=list(self.records),
                    sampling="post-pump CPU states; counts are repeated observations, not unique jobs",
                    causal_denial_attribution=False, cuda_event_queries=0,
                    inferred_gpu_release_bytes=None, native_scheduling_modified=False)

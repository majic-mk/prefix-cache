"""Bounded reactor-owner observation, not automatically installed in the hot path."""
from dataclasses import dataclass
from itertools import islice
from threading import get_ident
from time import monotonic_ns
from .dependencies import Parent

@dataclass(frozen=True)
class ReactorSnapshot:
    run_id: str
    epoch: int
    monotonic_ns: int
    actual_staging_bytes: int | None
    free_slots: int
    active_parent_count: int
    parents: tuple[Parent, ...]
    parents_truncated: bool
    ssd_read_ops: int | None
    ssd_write_ops: int | None
    h2d_inflight_bytes: int | None
    d2h_inflight_bytes: int | None
    gpu_immediately_reusable_bytes: int | None = None

    @property
    def available_fields(self) -> frozenset[str]:
        # Explicit capability names; an unavailable counter is never a measured zero.
        names = ("actual_staging_bytes", "free_slots", "active_parent_count",
                 "parents", "ssd_read_ops", "ssd_write_ops", "h2d_inflight_bytes",
                 "d2h_inflight_bytes", "gpu_immediately_reusable_bytes")
        return frozenset(name for name in names if getattr(self, name) is not None)

    def fresh(self, *, run_id: str, now_ns: int, max_age_ns: int) -> bool:
        return (self.run_id == run_id and max_age_ns > 0
                and 0 <= now_ns - self.monotonic_ns <= max_age_ns)

def capture_reactor(reactor, *, run_id: str, epoch: int) -> ReactorSnapshot:
    """Call only on the existing reactor thread. Never scan the GPU block pool.

    Observation bounds do not cap native concurrency. Counts whose scan would
    exceed the window remain None; they are never fabricated as zero.
    """
    if not run_id or type(epoch) is not int or epoch < 0:
        raise ValueError("run and nonnegative epoch are required")
    if reactor._worker.ident != get_ident():
        raise RuntimeError("snapshot must be captured by the reactor owner")
    jobs = tuple(islice(reactor._active, 32))
    parents = tuple(Parent(run_id, j.job_id, j.total_files, j.done_files,
                           j.inflight_files, j.future_set, j.failed is not None, None)
                    for j in jobs)
    if len(reactor._inflight) <= 64:
        reads = sum(op.op_kind == "read" for op in reactor._inflight.values())
        writes = sum(op.op_kind == "write" for op in reactor._inflight.values())
    else:
        reads = writes = None
    if len(reactor._pending_copies) <= 64:
        h2d = sum(copy.nbytes for copy in reactor._pending_copies if not copy.is_store)
        d2h = sum(copy.nbytes for copy in reactor._pending_copies if copy.is_store)
    else:
        h2d = d2h = None
    return ReactorSnapshot(run_id, epoch, monotonic_ns(),
        getattr(reactor, "actual_staging_bytes", None), reactor.staging_pool.free_count,
        len(reactor._active), parents, len(reactor._active) > len(parents),
        reads, writes, h2d, d2h)

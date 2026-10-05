from __future__ import annotations

import collections
import logging
import os
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .break_even import BreakEvenThresholds, should_load
from .file_mapper import FileMapper
from .fs_config import SharedFileConfig
from .liburing_file import (
    DirectIoFileStore,
    LiburingRing,
    align_up,
)
from .profiling import add_event, now_ns
from .staging import StagingPool
from .staging_cache import StagingDataCache, _CacheSlot
from .transfer import (
    ParsedKvLayout,
    _TransferProfile,
    block_ids_of,
    build_block_mapping,
    emit_transfer_events,
    first_group_block_index,
    safe_num_blocks,
    split_block_ids_for_files,
)

try:
    import torch
    from vllm import _custom_ops as ops

    TORCH_COPY_AVAILABLE = True
except Exception:
    torch = Any
    ops = None
    TORCH_COPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class NativeDrainUnknown(RuntimeError):
    """Native DMA/I/O completion was not proved; retain owners and stop service."""


class LoadDeclined(Exception):
    """Load skipped by break-even; block ids must be recomputed."""

    def __init__(self, block_ids: list[int], req_id: str = "") -> None:
        super().__init__("load declined by prefix-cache break-even gate")
        self.block_ids = block_ids
        self.req_id = req_id


@dataclass
class _ReactorJob:
    job_id: int
    is_store: bool
    block_hashes: list[bytes]
    # Per-file GPU block ids: source blocks for store, destination blocks for load.
    block_chunks: list[np.ndarray]
    profile: _TransferProfile
    future: "Future[int]"
    total_files: int
    transfer_size: int
    num_blocks: int
    next_file_index: int = 0
    inflight_files: int = 0
    done_files: int = 0
    failed: BaseException | None = None
    future_set: bool = False
    file_samples: list = field(default_factory=list)
    cuda_samples: list = field(default_factory=list)
    n_from_file: int = 0
    n_from_preload: int = 0
    n_from_cache: int = 0
    # Store-only: compute-stream event the store copy waits on before reading GPU KV.
    compute_event: Any = None
    # Common admission bookkeeping, never a resource-release witness.
    accepted_parent_sequence: int | None = None
    accepted_parent_retired: bool = False


@dataclass(frozen=True)
class _MandatoryWait:
    # Identity is local to one reactor instance; integer job IDs alone can recur.
    token: object
    futures: frozenset[Future[int]]


@dataclass(frozen=True)
class _AdmissionDrain:
    token: object


@dataclass(frozen=True)
class _P4Deferred:
    action: str = "defer"
    attempt: Any = None

@dataclass(frozen=True)
class _StageDispatch:
    decision: Any
    work_id: Any

    @property
    def action(self):
        return self.decision.action

    @property
    def attempt(self):
        return self.decision.attempt


@dataclass(frozen=True)
class _OwnerSnapshot:
    token: object
    future: Future


@dataclass(frozen=True)
class _P4Inspect:
    token: object
    future: Future

@dataclass(frozen=True)
class _P4Publish:
    token: object
    publication: Any
    future: Future

@dataclass
class _ProgressState:
    run_id: str
    token: object = field(default_factory=object)
    # Scheduling signal only: native jobs, slots, queues and Futures retain ownership.
    required: set[Future[int]] = field(default_factory=set)


@dataclass
class _PreloadRequest:
    block_hashes: list[bytes]
    preload_id: str = ""
    req_id: str = ""
    profile_tid: str = "kv_preload"


@dataclass(frozen=True)
class _PreloadInfo:
    block_hash: bytes
    preload_id: str
    req_id: str
    profile_tid: str
    file_index: int
    total_files: int


@dataclass
class _SharedPreloadSlot:
    # Released only when not cached and no copy still reads from it.
    block_hash: bytes
    slot_index: int
    preload_info: "_PreloadInfo | None"
    copies_inflight: int = 0
    cached: bool = True


@dataclass
class _RingOp:
    job: _ReactorJob | None  # None for preload reads
    file_index: int
    slot_index: int
    fd: int
    is_write: bool
    start_ns: int
    op_kind: str = "read"  # "open" | "read" | "write" | "close"
    final_path: str | None = None
    temp_path: str | None = None
    preload_hash: bytes | None = None
    preload_info: _PreloadInfo | None = None
    path_buf: Any = None  # keeps openat path alive until open CQE is reaped


@dataclass
class _ReadyFd:
    fd: int
    job: _ReactorJob | None
    file_index: int
    preload_hash: bytes | None
    open_start_ns: int
    preload_info: _PreloadInfo | None = None
    sequence: int = 0


@dataclass
class _CopyOp:
    job: _ReactorJob | None
    file_index: int
    slot_index: int
    is_store: bool
    start_ns: int
    start_event: Any
    end_event: Any
    nbytes: int
    members: list[tuple[int, "_ReactorJob", int, "_SharedPreloadSlot | None"]] | None = None
    accounting_copy_accepted: bool = True


@dataclass
class _ReadyCopy:
    job: _ReactorJob
    file_index: int
    slot_index: int
    mapping_len: int
    nbytes: int
    # Per-copy snapshot so shared siblings reusing one slot_index don't clobber it.
    mapping: Any = None
    shared: "_SharedPreloadSlot | None" = None
    cache: "_CacheSlot | None" = None


class IoReactor:
    _STOP = object()

    def __init__(
        self,
        *,
        config: SharedFileConfig,
        file_mapper: FileMapper,
        layout: ParsedKvLayout,
        break_even: BreakEvenThresholds | None = None,
        storage_block_tokens: int = 0,
        progress_run_id: str | None = None,
        observation_sink: Any = None,
        start_budget: Any = None,
        stage_accounting: Any = None,
        store_order: Any = None,
        dispatch_shadow: Any = None,
        max_accepted_parents: int | None = None,
        dispatch_controller: Any = None,
        p4_bridge: Any = None,
    ) -> None:
        if max_accepted_parents is not None and (
            type(max_accepted_parents) is not int or not 1 <= max_accepted_parents <= 64
        ):
            raise ValueError("common parent limit must be None or integer 1..64")
        if max_accepted_parents is not None and progress_run_id is None:
            raise ValueError("bounded admission requires the native progress bridge")
        if progress_run_id is not None and (
            type(progress_run_id) is not str or not progress_run_id.strip()
        ):
            raise ValueError("progress_run_id must be a nonempty string or None")
        if observation_sink is not None and not callable(observation_sink):
            raise TypeError("observation_sink must be callable or None")
        if p4_bridge is not None:
            from prefix_io_control.p4_bridge import NativeP4Bridge
            if type(p4_bridge) is not NativeP4Bridge:
                raise TypeError("exact isolated native P4 bridge required")
            if progress_run_id != p4_bridge.run_id or max_accepted_parents is None or stage_accounting is None:
                raise ValueError("P4 requires same-run common progress/admission/accounting")
            if any(x is not None for x in (start_budget,store_order,dispatch_shadow)):
                raise ValueError("P4 excludes legacy experimental policies")
            if p4_bridge.mode == "dependency_only":
                if dispatch_controller is None or dispatch_controller.config.mode != "fixed":
                    raise ValueError("dependency-only requires original fixed controller")
            elif dispatch_controller is not None:
                raise ValueError("shadow/unsupported I/J must retain original U allowances")
            p4_bridge.bind()
        if dispatch_controller is not None:
            from prefix_io_control.simple_stage_policy import DispatchController
            if not isinstance(dispatch_controller, DispatchController):
                raise TypeError("dispatch_controller must be DispatchController or None")
            if progress_run_id is None or dispatch_controller.run_id != progress_run_id:
                raise ValueError("stage controller requires the same native progress identity")
            if stage_accounting is None or max_accepted_parents is None:
                raise ValueError("stage controller requires common accounting and bounded admission")
            if any(value is not None for value in (dispatch_shadow, start_budget, store_order)):
                raise ValueError("stage controller qualification excludes other optional policies")
        if dispatch_shadow is not None:
            from prefix_io_control.dispatch_shadow import DispatchShadow
            if not isinstance(dispatch_shadow, DispatchShadow):
                raise TypeError("dispatch_shadow must be DispatchShadow or None")
            if progress_run_id is None or dispatch_shadow.run_id != progress_run_id:
                raise ValueError("dispatch shadow requires the same native progress identity")
            if stage_accounting is None:
                raise ValueError("dispatch shadow requires passive native stage accounting")
            if start_budget is not None or store_order is not None:
                raise ValueError("dispatch shadow qualification excludes other P3 policies")
        if stage_accounting is not None:
            from prefix_io_control.stage_accounting import StageAccounting
            if not isinstance(stage_accounting, StageAccounting):
                raise TypeError('stage_accounting must be StageAccounting or None')
            stage_accounting.bind()
        if start_budget is not None:
            from prefix_io_control.start_budget import StartBudget
            if not isinstance(start_budget, StartBudget):
                raise TypeError("start_budget must be StartBudget or None")
            if progress_run_id is None:
                raise ValueError("ordinary starts require the native mandatory bridge")
            start_budget.bind()
        if store_order is not None:
            from prefix_io_control.store_order import MandatoryStoreOrder
            if not isinstance(store_order, MandatoryStoreOrder):
                raise TypeError("store_order must be MandatoryStoreOrder or None")
            if progress_run_id is None:
                raise ValueError("store order requires the native mandatory bridge")
            if start_budget is not None:
                raise ValueError("P3 order qualification excludes start allowances")
            store_order.bind()
        if dispatch_shadow is not None:
            dispatch_shadow.bind()
        if dispatch_controller is not None:
            dispatch_controller.bind()
        if not TORCH_COPY_AVAILABLE:
            raise RuntimeError("vLLM and torch are required for the I/O reactor")
        self.config = config
        self.file_mapper = file_mapper
        self.layout = layout
        # Load-side break-even gate: decline sub-threshold loads to GPU recompute.
        # Disabled (no gating) when thresholds are off or block-token size unknown.
        self._break_even = break_even or BreakEvenThresholds()
        self._storage_block_tokens = storage_block_tokens
        payload_size = layout.storage_block_bytes
        io_size = align_up(payload_size)
        self.file_store = DirectIoFileStore(config, payload_size=payload_size)
        self.iodepth = max(1, config.iodepth)
        # Floor above iodepth so slots held by CUDA copies don't starve new reads.
        self._copy_headroom = max(4, self.iodepth // 2)
        min_slots = self.iodepth + self._copy_headroom
        slot_count = StagingPool.compute_slot_count(
            staging_mem_gib=config.staging_mem,
            io_size=io_size,
            min_slots=min_slots,
        )
        self.staging_buffer = layout.allocate_staging_buffer(slot_count)
        self.actual_staging_bytes = self.staging_buffer.untyped_storage().nbytes()
        self.staging_budget_bytes = int(config.staging_mem * (1 << 30))
        if self.actual_staging_bytes > self.staging_budget_bytes:
            raise ValueError(
                f"actual staging backing {self.actual_staging_bytes} exceeds "
                f"budget {self.staging_budget_bytes}"
            )
        self._staging_base_ptr = self.staging_buffer.data_ptr()
        self.staging_pool = StagingPool(slot_count=slot_count)
        if slot_count <= min_slots:
            logger.warning(
                "staging slots floored to %d (iodepth=%d + copy_headroom=%d); "
                "raise staging_mem so reads do not stall on copy-held slots",
                slot_count,
                self.iodepth,
                self._copy_headroom,
            )
        self.open_lookahead = max(1, config.open_lookahead or self.iodepth)
        ring_depth = max(self.iodepth + self.open_lookahead + 16, 16)
        if config.io_backend == "io_uring":
            self.ring = LiburingRing(ring_depth, ring_id=0)
        else:
            from .linux_aio import LinuxAioRing

            self.ring = LinuxAioRing(
                ring_depth, ring_id=0, metadata_workers=config.aio_metadata_workers
            )

        self._streams = [torch.cuda.Stream() for _ in range(slot_count)]
        self._copy_stream = torch.cuda.Stream()
        max_mapping_entries = max(1, layout.storage_block_size_factor)
        self._mapping_buffers = [
            np.empty((max_mapping_entries, 2), dtype=np.int64) for _ in range(slot_count)
        ]
        self._mapping_tensors = [torch.from_numpy(buffer) for buffer in self._mapping_buffers]
        self._slot_block_ids = [
            np.asarray([slot_index], dtype=np.int64) for slot_index in range(slot_count)
        ]
        max_copy_entries = max(1, len(layout.gpu_tensors) * max_mapping_entries)
        self._batch_src_ptrs = [
            np.empty(max_copy_entries, dtype=np.int64) for _ in range(slot_count)
        ]
        self._batch_dst_ptrs = [
            np.empty(max_copy_entries, dtype=np.int64) for _ in range(slot_count)
        ]
        self._batch_sizes = [np.empty(max_copy_entries, dtype=np.int64) for _ in range(slot_count)]
        self._batch_src_tensors = [torch.from_numpy(buffer) for buffer in self._batch_src_ptrs]
        self._batch_dst_tensors = [torch.from_numpy(buffer) for buffer in self._batch_dst_ptrs]
        self._batch_size_tensors = [torch.from_numpy(buffer) for buffer in self._batch_sizes]

        self._incoming: "queue.Queue[_ReactorJob | object]" = queue.Queue()
        self._active: list[_ReactorJob] = []
        self._inflight: dict[int, _RingOp] = {}
        self._pending_copies: list[_CopyOp] = []
        self._copy_ready: list[_ReadyCopy] = []
        self._fused_capacity = 0
        self._fused_src_np: np.ndarray | None = None
        self._fused_dst_np: np.ndarray | None = None
        self._fused_sizes_np: np.ndarray | None = None
        self._fused_src_t: Any = None
        self._fused_dst_t: Any = None
        self._fused_sizes_t: Any = None
        self._next_user_data = 0
        self._ready_sequence = 0
        self._stop = False
        self._closed = False
        self._submit_lock = threading.Lock()
        self._init_parent_admission(max_accepted_parents)

        self._data_inflight = 0
        self._open_inflight = 0
        self._ready_fds_load: collections.deque[_ReadyFd] = collections.deque()
        self._ready_fds_preload: collections.deque[_ReadyFd] = collections.deque()

        # Shared preload: one read + one slot per hash, released by refcount.
        self._share_preload = bool(getattr(config, "preload_share_staging", False))
        # hash -> count of distinct outstanding requests demanding it.
        self._preload_refcount: dict[bytes, int] = {}
        self._shared_cached: dict[bytes, _SharedPreloadSlot] = {}

        # Multi-copy preload: each requesting req_id gets its own staged copy of a
        # block hash (dedup per (req_id, hash) via _preload_owned).
        self._preload_pending: "collections.deque[_PreloadInfo]" = collections.deque()
        # Live and cancel counts per hash for pending copies.
        self._preload_pending_count: dict[bytes, int] = {}
        self._preload_pending_cancel: dict[bytes, int] = {}
        # hash -> count of open/read ops in flight.
        self._preload_inflight_hashes: dict[bytes, int] = {}
        # hash -> deque of (slot_index, _PreloadInfo); FIFO so oldest is evicted first.
        self._preload_slots: "collections.OrderedDict[bytes, collections.deque[tuple[int, _PreloadInfo]]]" = collections.OrderedDict()
        self._preload_cached_total = 0
        self._preload_inflight_total = 0
        # FIFO-capped per-(req_id, hash) dedup set.
        self._preload_owned: "collections.OrderedDict[tuple[str, bytes], None]" = (
            collections.OrderedDict()
        )
        self._max_preload_owned = 16384
        # Reserve iodepth slots for foreground loads.
        self._max_preload_slots = max(0, self.staging_pool.slot_count - self.iodepth)
        # hash -> deque of (job, file_index); waiter count <= inflight count for hash.
        self._preload_waiters: dict[bytes, collections.deque[tuple[_ReactorJob, int]]] = {}
        # hash -> count of in-flight stores; preload defers until write completes.
        self._store_inflight: dict[bytes, int] = {}
        # hash -> deque of _PreloadInfo parked on an in-flight store.
        self._preload_blocked_on_write: dict[bytes, collections.deque[_PreloadInfo]] = {}
        # hash -> deque of _PreloadInfo for openat misses; re-armed on store completion.
        self._known_missing: "collections.OrderedDict[bytes, collections.deque[_PreloadInfo]]" = (
            collections.OrderedDict()
        )
        self._max_known_missing = 4096

        cache_kind = str(getattr(config, "staging_cache", "off") or "off").lower()
        self._max_cache_slots = max(0, self.staging_pool.slot_count - self.iodepth)
        self._staging_cache: StagingDataCache | None = (
            StagingDataCache(policy=cache_kind, capacity=self._max_cache_slots)
            if cache_kind != "off" and self._max_cache_slots > 0
            else None
        )

        # None keeps ordinary deployment disabled; no parent signal refs are created.
        self._prefix_progress = (
            _ProgressState(progress_run_id) if progress_run_id is not None else None
        )
        self._prefix_start_budget = start_budget
        self._prefix_stage_accounting = stage_accounting
        self._prefix_store_order = store_order
        self._prefix_dispatch_shadow = dispatch_shadow
        self._prefix_dispatch_controller = dispatch_controller
        self._prefix_p4_bridge = p4_bridge
        self._prefix_p4_controls_pending = 0
        self._prefix_init_capacity_observation()
        self._observation_sink = observation_sink
        self._observation_failures = 0
        self._worker = threading.Thread(target=self._run, name="py-kvcache-reactor", daemon=True)
        self._worker.start()
        logger.info(
            "IoReactor started: iodepth=%d staging_slots=%d payload_size=%d (direct-staging)",
            self.iodepth,
            slot_count,
            payload_size,
        )

    def _init_parent_admission(self, limit):
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 64):
            raise ValueError("common parent limit must be None or integer 1..64")
        self._parent_cv = threading.Condition(self._submit_lock)
        self._max_accepted_parents = limit
        self._accepted_parent_count = 0
        self._parent_sequence = 0
        self._parent_count_valid = True
        self._parent_retired_count = 0
        self._parent_peak = 0
        self._parent_waiters = 0
        self._parent_backpressure_waits = 0
        self._admission_token = object()
        self._admission_drain_pending = False
        self._owner_snapshot_pending = 0
        self._native_drain_unknown = False
        self._native_fatal_reason = None
        self._native_fatal_drain_verified = False
        self._native_failure_sync_total = 0
        self._native_failure_syncs = {"h2d": 0, "d2h": 0}

    def raise_if_native_fatal(self):
        reason = getattr(self, "_native_fatal_reason", None)
        if reason is not None:
            if getattr(self, "_native_drain_unknown", False):
                raise NativeDrainUnknown(reason)
            raise RuntimeError(reason)

    def _freeze_native_drain_unknown(self, reason):
        self._stop = True
        self._prefix_stage_unknown("native_drain", NativeDrainUnknown(reason))
        cv = getattr(self, "_parent_cv", self._submit_lock)
        with cv:
            self._closed = True
            self._native_drain_unknown = True
            self._native_fatal_reason = str(reason)[:240]
            self._parent_count_valid = False
            if hasattr(cv, "notify_all"):
                cv.notify_all()
        # No Future completion, slot release, queue clearing or cache credit.

    def accepted_parent_count(self):
        if self._worker.ident != threading.get_ident():
            raise RuntimeError("accepted parent count must be captured by reactor owner")
        cv = getattr(self, "_parent_cv", self._submit_lock)
        with cv:
            return (self._accepted_parent_count
                    if getattr(self, "_parent_count_valid", False) else None)

    def parent_admission_snapshot(self, *, native_shutdown=False):
        if type(native_shutdown) is not bool:
            raise ValueError("explicit shutdown state required")
        if native_shutdown and self._worker.is_alive():
            raise RuntimeError("native owner is still running")
        if self._worker.is_alive() and self._worker.ident != threading.get_ident():
            raise RuntimeError("parent snapshot must be captured by reactor owner")
        cv = getattr(self, "_parent_cv", self._submit_lock)
        with cv:
            valid = getattr(self, "_parent_count_valid", False)
            return dict(
                count_valid=valid,
                accepted_parents=self._accepted_parent_count if valid else None,
                accepted_count_lower_bound=getattr(self, "_accepted_parent_count", 0),
                max_accepted_parents=getattr(self, "_max_accepted_parents", None),
                peak_accepted_parents=getattr(self, "_parent_peak", 0),
                retired_parents=getattr(self, "_parent_retired_count", 0),
                waiting_submitters=getattr(self, "_parent_waiters", 0),
                backpressure_waits=getattr(self, "_parent_backpressure_waits", 0),
                native_drain_unknown=getattr(self, "_native_drain_unknown", False),
                fatal_drain_verified=getattr(self, "_native_fatal_drain_verified", False),
                fatal_reason=getattr(self, "_native_fatal_reason", None),
                failure_stream_syncs=getattr(self, "_native_failure_sync_total", 0),
                failure_sync_by_stage=dict(getattr(self, "_native_failure_syncs", {})),
                source_gpu_reuse_inferred=False, staging_release_inferred=False,
            )

    def _retire_accepted_parent(self, job):
        if getattr(job, "accepted_parent_sequence", None) is None:
            return  # Legacy CPU fixtures and zero-file jobs were never admitted.
        cv = getattr(self, "_parent_cv", None)
        if cv is None:
            return
        with cv:
            if job.accepted_parent_retired:
                raise RuntimeError("accepted parent retired twice")
            if not self._parent_count_valid:
                return  # Unknown drain cannot turn into reusable capacity.
            if self._accepted_parent_count <= 0:
                self._parent_count_valid = False
                self._native_fatal_reason = "accepted parent accounting underflow"
                cv.notify_all()
                raise RuntimeError(self._native_fatal_reason)
            job.accepted_parent_retired = True
            self._accepted_parent_count -= 1
            self._parent_retired_count += 1
            cv.notify_all()

    def submit_job(self, job: _ReactorJob) -> None:
        cv = getattr(self, "_parent_cv", self._submit_lock)
        error = None
        with cv:
            limit = getattr(self, "_max_accepted_parents", None)
            while (not self._closed and limit is not None and
                   self._parent_count_valid and self._accepted_parent_count >= limit):
                if self._worker.ident == threading.get_ident():
                    # An owner callback must never block the sole thread capable
                    # of draining already accepted parents. This candidate has
                    # not been accepted and carries a concrete failure Future.
                    error = RuntimeError("reactor-owner reentrant admission would block")
                    break
                if not self._admission_drain_pending:
                    self._admission_drain_pending = True
                    self._incoming.put(_AdmissionDrain(self._admission_token))
                self._parent_waiters += 1
                self._parent_backpressure_waits += 1
                try:
                    cv.wait()
                finally:
                    self._parent_waiters -= 1
            if error is not None:
                pass
            elif self._closed or (limit is not None and not self._parent_count_valid):
                error = (NativeDrainUnknown(self._native_fatal_reason)
                         if getattr(self, "_native_drain_unknown", False) else
                         RuntimeError(getattr(self, "_native_fatal_reason", None)
                                      or "reactor is shut down"))
            else:
                if hasattr(self, "_parent_cv"):
                    if job.accepted_parent_sequence is not None:
                        raise ValueError("the same native parent cannot be admitted twice")
                    self._parent_sequence += 1
                    job.accepted_parent_sequence = self._parent_sequence
                    self._accepted_parent_count += 1
                    self._parent_peak = max(self._parent_peak, self._accepted_parent_count)
                self._incoming.put(job)
        # Future callbacks can re-enter submission. Never invoke them under the lock.
        if error is not None and not job.future_set:
            job.future_set = True
            job.future.set_exception(error)

    def request_owner_snapshot(self):
        """Metadata request through original incoming; no new resource/job queue."""
        future = Future()
        error = None
        cv = getattr(self, "_parent_cv", self._submit_lock)
        with cv:
            if self._closed:
                error = RuntimeError("reactor is shut down")
            elif not hasattr(self, "_admission_token"):
                error = RuntimeError("owner snapshot requires common admission initialization")
            elif self._owner_snapshot_pending >= 8:
                error = RuntimeError("bounded owner snapshot requests exhausted")
            else:
                self._owner_snapshot_pending += 1
                self._incoming.put(_OwnerSnapshot(self._admission_token, future))
        if error is not None:
            future.set_exception(error)
        return future

    def inspect_snapshot(self, timeout=5.0):
        if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
            raise ValueError("finite positive snapshot timeout required")
        self.raise_if_native_fatal()
        if self._worker.ident == threading.get_ident():
            return self._capture_owner_snapshot()
        if not self._worker.is_alive():
            return self._capture_owner_snapshot(native_shutdown=True)
        return self.request_owner_snapshot().result(timeout=timeout)

    def _capture_owner_snapshot(self, *, native_shutdown=False):
        if native_shutdown:
            if self._worker.is_alive():
                raise RuntimeError("native owner is still running")
        elif self._worker.ident != threading.get_ident():
            raise RuntimeError("snapshot must be captured by reactor owner")
        controller = getattr(self, "_prefix_dispatch_controller", None)
        accounting = getattr(self, "_prefix_stage_accounting", None)
        admission = self.parent_admission_snapshot(native_shutdown=native_shutdown)
        aio_snapshot = getattr(self.ring, "snapshot", None)
        capacity = self._prefix_capacity_snapshot(native_shutdown=native_shutdown)
        return dict(
            captured_ns=time.monotonic_ns(),
            run_id=self._prefix_progress.run_id if self._prefix_progress is not None else None,
            owner_capture=not native_shutdown, native_shutdown_read=native_shutdown,
            parent_admission=admission, admission=admission,
            aio=aio_snapshot() if callable(aio_snapshot) else None,
            native=dict(
                active_parents=len(self._active), ring_ops=len(self._inflight),
                pending_copies=len(self._pending_copies), copy_ready=len(self._copy_ready),
                ready_load_fds=len(self._ready_fds_load), ready_preload_fds=len(self._ready_fds_preload),
            ),
            stage_accounting=accounting.snapshot() if accounting is not None else None,
            controller=controller.snapshot(native_shutdown=native_shutdown) if controller is not None else None,
            free_reclaimable_staging_bytes=capacity["free_reclaimable_staging_bytes"],
            staging_capacity=capacity,
            p4=getattr(self,'_prefix_p4_bridge',None).snapshot(native_shutdown=native_shutdown)
                if getattr(self,'_prefix_p4_bridge',None) is not None else None,
            physical_drain_inferred=False, gpu_release_credit=False,
        )

    @property
    def progress_bridge_enabled(self) -> bool:
        return getattr(self, "_prefix_progress", None) is not None

    def request_mandatory(self, futures: list[Future[int]]) -> bool:
        state = getattr(self, "_prefix_progress", None)
        if state is None:
            return False
        if any(not isinstance(future, Future) for future in futures):
            raise TypeError("mandatory signals must refer to existing parent Futures")
        requested = frozenset(futures)
        if not requested:
            return False
        with self._submit_lock:
            if self._closed:
                return False
            # Reuse the native incoming channel. FIFO puts accepted jobs before waits.
            self._incoming.put(_MandatoryWait(state.token, requested))
        return True

    def is_mandatory(self, future: Future[int]) -> bool:
        state = getattr(self, "_prefix_progress", None)
        if state is None:
            return False
        if threading.get_ident() != self._worker.ident:
            raise RuntimeError("mandatory membership is reactor-owner state")
        return future in state.required

    def enqueue_preload(
        self,
        block_hashes: list[bytes],
        *,
        preload_id: str = "",
        req_id: str = "",
        profile_tid: str = "kv_preload",
    ) -> None:
        with self._submit_lock:
            if self._closed:
                return
            self._incoming.put(
                _PreloadRequest(
                    block_hashes,
                    preload_id=preload_id,
                    req_id=req_id,
                    profile_tid=profile_tid,
                )
            )

    def shutdown(self, wait: bool = True) -> None:
        cv = getattr(self, "_parent_cv", self._submit_lock)
        with cv:
            if not self._closed:
                self._closed = True
                self._incoming.put(self._STOP)
            if hasattr(cv, "notify_all"):
                cv.notify_all()
        if wait:
            self._worker.join()
            if getattr(self, "_native_drain_unknown", False):
                raise NativeDrainUnknown(self._native_fatal_reason)
            self.staging_pool.close()
            self.ring.close()

    def _preload_cache_push(
        self, block_hash: bytes, slot_index: int, info: "_PreloadInfo | None"
    ) -> None:
        slots = self._preload_slots.get(block_hash)
        if slots is None:
            slots = collections.deque()
            self._preload_slots[block_hash] = slots
        slots.append((slot_index, info))
        self._preload_cached_total += 1

    def _preload_cache_pop(self, block_hash: bytes) -> "tuple[int, _PreloadInfo | None] | None":
        slots = self._preload_slots.get(block_hash)
        if not slots:
            return None
        entry = slots.popleft()
        self._preload_cached_total -= 1
        if not slots:
            del self._preload_slots[block_hash]
        return entry

    def _has_cached_preload(self, block_hash: bytes) -> bool:
        return bool(self._preload_slots.get(block_hash))

    def _preload_inflight_add(self, block_hash: bytes) -> None:
        self._preload_inflight_hashes[block_hash] = (
            self._preload_inflight_hashes.get(block_hash, 0) + 1
        )
        self._preload_inflight_total += 1

    def _preload_inflight_remove(self, block_hash: bytes) -> None:
        count = self._preload_inflight_hashes.get(block_hash, 0)
        if count <= 0:
            return
        self._preload_inflight_total -= 1
        if count == 1:
            del self._preload_inflight_hashes[block_hash]
        else:
            self._preload_inflight_hashes[block_hash] = count - 1

    def _has_inflight_preload(self, block_hash: bytes) -> bool:
        return self._preload_inflight_hashes.get(block_hash, 0) > 0

    def _preload_waiter_add(self, block_hash: bytes, job: "_ReactorJob", file_index: int) -> None:
        waiters = self._preload_waiters.get(block_hash)
        if waiters is None:
            waiters = collections.deque()
            self._preload_waiters[block_hash] = waiters
        waiters.append((job, file_index))

    def _preload_waiter_pop(self, block_hash: bytes):
        waiters = self._preload_waiters.get(block_hash)
        if not waiters:
            return None
        item = waiters.popleft()
        if not waiters:
            del self._preload_waiters[block_hash]
        return item

    def _has_preload_waiter(self, block_hash: bytes) -> bool:
        return bool(self._preload_waiters.get(block_hash))

    def _preload_waiter_room(self, block_hash: bytes) -> bool:
        # Waiter count must not exceed inflight count: every waiter is served by some CQE.
        return len(self._preload_waiters.get(block_hash, ())) < (
            self._preload_inflight_hashes.get(block_hash, 0)
        )

    def _preload_own(self, req_id: str, block_hash: bytes) -> bool:
        key = (req_id, block_hash)
        if key in self._preload_owned:
            return False
        self._preload_owned[key] = None
        while len(self._preload_owned) > self._max_preload_owned:
            self._preload_owned.popitem(last=False)
        return True

    def _pending_push(self, info: "_PreloadInfo") -> None:
        self._preload_pending.append(info)
        self._preload_pending_count[info.block_hash] = (
            self._preload_pending_count.get(info.block_hash, 0) + 1
        )

    def _pending_dec(self, block_hash: bytes) -> None:
        count = self._preload_pending_count.get(block_hash, 0)
        if count <= 1:
            self._preload_pending_count.pop(block_hash, None)
        else:
            self._preload_pending_count[block_hash] = count - 1

    def _cancel_one_pending(self, block_hash: bytes) -> bool:
        # The load now reads this hash itself; the pending copy is redundant.
        live = self._preload_pending_count.get(block_hash, 0) - self._preload_pending_cancel.get(
            block_hash, 0
        )
        if live <= 0:
            return False
        self._preload_pending_cancel[block_hash] = (
            self._preload_pending_cancel.get(block_hash, 0) + 1
        )
        return True

    # --- Shared (ref-counted) preload helpers -------------------------------

    def _has_any_shared_state(self, block_hash: bytes) -> bool:
        return (
            block_hash in self._shared_cached
            or self._preload_inflight_hashes.get(block_hash, 0) > 0
            or self._preload_pending_count.get(block_hash, 0) > 0
            or block_hash in self._preload_blocked_on_write
            or block_hash in self._known_missing
        )

    def _preload_has_copy(self, block_hash: bytes) -> bool:
        if self._staging_cache is not None and block_hash in self._staging_cache:
            return True
        if self._has_inflight_preload(block_hash):
            return True
        if self._share_preload:
            return block_hash in self._shared_cached
        return self._has_cached_preload(block_hash)

    def _make_preload_info_for_job(self, job: "_ReactorJob", file_index: int) -> "_PreloadInfo":
        return _PreloadInfo(
            block_hash=job.block_hashes[file_index],
            preload_id="",
            req_id=job.profile.req_id,
            profile_tid="kv_preload",
            file_index=file_index,
            total_files=job.total_files,
        )

    def _prefix_invalidate_capacity(self, reason) -> None:
        try:
            message = str(reason)[:160] or "capacity observation unavailable"
        except Exception:
            message = "capacity observation unavailable"
        self._prefix_capacity_valid = False
        if getattr(self, "_prefix_capacity_error", None) is None:
            self._prefix_capacity_error = message

    def _prefix_init_capacity_observation(self) -> None:
        if hasattr(self, "_prefix_capacity_valid"):
            return  # Only a fresh reactor may initialize; invalidity is sticky.
        # Init is not a scan/reconstruction of pre-existing shared busy ownership.
        self._prefix_capacity_valid = True
        self._prefix_capacity_error = None
        self._prefix_shared_cached_pinned_slots = 0
        self._prefix_shared_cached_observed_count = 0
        try:
            if (len(getattr(self, "_shared_cached", {})) or
                    len(getattr(self, "_preload_slots", {})) or
                    getattr(self, "_preload_cached_total", 0) != 0):
                raise ValueError("capacity observation requires empty preload seed")
        except Exception as exc:
            self._prefix_invalidate_capacity(type(exc).__name__ + ": " + str(exc))

    def _prefix_observe_capacity(self, callback, *args) -> None:
        try:
            callback(*args)
        except Exception as exc:
            self._prefix_invalidate_capacity(type(exc).__name__ + ": " + str(exc))

    def _prefix_observe_shared_copy(self, slot, delta) -> None:
        if getattr(self, "_prefix_capacity_valid", False) is not True:
            return
        count = slot.copies_inflight
        registered = self._shared_cached.get(slot.block_hash) is slot
        if (type(count) is not int or count < 0 or
                type(slot.cached) is not bool or registered != slot.cached or
                (delta == 1 and count - 1 < 0) or
                (delta == -1 and count + 1 <= 0)):
            raise ValueError("stale/invalid shared copy reference")
        if registered:
            if delta == 1 and count == 1:
                self._prefix_shared_cached_pinned_slots += 1
            elif delta == -1 and count == 0:
                self._prefix_shared_cached_pinned_slots -= 1
        if (len(self._shared_cached) != self._prefix_shared_cached_observed_count or
                type(self._prefix_shared_cached_pinned_slots) is not int or
                not 0 <= self._prefix_shared_cached_pinned_slots <= len(self._shared_cached)):
            raise ValueError("shared busy scalar/cardinality mismatch")

    def _prefix_observe_shared_registry(self, slot, delta) -> None:
        if getattr(self, "_prefix_capacity_valid", False) is not True:
            return
        if (type(slot.copies_inflight) is not int or slot.copies_inflight < 0 or
                len(self._shared_cached) != self._prefix_shared_cached_observed_count + delta):
            raise ValueError("unexpected shared registry mutation")
        self._prefix_shared_cached_observed_count += delta
        if slot.copies_inflight > 0:
            self._prefix_shared_cached_pinned_slots += delta
        if (type(self._prefix_shared_cached_pinned_slots) is not int or
                not 0 <= self._prefix_shared_cached_pinned_slots <= len(self._shared_cached)):
            raise ValueError("shared registry busy scalar mismatch")

    def _shared_pin(self, slot: "_SharedPreloadSlot") -> None:
        slot.copies_inflight += 1  # Native ref operation always runs exactly once.
        self._prefix_observe_capacity(self._prefix_observe_shared_copy, slot, 1)

    def _shared_unpin(self, slot: "_SharedPreloadSlot") -> None:
        slot.copies_inflight -= 1
        self._prefix_observe_capacity(self._prefix_observe_shared_copy, slot, -1)

    def _shared_cache_insert(self, slot: "_SharedPreloadSlot") -> None:
        slot.cached = True
        self._shared_cached[slot.block_hash] = slot
        self._preload_cached_total += 1
        self._prefix_observe_capacity(self._prefix_observe_shared_registry, slot, 1)

    def _shared_uncache(self, slot: "_SharedPreloadSlot") -> None:
        removed = self._shared_cached.get(slot.block_hash) is slot
        if removed:
            del self._shared_cached[slot.block_hash]
            self._preload_cached_total -= 1
        slot.cached = False
        if removed:
            self._prefix_observe_capacity(self._prefix_observe_shared_registry, slot, -1)

    def _maybe_release_shared(self, slot: "_SharedPreloadSlot") -> None:
        if slot.cached or slot.copies_inflight > 0:
            return
        self._release_or_cache(slot.block_hash, slot.slot_index)

    def _purge_owned_for_hash(self, block_hash: bytes) -> None:
        stale = [k for k in self._preload_owned if k[1] == block_hash]
        for k in stale:
            del self._preload_owned[k]

    def _shared_decref(self, req_id: str, block_hash: bytes) -> None:
        # Only an owner's claim consumes demand; over-counts fall to eviction.
        key = (req_id, block_hash)
        if key not in self._preload_owned:
            return
        del self._preload_owned[key]
        count = self._preload_refcount.get(block_hash, 0)
        if count > 1:
            self._preload_refcount[block_hash] = count - 1
            return
        self._preload_refcount.pop(block_hash, None)
        slot = self._shared_cached.get(block_hash)
        if slot is not None:
            self._shared_uncache(slot)
            self._maybe_release_shared(slot)

    def _fail_all_preload_waiters(self, block_hash: bytes, exc: BaseException) -> bool:
        failed = False
        while True:
            waiter = self._preload_waiter_pop(block_hash)
            if waiter is None:
                break
            self._file_terminal(waiter[0], ok=False, exc=exc)
            failed = True
        return failed

    def _evict_one_shared_slot(self) -> bool:
        for block_hash, slot in list(self._shared_cached.items()):
            if slot.copies_inflight > 0:
                continue  # in use, not actually free
            had_ref = self._preload_refcount.get(block_hash, 0) > 0
            self._shared_uncache(slot)
            self._preload_refcount.pop(block_hash, None)
            self._purge_owned_for_hash(block_hash)
            add_event(
                "py_kvcache.preload.evict_slot",
                "kv_preload",
                now_ns(),
                0,
                tid=slot.preload_info.profile_tid if slot.preload_info else None,
                args=self._preload_event_args(
                    block_hash,
                    slot.preload_info,
                    reason="refcount_leak_reclaim" if had_ref else "load_slot_pressure",
                ),
            )
            self.staging_pool.release(slot.slot_index)
            return True
        return False

    def _has_work(self) -> bool:
        return bool(
            self._active
            or self._inflight
            or self._pending_copies
            or self._copy_ready
            or self._preload_pending
            or self._preload_slots
            or self._shared_cached
            or self._ready_fds_load
            or self._ready_fds_preload
        )

    def _run(self) -> None:
        try:
            while not (self._stop and not self._has_work()):
                self._drain_incoming(block=not self._has_work())
                if self._stop and not self._has_work():
                    break
                self._pump_once()
                if getattr(self, "_observation_sink", None) is not None:
                    self._observe_once()
            # Reachable only after normal STOP and original native work drains.
            self._cleanup_retained_cache_after_stop()
        except BaseException as exc:
            # Failure can release worker protection only after all native
            # stages have a physical drain witness; unknown drain is fail-closed.
            logger.exception("py-kvcache reactor thread crashed; proving native drain")
            self._fail_everything(exc)
        finally:
            budget = getattr(self, "_prefix_start_budget", None)
            if budget is not None:
                budget.clear()

    def _cleanup_retained_cache_after_stop(self) -> None:
        if not self._stop or self._has_work():
            raise RuntimeError("normal cache cleanup requires original work drain")
        cache = self._staging_cache
        if cache is None:
            return
        for slot_index in cache.drain_unpinned():
            self.staging_pool.release(slot_index)
        if len(cache):
            # Orphan positive refs are not physical completion evidence.
            self._freeze_native_drain_unknown("pinned cache remains after native STOP drain")
            raise NativeDrainUnknown(self._native_fatal_reason)
        cache.drain()  # Original final policy clear; no retained indices remain.

    def _observe_once(self) -> None:
        """Optional bounded CPU observer; native work/completion has already run."""
        sink = self._observation_sink
        if sink is None:
            return
        try:
            sink(self)
        except Exception:
            # A broken observer cannot fail native parent Futures or flood logs.
            self._observation_failures += 1
            self._observation_sink = None

    def _sync_copy_failure(self, stream, stage, exc):
        """Exception path only; synchronous proof precedes any native cleanup."""
        try:
            stream.synchronize()
        except BaseException as sync_exc:
            self._freeze_native_drain_unknown(
                stage + " stream drain failed after " + type(exc).__name__ +
                ": " + str(exc)
            )
            raise NativeDrainUnknown(self._native_fatal_reason) from sync_exc
        if hasattr(self, "_native_failure_sync_total"):
            self._native_failure_sync_total += 1
            self._native_failure_syncs[stage] += 1

    def _drain_native_before_fatal_failure(self):
        # Fatal only: no new operation may be submitted while proving all old work.
        streams = list(getattr(self, "_streams", ()))
        copy_stream = getattr(self, "_copy_stream", None)
        if copy_stream is not None:
            streams.append(copy_stream)
        seen = set()
        for stream in streams:
            if id(stream) not in seen:
                stream.synchronize()
                seen.add(id(stream))
        if self._pending_copies and not streams:
            raise NativeDrainUnknown("native copy streams unavailable for fatal drain")
        # Linux AIO.close joins its existing worker and refuses unsafe buffer reuse.
        self.ring.close()

    def _cleanup_after_verified_fatal_drain(self):
        # Called only after every native CUDA stream and the original AIO ring
        # drained. The original pool/free list defines the unique held slots;
        # shared consumers do not manufacture one release per copy member.
        closed_fds = set()
        for op in self._inflight.values():
            if op.fd < 0 or op.op_kind in ("open", "close") or op.fd in closed_fds:
                continue
            closed_fds.add(op.fd)
            try:
                if op.is_write and op.temp_path is not None:
                    self.file_store.cleanup_temp(fd=op.fd, temp_path=op.temp_path)
                else:
                    os.close(op.fd)
            except OSError:
                pass
        for ready in (*self._ready_fds_load, *self._ready_fds_preload):
            if ready.fd >= 0 and ready.fd not in closed_fds:
                closed_fds.add(ready.fd)
                try:
                    os.close(ready.fd)
                except OSError:
                    pass
        free = set(self.staging_pool._free)
        for index in range(self.staging_pool.slot_count):
            if index not in free:
                self.staging_pool.release(index)
        self._inflight.clear()
        self._pending_copies.clear()
        self._copy_ready.clear()
        self._ready_fds_load.clear()
        self._ready_fds_preload.clear()
        self._preload_slots.clear()
        self._shared_cached.clear()
        if self._staging_cache is not None:
            self._staging_cache._slots.clear()
            invalidate = getattr(self._staging_cache, "invalidate_observation", None)
            if callable(invalidate):
                self._prefix_observe_capacity(invalidate, "verified fatal cleanup")
        self._prefix_invalidate_capacity("verified fatal cleanup")
        self._preload_pending.clear()
        self._preload_inflight_hashes.clear()
        self._preload_inflight_total = self._preload_cached_total = 0
        self._data_inflight = self._open_inflight = 0

    def _fail_everything(self, exc: BaseException) -> None:
        self._stop = True
        cv = getattr(self, "_parent_cv", self._submit_lock)
        with cv:
            self._closed = True
            if hasattr(cv, "notify_all"):
                cv.notify_all()
        if isinstance(exc, NativeDrainUnknown) or getattr(self, "_native_drain_unknown", False):
            self._freeze_native_drain_unknown(str(exc))
            return  # Preserve existing owners/Futures until supervised context teardown.
        try:
            self._drain_native_before_fatal_failure()
        except BaseException as drain_exc:
            self._freeze_native_drain_unknown(
                "fatal physical drain not proved: " + type(drain_exc).__name__ +
                ": " + str(drain_exc)
            )
            return
        self._native_fatal_drain_verified = True
        self._cleanup_after_verified_fatal_drain()
        self._native_fatal_reason = "py-kvcache reactor thread crashed after verified drain"
        wrapped = RuntimeError(self._native_fatal_reason)
        wrapped.__cause__ = exc

        def fail_parent(job):
            if not job.future_set:
                self._retire_accepted_parent(job)  # Fatal native drain was proved above.
                job.future_set = True
                try:
                    job.future.set_exception(wrapped)
                except BaseException:
                    pass
            if not getattr(job, "accepted_parent_retired", False):
                self._retire_accepted_parent(job)

        for job in self._active:
            fail_parent(job)
        self._active = []
        for waiters in list(self._preload_waiters.values()):
            for job, _fi in waiters:
                fail_parent(job)
        self._preload_waiters.clear()
        # No Future callback executes under submit/admission lock.
        while True:
            try:
                item = self._incoming.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, _ReactorJob):
                fail_parent(item)
            elif isinstance(item, (_OwnerSnapshot,_P4Inspect,_P4Publish)):
                if not item.future.done():
                    item.future.set_exception(wrapped)
        self._owner_snapshot_pending = 0
        self._prefix_p4_controls_pending = 0
        state = getattr(self, "_prefix_progress", None)
        if state is not None:
            state.required.clear()

    def _drain_incoming(self, *, block: bool) -> None:
        if block:
            try:
                item = self._incoming.get(timeout=0.5)
            except queue.Empty:
                return
            self._intake(item)
        while True:
            try:
                item = self._incoming.get_nowait()
            except queue.Empty:
                return
            self._intake(item)

    def _intake(self, item: Any) -> None:
        state = getattr(self, "_prefix_progress", None)
        if isinstance(item, (_P4Inspect,_P4Publish)):
            try:
                if state is None or item.token is not state.token:
                    raise ValueError("P4 publication native run token differs")
                bridge = getattr(self,"_prefix_p4_bridge",None)
                if bridge is None:
                    raise RuntimeError("P4 bridge disabled")
                view = self._prefix_p4_collect()
                if view is None:
                    raise ValueError("actual native observation exceeds supported window")
                result = view if isinstance(item,_P4Inspect) else bridge.publish(
                    item.publication,view,now_ns=time.monotonic_ns())
                item.future.set_result(result)
            except Exception as diagnostic:
                if not item.future.done():
                    item.future.set_exception(diagnostic)
            finally:
                with self._submit_lock:
                    self._prefix_p4_controls_pending = max(0,getattr(self,"_prefix_p4_controls_pending",0)-1)
            return
        if isinstance(item, _AdmissionDrain):
            if item.token is getattr(self, "_admission_token", None):
                with self._parent_cv:
                    self._admission_drain_pending = False
                # Common capacity progress must not await a new worker/controller epoch.
                if state is not None:
                    target = next((job for job in self._active
                                   if not job.future_set and job.failed is None), None)
                    if target is not None:
                        state.required.add(target.future)
            return
        if isinstance(item, _OwnerSnapshot):
            if item.token is getattr(self, "_admission_token", None):
                with self._parent_cv:
                    self._owner_snapshot_pending -= 1
                try:
                    item.future.set_result(self._capture_owner_snapshot())
                except Exception as snapshot_exc:
                    item.future.set_exception(snapshot_exc)
            return
        if isinstance(item, _MandatoryWait):
            if state is not None and item.token is state.token:
                # Unknown, drained and prior-run Futures cannot acquire urgency.
                state.required.update(
                    job.future for job in self._active if job.future in item.futures
                )
            return
        if item is self._STOP:
            if state is not None:
                # No future scheduler epoch is needed to drain accepted work.
                state.required.update(job.future for job in self._active)
            self._stop = True
            unclaimed = self._preload_cached_total
            if unclaimed:
                add_event(
                    "py_kvcache.preload.unclaimed_at_shutdown",
                    "kv_preload",
                    now_ns(),
                    0,
                    args={"count": unclaimed},
                )
            for slots in self._preload_slots.values():
                for slot_index, _info in slots:
                    self.staging_pool.release(slot_index)
            self._preload_slots.clear()
            while self._shared_cached:
                slot = next(iter(self._shared_cached.values()))
                self._shared_uncache(slot)
                self._maybe_release_shared(slot)
            if self._staging_cache is not None:
                retained = len(self._staging_cache)
                for slot_index in self._staging_cache.drain_unpinned():
                    self.staging_pool.release(slot_index)
                if retained:
                    add_event(
                        "py_kvcache.staging_cache.retained_at_shutdown",
                        "kv_staging_cache",
                        now_ns(),
                        0,
                        args={"count": retained},
                    )
            self._preload_refcount.clear()
            self._preload_cached_total = 0
            # Original open/read ownership retires via _preload_inflight_remove.
            self._preload_pending.clear()
            self._preload_pending_count.clear()
            self._preload_pending_cancel.clear()
            self._preload_owned.clear()
            self._preload_blocked_on_write.clear()
            self._known_missing.clear()
            err = RuntimeError("reactor shutting down")
            for _h, waiters in list(self._preload_waiters.items()):
                for job, _fi in waiters:
                    self._file_terminal(job, ok=False, exc=err)
            self._preload_waiters.clear()
            return
        if isinstance(item, _PreloadRequest):
            total_files = len(item.block_hashes)
            for file_index, h in enumerate(item.block_hashes):
                if self._staging_cache is not None and h in self._staging_cache:
                    continue  # already resident; the real load will hit the cache
                if not self._preload_own(item.req_id, h):
                    continue
                info = _PreloadInfo(
                    block_hash=h,
                    preload_id=item.preload_id,
                    req_id=item.req_id,
                    profile_tid=item.profile_tid,
                    file_index=file_index,
                    total_files=total_files,
                )
                if self._share_preload:
                    # Demand counts; a read is queued only for the first demander.
                    self._preload_refcount[h] = self._preload_refcount.get(h, 0) + 1
                    if not self._has_any_shared_state(h):
                        self._pending_push(info)
                    continue
                known = self._known_missing.get(h)
                if known is not None:
                    # Hash already missed; join the deque so store completion re-arms this copy too.
                    known.append(info)
                    self._known_missing.move_to_end(h)
                else:
                    self._pending_push(info)
            return
        job = item
        if not job.is_store:
            if self._break_even.enabled and self._should_decline_load(job):
                self._decline_load(job)
                return
            # Clear stale known-missing marks; in-flight/cached copies stay for their owners.
            for h in job.block_hashes:
                self._known_missing.pop(h, None)
        else:
            producer = job.profile.req_id
            for h in job.block_hashes:
                # A request never loads a prefix it produced; withdraw its own demand.
                if self._share_preload:
                    self._shared_decref(producer, h)
                missed = self._known_missing.pop(h, None)
                if missed:
                    if self._share_preload:
                        rearm = missed if self._preload_refcount.get(h, 0) > 0 else None
                    else:
                        rearm = collections.deque(
                            info for info in missed if info.req_id != producer
                        )
                        rearm = rearm or None
                    if rearm:
                        blocked = self._preload_blocked_on_write.get(h)
                        if blocked is None:
                            self._preload_blocked_on_write[h] = rearm
                        else:
                            blocked.extend(rearm)
                self._store_inflight[h] = self._store_inflight.get(h, 0) + 1
        self._active.append(job)

    def _should_decline_load(self, job: "_ReactorJob") -> bool:
        # RAM medium only if every block is already staged/cached; else SSD.
        prefix_tokens = job.total_files * self._storage_block_tokens
        if prefix_tokens <= 0:
            return False
        ram_resident = all(self._preload_has_copy(h) for h in job.block_hashes)
        return not should_load(prefix_tokens, self._break_even, ram_resident=ram_resident)

    def _decline_load(self, job: "_ReactorJob") -> None:
        block_ids = [
            int(b)
            for chunk in job.block_chunks
            for b in np.asarray(chunk).reshape(-1)
        ]
        add_event(
            "py_kvcache.break_even.decline_load",
            "kv_offload",
            now_ns(),
            0,
            tid=job.profile.profile_tid,
            args={
                "req_id": job.profile.req_id,
                "job_id": job.job_id,
                "prefix_tokens": job.total_files * self._storage_block_tokens,
                "num_blocks": len(block_ids),
            },
        )
        job.future_set = True
        self._retire_accepted_parent(job)  # No native I/O/copy was issued.
        job.future.set_exception(LoadDeclined(block_ids, req_id=job.profile.req_id))

    def _pump_once(self) -> bool:
        made = False
        made = self._drain_cuda_copies() or made
        made = self._poll_ring_completions() or made
        made = self._schedule_work() or made
        # Fuse all load copies queued this pump into one swap_blocks_batch.
        made = self._flush_copy_batch() or made
        self.ring.submit_pending()
        made = self._finish_jobs() or made
        if not made and self._has_work():
            time.sleep(0)
        return made

    @staticmethod
    def _preload_event_args(
        block_hash: bytes,
        info: _PreloadInfo | None,
        **extra: Any,
    ) -> dict[str, Any]:
        args: dict[str, Any] = {
            "block_hash_prefix": block_hash.hex()[:16],
        }
        if info is not None:
            args.update(
                {
                    "req_id": info.req_id,
                    "preload_id": info.preload_id,
                    "preload_file_index": info.file_index,
                    "preload_num_files": info.total_files,
                }
            )
        args.update(extra)
        return args

    def _drain_cuda_copies(self) -> bool:
        if not self._pending_copies:
            return False
        made = False
        for copy in list(self._pending_copies):
            if not copy.end_event.query():
                continue
            self._pending_copies.remove(copy)
            if copy.accounting_copy_accepted:
                self._prefix_stage('completed', 'd2h' if copy.is_store else 'h2d', id(copy.end_event))
            duration_ns = int(copy.start_event.elapsed_time(copy.end_event) * 1_000_000)
            if copy.is_store:
                assert copy.job is not None
                copy.job.cuda_samples.append((copy.start_ns, duration_ns, copy.nbytes, 0))
                self._on_store_copy_done(copy)
            else:
                assert copy.members is not None
                per_job: dict[int, list[Any]] = {}
                order: list[int] = []
                for _slot_index, job, _file_index, _shared, _cache in copy.members:
                    rec = per_job.get(id(job))
                    if rec is None:
                        per_job[id(job)] = [job, 1]
                        order.append(id(job))
                    else:
                        rec[1] += 1
                for key in order:
                    job, count = per_job[key]
                    job.cuda_samples.append(
                        (
                            copy.start_ns,
                            duration_ns,
                            count * self.layout.storage_block_bytes,
                            0,
                        )
                    )
                for slot_index, job, file_index, shared, cache in copy.members:
                    self._settle_load_slot(job, file_index, slot_index, shared, cache)
                    self._file_terminal(job, ok=True)
            made = True
        return made

    def _on_store_copy_done(self, copy: _CopyOp) -> None:
        job = copy.job
        io_view = self._slot_view(copy.slot_index)
        final_path = self.file_mapper.get_file_name(job.block_hashes[copy.file_index])
        try:
            fd, temp_path = self.file_store.open_temp_write(final_path)
        except BaseException as exc:
            self._data_inflight -= 1
            self.staging_pool.release(copy.slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            self._release_store_inflight(job.block_hashes[copy.file_index], ok=False)
            return
        user_data = self._new_user_data()
        start_ns = now_ns()
        shadow_attempt = self._prefix_shadow_begin(
            "ssd_write", len(io_view), job, continuation=True
        )
        stage_dispatch = self._prefix_stage_decide(
            "ssd_write", len(io_view), ("ssd_write", user_data), job, continuation=True
        )
        try:
            self.file_store.queue_write(
                self.ring,
                user_data=user_data,
                fd=fd,
                io_array=io_view,
            )
        except BaseException as exc:
            self._prefix_stage_unknown("ssd_write", exc)
            self._prefix_shadow_settle(shadow_attempt, "uncertain", exc)
            self._prefix_stage_settle(stage_dispatch, "uncertain", exc)
            # Preserve a possibly accepted descriptor plus its original slot/FD
            # until the same fatal physical drain used for all parent siblings.
            self._inflight[user_data] = _RingOp(
                job=job, file_index=copy.file_index, slot_index=copy.slot_index,
                fd=fd, is_write=True, start_ns=start_ns, op_kind="write",
                final_path=final_path, temp_path=temp_path,
            )
            raise
        self._prefix_shadow_settle(shadow_attempt, "accepted")
        self._prefix_stage_settle(stage_dispatch, "accepted")
        self._prefix_stage('accepted', 'ssd_write', user_data, len(io_view))
        self._inflight[user_data] = _RingOp(
            job=job,
            file_index=copy.file_index,
            slot_index=copy.slot_index,
            fd=fd,
            is_write=True,
            start_ns=start_ns,
            op_kind="write",
            final_path=final_path,
            temp_path=temp_path,
        )

    def _poll_ring_completions(self) -> bool:
        completions = self.ring.poll_all()
        if not completions:
            return False
        for user_data, result in completions:
            op = self._inflight.pop(user_data, None)
            if op is None:
                continue
            kind = op.op_kind
            if kind in ('read', 'write'):
                self._prefix_stage('completed', 'ssd_' + kind, user_data, result)
            if kind == "read":
                self._data_inflight -= 1
                self._on_read_complete(op, result)
            elif kind == "open":
                self._on_open_complete(op, result)
            elif kind == "write":
                self._data_inflight -= 1
                self._on_write_complete(op, result)
            # else: close — result ignored.
        return True

    def _on_open_complete(self, op: _RingOp, result_fd: int) -> None:
        self._open_inflight -= 1
        op.path_buf = None
        preload_info = op.preload_info
        if result_fd < 0:
            exc = OSError(-result_fd, "io_uring openat failed")
            if op.preload_hash is not None:
                h = op.preload_hash
                self._preload_inflight_remove(h)
                if self._share_preload:
                    if self._fail_all_preload_waiters(h, exc):
                        return
                    if h in self._store_inflight:
                        blocked = self._preload_blocked_on_write.get(h)
                        if blocked is None:
                            blocked = collections.deque()
                            self._preload_blocked_on_write[h] = blocked
                        if preload_info is not None:
                            blocked.append(preload_info)
                    elif preload_info is not None:
                        self._remember_known_missing(preload_info)
                    return
                waiter = self._preload_waiter_pop(h)
                if waiter is not None:
                    job, _fi = waiter
                    self._file_terminal(job, ok=False, exc=exc)
                elif h in self._store_inflight:
                    # Store started while open was in flight; arm a readback when the write completes.
                    blocked = self._preload_blocked_on_write.get(h)
                    if blocked is None:
                        blocked = collections.deque()
                        self._preload_blocked_on_write[h] = blocked
                    if preload_info is not None:
                        blocked.append(preload_info)
                elif preload_info is not None:
                    self._remember_known_missing(preload_info)
            else:
                assert op.job is not None
                self._file_terminal(op.job, ok=False, exc=exc)
            return
        self._ready_sequence = getattr(self, "_ready_sequence", 0) + 1
        ready = _ReadyFd(
            fd=result_fd,
            job=op.job,
            file_index=op.file_index,
            preload_hash=op.preload_hash,
            open_start_ns=op.start_ns,
            preload_info=preload_info,
            sequence=self._ready_sequence,
        )
        if op.preload_hash is not None:
            self._ready_fds_preload.append(ready)
        else:
            self._ready_fds_load.append(ready)

    def _on_read_complete(self, op: _RingOp, result_nbytes: int) -> None:
        if op.preload_hash is not None:
            self._on_preload_read_complete(op, result_nbytes)
            return
        job = op.job
        assert job is not None
        try:
            if result_nbytes < 0:
                raise OSError(-result_nbytes, "io_uring read failed")
            if result_nbytes != self.file_store.io_size:
                raise IOError(
                    f"io_uring read transferred {result_nbytes} bytes, "
                    f"expected {self.file_store.io_size}"
                )
            duration_ns = now_ns() - op.start_ns
            job.file_samples.append((op.start_ns, duration_ns, self.layout.storage_block_bytes, op.file_index))
            self._queue_load_copy(job, op.file_index, op.slot_index)
        except BaseException as exc:
            self.staging_pool.release(op.slot_index)
            self._file_terminal(job, ok=False, exc=exc)
        finally:
            self._async_close(op.fd)

    def _on_preload_read_complete(self, op: _RingOp, result_nbytes: int) -> None:
        if self._share_preload:
            self._on_shared_preload_read_complete(op, result_nbytes)
            return
        block_hash = op.preload_hash
        assert block_hash is not None
        preload_info = op.preload_info
        self._preload_inflight_remove(block_hash)
        try:
            if result_nbytes < 0:
                raise OSError(-result_nbytes, "io_uring preload read failed")
            if result_nbytes != self.file_store.io_size:
                raise IOError(
                    f"io_uring preload read transferred {result_nbytes} bytes, "
                    f"expected {self.file_store.io_size}"
                )
            duration_ns = now_ns() - op.start_ns
            waiter = self._preload_waiter_pop(block_hash)
            add_event(
                "py_kvcache.preload.file_read",
                "kv_preload",
                op.start_ns,
                duration_ns,
                tid=preload_info.profile_tid if preload_info else None,
                args=self._preload_event_args(
                    block_hash,
                    preload_info,
                    success=True,
                    nbytes=result_nbytes,
                    has_waiter=waiter is not None,
                    cached=waiter is None and not self._stop,
                ),
            )
            if waiter is not None:
                job, file_index = waiter
                add_event(
                    "py_kvcache.preload.place_from_read",
                    "kv_preload",
                    now_ns(),
                    0,
                    tid=job.profile.profile_tid,
                    args=self._preload_event_args(
                        block_hash,
                        preload_info,
                        load_job_id=job.job_id,
                        load_req_id=job.profile.req_id,
                        load_file_index=file_index,
                    ),
                )
                self._queue_load_copy(job, file_index, op.slot_index, source="preload")
            elif self._stop:
                self.staging_pool.release(op.slot_index)
            else:
                self._preload_cache_push(block_hash, op.slot_index, preload_info)
        except BaseException as exc:
            add_event(
                "py_kvcache.preload.file_read",
                "kv_preload",
                op.start_ns,
                now_ns() - op.start_ns,
                tid=preload_info.profile_tid if preload_info else None,
                args=self._preload_event_args(
                    block_hash,
                    preload_info,
                    success=False,
                    error=type(exc).__name__,
                ),
            )
            self.staging_pool.release(op.slot_index)
            waiter = self._preload_waiter_pop(block_hash)
            if waiter is not None:
                job, _fi = waiter
                self._file_terminal(job, ok=False, exc=exc)
            else:
                logger.warning("py-kvcache preload read failed (no waiter): %s", exc)
        finally:
            self._async_close(op.fd)

    def _on_shared_preload_read_complete(self, op: _RingOp, result_nbytes: int) -> None:
        block_hash = op.preload_hash
        assert block_hash is not None
        preload_info = op.preload_info
        slot_index = op.slot_index
        self._preload_inflight_remove(block_hash)
        try:
            if result_nbytes < 0:
                raise OSError(-result_nbytes, "io_uring preload read failed")
            if result_nbytes != self.file_store.io_size:
                raise IOError(
                    f"io_uring preload read transferred {result_nbytes} bytes, "
                    f"expected {self.file_store.io_size}"
                )
            duration_ns = now_ns() - op.start_ns
            has_waiter = self._has_preload_waiter(block_hash)
            add_event(
                "py_kvcache.preload.file_read",
                "kv_preload",
                op.start_ns,
                duration_ns,
                tid=preload_info.profile_tid if preload_info else None,
                args=self._preload_event_args(
                    block_hash,
                    preload_info,
                    success=True,
                    nbytes=result_nbytes,
                    has_waiter=has_waiter,
                    shared=True,
                ),
            )
            slot = _SharedPreloadSlot(
                block_hash=block_hash,
                slot_index=slot_index,
                preload_info=preload_info,
                cached=False,
            )
            # One read serves every parked waiter for this hash.
            while True:
                waiter = self._preload_waiter_pop(block_hash)
                if waiter is None:
                    break
                job, file_index = waiter
                add_event(
                    "py_kvcache.preload.place_from_read",
                    "kv_preload",
                    now_ns(),
                    0,
                    tid=job.profile.profile_tid,
                    args=self._preload_event_args(
                        block_hash,
                        preload_info,
                        load_job_id=job.job_id,
                        load_req_id=job.profile.req_id,
                        load_file_index=file_index,
                        shared=True,
                    ),
                )
                self._shared_pin(slot)
                try:
                    self._queue_load_copy(job, file_index, slot_index, shared=slot, source="preload")
                except BaseException as exc:
                    self._shared_unpin(slot)
                    self._file_terminal(job, ok=False, exc=exc)
                self._shared_decref(job.profile.req_id, block_hash)
            if self._stop:
                self._maybe_release_shared(slot)
            elif self._preload_refcount.get(block_hash, 0) > 0:
                self._shared_cache_insert(slot)
            else:
                self._maybe_release_shared(slot)
        except BaseException as exc:
            add_event(
                "py_kvcache.preload.file_read",
                "kv_preload",
                op.start_ns,
                now_ns() - op.start_ns,
                tid=preload_info.profile_tid if preload_info else None,
                args=self._preload_event_args(
                    block_hash,
                    preload_info,
                    success=False,
                    error=type(exc).__name__,
                    shared=True,
                ),
            )
            if not self._fail_all_preload_waiters(block_hash, exc):
                logger.warning("py-kvcache shared preload read failed (no waiter): %s", exc)
            self.staging_pool.release(slot_index)
        finally:
            self._async_close(op.fd)

    def _on_write_complete(self, op: _RingOp, result_nbytes: int) -> None:
        job = op.job
        try:
            if result_nbytes < 0:
                raise OSError(-result_nbytes, "io_uring write failed")
            if result_nbytes != self.file_store.io_size:
                raise IOError(
                    f"io_uring write transferred {result_nbytes} bytes, "
                    f"expected {self.file_store.io_size}"
                )
            self.file_store.finish_write(
                fd=op.fd,
                temp_path=op.temp_path,
                final_path=op.final_path,
            )
            duration_ns = now_ns() - op.start_ns
            job.file_samples.append((op.start_ns, duration_ns, self.layout.storage_block_bytes, op.file_index))
            # The store slot view is byte-identical to a read of the same hash, so
            # retain it as a cache entry instead of releasing (write-back).
            self._release_or_cache(job.block_hashes[op.file_index], op.slot_index)
            self._file_terminal(job, ok=True)
            self._release_store_inflight(job.block_hashes[op.file_index], ok=True)
        except BaseException as exc:
            self.file_store.cleanup_temp(fd=op.fd, temp_path=op.temp_path)
            self.staging_pool.release(op.slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            self._release_store_inflight(job.block_hashes[op.file_index], ok=False)

    def _can_issue_store(self) -> bool:
        # Device budget releases at CQE reap, not CUDA copy completion.
        return self._data_inflight < self.iodepth

    def _can_open(self) -> bool:
        primed = self._open_inflight + len(self._ready_fds_load) + len(self._ready_fds_preload)
        return primed < self.open_lookahead

    def _has_real_load_pressure(self) -> bool:
        # CUDA copies (staging->GPU) don't count: they're async DMA, not disk I/O.
        if self._ready_fds_load:
            return True
        if any(
            op.op_kind in ("open", "read") and op.job is not None and not op.job.is_store
            for op in self._inflight.values()
        ):
            return True
        for job in self._active:
            if job.is_store or job.failed is not None or job.future_set:
                continue
            idx = job.next_file_index
            if idx < job.total_files:
                h = job.block_hashes[idx]
                if not self._preload_has_copy(h):
                    return True
        return False

    def _can_issue_speculative_preload(self) -> bool:
        return not self._has_real_load_pressure()

    def _preload_slots_available(self) -> bool:
        occupied = self._preload_cached_total + self._preload_inflight_total
        return occupied < self._max_preload_slots

    def _evict_one_preload_slot(self) -> bool:
        if self._share_preload:
            return self._evict_one_shared_slot()
        if not self._preload_slots:
            return False
        block_hash = next(iter(self._preload_slots))
        entry = self._preload_cache_pop(block_hash)
        assert entry is not None
        slot_index, preload_info = entry
        add_event(
            "py_kvcache.preload.evict_slot",
            "kv_preload",
            now_ns(),
            0,
            tid=preload_info.profile_tid if preload_info else None,
            args=self._preload_event_args(block_hash, preload_info, reason="load_slot_pressure"),
        )
        self.staging_pool.release(slot_index)
        return True

    def _evict_one_cache_slot(self, reason: str) -> bool:
        if self._staging_cache is None:
            return False
        slot_index = self._staging_cache.evict_one()
        if slot_index is None:
            return False
        self.staging_pool.release(slot_index)
        add_event(
            "py_kvcache.staging_cache.evict",
            "kv_staging_cache",
            now_ns(),
            0,
            args={"reason": reason, "size": len(self._staging_cache)},
        )
        return True

    def _reserve_foreground_slot(self):
        # Slot priority: foreground > preload > cache, so reclaim cache then preload.
        slot = self.staging_pool.try_reserve()
        if slot is None and self._evict_one_cache_slot("foreground_pressure"):
            slot = self.staging_pool.try_reserve()
        if slot is None and self._evict_one_preload_slot():
            slot = self.staging_pool.try_reserve()
        self._prefix_p4_note_reservation(slot)
        return slot

    def _reserve_preload_slot(self):
        # Preload outranks the cache: a speculative read may reclaim a cache slot.
        slot = self.staging_pool.try_reserve()
        if slot is None and self._evict_one_cache_slot("preload_pressure"):
            slot = self.staging_pool.try_reserve()
        self._prefix_p4_note_reservation(slot)
        return slot

    def _release_or_cache(self, block_hash: bytes, slot_index: int) -> None:
        if getattr(self, '_stop', False) or self._staging_cache is None:
            self.staging_pool.release(slot_index)
            return
        to_release = self._staging_cache.put(block_hash, slot_index)
        if to_release is not None:
            self.staging_pool.release(to_release)
        if to_release != slot_index:
            add_event(
                "py_kvcache.staging_cache.insert",
                "kv_staging_cache",
                now_ns(),
                0,
                args={"size": len(self._staging_cache)},
            )

    def _settle_load_slot(
        self,
        job: "_ReactorJob",
        file_index: int,
        slot_index: int,
        shared: "_SharedPreloadSlot | None",
        cache: "_CacheSlot | None",
    ) -> None:
        # Immutable hash: disk bytes stay valid regardless of GPU copy outcome.
        if cache is not None:
            self._staging_cache.unpin(cache)
        elif shared is None:
            self._release_or_cache(job.block_hashes[file_index], slot_index)
        else:
            self._shared_unpin(shared)
            self._maybe_release_shared(shared)

    def _schedule_cache_hit(
        self, job: "_ReactorJob", file_index: int, cache_slot: "_CacheSlot"
    ) -> None:
        # Non-destructive: the slot stays in the cache, pinned for the copy's DMA.
        add_event(
            "py_kvcache.staging_cache.hit",
            "kv_staging_cache",
            now_ns(),
            0,
            tid=job.profile.profile_tid,
            args={"load_job_id": job.job_id, "load_req_id": job.profile.req_id},
        )
        self._staging_cache.pin(cache_slot)
        job.next_file_index += 1
        job.inflight_files += 1
        try:
            self._queue_load_copy(job, file_index, cache_slot.slot_index, cache=cache_slot, source="cache")
        except BaseException as exc:
            self._staging_cache.unpin(cache_slot)
            self._file_terminal(job, ok=False, exc=exc)

    def _prefix_stage(self, event, stage, key, value=None):
        accounting = getattr(self, "_prefix_stage_accounting", None)
        if accounting is None or not accounting.valid:
            return
        try:
            if event == "accepted":
                accounting.accepted(stage, key, value)
            else:
                accounting.completed(stage, key, value)
        except Exception as exc:
            accounting.invalidate(type(exc).__name__ + ": " + str(exc))

    def _prefix_stage_unknown(self, stage, exc):
        accounting = getattr(self, "_prefix_stage_accounting", None)
        if accounting is not None and accounting.valid:
            # Legacy invalidate clears records. At ambiguous native boundaries
            # retain the existing bounded accepted-key/byte facts as evidence.
            # Sticky invalidity cannot establish zero execution or completion.
            accounting.valid = False
            accounting.error = (stage + " physical completion/acceptance unknown: " +
                                type(exc).__name__ + ": " + str(exc))[:160]

    def _prefix_stage_copy(self, stage, end_event, sizes, count):
        accounting = getattr(self, "_prefix_stage_accounting", None)
        if accounting is None or not accounting.valid:
            return
        try:
            accounting.accepted(stage, id(end_event), int(sizes[:count].sum()))
        except Exception as exc:
            accounting.invalidate(type(exc).__name__ + ": " + str(exc))

    def _prefix_capacity_components(self):
        """O(1) owner observation; future.done is never a resource witness."""
        if getattr(self, "_prefix_capacity_valid", False) is False:
            return None
        try:
            if type(self._prefix_capacity_valid) is not bool:
                raise ValueError("capacity validity flag must be bool")
            if getattr(self, "_worker", None) is None or self._worker.ident != threading.get_ident():
                raise ValueError("capacity sample requires actual native owner")
            if (self._stop or getattr(self.staging_pool, "_closed", False) or
                    getattr(self, "_native_drain_unknown", False) or
                    getattr(self, "_native_fatal_reason", None) is not None):
                return None
            cache = self._staging_cache
            cache_count = len(cache) if cache is not None else 0
            cache_pinned = cache.pinned_slot_count if cache is not None else 0
            if cache is not None and not cache.observation_valid:
                raise ValueError(cache.observation_error or "cache observation unavailable")
            shared_count = len(self._shared_cached)
            shared_pinned = self._prefix_shared_cached_pinned_slots
            preload_total = self._preload_cached_total
            free = self.staging_pool.free_count
            slot_count = self.staging_pool.slot_count
            io_size = self.file_store.io_size
            if any(type(v) is not int for v in
                   (cache_count, cache_pinned, shared_count, shared_pinned, preload_total,
                    free, slot_count, io_size, self._prefix_shared_cached_observed_count)):
                raise ValueError("capacity counts must be exact ints")
            ordinary = preload_total - shared_count
            if (slot_count <= 0 or io_size <= 0 or
                    not 0 <= free <= slot_count or not 0 <= cache_pinned <= cache_count or
                    not 0 <= shared_pinned <= shared_count or ordinary < 0 or
                    self._prefix_shared_cached_observed_count != shared_count or
                    len(self._preload_slots) > ordinary or
                    cache_count + preload_total + free > slot_count):
                raise ValueError("capacity scalar conservation/cardinality mismatch")
            cache_clean = cache_count - cache_pinned
            shared_clean = shared_count - shared_pinned
            reclaimable = free + cache_clean + ordinary + shared_clean
            if not 0 <= reclaimable <= slot_count:
                raise ValueError("reclaimable capacity outside original pool")
            return (slot_count, io_size, free, cache_count, cache_pinned, cache_clean,
                    shared_count, shared_pinned, shared_clean, preload_total, ordinary,
                    reclaimable)
        except Exception as exc:
            self._prefix_invalidate_capacity(type(exc).__name__ + ": " + str(exc))
            return None

    def _prefix_capacity_snapshot(self, *, native_shutdown=False):
        keys = ("slot_count", "io_size", "free_slots", "cache_slots",
                "cache_pinned_slots", "cache_clean_slots", "shared_cached_slots",
                "shared_cached_pinned_slots", "shared_clean_slots", "preload_cached_slots",
                "ordinary_preload_clean_slots", "foreground_reclaimable_slots")
        values = None
        if not native_shutdown:
            try:
                values = self._prefix_capacity_components()
            except Exception as exc:
                self._prefix_invalidate_capacity(type(exc).__name__ + ": " + str(exc))
        valid = values is not None
        result = dict(zip(keys, values if valid else (None,) * len(keys)))
        sticky_valid = getattr(self, "_prefix_capacity_valid", False) is True
        reason = getattr(self, "_prefix_capacity_error", None)
        if not valid and reason is None:
            reason = ("native_shutdown" if native_shutdown else
                      "native_stop" if getattr(self, "_stop", False) else
                      "native_capacity_unavailable")
        result.update(valid=valid, observation_valid=sticky_valid, error=reason,
                      free_reclaimable_staging_bytes=(values[-1] * values[1] if valid else None),
                      capacity_semantics="foreground_clean_reclaimable",
                      physical_release_credit=False)
        return result

    def _prefix_clean_reclaimable_bytes(self):
        try:
            values = self._prefix_capacity_components()
            return values[-1] * values[1] if values is not None else None
        except Exception as exc:
            self._prefix_invalidate_capacity(type(exc).__name__ + ": " + str(exc))
            return None

    def _prefix_stage_decide(self, stage, nbytes, work_id, job=None, *,
                             continuation=False, support=False, reserved_bytes=0, ready=None):
        controller = getattr(self, "_prefix_dispatch_controller", None)
        p4 = getattr(self, "_prefix_p4_bridge", None)
        if controller is None and p4 is None:
            return None  # Off: no clock, snapshot, policy import or copy-byte sum.
        try:
            from prefix_io_control.dispatch_budget import Amount, STAGES
            from prefix_io_control.dispatch_shadow import ShadowState
            now = time.monotonic_ns()
            accounting = self._prefix_stage_accounting
            inflight = None
            if accounting.valid:
                accounting._owner()
                inflight = tuple(
                    Amount(accounting.stats[name]["inflight_ops"],
                           accounting.stats[name]["inflight_bytes"])
                    for name in STAGES
                )
            free = self._prefix_clean_reclaimable_bytes()
            if free is not None:
                # The candidate's original native reserve succeeded; reconstruct
                # its clean capacity before that reserve, not a new allocation.
                free += reserved_bytes
            parents = self.accepted_parent_count()
            sample = ShadowState(
                controller.run_id if controller is not None else p4.run_id, now, inflight=inflight,
                free_staging_bytes=free, accepted_parents=parents,
                # Original device/slot/FD checks succeeded. The unchanged copy
                # launch still enforces compute-event stream ordering before DMA.
                native_issue_safe=True,
            )
            progress = (
                "shutdown" if self._stop else
                "continuation" if continuation or job is not None and job.failed is not None else
                "mandatory" if job is not None and self.is_mandatory(job.future) else
                "mandatory_support" if support else None
            )
            if p4 is not None:
                try:
                    from dataclasses import replace
                    from prefix_io_control.p4_types import WorkDescriptor
                    single_preload = None
                    if (p4.policy.single_file is not None and type(ready) is _ReadyFd and
                            ready.job is None and ready.file_index == -1 and
                            type(ready.sequence) is int and ready.sequence > 0 and
                            type(ready.open_start_ns) is int and 0 < ready.open_start_ns <= now and
                            type(ready.preload_hash) is bytes and len(ready.preload_hash) == 32 and
                            type(ready.preload_info) is _PreloadInfo and
                            ready.preload_info.block_hash == ready.preload_hash and
                            ready.preload_info.total_files == 1 and ready.preload_info.file_index == 0 and
                            bool(ready.preload_info.preload_id) and bool(ready.preload_info.req_id)):
                        # Original immutable open time and owner sequence, never
                        # a policy-created/restarted age or a synthetic parent.
                        single_preload = ("preload:" + str(ready.sequence) + ":" +
                                          ready.preload_hash.hex(), ready.open_start_ns)
                    view = self._prefix_p4_collect()
                    if view is not None and (job is not None or progress is not None or single_preload is not None):
                        # Unbound ordinary preload has no native arrival proof:
                        # keep its original path, rather than reset a policy age.
                        preview_now = time.monotonic_ns()
                        snapshot = replace(view.snapshot,native_state=sample)
                        child = work_id[-1] if type(work_id) is tuple else work_id
                        if type(child) not in (int,str) or type(child) is int and child < 0:
                            child = str(work_id)[:128]
                        if single_preload is not None:
                            child = single_preload[0]
                        actual = WorkDescriptor(p4.run_id,snapshot.snapshot_epoch,
                            job.accepted_parent_sequence if job is not None else 0,
                            child,stage,nbytes,0,
                            job.profile.start_ns if job is not None else
                                single_preload[1] if single_preload is not None else preview_now,
                            False,True,reserved_bytes,
                            progress=progress,minimum_unit_bytes=(
                                self.file_store.io_size if stage in ("ssd_read","ssd_write") else
                                sum(self.layout.bytes_per_kernel_block)))
                        preview = p4.preview_issue(actual,snapshot,now_ns=preview_now)
                        if preview.action == "defer" and progress is None:
                            if (p4.policy.single_file is not None and
                                    not p4.record_single_file_deferral(actual.work_id, at_ns=time.monotonic_ns())):
                                return None
                            return _StageDispatch(_P4Deferred(),work_id)
                except Exception as p4_exc:
                    p4.fail(type(p4_exc).__name__+": "+str(p4_exc))
            if controller is None:
                return None  # unsupported I/J preserve U; no new allowance owner
            decision = controller.decide(
                stage, nbytes, sample, now_ns=now, work_id=work_id,
                staging_bytes_needed=reserved_bytes, progress=progress
            )
            return _StageDispatch(decision, work_id)
        except Exception as exc:
            if controller is not None:
                controller.fail(type(exc).__name__ + ": " + str(exc))
            elif p4 is not None:
                p4.fail(type(exc).__name__ + ": " + str(exc))
            return None  # Failed optional performance control uses original dispatch.

    def _prefix_stage_settle(self, dispatch, outcome, exc=None):
        if dispatch is None or dispatch.attempt is None:
            return
        controller = self._prefix_dispatch_controller
        try:
            if outcome == "accepted":
                controller.accepted(dispatch.attempt)
            elif outcome == "rejected":
                controller.rejected(dispatch.attempt, type(exc).__name__ + ": " + str(exc),
                                    proved=True)
            else:
                controller.uncertain(dispatch.attempt, type(exc).__name__ + ": " + str(exc))
            controller.retire_work(dispatch.attempt.stage, dispatch.work_id)
        except Exception as audit_exc:
            controller.fail(type(audit_exc).__name__ + ": " + str(audit_exc))

    def _prefix_stage_retire(self, stage, work_id):
        controller = getattr(self, "_prefix_dispatch_controller", None)
        if controller is not None:
            try:
                controller.retire_work(stage, work_id)
            except Exception as exc:
                controller.fail(type(exc).__name__ + ": " + str(exc))

    def _prefix_stage_completion_unknown(self, stage, dispatch, exc):
        if dispatch is None:
            return
        controller = self._prefix_dispatch_controller
        try:
            controller.mark_completion_unknown(stage, type(exc).__name__ + ": " + str(exc))
        except Exception as audit_exc:
            controller.fail(type(audit_exc).__name__ + ": " + str(audit_exc))

    def _prefix_ready_read_decision(self, ready):
        controller = getattr(self, "_prefix_dispatch_controller", None)
        if controller is None and getattr(self,"_prefix_p4_bridge",None) is None:
            return None
        if ready.sequence <= 0:
            self._ready_sequence = getattr(self, "_ready_sequence", 0) + 1
            ready.sequence = self._ready_sequence
        support = False
        if ready.preload_hash is not None:
            waiters = self._preload_waiters.get(ready.preload_hash, ())
            if len(waiters) <= 64:
                support = any(self.is_mandatory(job.future) for job, _index in waiters)
            elif self._has_preload_waiter(ready.preload_hash):
                support = True  # Conservative progress superset, never release credit.
        return self._prefix_stage_decide(
            "ssd_read", self.file_store.io_size, ("ssd_read", ready.sequence), ready.job,
            support=support, reserved_bytes=self.file_store.io_size, ready=ready
        )

    def _prefix_shadow_begin(self, stage, nbytes, job=None, *, continuation=False):
        shadow = getattr(self, "_prefix_dispatch_shadow", None)
        if shadow is None or shadow.mode != "shadow" or shadow.faulted:
            return None  # Off: no clock, snapshot, ticket, or policy import.
        try:
            from prefix_io_control.dispatch_budget import Amount, STAGES
            from prefix_io_control.dispatch_shadow import ShadowState
            accounting = self._prefix_stage_accounting
            captured = time.monotonic_ns()
            inflight = None
            if accounting.valid:
                accounting._owner()
                inflight = tuple(
                    Amount(accounting.stats[name]["inflight_ops"],
                           accounting.stats[name]["inflight_bytes"])
                    for name in STAGES
                )
            sample = ShadowState(
                shadow.run_id, captured, inflight=inflight,
                free_staging_bytes=None, accepted_parents=None,
                # Called only immediately before an existing native API after its
                # original slot, FD, mapping and stream dependency path.
                native_issue_safe=True,
            )
            progress = (
                "shutdown" if self._stop else
                "mandatory" if job is not None and self.is_mandatory(job.future) else
                "continuation" if continuation else None
            )
            return shadow.begin(stage, nbytes, sample, now_ns=captured, progress=progress)
        except Exception as exc:
            shadow.fail(type(exc).__name__ + ": " + str(exc))
            return None  # Optional observation cannot block native dispatch.

    def _prefix_shadow_copy_begin(self, stage, sizes, count, job=None, *, continuation=False):
        shadow = getattr(self, "_prefix_dispatch_shadow", None)
        if shadow is None or shadow.mode != "shadow" or shadow.faulted:
            return None  # Never evaluate copy sizes on the original off path.
        try:
            return self._prefix_shadow_begin(
                stage, int(sizes[:count].sum()), job, continuation=continuation
            )
        except Exception as exc:
            shadow.fail(type(exc).__name__ + ": " + str(exc))
            return None

    def _prefix_shadow_settle(self, attempt, outcome, exc=None):
        if attempt is None:
            return
        shadow = self._prefix_dispatch_shadow
        try:
            if outcome == "accepted":
                shadow.accepted(attempt)
            else:
                # A backend exception is not evidence of pre-acceptance rejection.
                shadow.uncertain(attempt, type(exc).__name__ + ": " + str(exc))
        except Exception as audit_exc:
            shadow.fail(type(audit_exc).__name__ + ": " + str(audit_exc))

    def _prefix_shadow_completion_unknown(self, stage, attempt, exc):
        if attempt is None:
            return
        shadow = self._prefix_dispatch_shadow
        try:
            shadow.mark_completion_unknown(stage, type(exc).__name__ + ": " + str(exc))
        except Exception as audit_exc:
            shadow.fail(type(audit_exc).__name__ + ": " + str(audit_exc))

    def _prefix_allow_start(self, kind, job=None):
        budget = getattr(self, "_prefix_start_budget", None)
        if budget is None:
            return None  # Off: no ticket, clock, counter, or policy import.
        if budget.faulted:
            return None  # A failed optional gate stays on the native path.
        try:
            state = self._prefix_progress
            return budget.allow(
                kind, job.future if job is not None else None,
                now_ns=time.monotonic_ns(), free_slots=self.staging_pool.free_count,
                mandatory=job is not None and self.is_mandatory(job.future),
                # Conservative support superset, not a release-dependency policy:
                # any active mandatory wait permits native stores to make progress.
                support=kind == "store" and bool(state.required),
                continuation=(
                    kind == "load" and self._has_inflight_preload(
                        job.block_hashes[job.next_file_index]
                    )
                ), shutdown=self._stop,
            )
        except Exception as exc:
            budget.fail(type(exc).__name__)
            return None  # Optional performance gate cannot fail native work.

    def _prefix_commit_start(self, ticket):
        if ticket is None:
            return
        budget = self._prefix_start_budget
        try:
            budget.commit(ticket)
        except Exception as exc:
            budget.fail(type(exc).__name__)

    def _prefix_p4_note_reservation(self, slot):
        bridge = getattr(self, "_prefix_p4_bridge", None)
        if bridge is None or slot is None:
            return
        try:
            bridge.note_reservation(slot.index)
        except Exception as exc:
            bridge.fail(type(exc).__name__ + ": " + str(exc))

    def _prefix_p4_parent_terminal(self, job, *, successful):
        bridge = getattr(self, "_prefix_p4_bridge", None)
        if bridge is None:
            return  # off before clock/import or scalar owner inspection
        pid = getattr(job, "accepted_parent_sequence", None)
        if type(pid) is not int or pid < 1:
            return
        try:
            if self._worker.ident != threading.get_ident():
                raise RuntimeError("closure drain sampling requires native owner")
            physical_success = (successful and job.failed is None and
                job.done_files == job.total_files and job.inflight_files == 0 and
                job.next_file_index == job.total_files and
                not getattr(job, "accepted_parent_retired", False))
            bridge.complete_native_closure(pid, now_ns=time.monotonic_ns(),
                successful=bool(physical_success),
                drain_known=not getattr(self, "_native_drain_unknown", False))
        except Exception as exc:
            bridge.fail(type(exc).__name__ + ": " + str(exc))


    def publish_p4_scheduled_load(self, observation):
        """One immutable scalar publication; no work queue or awaited Future."""
        bridge = getattr(self, "_prefix_p4_bridge", None)
        if bridge is None:
            return False  # off before optional imports, clocks or metadata scan
        from prefix_io_control.p4_load_observation import SchedulerLoadObservation, SchedulerLoadUnavailable
        if type(observation) not in (SchedulerLoadObservation, SchedulerLoadUnavailable):
            raise TypeError("strict immutable scheduler work values required")
        if observation.run_id != bridge.run_id:
            raise ValueError("scheduler work belongs to a different run")
        with self._submit_lock:
            if self._closed:
                raise RuntimeError("native intake closed")
            previous = getattr(self, "_prefix_p4_load_frame", None)
            sequence = getattr(self, "_prefix_p4_load_sequence", 0)
            if observation.sequence < sequence:
                raise ValueError("scheduler work sequence moved backwards")
            if observation.sequence == sequence:
                if observation == previous:
                    return False
                # Conflicting same-step metadata invalidates the previous view.
                self._prefix_p4_load_frame = None
                raise ValueError("conflicting same-sequence scheduled work")
            self._prefix_p4_load_frame = observation
            self._prefix_p4_load_sequence = observation.sequence
        return True

    def _prefix_p4_load_values(self, now_ns):
        bridge = getattr(self, "_prefix_p4_bridge", None)
        if bridge is None:
            return (), None
        lock = getattr(self, "_submit_lock", None)
        if lock is None:
            frame = getattr(self, "_prefix_p4_load_frame", None)
        else:
            with lock:
                frame = getattr(self, "_prefix_p4_load_frame", None)
        from prefix_io_control.p4_load_observation import SchedulerLoadObservation
        if type(frame) is not SchedulerLoadObservation or not frame.fresh(
                run_id=bridge.run_id, now_ns=now_ns,
                max_age_ns=bridge.policy.config.sample_max_age_ns):
            return (), None
        return frame.signature, frame

    def _prefix_p4_existing_io_state(self, now_ns, free):
        accounting = getattr(self, "_prefix_stage_accounting", None)
        if accounting is None or not accounting.valid:
            return None
        try:
            from prefix_io_control.dispatch_budget import Amount, STAGES
            from prefix_io_control.dispatch_shadow import ShadowState
            accounting._owner()
            return ShadowState(self._prefix_p4_bridge.run_id, now_ns,
                inflight=tuple(Amount(accounting.stats[s]["inflight_ops"],
                    accounting.stats[s]["inflight_bytes"]) for s in STAGES),
                free_staging_bytes=free,
                accepted_parents=self.accepted_parent_count(),
                native_issue_safe=None)
        except Exception:
            return None  # missing owner facts are unknown rather than zero

    def _prefix_p4_read_batch_ids(self):
        bridge = getattr(self, "_prefix_p4_bridge", None)
        if bridge is None:
            return None
        try:
            view = self._prefix_p4_collect()
            if view is None:
                return None
            ids = tuple((r.job.accepted_parent_sequence, r.file_index, "ssd_read")
                for r in self._ready_fds_load if r.job is not None and r.job.failed is None)
            by_id = {w.work_id:w for w in view.works}
            works = tuple(by_id[wid] for wid in ids if wid in by_id)
            if not works:
                return None
            return bridge.batch_prefix(view, works, now_ns=time.monotonic_ns())
        except Exception as exc:
            bridge.fail(type(exc).__name__+": "+str(exc))
            return None  # optional advice cannot drop accepted native work

    def _prefix_p4_collect(self):
        """Owner-only bounded values; original queues and owners stay untouched."""
        bridge = getattr(self, "_prefix_p4_bridge", None)
        if bridge is None:
            return None  # off before clock, snapshot, imports, or scanning
        from prefix_io_control.p4_bridge import NativePublication
        from prefix_io_control.p4_eta import ClosureGeometry
        from prefix_io_control.p4_types import SystemSnapshot, WorkDescriptor, WaitingTarget, ReleaseWitness
        from prefix_io_control.dependencies import Parent, Resource, ResourceId
        if self._worker.ident != threading.get_ident():
            raise RuntimeError("P4 collection requires actual native owner")
        if len(self._active) > 32 or (len(self._ready_fds_load) + len(self._copy_ready) +
                sum(1 for j in self._active if j.is_store and j.failed is None and
                    j.next_file_index < j.total_files)) > 64:
            bridge.last_reason = "native_observation_window_exceeded"
            return None  # no truncation or dropping of native jobs
        now = time.monotonic_ns()
        run = bridge.run_id
        load_signature, load_frame = self._prefix_p4_load_values(now)
        jobs, order = {}, {}
        for index, job in enumerate(self._active):
            pid = job.accepted_parent_sequence
            if type(pid) is not int or pid < 1 or job.accepted_parent_retired:
                bridge.last_reason = "unknown_native_parent_identity"
                return None
            jobs[pid] = job
            order[pid] = index
        raw = []
        for ready in self._ready_fds_load:
            if ready.job is None or ready.job.failed is not None:
                continue
            if ready.sequence <= 0:
                self._ready_sequence = getattr(self, "_ready_sequence", 0) + 1
                ready.sequence = self._ready_sequence
            raw.append((ready.job, ready.file_index, "ssd_read", self.file_store.io_size,
                        self.file_store.io_size, None, None))
        for job in self._active:
            if job.is_store and job.failed is None and job.next_file_index < job.total_files:
                raw.append((job, job.next_file_index, "d2h", self.layout.storage_block_bytes,
                            self.file_store.io_size, None, None))
        for rc in self._copy_ready:
            generation = bridge.generation(rc.slot_index)
            identity = (ResourceId(run, "cpu-staging", 0, rc.slot_index, generation)
                        if generation is not None else None)
            raw.append((rc.job, rc.file_index, "h2d", rc.nbytes, 0, generation, identity))
        by_parent = {pid: [] for pid in jobs}
        for item in raw:
            pid = item[0].accepted_parent_sequence
            if pid not in jobs:
                bridge.last_reason = "ready_parent_outside_native_owner_window"
                return None
            if item[2] != "d2h":
                by_parent[pid].append(item)
        parents = []
        complete_ready = set()
        for pid, job in jobs.items():
            rows = by_parent[pid]
            files = {x[1] for x in rows}
            complete = (job.failed is None and not job.future_set and
                        job.next_file_index == job.total_files and
                        len(files) == len(rows) == job.inflight_files and
                        job.done_files + len(files) == job.total_files and len(files) > 0)
            depth = max((2 if x[2] == "ssd_read" else 1 for x in rows), default=0) if complete else None
            if complete:
                complete_ready.add(pid)
                # Measure the native whole closure, including original queues and
                # fused copies. Per-file timings cannot predict future fusion.
                grouped_rows = {}
                for row in rows:
                    quantum = self.file_store.io_size if row[2] == "ssd_read" else row[3]
                    key = (row[2], row[3], quantum)
                    grouped_rows[key] = grouped_rows.get(key, 0) + 1
                geometry = ClosureGeometry(tuple(sorted(
                    key + (count,) for key, count in grouped_rows.items())))
                context = ("native-ready-closure-v1", self.file_store.io_size,
                    self.layout.storage_block_bytes,
                    len(self._active), len(self._ready_fds_load), len(self._copy_ready),
                    len(getattr(self, "_pending_copies", ())),
                    len(getattr(self, "_inflight", ())),
                    getattr(self, "_data_inflight", 0),
                    getattr(self, "_open_inflight", 0),
                    getattr(self, "iodepth", 0),
                    load_frame.source_sha256 if load_frame is not None else "scheduler-source-unknown",
                    load_frame.geometry_sha256 if load_frame is not None else "scheduled-work-unknown")
                bridge.observe_native_closure(pid, geometry, context, now_ns=now)
            parents.append(Parent(run, pid, job.total_files, job.done_files, job.inflight_files,
                bool(job.future_set), job.failed is not None, depth, None))
        free = self._prefix_clean_reclaimable_bytes()
        native_state = self._prefix_p4_existing_io_state(now, free)
        io_signature = tuple((a.ops,a.nbytes) for a in native_state.inflight) if native_state is not None else ()
        signature = (tuple((p.job_id,p.total_files,p.done_files,p.inflight_files,p.future_done,p.failed,
                           p.remaining_stages) for p in parents),
                     tuple((j.accepted_parent_sequence,fi,stage,nb,need,gen) for j,fi,stage,nb,need,gen,rid in raw),
                     tuple((rc.slot_index,rc.shared.copies_inflight if rc.shared else
                            rc.cache.copies_inflight if rc.cache else 1,
                            bool(rc.shared.cached) if rc.shared else False)
                           for rc in self._copy_ready), free,
                     bool(self._stop), tuple(sorted(j.accepted_parent_sequence for j in self._active
                                               if self.is_mandatory(j.future))),
                     io_signature, load_signature,
                     load_frame.sequence if load_frame is not None else None)
        epoch = bridge.epoch(signature, now)
        works = []
        for idx,(job,fi,stage,nbytes,need,generation,identity) in enumerate(raw):
            progress = ("shutdown" if self._stop else "continuation" if stage == "h2d" else
                        "mandatory" if self.is_mandatory(job.future) else None)
            works.append(WorkDescriptor(run, epoch, job.accepted_parent_sequence, fi, stage,
                nbytes, idx, job.profile.start_ns, False, True, need, generation, progress,
                (self.file_store.io_size if stage == "ssd_read" else
                 self.layout.storage_block_bytes if stage == "d2h" else nbytes), identity))
        targets = []
        for pid, job in jobs.items():
            if job.failed is not None or job.future_set:
                continue
            if not job.is_store:
                targets.append(WaitingTarget("restore:"+str(pid),order[pid],
                    job.profile.start_ns,restore_parent_id=pid))
            elif free == 0 and job.next_file_index < job.total_files:
                targets.append(WaitingTarget("cpu-slot:"+str(pid),order[pid],
                    job.profile.start_ns,need_cpu_bytes=self.file_store.io_size))
        parent_map = {p.job_id:p for p in parents}
        generations, witnesses = {}, []
        for work in works:
            if work.resource_identity is not None:
                generations[work.resource_identity] = work.generation
        for target in targets:
            pid = target.restore_parent_id
            if pid not in complete_ready or len(witnesses) == 8:
                continue
            rows = tuple(w.work_id for w in works if w.parent_id == pid and w.stage in ("ssd_read","h2d"))
            identity = ResourceId(run,"restore-parent",0,pid,pid)
            if identity not in generations and len(generations) >= 64:
                bridge.generation_witness_omissions += 1
                continue  # omit optional witness, retain full native ready window
            generations[identity] = pid
            resource = Resource(identity,jobs[pid].transfer_size,0,frozenset((pid,)),
                                "observed_blocking",False,None)
            witnesses.append(ReleaseWitness(target.target_id,resource,(parent_map[pid],),
                "restore",rows,"py_kvcache.reactor.restore-parent",None,
                completed_restore_parent_id=pid))
        # CPU source pin evidence is conservative parent-AND. Continuations are
        # never delayed/reordered by policy even if this potential path is seen.
        cpu_target = next((t for t in targets if t.need_cpu_bytes),None)
        grouped = {}
        for rc in self._copy_ready:
            grouped.setdefault(rc.slot_index,[]).append(rc)
        for slot_index, members in grouped.items():
            if cpu_target is None or len(witnesses) == 8:
                break
            generation = bridge.generation(slot_index)
            pids = frozenset(rc.job.accepted_parent_sequence for rc in members)
            if generation is None or not pids.issubset(complete_ready):
                continue
            shared = members[0].shared
            cache = members[0].cache
            if any(rc.shared is not shared or rc.cache is not cache for rc in members):
                continue
            if shared is not None and (shared.slot_index != slot_index or
                    shared.copies_inflight != len(members) or
                    shared.cached and self._shared_cached.get(shared.block_hash) is not shared):
                continue
            if cache is not None and (cache.slot_index != slot_index or
                    cache.copies_inflight != len(members) or
                    self._staging_cache is None or
                    self._staging_cache._slots.get(cache.block_hash) is not cache):
                continue
            # Original raw read slots have one copy owner; no unknown consumers.
            if shared is None and cache is None and len(members) != 1:
                continue
            identity = ResourceId(run,"cpu-staging",0,slot_index,generation)
            generations[identity] = generation
            rows = tuple(w.work_id for w in works if w.parent_id in pids and w.stage in ("ssd_read","h2d"))
            resource = Resource(identity,self.file_store.io_size,0,pids,
                                "observed_blocking",False,None)
            witnesses.append(ReleaseWitness(cpu_target.target_id,resource,
                tuple(parent_map[pid] for pid in sorted(pids)),"cpu",rows,
                "py_kvcache.reactor.cpu-staging",None,
                release_scope="multi_protector" if len(pids)>1 else "parent_job"))
        caps = frozenset(("native_ready_work","owner_release_protocol",
            "restore_parent_completion","cpu_owner_generation","cpu_active_refs","cpu_protectors"))
        if native_state is not None:
            caps |= frozenset(("native_existing_io",))
        if load_frame is not None:
            caps |= frozenset(("scheduled_work_observation",))
        snapshot = SystemSnapshot(run,epoch,now,caps,tuple(parents),native_state,
            tuple(targets),tuple(generations.items()),None,free,load_signature)
        return NativePublication(snapshot,tuple(works),tuple(witnesses))

    def _prefix_p4_order(self, entries, kind):
        bridge = getattr(self, "_prefix_p4_bridge", None)
        if bridge is None:
            return entries  # off before clock, snapshot or candidate iteration
        try:
            view = self._prefix_p4_collect()
            if view is None:
                return entries
            if kind == "stores":
                ids = [(j.accepted_parent_sequence,j.next_file_index,"d2h") for j in entries
                       if j.is_store and j.failed is None and j.next_file_index < j.total_files]
            else:
                ids = [(r.job.accepted_parent_sequence,r.file_index,"ssd_read") for r in entries
                       if r.job is not None and r.job.failed is None]
            candidates = tuple(w for w in view.works if w.work_id in set(ids))
            if not candidates:
                return entries  # empty pump is not a dispatch boundary
            selected = bridge.order(view,candidates,now_ns=time.monotonic_ns())
            if tuple(ids) == selected:
                return entries
            eligible = ([x for x in entries if x.is_store and x.failed is None and
                         x.next_file_index < x.total_files] if kind=="stores" else
                        [x for x in entries if x.job is not None and x.job.failed is None])
            by_id = dict(zip(ids,eligible))
            rank = {wid:i for i,wid in enumerate(selected)}
            # Local bounded permutation only; no retained queue or ownership.
            chosen = iter(sorted(by_id.values(),key=lambda x:rank[
                (x.accepted_parent_sequence,x.next_file_index,"d2h") if kind=="stores" else
                (x.job.accepted_parent_sequence,x.file_index,"ssd_read")]))
            return tuple(next(chosen) if (
                x.is_store and x.failed is None and x.next_file_index<x.total_files
                if kind=="stores" else x.job is not None and x.job.failed is None) else x
                for x in entries)
        except Exception as exc:
            bridge.fail(type(exc).__name__+": "+str(exc))
            return entries  # optional strategy errors cannot fail native jobs

    def inspect_p4_view(self, timeout=5.0):
        if getattr(self,"_prefix_p4_bridge",None) is None:
            raise RuntimeError("P4 bridge is disabled")
        future = Future()
        with self._submit_lock:
            if self._closed:
                raise RuntimeError("native intake closed")
            if getattr(self,"_prefix_p4_controls_pending",0) >= 2:
                raise RuntimeError("bounded P4 metadata control inbox full")
            self._prefix_p4_controls_pending = getattr(self,"_prefix_p4_controls_pending",0)+1
            self._incoming.put(_P4Inspect(self._prefix_progress.token,future))
        return future

    def request_p4_publication(self, publication):
        bridge = getattr(self,"_prefix_p4_bridge",None)
        if bridge is None:
            raise RuntimeError("P4 bridge is disabled")
        from prefix_io_control.p4_bridge import NativePublication
        if type(publication) is not NativePublication:
            raise TypeError("strict immutable owner publication required")
        future = Future()
        with self._submit_lock:
            if self._closed:
                raise RuntimeError("native intake closed")
            if getattr(self,"_prefix_p4_controls_pending",0) >= 2:
                raise RuntimeError("bounded P4 metadata control inbox full")
            self._prefix_p4_controls_pending = getattr(self,"_prefix_p4_controls_pending",0)+1
            self._incoming.put(_P4Publish(self._prefix_progress.token,publication,future))
        return future

    def _prefix_ordered_stores(self):
        order = getattr(self, "_prefix_store_order", None)
        if order is None or order.faulted:
            return self._prefix_p4_order(self._active, "stores")
        try:
            return order.ordered(self._active, self._prefix_progress.required)
        except Exception as exc:
            order.fail(type(exc).__name__ + ": " + str(exc))
            return self._active  # Optional ordering cannot fail or drop native work.

    def _schedule_work(self) -> bool:
        made = False

        # 0. Drain opened load fds first (latency-critical).
        made = self._drain_ready_load_fds() or made

        # 1. Load jobs: claim cached/inflight preload copies or open files.
        for job in self._active:
            if job.failed is not None or job.is_store:
                continue
            while job.failed is None and job.next_file_index < job.total_files:
                file_index = job.next_file_index
                h = job.block_hashes[file_index]
                ticket = self._prefix_allow_start("load", job)
                if ticket is False:
                    break
                if self._staging_cache is not None:
                    cs = self._staging_cache.get(h)
                    if cs is not None:
                        self._prefix_commit_start(ticket)
                        self._schedule_cache_hit(job, file_index, cs)
                        made = True
                        continue
                if self._share_preload:
                    if h in self._shared_cached:
                        self._prefix_commit_start(ticket)
                        self._shared_claim_from_cache(job, file_index)
                        made = True
                    elif self._has_inflight_preload(h):
                        # One read serves all waiters, so no waiter_room cap here.
                        self._prefix_commit_start(ticket)
                        self._schedule_preload_inflight(job, file_index)
                        made = True
                    elif self._preload_pending_count.get(h, 0) > 0:
                        # Queued but unopened: promote to a shared read, don't wait on the gate.
                        if not self._can_open():
                            break
                        self._prefix_commit_start(ticket)
                        self._promote_pending_to_shared_read(job, file_index)
                        made = True
                    else:
                        if not self._can_open():
                            break
                        self._prefix_commit_start(ticket)
                        self._submit_open_read(job, file_index)
                        self._shared_decref(job.profile.req_id, h)
                        made = True
                    continue
                if self._has_cached_preload(h):
                    self._prefix_commit_start(ticket)
                    self._schedule_preload_done(job, file_index)
                    made = True
                elif self._has_inflight_preload(h) and self._preload_waiter_room(h):
                    self._prefix_commit_start(ticket)
                    self._schedule_preload_inflight(job, file_index)
                    made = True
                else:
                    if not self._can_open():
                        break
                    self._prefix_commit_start(ticket)
                    self._submit_open_read(job, file_index)
                    self._cancel_one_pending(h)
                    made = True

        # 2. Store jobs (background relative to loads).
        for job in self._prefix_ordered_stores():
            if job.failed is not None or not job.is_store:
                continue
            while (
                job.failed is None
                and job.next_file_index < job.total_files
                and self._can_issue_store()
            ):
                if not self._schedule_one(job):
                    break
                made = True

        # 3. Speculative preloads: only during store windows or idle.
        made = self._drain_ready_preload_fds() or made
        self._drain_cancelled_pending()
        if self._can_issue_speculative_preload():
            while self._preload_pending and self._can_open() and self._preload_slots_available():
                info = self._preload_pending[0]
                h = info.block_hash
                # No cross-copy dedup: each pending entry is a distinct request's
                # own demand (intake already deduped per (req_id, hash)).
                if self._preload_pending_cancel.get(h, 0) > 0:
                    self._preload_pending.popleft()
                    self._consume_pending_cancel(h)
                    self._pending_dec(h)
                    continue
                if h in self._store_inflight:
                    self._preload_pending.popleft()
                    self._pending_dec(h)
                    blocked = self._preload_blocked_on_write.get(h)
                    if blocked is None:
                        blocked = collections.deque()
                        self._preload_blocked_on_write[h] = blocked
                    blocked.append(info)
                    add_event(
                        "py_kvcache.preload.defer_on_write",
                        "kv_preload",
                        now_ns(),
                        0,
                        tid=info.profile_tid,
                        args=self._preload_event_args(h, info),
                    )
                    continue
                ticket = self._prefix_allow_start("preload")
                if ticket is False:
                    break
                self._prefix_commit_start(ticket)
                self._preload_pending.popleft()
                self._pending_dec(h)
                self._submit_open_preload(info)
                made = True

        return made

    def _consume_pending_cancel(self, block_hash: bytes) -> None:
        count = self._preload_pending_cancel.get(block_hash, 0)
        if count <= 1:
            self._preload_pending_cancel.pop(block_hash, None)
        else:
            self._preload_pending_cancel[block_hash] = count - 1

    def _drain_cancelled_pending(self) -> None:
        while self._preload_pending and (
            self._preload_pending_cancel.get(self._preload_pending[0].block_hash, 0) > 0
        ):
            h = self._preload_pending.popleft().block_hash
            self._consume_pending_cancel(h)
            self._pending_dec(h)

    def _drain_ready_load_fds(self) -> bool:
        made = False
        ordered = self._prefix_p4_order(self._ready_fds_load,"reads")
        if ordered is not self._ready_fds_load:
            for index, ready in enumerate(ordered):
                self._ready_fds_load[index] = ready
        p4_batch_ids = self._prefix_p4_read_batch_ids()
        while self._ready_fds_load and self._data_inflight < self.iodepth:
            if p4_batch_ids is not None:
                item = self._ready_fds_load[0]
                if item.job is not None and (item.job.accepted_parent_sequence, item.file_index, "ssd_read") not in p4_batch_ids:
                    return made  # retain tail in its original native queue
            slot = self._reserve_foreground_slot()
            if slot is None:
                return made
            ready = self._ready_fds_load[0]
            dispatch = self._prefix_ready_read_decision(ready)
            if dispatch is not None and dispatch.action == "defer":
                self.staging_pool.release(slot.index)
                return made
            self._ready_fds_load.popleft()
            self._submit_read_from_ready(ready, slot.index, stage_dispatch=dispatch)
            made = True

        return made

    def _requeue_preload_hash(self, info: "_PreloadInfo") -> None:
        self._pending_push(info)

    def _remember_known_missing(self, info: "_PreloadInfo") -> None:
        block_hash = info.block_hash
        missed = self._known_missing.get(block_hash)
        if missed is None:
            missed = collections.deque()
            self._known_missing[block_hash] = missed
        missed.append(info)
        self._known_missing.move_to_end(block_hash)
        while len(self._known_missing) > self._max_known_missing:
            self._known_missing.popitem(last=False)

    def _release_store_inflight(self, block_hash: bytes, *, ok: bool) -> None:
        count = self._store_inflight.get(block_hash, 0) - 1
        if count > 0:
            self._store_inflight[block_hash] = count
            return
        self._store_inflight.pop(block_hash, None)
        blocked = self._preload_blocked_on_write.pop(block_hash, None)
        if not blocked:
            return
        if not ok or self._stop:
            return
        if self._staging_cache is not None and block_hash in self._staging_cache:
            # Store already cached the bytes; a reuse load hits the cache, so the
            # re-arm read is redundant. Clear shared demand so refcount does not leak.
            if self._share_preload:
                self._purge_owned_for_hash(block_hash)
                self._preload_refcount.pop(block_hash, None)
            return
        if self._share_preload:
            # refcount carries multiplicity, so one read-back serves all refholders.
            self._requeue_preload_hash(blocked[0])
        else:
            for info in blocked:
                self._requeue_preload_hash(info)
        add_event(
            "py_kvcache.preload.wake_after_write",
            "kv_preload",
            now_ns(),
            0,
            tid=blocked[0].profile_tid,
            args=self._preload_event_args(block_hash, blocked[0], copies=len(blocked)),
        )

    def _drain_ready_preload_fds(self) -> bool:
        made = False
        while self._ready_fds_preload:
            ready = self._ready_fds_preload[0]
            preload_hash = ready.preload_hash
            waited = preload_hash is not None and self._has_preload_waiter(preload_hash)
            if waited:
                if self._data_inflight >= self.iodepth:
                    return made
                slot = self._reserve_foreground_slot()
                if slot is None:
                    return made
                dispatch = self._prefix_ready_read_decision(ready)
                if dispatch is not None and dispatch.action == "defer":
                    self.staging_pool.release(slot.index)
                    return made
                self._ready_fds_preload.popleft()
                self._submit_read_from_ready(ready, slot.index, stage_dispatch=dispatch)
                made = True
                continue
            if preload_hash is not None and self._has_real_load_pressure():
                preload_info = ready.preload_info
                self._ready_fds_preload.popleft()
                self._preload_inflight_remove(preload_hash)
                if preload_info is not None:
                    self._requeue_preload_hash(preload_info)
                add_event(
                    "py_kvcache.preload.yield_open_fd",
                    "kv_preload",
                    now_ns(),
                    0,
                    tid=preload_info.profile_tid if preload_info else None,
                    args=self._preload_event_args(
                        preload_hash,
                        preload_info,
                        reason="real_load_pressure",
                    ),
                )
                controller = getattr(self, "_prefix_dispatch_controller", None)
                if controller is not None:
                    self._prefix_stage_retire("ssd_read", ("ssd_read", ready.sequence))
                self._async_close(ready.fd)
                made = True
                continue
            if self._data_inflight >= self.iodepth or not self._can_issue_speculative_preload():
                return made
            slot = self._reserve_preload_slot()
            if slot is None:
                return made
            dispatch = self._prefix_ready_read_decision(ready)
            if dispatch is not None and dispatch.action == "defer":
                self.staging_pool.release(slot.index)
                return made
            self._ready_fds_preload.popleft()
            self._submit_read_from_ready(ready, slot.index, stage_dispatch=dispatch)
            made = True
        return made

    def _schedule_one(self, job: _ReactorJob) -> bool:
        ticket = self._prefix_allow_start("store", job)
        if ticket is False:
            return False
        slot = self._reserve_foreground_slot()
        if slot is None:
            return False
        controller = getattr(self, "_prefix_dispatch_controller", None)
        if controller is not None or getattr(self,"_prefix_p4_bridge",None) is not None:
            try:
                mapping = self._build_mapping(
                    slot.index, job.block_chunks[job.next_file_index], 1,
                    self._slot_block_ids[slot.index], self.layout.storage_block_size_factor
                )
                mapping_len = int(mapping.shape[0])
                offset = self._fill_swap_ptrs(
                    self._mapping_buffers[slot.index][:mapping_len], slot.index, False,
                    self._batch_src_ptrs[slot.index], self._batch_dst_ptrs[slot.index],
                    self._batch_sizes[slot.index], 0
                ) if mapping_len else 0
                dispatch = self._prefix_stage_decide(
                    "d2h", int(self._batch_sizes[slot.index][:offset].sum()),
                    ("d2h", job.accepted_parent_sequence or job.job_id, job.next_file_index),
                    job, reserved_bytes=self.file_store.io_size
                ) if offset else None
            except BaseException as exc:
                # The physical backend was never entered. Preserve the original
                # per-file failure semantics while retaining other parent stages.
                file_index = job.next_file_index
                job.next_file_index += 1
                job.inflight_files += 1
                self.staging_pool.release(slot.index)
                self._file_terminal(job, ok=False, exc=exc)
                self._release_store_inflight(job.block_hashes[file_index], ok=False)
                return False
            if dispatch is not None and dispatch.action == "defer":
                self.staging_pool.release(slot.index)
                return False
            self._prefix_commit_start(ticket)
            return self._schedule_store_copy(
                job, job.next_file_index, slot.index, prepared_mapping=mapping,
                prepared_copy_count=offset, stage_dispatch=dispatch
            )
        self._prefix_commit_start(ticket)
        self._schedule_store_copy(job, job.next_file_index, slot.index)
        return True

    def _schedule_preload_done(self, job: _ReactorJob, file_index: int) -> None:
        h = job.block_hashes[file_index]
        entry = self._preload_cache_pop(h)
        assert entry is not None
        slot_index, preload_info = entry
        add_event(
            "py_kvcache.preload.place_from_cache",
            "kv_preload",
            now_ns(),
            0,
            tid=job.profile.profile_tid,
            args=self._preload_event_args(
                h,
                preload_info,
                load_job_id=job.job_id,
                load_req_id=job.profile.req_id,
                load_file_index=file_index,
            ),
        )
        job.next_file_index += 1
        job.inflight_files += 1
        try:
            self._queue_load_copy(job, file_index, slot_index, source="preload")
        except BaseException as exc:
            self.staging_pool.release(slot_index)
            self._file_terminal(job, ok=False, exc=exc)

    def _schedule_preload_inflight(self, job: _ReactorJob, file_index: int) -> None:
        h = job.block_hashes[file_index]
        add_event(
            "py_kvcache.preload.wait_for_inflight",
            "kv_preload",
            now_ns(),
            0,
            tid=job.profile.profile_tid,
            args=self._preload_event_args(
                h,
                None,
                load_job_id=job.job_id,
                load_req_id=job.profile.req_id,
                load_file_index=file_index,
            ),
        )
        job.next_file_index += 1
        job.inflight_files += 1
        self._preload_waiter_add(h, job, file_index)

    def _shared_claim_from_cache(self, job: _ReactorJob, file_index: int) -> None:
        h = job.block_hashes[file_index]
        slot = self._shared_cached[h]
        add_event(
            "py_kvcache.preload.place_from_cache",
            "kv_preload",
            now_ns(),
            0,
            tid=job.profile.profile_tid,
            args=self._preload_event_args(
                h,
                slot.preload_info,
                load_job_id=job.job_id,
                load_req_id=job.profile.req_id,
                load_file_index=file_index,
                shared=True,
                refcount=self._preload_refcount.get(h, 0),
            ),
        )
        job.next_file_index += 1
        job.inflight_files += 1
        self._shared_pin(slot)
        try:
            self._queue_load_copy(job, file_index, slot.slot_index, shared=slot, source="preload")
        except BaseException as exc:
            self._shared_unpin(slot)
            self._file_terminal(job, ok=False, exc=exc)
            self._maybe_release_shared(slot)
        self._shared_decref(job.profile.req_id, h)

    def _promote_pending_to_shared_read(self, job: _ReactorJob, file_index: int) -> None:
        h = job.block_hashes[file_index]
        self._cancel_one_pending(h)
        info = self._make_preload_info_for_job(job, file_index)
        self._submit_open_preload(info)
        self._schedule_preload_inflight(job, file_index)

    def _submit_open_read(self, job: _ReactorJob, file_index: int) -> None:
        # Slot reserved at read issue, not here — slow opens never tie up buffers.
        job.next_file_index += 1
        job.inflight_files += 1
        final_path = self.file_mapper.get_file_name(job.block_hashes[file_index])
        user_data = self._new_user_data()
        start_ns = now_ns()
        try:
            path_buf = self.file_store.queue_open_read(
                self.ring, user_data=user_data, path=final_path
            )
        except BaseException as exc:
            self._file_terminal(job, ok=False, exc=exc)
            return
        self._open_inflight += 1
        self._inflight[user_data] = _RingOp(
            job=job,
            file_index=file_index,
            slot_index=-1,
            fd=-1,
            is_write=False,
            start_ns=start_ns,
            op_kind="open",
            path_buf=path_buf,
        )

    def _submit_open_preload(self, info: "_PreloadInfo") -> None:
        # The async openat IS the existence check; a miss flows to _known_missing.
        # Do not pre-gate with os.path.exists -- it stalls the pump.
        block_hash = info.block_hash
        final_path = self.file_mapper.get_file_name(block_hash)
        user_data = self._new_user_data()
        start_ns = now_ns()
        try:
            path_buf = self.file_store.queue_open_read(
                self.ring, user_data=user_data, path=final_path
            )
        except BaseException as exc:
            logger.warning("py-kvcache preload openat submit failed: %s", exc)
            return
        self._open_inflight += 1
        self._preload_inflight_add(block_hash)
        self._inflight[user_data] = _RingOp(
            job=None,
            file_index=-1,
            slot_index=-1,
            fd=-1,
            is_write=False,
            start_ns=start_ns,
            op_kind="open",
            preload_hash=block_hash,
            preload_info=info,
            path_buf=path_buf,
        )

    def _submit_read_from_ready(self, ready: _ReadyFd, slot_index: int, *,
                                stage_dispatch=None) -> None:
        io_view = self._slot_view(slot_index)
        preload_info = ready.preload_info
        user_data = self._new_user_data()
        start_ns = now_ns()
        shadow_attempt = self._prefix_shadow_begin(
            "ssd_read", len(io_view), ready.job, continuation=True
        )
        try:
            self.file_store.queue_read(
                self.ring,
                user_data=user_data,
                fd=ready.fd,
                io_array=io_view,
            )
        except BaseException as exc:
            self._prefix_stage_unknown("ssd_read", exc)
            self._prefix_shadow_settle(shadow_attempt, "uncertain", exc)
            self._prefix_stage_settle(stage_dispatch, "uncertain", exc)
            # An enqueue exception may follow descriptor registration. Retain
            # the original FD/staging owner in the original ring-op map until
            # fatal close proves all accepted AIO and DMA drained.
            self._inflight[user_data] = _RingOp(
                job=ready.job, file_index=ready.file_index, slot_index=slot_index,
                fd=ready.fd, is_write=False, start_ns=start_ns, op_kind="read",
                preload_hash=ready.preload_hash, preload_info=preload_info,
            )
            self._data_inflight += 1
            raise
        self._prefix_shadow_settle(shadow_attempt, "accepted")
        self._prefix_stage_settle(stage_dispatch, "accepted")
        self._prefix_stage('accepted', 'ssd_read', user_data, len(io_view))
        self._data_inflight += 1
        self._inflight[user_data] = _RingOp(
            job=ready.job,
            file_index=ready.file_index,
            slot_index=slot_index,
            fd=ready.fd,
            is_write=False,
            start_ns=start_ns,
            op_kind="read",
            preload_hash=ready.preload_hash,
            preload_info=preload_info,
        )

    def _async_close(self, fd: int) -> None:
        # Tracked in _inflight so shutdown drains it cleanly.
        if fd < 0:
            return
        user_data = self._new_user_data()
        try:
            self.file_store.queue_close(self.ring, user_data=user_data, fd=fd)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            return
        self._inflight[user_data] = _RingOp(
            job=None,
            file_index=-1,
            slot_index=-1,
            fd=fd,
            is_write=False,
            start_ns=now_ns(),
            op_kind="close",
        )

    def _schedule_store_copy(self, job: _ReactorJob, file_index: int, slot_index: int, *,
                             prepared_mapping=None, prepared_copy_count=None,
                             stage_dispatch=None) -> bool:
        job.next_file_index += 1
        job.inflight_files += 1
        try:
            src_to_dst = prepared_mapping
            if src_to_dst is None:
                src_to_dst = self._build_mapping(
                    slot_index,
                    job.block_chunks[file_index],
                    1,
                    self._slot_block_ids[slot_index],
                    self.layout.storage_block_size_factor,
                )
            launch_options = {}
            if prepared_copy_count is not None or stage_dispatch is not None:
                launch_options = dict(prepared_copy_count=prepared_copy_count,
                                      stage_dispatch=stage_dispatch)
            copy = self._launch_swap_blocks(
                slot_index,
                src_to_dst,
                self.layout.storage_block_bytes,
                is_store=True,
                job=job,
                file_index=file_index,
                **launch_options,
            )
        except NativeDrainUnknown:
            raise  # Existing pool/job/protector ownership must remain intact.
        except BaseException as exc:
            controller = getattr(self, "_prefix_dispatch_controller", None)
            if stage_dispatch is not None and controller.pending is stage_dispatch.attempt:
                # An event/mapping failure before THIS backend call was entered.
                self._prefix_stage_settle(stage_dispatch, "rejected", exc)
            self.staging_pool.release(slot_index)
            self._file_terminal(job, ok=False, exc=exc)
            self._release_store_inflight(job.block_hashes[file_index], ok=False)
            return False
        # A store occupies one device-budget unit from copy-launch through
        # write-completion, else _can_issue_store over-launches and overflows the SQ.
        self._data_inflight += 1
        self._pending_copies.append(copy)
        return True

    def _build_mapping(
        self,
        slot_index: int,
        src_blocks: np.ndarray,
        src_block_size_factor: int,
        dst_blocks: np.ndarray,
        dst_block_size_factor: int,
    ) -> Any:
        output = self._mapping_buffers[slot_index]
        mapping_length = build_block_mapping(
            src_blocks,
            src_block_size_factor,
            dst_blocks,
            dst_block_size_factor,
            output,
        )
        return self._mapping_tensors[slot_index][:mapping_length]

    def _queue_load_copy(
        self,
        job: _ReactorJob,
        file_index: int,
        slot_index: int,
        *,
        shared: "_SharedPreloadSlot | None" = None,
        cache: "_CacheSlot | None" = None,
        source: str = "file",
    ) -> None:
        if source == "preload":
            job.n_from_preload += 1
        elif source == "cache":
            job.n_from_cache += 1
        else:
            job.n_from_file += 1
        output = self._mapping_buffers[slot_index]
        mapping_length = build_block_mapping(
            self._slot_block_ids[slot_index],
            self.layout.storage_block_size_factor,
            job.block_chunks[file_index],
            1,
            output,
        )
        # Snapshot: the per-slot buffer is reused, including by shared siblings in a pump.
        mapping = output[:mapping_length].copy()
        self._copy_ready.append(
            _ReadyCopy(
                job=job,
                file_index=file_index,
                slot_index=slot_index,
                mapping_len=mapping_length,
                nbytes=self.layout.storage_block_bytes,
                mapping=mapping,
                shared=shared,
                cache=cache,
            )
        )

    def _ensure_fused_capacity(self, entries: int) -> None:
        if entries <= self._fused_capacity:
            return
        cap = max(entries, self._fused_capacity * 2, 1024)
        self._fused_src_np = np.empty(cap, dtype=np.int64)
        self._fused_dst_np = np.empty(cap, dtype=np.int64)
        self._fused_sizes_np = np.empty(cap, dtype=np.int64)
        self._fused_src_t = torch.from_numpy(self._fused_src_np)
        self._fused_dst_t = torch.from_numpy(self._fused_dst_np)
        self._fused_sizes_t = torch.from_numpy(self._fused_sizes_np)
        self._fused_capacity = cap

    def _fill_swap_ptrs(
        self,
        mapping: np.ndarray,
        slot_index: int,
        staging_is_src: bool,
        src_arr: np.ndarray,
        dst_arr: np.ndarray,
        sizes_arr: np.ndarray,
        offset: int,
    ) -> int:
        """Fill swap_blocks pointer arrays for one staging slot."""
        layout = self.layout
        factor = layout.storage_block_size_factor
        slot_base = self._staging_base_ptr + slot_index * layout.storage_block_bytes
        if staging_is_src:
            staging_local = mapping[:, 0] - slot_index * factor
            gpu_blocks = mapping[:, 1]
        else:
            gpu_blocks = mapping[:, 0]
            staging_local = mapping[:, 1] - slot_index * factor
        rows = mapping.shape[0]
        end = offset
        for gpu_tensor, block_bytes, layer_offset in zip(
            layout.gpu_tensors, layout.bytes_per_kernel_block, layout.staging_layer_offsets
        ):
            start = end
            end = start + rows
            staging_ptr = slot_base + layer_offset + staging_local * block_bytes
            gpu_ptr = gpu_tensor.data_ptr() + gpu_blocks * block_bytes
            if staging_is_src:
                src_arr[start:end] = staging_ptr
                dst_arr[start:end] = gpu_ptr
            else:
                src_arr[start:end] = gpu_ptr
                dst_arr[start:end] = staging_ptr
            sizes_arr[start:end] = block_bytes
        return end

    def _flush_copy_batch(self) -> bool:
        ready = self._copy_ready
        if not ready:
            return False
        total = sum(rc.mapping_len for rc in ready)
        if total <= 0:
            self._copy_ready = []
            for rc in ready:
                self._settle_load_slot(rc.job, rc.file_index, rc.slot_index, rc.shared, rc.cache)
                self._file_terminal(rc.job, ok=True)
            return True
        self._ensure_fused_capacity(total * len(self.layout.gpu_tensors))
        src = self._fused_src_np
        dst = self._fused_dst_np
        sizes = self._fused_sizes_np
        start_ns = now_ns()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        shadow_attempt = None
        shadow_accepted = False
        stage_dispatch = None
        backend_called = False
        try:
            with torch.cuda.stream(self._copy_stream):
                start_event.record(self._copy_stream)
                offset = 0
                for rc in ready:
                    offset = self._fill_swap_ptrs(
                        rc.mapping,
                        rc.slot_index,
                        True,
                        src,
                        dst,
                        sizes,
                        offset,
                    )
                shadow_attempt = self._prefix_shadow_copy_begin(
                    "h2d", sizes, offset, continuation=True
                )
                if (getattr(self, "_prefix_dispatch_controller", None) is not None or
                        getattr(self,"_prefix_p4_bridge",None) is not None):
                    stage_dispatch = self._prefix_stage_decide(
                        "h2d", int(sizes[:offset].sum()), ("h2d", id(ready)),
                        continuation=True
                    )
                backend_called = True
                ops.swap_blocks_batch(
                    self._fused_src_t[:offset],
                    self._fused_dst_t[:offset],
                    self._fused_sizes_t[:offset],
                )
                self._prefix_shadow_settle(shadow_attempt, "accepted")
                self._prefix_stage_settle(stage_dispatch, "accepted")
                self._prefix_stage_copy("h2d", end_event, sizes, offset)
                shadow_accepted = True
                end_event.record(self._copy_stream)
        except BaseException as exc:
            if backend_called:
                self._prefix_stage_unknown("h2d", exc)
            if shadow_accepted:
                self._prefix_shadow_completion_unknown("h2d", shadow_attempt, exc)
                self._prefix_stage_completion_unknown("h2d", stage_dispatch, exc)
            else:
                self._prefix_shadow_settle(shadow_attempt, "uncertain", exc)
                self._prefix_stage_settle(stage_dispatch,
                                          "uncertain" if backend_called else "rejected", exc)
            self._sync_copy_failure(self._copy_stream, "h2d", exc)
            self._copy_ready = []
            for rc in ready:
                self._settle_load_slot(rc.job, rc.file_index, rc.slot_index, rc.shared, rc.cache)
                self._file_terminal(rc.job, ok=False, exc=exc)
            return True
        self._copy_ready = []
        self._pending_copies.append(
            _CopyOp(
                job=None,
                file_index=-1,
                slot_index=-1,
                is_store=False,
                start_ns=start_ns,
                start_event=start_event,
                end_event=end_event,
                nbytes=len(ready) * self.layout.storage_block_bytes,
                members=[
                    (rc.slot_index, rc.job, rc.file_index, rc.shared, rc.cache) for rc in ready
                ],
            )
        )
        return True

    def _launch_swap_blocks(
        self,
        slot_index: int,
        src_to_dst: Any,
        sample_nbytes: int,
        *,
        is_store: bool,
        job: _ReactorJob,
        file_index: int,
        prepared_copy_count=None,
        stage_dispatch=None,
    ) -> _CopyOp:
        stream = self._streams[slot_index]
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_ns = now_ns()
        shadow_attempt = None
        with torch.cuda.stream(stream):
            if is_store and job.compute_event is not None:
                stream.wait_event(job.compute_event)
            start_event.record(stream)
            mapping_len = int(src_to_dst.shape[0])
            if mapping_len > 0:
                mapping = self._mapping_buffers[slot_index][:mapping_len]
                offset = prepared_copy_count
                if offset is None:
                    offset = self._fill_swap_ptrs(
                        mapping,
                        slot_index,
                        not is_store,
                        self._batch_src_ptrs[slot_index],
                        self._batch_dst_ptrs[slot_index],
                        self._batch_sizes[slot_index],
                        0,
                    )
                shadow_attempt = self._prefix_shadow_copy_begin(
                    "d2h" if is_store else "h2d", self._batch_sizes[slot_index],
                    offset, job, continuation=not is_store
                )
                try:
                    ops.swap_blocks_batch(
                        self._batch_src_tensors[slot_index][:offset],
                        self._batch_dst_tensors[slot_index][:offset],
                        self._batch_size_tensors[slot_index][:offset],
                    )
                except BaseException as exc:
                    self._prefix_stage_unknown("d2h" if is_store else "h2d", exc)
                    self._prefix_shadow_settle(shadow_attempt, "uncertain", exc)
                    self._prefix_stage_settle(stage_dispatch, "uncertain", exc)
                    self._sync_copy_failure(stream, "d2h" if is_store else "h2d", exc)
                    raise
                self._prefix_shadow_settle(shadow_attempt, "accepted")
                self._prefix_stage_settle(stage_dispatch, "accepted")
                self._prefix_stage_copy(
                    "d2h" if is_store else "h2d", end_event,
                    self._batch_sizes[slot_index], offset
                )
            try:
                end_event.record(stream)
            except BaseException as exc:
                if mapping_len > 0:
                    self._prefix_stage_unknown("d2h" if is_store else "h2d", exc)
                self._prefix_shadow_completion_unknown(
                    "d2h" if is_store else "h2d", shadow_attempt, exc
                )
                self._prefix_stage_completion_unknown(
                    "d2h" if is_store else "h2d", stage_dispatch, exc
                )
                self._sync_copy_failure(stream, "d2h" if is_store else "h2d", exc)
                raise
        return _CopyOp(
            job=job,
            file_index=file_index,
            slot_index=slot_index,
            is_store=is_store,
            start_ns=start_ns,
            start_event=start_event,
            end_event=end_event,
            nbytes=sample_nbytes,
            accounting_copy_accepted=mapping_len > 0,
        )

    def _file_terminal(
        self, job: _ReactorJob, *, ok: bool, exc: BaseException | None = None
    ) -> None:
        job.inflight_files -= 1
        if ok:
            job.done_files += 1
        if job.future_set:
            return
        if exc is not None:
            # A Future is the worker's whole-parent protection signal. A drained
            # file does not prove sibling DMA/SSD stages have stopped using GPU
            # source/destination blocks. Publish failure only in _finish_jobs.
            if job.failed is None:
                job.failed = exc
                logger.exception("py-kvcache transfer job %s failed", job.job_id, exc_info=exc)
        elif job.failed is None and job.done_files == job.total_files:
            self._prefix_p4_parent_terminal(job, successful=True)
            self._retire_accepted_parent(job)  # Physical whole-parent drain before callbacks.
            job.future_set = True
            self._emit(job, success=True)
            job.future.set_result(job.transfer_size)

    def _finish_jobs(self) -> bool:
        if not self._active:
            return False
        remaining: list[_ReactorJob] = []
        removed = False
        for job in self._active:
            drained = job.inflight_files == 0 and (
                job.next_file_index >= job.total_files or job.failed is not None
            )
            if drained and job.failed is not None and not job.future_set:
                if getattr(self, "_native_drain_unknown", False):
                    remaining.append(job)
                    continue
                self._prefix_p4_parent_terminal(job, successful=False)
                self._retire_accepted_parent(job)  # Safe capacity before reentrant callbacks.
                job.future_set = True
                self._emit(job, success=False)
                job.future.set_exception(job.failed)
            if drained and job.future_set:
                controller = getattr(self, "_prefix_dispatch_controller", None)
                if controller is not None:
                    self._prefix_stage_retire(
                        "d2h", ("d2h", job.accepted_parent_sequence or job.job_id, job.next_file_index)
                    )
                if not job.accepted_parent_retired:
                    self._retire_accepted_parent(job)
                state = getattr(self, "_prefix_progress", None)
                if state is not None:
                    # Both success and failure retire after all physical stages.
                    state.required.discard(job.future)
                budget = getattr(self, "_prefix_start_budget", None)
                if budget is not None and not budget.faulted:
                    try:
                        budget.retire(job.future)
                    except Exception as exc:
                        budget.fail(type(exc).__name__)
                removed = True
                continue
            remaining.append(job)
        if removed:
            self._active = remaining
        return removed

    def _emit(self, job: _ReactorJob, *, success: bool) -> None:
        emit_transfer_events(
            self.layout,
            job.profile,
            success=success,
            transfer_size=job.transfer_size if success else 0,
            file_samples=job.file_samples,
            cuda_samples=job.cuda_samples,
            num_files=job.total_files,
            num_blocks=job.num_blocks,
            n_from_file=job.n_from_file,
            n_from_preload=job.n_from_preload,
            n_from_cache=job.n_from_cache,
        )

    def _slot_view(self, slot_index: int) -> np.ndarray[Any, np.dtype[np.uint8]]:
        return self.layout.storage_slot_view(self.staging_buffer, slot_index)

    def _new_user_data(self) -> int:
        self._next_user_data += 1
        return self._next_user_data


class TransferCoordinator:
    def __init__(
        self,
        *,
        config: SharedFileConfig,
        file_mapper: FileMapper,
        layout: ParsedKvLayout,
        break_even: BreakEvenThresholds | None = None,
        storage_block_tokens: int = 0,
        progress_run_id: str | None = None,
        observation_sink: Any = None,
        start_budget: Any = None,
        stage_accounting: Any = None,
        store_order: Any = None,
        dispatch_shadow: Any = None,
        max_accepted_parents: int | None = None,
        dispatch_controller: Any = None,
        p4_bridge: Any = None,
    ) -> None:
        self.layout = layout
        self.reactor = IoReactor(
            config=config,
            file_mapper=file_mapper,
            layout=layout,
            break_even=break_even,
            storage_block_tokens=storage_block_tokens,
            progress_run_id=progress_run_id,
            observation_sink=observation_sink,
            start_budget=start_budget,
            stage_accounting=stage_accounting,
            store_order=store_order,
            dispatch_shadow=dispatch_shadow,
            max_accepted_parents=max_accepted_parents,
            dispatch_controller=dispatch_controller,
            p4_bridge=p4_bridge,
        )

    @property
    def progress_bridge_enabled(self) -> bool:
        return self.reactor.progress_bridge_enabled

    def request_mandatory(self, futures: list[Future[int]]) -> bool:
        return self.reactor.request_mandatory(futures)

    def submit_store(
        self,
        src_spec: Any,
        dst_spec: Any,
        *,
        job_id: int = -1,
        profile_tid: str = "kv_store",
        req_id: str = "",
    ) -> "Future[int]":
        start_ns = now_ns()
        profile = _TransferProfile(
            job_id=job_id,
            direction="gpu_to_storage",
            profile_tid=profile_tid,
            req_id=req_id,
            start_ns=start_ns,
        )
        future: "Future[int]" = Future()
        block_hashes = list(dst_spec.block_hashes)
        total = len(block_hashes)
        if total == 0:
            future.set_result(0)
            return future
        src_chunks = split_block_ids_for_files(self.layout, block_ids_of(src_spec), total)
        # Capture the compute stream here (worker thread); the reactor thread cannot.
        compute_event = torch.cuda.Event()
        compute_event.record(torch.cuda.current_stream())
        job = _ReactorJob(
            job_id=job_id,
            is_store=True,
            block_hashes=block_hashes,
            block_chunks=src_chunks,
            profile=profile,
            future=future,
            total_files=total,
            transfer_size=total * self.layout.storage_block_bytes,
            num_blocks=safe_num_blocks(src_spec),
            compute_event=compute_event,
        )
        future.set_running_or_notify_cancel()
        self.reactor.submit_job(job)
        return future

    def submit_load(
        self,
        src_spec: Any,
        dst_spec: Any,
        *,
        job_id: int = -1,
        profile_tid: str = "kv_load",
        req_id: str = "",
    ) -> "Future[int]":
        start_ns = now_ns()
        profile = _TransferProfile(
            job_id=job_id,
            direction="storage_to_gpu",
            profile_tid=profile_tid,
            req_id=req_id,
            start_ns=start_ns,
        )
        future: "Future[int]" = Future()
        block_hashes = list(src_spec.block_hashes)
        total = len(block_hashes)
        if total == 0:
            future.set_result(0)
            return future
        dst_chunks = split_block_ids_for_files(
            self.layout,
            block_ids_of(dst_spec),
            total,
            first_group_block_index(dst_spec),
        )
        job = _ReactorJob(
            job_id=job_id,
            is_store=False,
            block_hashes=block_hashes,
            block_chunks=dst_chunks,
            profile=profile,
            future=future,
            total_files=total,
            transfer_size=total * self.layout.storage_block_bytes,
            num_blocks=safe_num_blocks(dst_spec),
        )
        future.set_running_or_notify_cancel()
        self.reactor.submit_job(job)
        return future

    def submit_preload(
        self,
        block_hashes: list[bytes],
        *,
        preload_id: str = "",
        req_id: str = "",
        profile_tid: str = "kv_preload",
    ) -> None:
        self.reactor.enqueue_preload(
            block_hashes,
            preload_id=preload_id,
            req_id=req_id,
            profile_tid=profile_tid,
        )

    def raise_if_native_fatal(self):
        self.reactor.raise_if_native_fatal()

    def inspect_p4_view(self):
        return self.reactor.inspect_p4_view()

    def publish_p4_scheduled_load(self, observation):
        return self.reactor.publish_p4_scheduled_load(observation)

    def request_p4_publication(self, publication):
        return self.reactor.request_p4_publication(publication)

    def request_owner_snapshot(self):
        return self.reactor.request_owner_snapshot()

    def inspect_snapshot(self, timeout=5.0):
        return self.reactor.inspect_snapshot(timeout)

    def shutdown(self) -> None:
        self.reactor.shutdown()


__all__ = ["IoReactor", "TransferCoordinator"]

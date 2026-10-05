"""CPU fixture for native reactor/handler code; never a replacement executor.

Native admission, intake, pump, completion, continuation and shutdown run
unchanged. Only CUDA/file/ring surfaces and the zero ordinary-credit test seam
are fake. There are no performance or GPU measurements here.
"""
import collections
from concurrent.futures import Future
from contextlib import contextmanager
import queue
import threading
import time
from types import SimpleNamespace as NS
import numpy as np
from py_kvcache import reactor as mod
from py_kvcache.break_even import BreakEvenThresholds
from py_kvcache.reactor import IoReactor, TransferCoordinator, _ProgressState, _CopyOp, _MandatoryWait
from py_kvcache.staging import StagingPool
from py_kvcache.transfer import ParsedKvLayout
from py_kvcache.vllm import GPULoadStoreSpec, SharedStorageLoadStoreSpec, NoopSharedStorageOffloadingHandler

class Pool(StagingPool):
    def __init__(self, slots):
        super().__init__(slot_count=slots)
        self.releases = []
    def release(self, index):
        self.releases.append(index)
        super().release(index)

class Ring:
    def __init__(self, rig):
        self.rig = rig
        self.pending = collections.deque()
        self.permits = threading.Semaphore(0)
        self.auto = True
        self.results = {}
        self.duplicate = False
    def poll_all(self):
        result = []
        while self.pending and (self.auto or self.permits.acquire(blocking=False)):
            ident = self.pending.popleft()
            cqe = (ident, self.results.get(ident, 4096))
            result.append(cqe)
            if self.duplicate:
                result.append(cqe)
        return result
    def submit_pending(self):
        pass
    def close(self):
        assert not self.pending, "ring closed before draining CPU completions"

class FileStore:
    io_size = 4096
    def __init__(self, rig):
        self.rig = rig
        self.finished = []
        self.cleaned = []
    def open_temp_write(self, final):
        return 10, final + ".temp"
    def queue_write(self, ring, *, user_data, fd, io_array):
        ring.pending.append(user_data)
        self.rig.writes += 1
        self.rig.write_queued.set()
    def finish_write(self, **kwargs):
        self.finished.append(kwargs["final_path"])
    def cleanup_temp(self, **kwargs):
        self.cleaned.append(kwargs)

class Rig:
    def __init__(self, monkeypatch, *, enabled=True, slots=2, iodepth=2):
        self.compute_ready = threading.Event()
        self.copy_ready = threading.Event()
        self.copy_polled = threading.Event()
        self.copy_issued = threading.Event()
        self.write_queued = threading.Event()
        self.marked = threading.Event()
        self.compute_ready.set()
        self.copy_ready.set()
        self.ordinary_credits = 0
        self.writes = 0
        self.issued = 0
        self.observer_ticks = 0
        self.observe = None
        self.snapshots = collections.deque(maxlen=128)
        rig = self

        class ComputeEvent:
            def record(self, stream):
                pass
            def query(self):
                return rig.compute_ready.is_set()
        # Exercise real TransferCoordinator.submit_store without touching CUDA.
        monkeypatch.setattr(mod, "torch", NS(cuda=NS(
            Event=ComputeEvent, current_stream=lambda: object())))

        r = self.r = IoReactor.__new__(IoReactor)
        r.layout = ParsedKvLayout([None], [4096], 1, False)
        r.iodepth = iodepth
        r.open_lookahead = iodepth
        r._copy_headroom = 4
        r.staging_pool = Pool(slots)
        r.actual_staging_bytes = slots*4096 + 4095
        r._share_preload = False
        r._staging_cache = None
        r._max_preload_slots = 0
        r._max_cache_slots = 0
        r._preload_cached_total = r._preload_inflight_total = 0
        r._data_inflight = r._open_inflight = 0
        r._next_user_data = 0
        r._break_even = BreakEvenThresholds()
        r._storage_block_tokens = 1
        r._active = []
        r._inflight = {}
        r._pending_copies = []
        r._copy_ready = []
        r._incoming = queue.Queue()
        r._submit_lock = threading.Lock()
        r._closed = r._stop = False
        r._prefix_progress = _ProgressState("cpu-run") if enabled else None
        for name in ("_preload_refcount", "_shared_cached", "_preload_pending_count",
                     "_preload_pending_cancel", "_preload_inflight_hashes", "_preload_waiters",
                     "_store_inflight", "_preload_blocked_on_write"):
            setattr(r, name, {})
        for name in ("_preload_slots", "_preload_owned", "_known_missing"):
            setattr(r, name, collections.OrderedDict())
        for name in ("_preload_pending", "_ready_fds_load", "_ready_fds_preload"):
            setattr(r, name, collections.deque())
        r._max_preload_owned = 16384
        r._max_known_missing = 4096
        r._slot_block_ids = [np.asarray([i]) for i in range(slots)]
        r.file_mapper = NS(get_file_name=lambda h: "fixture-" + h.hex())
        r.file_store = FileStore(self)
        r.ring = Ring(self)
        r._slot_view = lambda index: bytearray(4096)
        r._build_mapping = lambda *args: np.zeros((1, 2), dtype=np.int64)
        r._emit = lambda *args, **kwargs: None

        def launch(slot_index, mapping, nbytes, *, is_store, job, file_index):
            self.issued += 1
            self.copy_issued.set()
            def query_done():
                ready = self.copy_ready.is_set() and job.compute_event.query()
                self.copy_polled.set()
                return ready
            return _CopyOp(job, file_index, slot_index, is_store, time.perf_counter_ns(),
                           NS(elapsed_time=lambda other: 1.0),
                           NS(query=query_done), nbytes)
        r._launch_swap_blocks = launch
        original_schedule_one = r._schedule_one
        def zero_credit_test_seam(job):
            # Test-only quota shim. Native physical/event/slot checks stay below.
            if not self.ordinary_credits and not r.is_mandatory(job.future):
                return False
            return original_schedule_one(job)
        r._schedule_one = zero_credit_test_seam
        original_intake = r._intake
        def intake(item):
            original_intake(item)
            if r.progress_bridge_enabled and r._prefix_progress.required:
                self.marked.set()
        r._intake = intake
        original_pump = r._pump_once
        def pump():
            result = original_pump()
            if self.observe is not None:
                self.observer_ticks += 1
                self.snapshots.append(self.observe(r, run_id="cpu-run", epoch=self.observer_ticks))
            return result
        r._pump_once = pump
        r._worker = threading.Thread(target=r._run, name="cpu-native-reactor-fixture", daemon=True)
        self.coordinator = TransferCoordinator.__new__(TransferCoordinator)
        self.coordinator.layout = r.layout
        self.coordinator.reactor = r
        self.handler = NoopSharedStorageOffloadingHandler(coordinator=self.coordinator)

    def submit(self, files=2, job_id=7):
        try:
            gpu = GPULoadStoreSpec(list(range(files)), group_sizes=[files], block_indices=[0])
        except TypeError:
            gpu = GPULoadStoreSpec()
            gpu.block_ids = np.arange(files)
        storage = SharedStorageLoadStoreSpec([f"{job_id}-{i}".encode() for i in range(files)])
        assert self.handler.transfer_async(job_id, (gpu, storage), req_id="cpu-fixture")
        return self.handler._active[job_id][0]

    def start(self):
        self.r._worker.start()

    def waiter(self, jobs):
        thread = threading.Thread(target=self.handler.wait, args=(jobs,), daemon=True)
        thread.start()
        return thread

    def cleanup(self):
        # Test-only recovery ensures a failing assertion never leaves a thread hanging.
        self.ordinary_credits = 1
        self.compute_ready.set()
        self.copy_ready.set()
        self.r.ring.auto = True
        if self.r._worker.ident is None:
            self.r._worker.start()
        self.r.shutdown(wait=False)
        self.r._worker.join(timeout=3)
        assert not self.r._worker.is_alive(), "CPU fixture failed to terminate"

@contextmanager
def running_rig(monkeypatch, **kwargs):
    rig = Rig(monkeypatch, **kwargs)
    try:
        yield rig
    finally:
        rig.cleanup()

"""Real native pump/queues/futures; fake CUDA and file syscalls, no GPU metrics."""
import contextlib,threading,time
from types import SimpleNamespace as NS
import numpy as np
import torch
from py_kvcache import reactor as mod
from py_kvcache.reactor import IoReactor
from py_kvcache.vllm import GPULoadStoreSpec,SharedStorageLoadStoreSpec
from prefix_io_control.start_budget import StartBudget,StartBudgetConfig
from tests.prefix_io_v1_progress._cpu_backend import Rig

class BudgetRig(Rig):
    def __init__(self,monkeypatch,*,mode="fixed",units=1,slots=4,depth=2,reserve=0,
                 epoch=100,max_age=1000,enabled=True,share=False):
        super().__init__(monkeypatch,slots=slots,iodepth=depth,enabled=True)
        self.clock=[0]
        monkeypatch.setattr(mod.time,"monotonic_ns",lambda:self.clock[0])
        # Remove the previous test-only zero-credit shim; use the patched native method.
        self.r._schedule_one=IoReactor._schedule_one.__get__(self.r)
        self.budget=StartBudget(StartBudgetConfig(mode,epoch,units,max_age,reserve))
        if enabled:self.budget.bind()
        self.r._prefix_start_budget=self.budget if enabled else None
        self.reads=0;self.opens=0;self.h2d_launches=[]
        self.r._max_preload_slots=slots-depth
        self.r._share_preload=share
        self.r._mapping_buffers=[np.zeros((4,2),dtype=np.int64) for _ in range(slots)]
        self.r._staging_base_ptr=2_000_000
        self.r.layout.gpu_tensors[0]=NS(data_ptr=lambda:1_000_000)
        self.r._copy_stream=object()
        self.r._fused_capacity=0
        rig=self
        class Event:
            def __init__(self,**kwargs):pass
            def record(self,stream):pass
            def query(self):return rig.compute_ready.is_set() and rig.copy_ready.is_set()
            def elapsed_time(self,other):return 1.0
        cuda=NS(Event=Event,current_stream=lambda:object(),stream=lambda _:contextlib.nullcontext())
        monkeypatch.setattr(mod,"torch",NS(cuda=cuda,from_numpy=torch.from_numpy))
        def swap(src,dst,sizes):
            rig.h2d_launches.append(dict(entries=len(sizes),bytes=sum(sizes.tolist())))
        monkeypatch.setattr(mod,"ops",NS(swap_blocks_batch=swap))
        def open_read(ring,*,user_data,path):
            rig.opens+=1;ring.results[user_data]=10;ring.pending.append(user_data)
            return b"fake-path"
        def read(ring,*,user_data,fd,io_array):
            rig.reads+=1;ring.pending.append(user_data)
        def close(ring,*,user_data,fd):
            ring.results[user_data]=0;ring.pending.append(user_data)
        self.r.file_store.queue_open_read=open_read
        self.r.file_store.queue_read=read
        self.r.file_store.queue_close=close

    def load(self,files=1,job_id=20,hashes=None):
        gpu=GPULoadStoreSpec(list(range(files)),group_sizes=[files],block_indices=[0])
        storage=SharedStorageLoadStoreSpec(hashes or [f"read-{job_id}-{i}".encode() for i in range(files)])
        assert self.handler.transfer_async(job_id,(storage,gpu),req_id="cpu-load-"+str(job_id))
        return self.handler._active[job_id][0]

    def preload(self,hashes):
        self.coordinator.submit_preload(hashes,preload_id="cpu-preload",req_id="cpu-preload")

    def stop(self):
        self.r.shutdown()
        assert not self.r._active and not self.r._inflight and not self.r._pending_copies
        assert not self.r._prefix_progress.required
        if self.r._prefix_start_budget:
            assert not self.budget.waiting

@contextlib.contextmanager
def rig(monkeypatch,**kwargs):
    r=BudgetRig(monkeypatch,**kwargs)
    try:yield r
    finally:r.cleanup()

def eventually(predicate):
    deadline=time.monotonic()+2
    while not predicate() and time.monotonic()<deadline:time.sleep(.002)
    assert predicate(),"native CPU event path did not reach expected state"

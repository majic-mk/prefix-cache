import ctypes as C
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from py_kvcache.linux_aio import LinuxAioRing
from py_kvcache.liburing_file import DirectIoFileStore
from py_kvcache.fs_config import SharedFileConfig
from ._fixture import rig,eventually

def test_real_linux_aio_mixed_read_write_under_shared_start_budget(monkeypatch,tmp_path):
    """Real aligned disk bytes, native completions; explicitly fake CUDA only."""
    with rig(monkeypatch) as x:
        r=x.r
        ring=LinuxAioRing(4)
        r.ring=ring
        r.file_store=DirectIoFileStore(SharedFileConfig(str(tmp_path),io_backend="linux_aio"),payload_size=4096)
        r.file_mapper=NS(get_file_name=lambda h:str(tmp_path/h.hex()))
        owners=[C.create_string_buffer(8192) for _ in range(r.staging_pool.slot_count)]
        arrays=[]
        for owner in owners:
            address=(C.addressof(owner)+4095)&~4095
            arrays.append(np.ctypeslib.as_array((C.c_uint8*4096).from_address(address)))
        r._slot_view=lambda index:arrays[index]
        original=r._launch_swap_blocks
        def cpu_copy(slot,*args,**kwargs):
            if kwargs["is_store"]:arrays[slot].fill(173)
            return original(slot,*args,**kwargs)
        r._launch_swap_blocks=cpu_copy
        handoffs=[]
        original_queue=r._queue_load_copy
        def queue(job,index,slot,**kwargs):
            handoffs.append(arrays[slot].tobytes())
            return original_queue(job,index,slot,**kwargs)
        r._queue_load_copy=queue
        source=tmp_path/b"source".hex();source.write_bytes(b"x"*4096)
        load=x.load(hashes=[b"source"]);store=x.submit(files=1)
        x.start()
        assert load.result(timeout=3)==4096
        eventually(lambda:x.budget.denied_quota>0)
        target=tmp_path/b"7-0".hex()
        assert not target.exists() and not store.done()
        x.clock[0]=100
        assert store.result(timeout=3)==4096
        x.stop()
        assert source.read_bytes()==b"x"*4096
        assert target.read_bytes()==bytes([173])*4096
        assert handoffs==[b"x"*4096]
        assert len(x.h2d_launches)==1
        state=ring.snapshot()
        assert state["accepted"]==state["completed"]==state["reaped"] and state["outstanding"]==0
        assert x.budget.starts==dict(load=1,store=1,preload=0)
        assert not x.budget.faulted and not list(tmp_path.glob("*.tmp*"))

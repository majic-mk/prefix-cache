"""Real CPU mixed metadata/data and producer progress checks."""
import ctypes as C
import errno
import importlib.util
import os
from pathlib import Path
import threading
import time

import pytest
from py_kvcache.linux_aio import LinuxAioRing
from .helpers import Aligned, reap

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("aio_mixed_screen",
    ROOT/"experiments/prefix_io_v1/scripts/aio_cpu_mixed_screen.py")
mixed=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixed)

@pytest.mark.parametrize("size",[57344*16,57344*64,57344*256])
@pytest.mark.parametrize("depth,workers",[(1,1),(4,2)])
def test_production_sized_mixed_file_lifecycle(tmp_path,size,depth,workers):
    result=mixed.run_case(tmp_path/"case",size,depth,0,jobs=depth*2,
                          metadata_workers=workers)
    assert result["fd_leak"]==result["worker_leak"]==0
    assert result["operation_counts"]==dict(open=depth,read=depth,write=depth,close=depth)
    assert result["snapshot"]["max_outstanding"]<=2*result["ring_depth"]

def test_sync_publication_mixed_lifecycle(tmp_path):
    result=mixed.run_case(tmp_path/"sync",57344*16,4,0,jobs=8,sync=True)
    assert result["snapshot"]["accepted"]==16
    assert result["snapshot"]["outstanding"]==0

def test_empty_submit_does_not_wake_but_close_still_does(monkeypatch):
    ring=LinuxAioRing(2)
    calls=[]
    original=ring._wake
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(ring,"_wake",counted)
    try:
        for _ in range(1000):
            ring.submit_pending()
        assert calls==[], "empty reactor pumps must not cause eventfd wake storms"
    finally:
        ring.close()
    assert calls, "close must wake an idle worker even with no pending requests"
    assert ring.snapshot()["drained"]

def test_real_io_progress_while_metadata_workers_are_blocked(tmp_path,monkeypatch):
    p=tmp_path/"file";p.write_bytes(b"x"*4096)
    fd=os.open(p,os.O_RDWR|os.O_DIRECT)
    ring=LinuxAioRing(4,metadata_workers=2)
    barrier=threading.Barrier(3)
    release=threading.Event()
    original=ring._metadata
    def delayed(req):
        barrier.wait(timeout=5)
        assert release.wait(timeout=5)
        return original(req)
    monkeypatch.setattr(ring,"_metadata",delayed)
    path=C.create_string_buffer(os.fsencode(p))
    a,b=Aligned(4096,0),Aligned(4096,42)
    opened=[]
    try:
        for token in [1,2]:
            ring.queue_openat(user_data=token,path_ptr=C.addressof(path),open_flags=os.O_RDONLY)
        ring.submit_pending()
        barrier.wait(timeout=5)
        ring.queue_rw(user_data=3,fd=fd,ptr=a.ptr,nbytes=4096,write=False)
        ring.submit_pending()
        assert reap(ring,1)=={3:4096}
        assert a.bytes()==b"x"*4096
        ring.queue_rw(user_data=4,fd=fd,ptr=b.ptr,nbytes=4096,write=True)
        ring.submit_pending()
        assert reap(ring,1)=={4:4096}
        # Both metadata operations still own their accepted descriptors.
        assert ring.snapshot()["outstanding"]==2
        release.set()
        result=reap(ring,2)
        opened=list(result.values())
        assert all(x>=0 for x in opened)
    finally:
        release.set()
        ring.close()
        for x in opened:
            os.close(x)
        os.close(fd)

def test_real_metadata_backpressure_reap_then_retry(tmp_path):
    p=tmp_path/"file";p.write_bytes(b"abc")
    path=C.create_string_buffer(os.fsencode(p))
    ring=LinuxAioRing(2,metadata_workers=1)
    claimed=[]
    try:
        for token in range(ring.capacity):
            ring.queue_openat(user_data=token,path_ptr=C.addressof(path),open_flags=os.O_RDONLY)
        with pytest.raises(BlockingIOError) as exc:
            ring.queue_openat(user_data=4,path_ptr=C.addressof(path),open_flags=os.O_RDONLY)
        assert exc.value.errno==errno.EAGAIN
        # Accepted unreaped completions remain in the capacity budget.
        ring.submit_pending()
        result=reap(ring,ring.capacity)
        claimed=list(result.values())
        assert all(x>=0 for x in claimed)
        for token,fd in enumerate(claimed):
            ring.queue_close(user_data=token+10,fd=fd)
        ring.submit_pending()
        assert reap(ring,4)=={i+10:0 for i in range(4)}
        claimed=[]
        ring.queue_openat(user_data=20,path_ptr=C.addressof(path),open_flags=os.O_RDONLY)
        ring.submit_pending()
        claimed=[reap(ring,1)[20]]
        assert os.read(claimed[0],3)==b"abc"
        assert ring.snapshot()["max_outstanding"]==ring.capacity
    finally:
        ring.close()
        for fd in claimed:os.close(fd)

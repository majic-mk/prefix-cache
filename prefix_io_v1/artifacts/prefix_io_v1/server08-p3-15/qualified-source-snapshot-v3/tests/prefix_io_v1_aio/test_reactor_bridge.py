"""Actual author completion dispatch with real disk I/O and explicitly mocked CUDA handoff."""
import ctypes as C
import os
import time
from types import SimpleNamespace
import pytest
from py_kvcache.linux_aio import LinuxAioRing
from py_kvcache.liburing_file import DirectIoFileStore
from py_kvcache.fs_config import SharedFileConfig
from py_kvcache.reactor import IoReactor
from .helpers import Aligned

def reactor_shell(ring, store):
    r=IoReactor.__new__(IoReactor)
    r.ring=ring;r.file_store=store;r.layout=SimpleNamespace(storage_block_bytes=4096)
    r._inflight={};r._data_inflight=1
    ids=iter(range(100,200));r._new_user_data=lambda:next(ids)
    released=[];copies=[];terminals=[];stores=[];cached=[]
    r.staging_pool=SimpleNamespace(release=released.append)
    r._queue_load_copy=lambda *args:copies.append(args)  # No actual CUDA submission.
    r._file_terminal=lambda job,**kw:terminals.append(kw)
    r._release_store_inflight=lambda *args,**kw:stores.append(kw)
    r._release_or_cache=lambda *args:cached.append(args)
    return r,released,copies,terminals,stores,cached

def drain(r):
    r.ring.submit_pending()
    deadline=time.monotonic()+5
    while r._inflight:
        r._poll_ring_completions()
        r.ring.submit_pending()
        assert time.monotonic()<deadline
        time.sleep(.0001)

@pytest.mark.parametrize("case",["valid","short","bad_fd"])
def test_real_read_result_through_author_completion(tmp_path,case):
    b=Aligned(4096,0)
    p=tmp_path/"data";p.write_bytes(b"" if case=="short" else b"q"*4096)
    fd=-1 if case=="bad_fd" else os.open(p,os.O_RDONLY|os.O_DIRECT)
    ring=LinuxAioRing(4)
    store=DirectIoFileStore(SharedFileConfig(str(tmp_path),io_backend="linux_aio"),payload_size=4096)
    r,released,copies,terminals,_,_=reactor_shell(ring,store)
    job=SimpleNamespace(file_samples=[])
    op=SimpleNamespace(op_kind="read",preload_hash=None,job=job,start_ns=time.perf_counter_ns(),
                       fd=fd,file_index=0,slot_index=3)
    r._inflight[1]=op
    try:
        ring.queue_rw(user_data=1,fd=fd,ptr=b.ptr,nbytes=4096,write=False)
        drain(r)
        assert r._data_inflight==0
        if case=="valid":
            assert b.bytes()==b"q"*4096
            assert len(copies)==1 and released==[] and terminals==[]
        else:
            assert copies==[] and released==[3]
            assert len(terminals)==1 and terminals[0]["ok"] is False
        if fd>=0:
            with pytest.raises(OSError):os.fstat(fd)
    finally:
        ring.close()

@pytest.mark.parametrize("bad_fd",[False,True])
def test_real_write_result_publication_and_release_order(tmp_path,bad_fd):
    b=Aligned(4096,31)
    ring=LinuxAioRing(4)
    store=DirectIoFileStore(SharedFileConfig(str(tmp_path),io_backend="linux_aio"),payload_size=4096)
    r,released,_,terminals,stores,cached=reactor_shell(ring,store)
    final=str(tmp_path/"kv");fd,temp=store.open_temp_write(final)
    if bad_fd:
        os.close(fd);fd=-1
    job=SimpleNamespace(file_samples=[],block_hashes=[b"prefix"])
    op=SimpleNamespace(op_kind="write",job=job,fd=fd,temp_path=temp,final_path=final,
                       start_ns=time.perf_counter_ns(),file_index=0,slot_index=3)
    r._inflight[1]=op
    try:
        ring.queue_rw(user_data=1,fd=fd,ptr=b.ptr,nbytes=4096,write=True)
        assert not os.path.exists(final) and released==[] and cached==[] and terminals==[]
        drain(r)
        assert r._data_inflight==0
        if bad_fd:
            assert not os.path.exists(final) and released==[3] and cached==[]
            assert terminals[0]["ok"] is False and stores[0]["ok"] is False
        else:
            assert open(final,"rb").read()==b.bytes()
            assert released==[] and cached==[(b"prefix",3)]
            assert terminals==[{"ok":True}] and stores==[{"ok":True}]
        assert not os.path.exists(temp)
    finally:
        ring.close()

"""Real CPU file I/O tests; no mocked kernel, no CUDA."""
import ctypes as C
import errno
import gc
import os
import threading
from contextlib import ExitStack

import numpy as np
import pytest
from py_kvcache.linux_aio import LinuxAioRing
from py_kvcache.liburing_file import DirectIoFileStore
from py_kvcache.fs_config import SharedFileConfig
from .helpers import Aligned, reap

@pytest.mark.parametrize("depth", [1, 4, 16])
@pytest.mark.parametrize("size", [4096, 65536, 1048576])
def test_real_direct_batch_roundtrip(tmp_path, depth, size):
    with ExitStack() as stack:
        ring = LinuxAioRing(depth)
        stack.callback(ring.close)
        fds = []
        written = []
        restored = []
        for i in range(depth):
            fd = os.open(tmp_path / str(i), os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_DIRECT, 0o600)
            # Close only after the ring drains if a test assertion fails.
            fds.append(fd)
            written.append(Aligned(size, i + 17))
            restored.append(Aligned(size, 0))
        try:
            for i, fd in enumerate(fds):
                ring.queue_rw(user_data=i, fd=fd, ptr=written[i].ptr, nbytes=size, write=True)
            ring.submit_pending()
            assert reap(ring, depth) == {i: size for i in range(depth)}
            for i, fd in enumerate(fds):
                ring.queue_rw(user_data=i, fd=fd, ptr=restored[i].ptr, nbytes=size, write=False)
            ring.submit_pending()
            assert reap(ring, depth) == {i: size for i in range(depth)}
            assert all(a.bytes() == b.bytes() for a, b in zip(written, restored))
            assert ring.snapshot()["max_kernel_inflight"] <= depth
            assert ring.snapshot()["max_outstanding"] <= 2 * depth
        finally:
            ring.close()
            for fd in fds:
                os.close(fd)

def test_author_file_store_publish_read_and_duplicate(tmp_path):
    store = DirectIoFileStore(SharedFileConfig(root_dir=str(tmp_path),
                                              io_backend="linux_aio", sync_on_store=True),
                              payload_size=4101)
    a, b = Aligned(store.io_size, 39), Aligned(store.io_size, 0)
    av = np.ctypeslib.as_array((C.c_ubyte * a.size).from_address(a.address))
    bv = np.ctypeslib.as_array((C.c_ubyte * b.size).from_address(b.address))
    final = str(tmp_path / "kv.bin")
    ring = LinuxAioRing(4)
    try:
        for expected in [True, False]:
            fd, temp = store.open_temp_write(final)
            try:
                store.queue_write(ring, user_data=1, fd=fd, io_array=av)
                ring.submit_pending()
                assert reap(ring, 1)[1] == store.io_size
                assert store.finish_write(fd=fd, temp_path=temp, final_path=final) is expected
                fd = None
            finally:
                if fd is not None:
                    ring.close()
                    store.cleanup_temp(fd=fd, temp_path=temp)
        path_owner = store.queue_open_read(ring, user_data=2, path=final)
        ring.submit_pending()
        fd = reap(ring, 1)[2]
        assert fd >= 0
        store.queue_read(ring, user_data=3, fd=fd, io_array=bv)
        ring.submit_pending()
        assert reap(ring, 1)[3] == store.io_size
        assert a.bytes() == b.bytes()
        store.queue_close(ring, user_data=4, fd=fd)
        ring.submit_pending()
        assert reap(ring, 1)[4] == 0
        with pytest.raises(OSError):
            os.fstat(fd)
        assert path_owner is not None
        assert not list(tmp_path.glob("*.tmp-*"))
    finally:
        ring.close()

def test_missing_file_and_invalid_fd(tmp_path):
    ring = LinuxAioRing(4)
    b = Aligned(4096)
    path = C.create_string_buffer(os.fsencode(tmp_path / "missing"))
    try:
        ring.queue_openat(user_data=1, path_ptr=C.addressof(path), open_flags=os.O_RDONLY)
        ring.queue_close(user_data=2, fd=-1)
        ring.queue_rw(user_data=3, fd=-1, ptr=b.ptr, nbytes=4096, write=False)
        ring.submit_pending()
        assert reap(ring, 3) == {1: -errno.ENOENT, 2: -errno.EBADF, 3: -errno.EBADF}
    finally:
        ring.close()

def test_short_read_is_not_successful_full_read(tmp_path):
    p = tmp_path / "empty"
    p.write_bytes(b"")
    fd = os.open(p, os.O_RDONLY | os.O_DIRECT)
    ring = LinuxAioRing(2)
    b = Aligned(4096)
    try:
        ring.queue_rw(user_data=1, fd=fd, ptr=b.ptr, nbytes=4096, write=False)
        ring.submit_pending()
        assert reap(ring, 1)[1] == 0
    finally:
        ring.close()
        os.close(fd)

def test_actual_alignment_error_propagates(tmp_path):
    fd = os.open(tmp_path / "aligned", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_DIRECT, 0o600)
    ring = LinuxAioRing(2)
    b = Aligned(8192)
    try:
        ring.queue_rw(user_data=1, fd=fd, ptr=C.c_void_p(b.address + 1), nbytes=4096, write=True)
        ring.submit_pending()
        assert reap(ring, 1)[1] == -errno.EINVAL
    finally:
        ring.close()
        os.close(fd)

def test_close_flushes_pending_writes_and_preserves_completions(tmp_path):
    fd = os.open(tmp_path / "data", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_DIRECT, 0o600)
    b = Aligned(65536, 27)
    ring = LinuxAioRing(2)
    try:
        ring.queue_rw(user_data=0, fd=fd, ptr=b.ptr, nbytes=b.size, write=True)
        ring.close()
        assert ring.poll_all() == [(0, b.size)]
        assert (tmp_path / "data").read_bytes() == b.bytes()
        assert ring.snapshot()["drained"] is True
        ring.close()
        with pytest.raises(RuntimeError):
            ring.queue_close(user_data=5, fd=-1)
    finally:
        os.close(fd)

def test_unclaimed_open_closed_but_claimed_fd_owned_by_caller(tmp_path):
    p = tmp_path / "file"; p.write_bytes(b"x")
    baseline = len(os.listdir("/proc/self/fd"))
    ring = LinuxAioRing(2)
    path = C.create_string_buffer(os.fsencode(p))
    ring.queue_openat(user_data=1, path_ptr=C.addressof(path), open_flags=os.O_RDONLY)
    ring.close()
    assert ring.poll_all() == [(1, -errno.ECANCELED)]
    assert len(os.listdir("/proc/self/fd")) == baseline
    ring = LinuxAioRing(2)
    ring.queue_openat(user_data=1, path_ptr=C.addressof(path), open_flags=os.O_RDONLY)
    ring.submit_pending()
    fd = reap(ring, 1)[1]
    ring.close()
    assert os.read(fd, 1) == b"x"
    os.close(fd)
    assert len(os.listdir("/proc/self/fd")) == baseline

def test_repeat_lifecycle_no_fd_or_worker_leak(tmp_path):
    p = tmp_path / "file"; p.write_bytes(b"x")
    gc.collect()
    baseline = len(os.listdir("/proc/self/fd"))
    for _ in range(12):
        ring = LinuxAioRing(2)
        path = C.create_string_buffer(os.fsencode(p))
        ring.queue_openat(user_data=1, path_ptr=C.addressof(path), open_flags=os.O_RDONLY)
        ring.submit_pending()
        fd = reap(ring, 1)[1]
        ring.queue_close(user_data=2, fd=fd)
        ring.close()
        assert dict(ring.poll_all())[2] == 0
    gc.collect()
    assert len(os.listdir("/proc/self/fd")) == baseline
    assert not [t for t in threading.enumerate() if t.name.startswith("pykvcache-aio-")]

def test_relative_openat_and_owned_path_copy(tmp_path):
    (tmp_path / "file").write_bytes(b"yes")
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    path = C.create_string_buffer(b"file")
    ring = LinuxAioRing(2)
    try:
        ring.queue_openat(user_data=1, path_ptr=C.addressof(path), open_flags=os.O_RDONLY,
                          dirfd=directory)
        path.value = b"xxxx"
        ring.submit_pending()
        fd = reap(ring, 1)[1]
        assert os.read(fd, 3) == b"yes"
        os.close(fd)
    finally:
        ring.close()
        os.close(directory)

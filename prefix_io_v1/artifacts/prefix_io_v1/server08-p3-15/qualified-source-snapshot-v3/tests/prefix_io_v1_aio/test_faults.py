"""Deterministic fault injection; these results are CPU/mock, not disk measurements."""
import ctypes as C
import errno
import threading
import time
from collections import deque

import pytest
from py_kvcache.linux_aio import LinuxAioRing
from .helpers import Aligned, reap

class FakeKernel:
    def __init__(self, actions=(), reverse=False):
        self.actions = deque(actions)
        self.events = deque()
        self.reverse = reverse
        self.gate = threading.Event(); self.gate.set()
        self.entered = threading.Event()
        self.submit_gate = threading.Event(); self.submit_gate.set()
        self.closed = False
        self.seen = []
    def submit(self, requests):
        self.entered.set()
        assert self.submit_gate.wait(3), "test submit gate timed out"
        action = self.actions.popleft() if self.actions else len(requests)
        if isinstance(action, Exception):
            raise action
        count = min(action, len(requests))
        for req in requests[:count]:
            self.seen.append(req.user_data)
            self.events.append((req.user_data, req.nbytes))
        return count
    def poll(self):
        if not self.gate.is_set():
            return []
        result = list(self.events); self.events.clear()
        return result[::-1] if self.reverse else result
    def close(self):
        self.closed = True

def queue(ring, buf, ids):
    for token in ids:
        ring.queue_rw(user_data=token, fd=9, ptr=buf.ptr, nbytes=buf.size, write=False)

@pytest.mark.parametrize("actions", [
    [1, 1, 1], [0, 2], [OSError(errno.EAGAIN, "busy"), 1],
    [OSError(errno.EINTR, "signal"), 2],
])
def test_partial_zero_and_transient_submit_no_loss_or_duplicate(actions):
    fake = FakeKernel(actions, reverse=True)
    ring = LinuxAioRing(4, _kernel=fake)
    b = Aligned(4096)
    try:
        queue(ring, b, [10, 11, 12, 13]); ring.submit_pending()
        assert reap(ring, 4) == {i: 4096 for i in [10, 11, 12, 13]}
        assert sorted(fake.seen) == [10, 11, 12, 13]
        assert ring.snapshot()["outstanding"] == 0
    finally:
        ring.close()

def test_persistent_eagain_is_explicit_failure_and_close_finishes():
    fake = FakeKernel([OSError(errno.EAGAIN, "busy")] * 105)
    ring = LinuxAioRing(1, _kernel=fake)
    b = Aligned(4096)
    try:
        queue(ring, b, [1]); ring.submit_pending()
        assert reap(ring, 1) == {1: -errno.EAGAIN}
        assert fake.seen == []
    finally:
        ring.close()

def test_permanent_error_does_not_discard_other_requests():
    fake = FakeKernel([OSError(errno.EBADF, "bad"), 2])
    ring = LinuxAioRing(4, _kernel=fake)
    b = Aligned(4096)
    try:
        queue(ring, b, [1, 2, 3]); ring.submit_pending()
        assert reap(ring, 3) == {1: -errno.EBADF, 2: 4096, 3: 4096}
        assert fake.seen == [2, 3]
    finally:
        ring.close()

def test_budget_includes_pending_inflight_and_unreaped():
    fake = FakeKernel(); fake.gate.clear()
    ring = LinuxAioRing(2, _kernel=fake)
    b = Aligned(4096)
    try:
        queue(ring, b, range(4))
        with pytest.raises(BlockingIOError) as exc:
            queue(ring, b, [5])
        assert exc.value.errno == errno.EAGAIN
        with pytest.raises(ValueError):
            queue(ring, b, [0])
        assert ring.snapshot()["outstanding"] == 4
        fake.gate.set(); ring.submit_pending()
        assert reap(ring, 4) == {i: 4096 for i in range(4)}
        queue(ring, b, [0]); ring.submit_pending()
        assert reap(ring, 1) == {0: 4096}
        assert ring.snapshot()["max_outstanding"] == 4
    finally:
        fake.gate.set(); ring.close()

def test_blocking_kernel_submit_does_not_block_caller_or_release_early():
    fake = FakeKernel(); fake.submit_gate.clear()
    ring = LinuxAioRing(2, _kernel=fake)
    b = Aligned(4096)
    finished = threading.Event()
    def caller():
        queue(ring, b, [1]); ring.submit_pending(); finished.set()
    thread = threading.Thread(target=caller)
    try:
        thread.start()
        assert finished.wait(1)
        assert fake.entered.wait(1)
        assert ring.poll_all() == []
        assert ring.snapshot()["outstanding"] == 1
        fake.submit_gate.set()
        assert reap(ring, 1) == {1: 4096}
    finally:
        fake.submit_gate.set(); thread.join(); ring.close()

def test_close_waits_for_accepted_work_and_rejects_new_work():
    fake = FakeKernel(); fake.submit_gate.clear()
    ring = LinuxAioRing(2, _kernel=fake)
    b = Aligned(4096)
    queue(ring, b, [1]); ring.submit_pending()
    assert fake.entered.wait(1)
    finished = threading.Event()
    def closer():
        ring.close(); finished.set()
    thread = threading.Thread(target=closer)
    try:
        thread.start()
        deadline = time.monotonic() + 1
        while not ring._closing and time.monotonic() < deadline:
            time.sleep(.001)
        assert ring._closing
        assert not finished.is_set()
        assert not fake.closed
        with pytest.raises(RuntimeError):
            queue(ring, b, [2])
        fake.submit_gate.set()
        assert finished.wait(2)
        assert fake.closed
        assert ring.poll_all() == [(1, 4096)]
    finally:
        fake.submit_gate.set(); thread.join(); ring.close()

def test_out_of_order_completion_maps_to_original_tokens():
    fake = FakeKernel(reverse=True)
    ring = LinuxAioRing(4, _kernel=fake)
    b = Aligned(4096)
    try:
        queue(ring, b, [11, 25, 39, 40]); ring.submit_pending()
        assert reap(ring, 4) == {11:4096, 25:4096, 39:4096, 40:4096}
        assert len(set(fake.seen)) == 4
    finally:
        ring.close()

def test_unknown_completion_fails_closed_and_destroys_context():
    class Broken(FakeKernel):
        def poll(self):
            return [(999, 4096)]
    fake = Broken()
    ring = LinuxAioRing(1, _kernel=fake)
    b = Aligned(4096)
    queue(ring, b, [1]); ring.submit_pending()
    ring._worker.join(2)
    assert not ring._worker.is_alive()
    assert fake.closed
    with pytest.raises(RuntimeError):
        ring.poll_all()
    with pytest.raises(RuntimeError):
        ring.close()

@pytest.mark.parametrize("depth,workers", [(0,2), (-1,2), (True,2), (1,0), (1,9), (1,True)])
def test_invalid_budgets_rejected_before_context_creation(depth,workers):
    with pytest.raises(ValueError):
        LinuxAioRing(depth, metadata_workers=workers)

@pytest.mark.parametrize("token", [-1, 1 << 64, True, "1"])
def test_invalid_ids_rejected(token):
    ring = LinuxAioRing(1, _kernel=FakeKernel())
    b = Aligned(4096)
    try:
        with pytest.raises(ValueError):
            queue(ring, b, [token])
        assert ring.snapshot()["accepted"] == 0
    finally:
        ring.close()

def test_metadata_open_does_not_block_other_metadata(tmp_path):
    import os
    path = tmp_path / "file"; path.write_bytes(b"x")
    slow_entered = threading.Event()
    release = threading.Event()
    ring = LinuxAioRing(2, metadata_workers=2, _kernel=FakeKernel())
    original = ring._metadata
    def metadata(req):
        if req.user_data == 1:
            slow_entered.set()
            assert release.wait(3)
        return original(req)
    ring._metadata = metadata
    owner = C.create_string_buffer(os.fsencode(path))
    try:
        for token in [1, 2]:
            ring.queue_openat(user_data=token, path_ptr=C.addressof(owner), open_flags=os.O_RDONLY)
        ring.submit_pending()
        assert slow_entered.wait(1)
        first = reap(ring, 1)
        assert set(first) == {2}
        os.close(first[2])
        release.set()
        last = reap(ring, 1)
        os.close(last[1])
    finally:
        release.set(); ring.close()

def test_metadata_exception_does_not_leak_other_completed_open(tmp_path):
    import os
    path = tmp_path / "file"; path.write_bytes(b"x")
    baseline = len(os.listdir("/proc/self/fd"))
    second_opened = threading.Event()
    ring = LinuxAioRing(2, metadata_workers=2, _kernel=FakeKernel())
    original = ring._metadata
    def metadata(req):
        if req.user_data == 1:
            assert second_opened.wait(2)
            raise RuntimeError("injected metadata failure")
        result = original(req)
        second_opened.set()
        return result
    ring._metadata = metadata
    owner = C.create_string_buffer(os.fsencode(path))
    for token in [1, 2]:
        ring.queue_openat(user_data=token, path_ptr=C.addressof(owner), open_flags=os.O_RDONLY)
    ring.submit_pending()
    ring._worker.join(3)
    assert not ring._worker.is_alive()
    with pytest.raises(RuntimeError):
        ring.close()
    assert len(os.listdir("/proc/self/fd")) == baseline
    assert not [t for t in threading.enumerate() if t.name.startswith("pykvcache-aio-")]

def test_destroy_failure_retains_request_references_and_fails_closed():
    class Broken(FakeKernel):
        def close(self):
            raise OSError(errno.EIO, "injected destroy failure")
    ring = LinuxAioRing(1, _kernel=Broken())
    b = Aligned(4096)
    queue(ring, b, [1])
    with pytest.raises(RuntimeError, match="did not drain"):
        ring.close()
    assert ring.snapshot()["drained"] is False
    assert ring.snapshot()["outstanding"] == 1
    assert ring._ops[1].ptr is b.ptr
    with pytest.raises(RuntimeError):
        ring.poll_all()
    assert not [t for t in threading.enumerate() if t.name.startswith("pykvcache-aio-")]

@pytest.mark.parametrize("stage", ["selector", "worker"])
def test_constructor_failure_cleans_context_and_notification_fds(monkeypatch, stage):
    import os
    import py_kvcache.linux_aio as module
    fake = FakeKernel()
    baseline = len(os.listdir("/proc/self/fd"))
    def fail(*args, **kwargs):
        raise RuntimeError("injected constructor failure")
    if stage == "selector":
        monkeypatch.setattr(module.selectors, "DefaultSelector", fail)
    else:
        monkeypatch.setattr(module.threading.Thread, "start", fail)
    with pytest.raises(RuntimeError, match="injected constructor failure"):
        LinuxAioRing(1, _kernel=fake)
    assert fake.closed
    assert len(os.listdir("/proc/self/fd")) == baseline

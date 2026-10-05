import ctypes as C
import time

class Aligned:
    def __init__(self, size, byte=91):
        self.size = size
        self.owner = C.create_string_buffer(size + 4096)
        self.address = (C.addressof(self.owner) + 4095) & ~4095
        self.ptr = C.c_void_p(self.address)
        C.memset(self.address, byte, size)
    def bytes(self):
        return C.string_at(self.address, self.size)

def reap(ring, count, timeout=5):
    deadline = time.monotonic() + timeout
    result = []
    while len(result) < count:
        result.extend(ring.poll_all())
        if time.monotonic() >= deadline:
            raise AssertionError(("completion timeout", result, ring.snapshot()))
        if len(result) < count:
            time.sleep(0.0001)
    assert len(result) == count
    return dict(result)

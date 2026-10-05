"""CPU contract for DMA ownership. Actual completions must come from the adapter.

This ledger does not allocate GPU memory and is not a replacement for the native
allocator. Failure or cancellation never implies completion of an issued DMA.
"""
from dataclasses import dataclass
from enum import Enum
import threading


class Phase(str, Enum):
    RESERVED = "RESERVED"
    ISSUED = "ISSUED"
    CANCEL_PENDING = "CANCEL_PENDING"
    STORAGE_PENDING = "STORAGE_PENDING"
    QUARANTINED = "QUARANTINED"
    RELEASED = "RELEASED"


@dataclass
class Ticket:
    ticket_id: str
    direction: str
    generation: int
    gpu_bytes: int
    pinned_bytes: int
    optional: bool
    phase: Phase = Phase.RESERVED


class OwnershipLedger:
    def __init__(self, gpu_capacity: int, pinned_capacity: int):
        for n in (gpu_capacity, pinned_capacity):
            if type(n) is not int or n < 0:
                raise ValueError("invalid capacity")
        self.gpu_capacity = gpu_capacity
        self.pinned_capacity = pinned_capacity
        self.tickets = {}
        self.gpu_bytes = self.pinned_bytes = 0
        self.gpu_byte_ns = self.pinned_byte_ns = 0
        self.epoch = 0
        self.last_ns = None
        self._lock = threading.RLock()

    def _advance(self, now_ns):
        if type(now_ns) is not int or now_ns < 0:
            raise ValueError("integer monotonic time required")
        if self.last_ns is not None:
            if now_ns < self.last_ns:
                raise ValueError("clock moved backwards")
            self.gpu_byte_ns += self.gpu_bytes * (now_ns - self.last_ns)
            self.pinned_byte_ns += self.pinned_bytes * (now_ns - self.last_ns)
        self.last_ns = now_ns

    def reserve(self, ticket_id, direction, generation, gpu_bytes, pinned_bytes,
                optional, now_ns, expected_epoch):
        with self._lock:
            if expected_epoch != self.epoch:
                raise ValueError("stale reservation snapshot")
            if not ticket_id or ticket_id in self.tickets:
                raise ValueError("duplicate or empty ticket")
            if direction not in ("H2D", "D2H") or type(optional) is not bool:
                raise ValueError("invalid transfer")
            if any(type(v) is not int or v < 0 for v in (generation, gpu_bytes, pinned_bytes)):
                raise ValueError("invalid size or generation")
            if gpu_bytes + pinned_bytes == 0:
                raise ValueError("empty transfer")
            self._advance(now_ns)
            if self.gpu_bytes + gpu_bytes > self.gpu_capacity or self.pinned_bytes + pinned_bytes > self.pinned_capacity:
                return False  # All-or-nothing; no partial lease.
            self.tickets[ticket_id] = Ticket(ticket_id, direction, generation,
                                             gpu_bytes, pinned_bytes, optional)
            self.gpu_bytes += gpu_bytes
            self.pinned_bytes += pinned_bytes
            self.epoch += 1
            return True

    def _get(self, ticket_id, generation, now_ns):
        t = self.tickets[ticket_id]
        if t.generation != generation:
            raise ValueError("stale block generation")
        self._advance(now_ns)
        return t

    def _release(self, t):
        self.gpu_bytes -= t.gpu_bytes
        self.pinned_bytes -= t.pinned_bytes
        t.gpu_bytes = t.pinned_bytes = 0
        t.phase = Phase.RELEASED

    def issue(self, ticket_id, generation, now_ns):
        with self._lock:
            t = self._get(ticket_id, generation, now_ns)
            if t.phase != Phase.RESERVED:
                raise ValueError("transfer not reservable")
            t.phase = Phase.ISSUED
            self.epoch += 1

    def cancel(self, ticket_id, generation, now_ns):
        with self._lock:
            t = self._get(ticket_id, generation, now_ns)
            if not t.optional:
                raise ValueError("cannot drop required inference state")
            if t.phase == Phase.RESERVED:
                self._release(t)
            elif t.phase == Phase.ISSUED:
                t.phase = Phase.CANCEL_PENDING
            else:
                raise ValueError("cancellation not legal in this state")
            self.epoch += 1

    def dma_complete(self, ticket_id, generation, now_ns, storage_pending=False):
        with self._lock:
            t = self._get(ticket_id, generation, now_ns)
            if t.phase not in (Phase.ISSUED, Phase.CANCEL_PENDING):
                raise ValueError("no issued DMA")
            if storage_pending and (t.direction != "D2H" or t.phase == Phase.CANCEL_PENDING):
                raise ValueError("invalid storage ownership")
            if storage_pending:
                self.gpu_bytes -= t.gpu_bytes
                t.gpu_bytes = 0
                t.phase = Phase.STORAGE_PENDING
            else:
                self._release(t)
            self.epoch += 1

    def storage_complete(self, ticket_id, generation, now_ns):
        with self._lock:
            t = self._get(ticket_id, generation, now_ns)
            if t.phase != Phase.STORAGE_PENDING:
                raise ValueError("storage not pending")
            self._release(t)
            self.epoch += 1

    def fail(self, ticket_id, generation, now_ns):
        with self._lock:
            t = self._get(ticket_id, generation, now_ns)
            if t.phase in (Phase.ISSUED, Phase.CANCEL_PENDING, Phase.STORAGE_PENDING):
                t.phase = Phase.QUARANTINED
            elif t.phase == Phase.RESERVED:
                self._release(t)
            else:
                raise ValueError("invalid failure transition")
            self.epoch += 1

    def release_quarantined(self, ticket_id, generation, now_ns, *, completion_fenced):
        with self._lock:
            t = self._get(ticket_id, generation, now_ns)
            if t.phase != Phase.QUARANTINED or completion_fenced is not True:
                raise ValueError("must fence DMA AND storage before release")
            self._release(t)
            self.epoch += 1

    def snapshot(self, now_ns):
        with self._lock:
            self._advance(now_ns)
            return dict(epoch=self.epoch, gpu_bytes=self.gpu_bytes,
                        pinned_bytes=self.pinned_bytes, gpu_byte_ns=self.gpu_byte_ns,
                        pinned_byte_ns=self.pinned_byte_ns,
                        outstanding=sum(t.phase != Phase.RELEASED for t in self.tickets.values()))

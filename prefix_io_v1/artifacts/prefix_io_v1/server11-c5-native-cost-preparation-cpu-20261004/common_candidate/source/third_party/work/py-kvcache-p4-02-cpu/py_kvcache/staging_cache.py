"""Optional staging-data cache: retain completed CPU-staging buffers keyed by
block hash so a later load for the same hash is a staging-resident hit (one
staging->GPU copy, no disk read) instead of a re-open + re-read.

Safe because storage blocks are immutable and hash-addressed: a cached buffer is
always valid for its hash. This is read-side only -- store buffers are never
cached, so writes still persist to disk immediately.

Pure-Python (no torch/CUDA). The cache holds staging slot indices checked out of
the reactor's pool but never calls the pool itself -- it returns slot indices for
the reactor to release, preserving the pool's single-owner invariant.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Iterator, Protocol


class EvictionPolicy(Protocol):
    def admit(self, key: bytes) -> None: ...
    def touch(self, key: bytes) -> None: ...
    def evict(self, key: bytes) -> None: ...
    def victims(self) -> Iterator[bytes]: ...
    def clear(self) -> None: ...


class LruPolicy:
    """Least-recently-used ordering over resident keys."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._od: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()

    def admit(self, key: bytes) -> None:
        self._od[key] = None
        self._od.move_to_end(key)

    def touch(self, key: bytes) -> None:
        if key in self._od:
            self._od.move_to_end(key)

    def evict(self, key: bytes) -> None:
        self._od.pop(key, None)

    def victims(self) -> Iterator[bytes]:
        return iter(list(self._od))  # oldest first

    def clear(self) -> None:
        self._od.clear()


class ArcPolicy:
    """ARC-style adaptive policy (Megiddo & Modha): T1/T2 resident lists +
    B1/B2 ghost lists (keys only) with an adaptive target ``p`` for |T1|.

    Adapted for pull-based eviction: ``admit``/``touch`` drive adaptation on
    every access; the resident eviction (REPLACE) is deferred to ``victims()``
    (the reactor pulls a victim when it needs a slot), with evicted residents
    demoted to the matching ghost list. Scan resistance -- the reason to choose
    ARC over LRU -- comes from one-shot keys living in T1/B1 while reused keys
    are promoted to T2.
    """

    def __init__(self, capacity: int) -> None:
        self._c = max(1, capacity)
        self._p = 0
        self._t1: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()
        self._t2: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()
        self._b1: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()
        self._b2: "collections.OrderedDict[bytes, None]" = collections.OrderedDict()

    def admit(self, key: bytes) -> None:
        if key in self._b1:
            delta = 1 if not self._b1 else max(1, len(self._b2) // max(1, len(self._b1)))
            self._p = min(self._c, self._p + delta)
            self._b1.pop(key, None)
            self._t2[key] = None
        elif key in self._b2:
            delta = 1 if not self._b2 else max(1, len(self._b1) // max(1, len(self._b2)))
            self._p = max(0, self._p - delta)
            self._b2.pop(key, None)
            self._t2[key] = None
        else:
            self._t1[key] = None
        self._trim_ghosts()

    def touch(self, key: bytes) -> None:
        if key in self._t1:
            self._t1.pop(key, None)
            self._t2[key] = None
        elif key in self._t2:
            self._t2.move_to_end(key)

    def evict(self, key: bytes) -> None:
        if key in self._t1:
            self._t1.pop(key, None)
            self._b1[key] = None
        elif key in self._t2:
            self._t2.pop(key, None)
            self._b2[key] = None
        self._trim_ghosts()

    def victims(self) -> Iterator[bytes]:
        # REPLACE preference: evict from T1 when it exceeds the target p.
        if len(self._t1) > self._p:
            first, second = self._t1, self._t2
        else:
            first, second = self._t2, self._t1
        yield from list(first)  # LRU-first
        yield from list(second)

    def clear(self) -> None:
        self._t1.clear()
        self._t2.clear()
        self._b1.clear()
        self._b2.clear()
        self._p = 0

    @property
    def p(self) -> int:
        return self._p

    def _trim_ghosts(self) -> None:
        while len(self._b1) > self._c:
            self._b1.popitem(last=False)
        while len(self._b2) > self._c:
            self._b2.popitem(last=False)


@dataclass
class _CacheSlot:
    block_hash: bytes
    slot_index: int
    copies_inflight: int = 0  # loads mid staging->GPU DMA out of this slot


def make_policy(kind: str, capacity: int) -> EvictionPolicy:
    if kind == "lru":
        return LruPolicy(capacity)
    if kind == "arc":
        return ArcPolicy(capacity)
    raise ValueError(f"unknown staging_cache policy {kind!r}")


class StagingDataCache:
    """Hash-addressed retention of completed staging slots, bounded at
    ``capacity`` slots, evictable on demand. Never touches the staging pool:
    returns slot indices for the reactor to release."""

    def __init__(self, *, policy: str, capacity: int) -> None:
        self._capacity = max(0, capacity)
        self._policy = make_policy(policy, self._capacity)
        self._slots: dict[bytes, _CacheSlot] = {}
        # Observation only: distinct resident slots, not number of copy refs.
        self._pinned_slots = 0
        self._observed_slots = 0
        self._observation_valid = True
        self._observation_error: str | None = None

    @property
    def observation_valid(self) -> bool:
        return self._checked_observation()

    @property
    def observation_error(self) -> str | None:
        return self._observation_error

    def invalidate_observation(self, reason) -> None:
        # Diagnostics must not change a native cache/refcount operation.
        try:
            message = str(reason)[:160] or "cache observation unavailable"
        except Exception:
            message = "cache observation unavailable"
        self._observation_valid = False
        if self._observation_error is None:
            self._observation_error = message

    def _checked_observation(self) -> bool:
        if self._observation_valid is False:
            return False
        try:
            if type(self._observation_valid) is not bool:
                raise ValueError("cache validity flag must be bool")
            size = len(self._slots)
            if (type(self._pinned_slots) is not int or
                    type(self._observed_slots) is not int or
                    self._observed_slots != size or
                    not 0 <= self._pinned_slots <= size):
                raise ValueError("cache scalar/cardinality mismatch")
            return True
        except Exception as exc:
            self.invalidate_observation(type(exc).__name__ + ": " + str(exc))
            return False

    @property
    def pinned_slot_count(self) -> int | None:
        return self._pinned_slots if self._checked_observation() else None

    @property
    def clean_slot_count(self) -> int | None:
        return len(self._slots) - self._pinned_slots if self._checked_observation() else None

    def _observe_safely(self, callback, *args) -> None:
        # Also insulate the native operation from a failing observation hook.
        try:
            callback(*args)
        except Exception as exc:
            self.invalidate_observation(type(exc).__name__ + ": " + str(exc))

    def _observe_size_delta(self, delta: int) -> None:
        if not self._observation_valid:
            return
        try:
            if len(self._slots) != self._observed_slots + delta:
                raise ValueError("unexpected cache registry mutation")
            self._observed_slots += delta
            self._checked_observation()
        except Exception as exc:
            self.invalidate_observation(type(exc).__name__ + ": " + str(exc))

    def _observe_copy_delta(self, slot: _CacheSlot, delta: int) -> None:
        if not self._observation_valid:
            return
        try:
            count = slot.copies_inflight
            if (self._slots.get(slot.block_hash) is not slot or
                    type(count) is not int or count < 0 or
                    (delta == -1 and count + 1 <= 0) or
                    (delta == 1 and count - 1 < 0)):
                raise ValueError("stale/invalid cache copy reference")
            if delta == 1 and count == 1:
                self._pinned_slots += 1
            elif delta == -1 and count == 0:
                self._pinned_slots -= 1
            self._checked_observation()
        except Exception as exc:
            self.invalidate_observation(type(exc).__name__ + ": " + str(exc))

    def __contains__(self, block_hash: bytes) -> bool:
        return block_hash in self._slots

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def policy(self) -> EvictionPolicy:
        return self._policy

    def get(self, block_hash: bytes) -> _CacheSlot | None:
        slot = self._slots.get(block_hash)
        if slot is not None:
            self._policy.touch(block_hash)
        return slot

    def put(self, block_hash: bytes, slot_index: int) -> int | None:
        """Retain ``slot_index`` under ``block_hash``. Returns a slot index the
        caller must release: the passed index if the hash is already cached
        (dup -- keep the existing copy) or a self-evicted victim when at
        capacity; otherwise ``None``."""
        if self._capacity == 0:
            return slot_index
        if block_hash in self._slots:
            return slot_index
        freed: int | None = None
        if len(self._slots) >= self._capacity:
            freed = self.evict_one()
        self._slots[block_hash] = _CacheSlot(block_hash=block_hash, slot_index=slot_index)
        self._policy.admit(block_hash)
        self._observe_safely(self._observe_size_delta, 1)
        return freed

    def pin(self, slot: _CacheSlot) -> None:
        slot.copies_inflight += 1
        self._observe_safely(self._observe_copy_delta, slot, 1)

    def unpin(self, slot: _CacheSlot) -> None:
        slot.copies_inflight -= 1
        self._observe_safely(self._observe_copy_delta, slot, -1)

    def evict_one(self) -> int | None:
        """Drop the policy victim with no in-flight copy; return its slot index
        for the reactor to release, or ``None`` if empty / all pinned."""
        for block_hash in self._policy.victims():
            slot = self._slots.get(block_hash)
            if slot is None or slot.copies_inflight > 0:
                continue
            del self._slots[block_hash]
            self._policy.evict(block_hash)
            self._observe_safely(self._observe_size_delta, -1)
            if type(slot.copies_inflight) is not int or slot.copies_inflight != 0:
                self.invalidate_observation("eviction with invalid copy reference")
            return slot.slot_index
        return None

    def drain_unpinned(self) -> list[int]:
        """Stop cleanup only. Preserve native pinned objects and their policy keys."""
        indices = []
        for block_hash, slot in list(self._slots.items()):
            if type(slot.copies_inflight) is not int or slot.copies_inflight < 0:
                self.invalidate_observation("invalid copy reference during clean drain")
                continue
            if slot.copies_inflight != 0:
                continue
            del self._slots[block_hash]
            self._policy.evict(block_hash)
            self._observe_safely(self._observe_size_delta, -1)
            indices.append(slot.slot_index)
        return indices

    def drain(self) -> list[int]:
        indices = [slot.slot_index for slot in self._slots.values()]
        self._slots.clear()
        self._policy.clear()
        if self._observation_valid and self._pinned_slots == 0:
            self._observed_slots = 0
        else:
            self.invalidate_observation("cache drain without a known unpinned state")
        return indices


__all__ = [
    "ArcPolicy",
    "EvictionPolicy",
    "LruPolicy",
    "StagingDataCache",
    "make_policy",
]

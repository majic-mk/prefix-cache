from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StagingSlot:
    index: int


class StagingPool:
    def __init__(self, *, slot_count: int):
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")

        self.slot_count = slot_count
        self._closed = False
        self._slots = [StagingSlot(index=i) for i in range(slot_count)]
        self._free: list[int] = list(range(slot_count))

    @staticmethod
    def compute_slot_count(
        *,
        staging_mem_gib: float,
        io_size: int,
        min_slots: int = 1,
    ) -> int:
        if isinstance(staging_mem_gib, bool) or not isinstance(staging_mem_gib, (int, float)):
            raise ValueError("staging_mem_gib must be finite and positive")
        if not math.isfinite(staging_mem_gib) or staging_mem_gib <= 0:
            raise ValueError("staging_mem_gib must be finite and positive")
        if type(io_size) is not int or io_size <= 0:
            raise ValueError("io_size must be a positive integer")
        if type(min_slots) is not int or min_slots <= 0:
            raise ValueError("min_slots must be a positive integer")
        # ParsedKvLayout.allocate_staging_buffer retains an aligned view of a
        # backing allocation with DIRECT_IO_ALIGNMENT - 1 additional bytes.
        from .liburing_file import DIRECT_IO_ALIGNMENT
        overhead = DIRECT_IO_ALIGNMENT - 1
        staging_bytes = int(staging_mem_gib * (1 << 30))
        required = min_slots * io_size + overhead
        if required > staging_bytes:
            raise ValueError(
                f"staging budget {staging_bytes} bytes cannot fit minimum "
                f"{min_slots} slots: backing requires {required} bytes; "
                "freeze a larger budget or lower iodepth"
            )
        return (staging_bytes - overhead) // io_size

    @property
    def free_count(self) -> int:
        return len(self._free)

    def try_reserve(self) -> StagingSlot | None:
        if self._closed:
            raise RuntimeError("staging pool is closed")
        if not self._free:
            return None
        index = self._free.pop()
        return self._slots[index]

    def release(self, slot_index: int) -> None:
        if self._closed:
            return
        if slot_index < 0 or slot_index >= self.slot_count:
            raise ValueError(f"invalid staging slot index {slot_index}")
        self._free.append(slot_index)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._free.clear()


__all__ = ["StagingPool", "StagingSlot"]

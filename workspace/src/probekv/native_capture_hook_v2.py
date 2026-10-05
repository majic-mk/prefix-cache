"""Target-only taps of the actual native QKV projection.

This scope does not invoke projections, synchronize CUDA, classify provenance,
or publish a Source. A completed-block consumer must fence/copy these slices,
check finite values and match them to its real execution ledger. Capturing the
projection is not proof that the block's attention/MLP subsequently completed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class ProjectedKVSlice:
    layer_1based: int
    projection_positions: Tuple[int, ...]
    captured_positions: Tuple[int, ...]
    key: Any
    value: Any

    @property
    def full_projection_positions(self):
        """Explicit name for the full active-before layout (not captured rows)."""
        return self.projection_positions

    @property
    def owned_tensor_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.key, self.value))


def _positions(values, name):
    rows = tuple(values)
    if not rows or any(type(p) is not int or p < 0 for p in rows) or len(set(rows)) != len(rows):
        raise ValueError(name + " must contain distinct nonnegative absolute positions")
    return rows


class CurrentLayerCapture:
    """Scope one actual BF16 native ``qkv_proj`` call before in-place RoPE.

    Validation/capacity failures skip this capture, not the answer. Real runtime
    faults are not disguised as ordinary capture rejection. Output copies use
    the projection's current stream; the consumer is responsible for the normal
    completed-block event/fence before CPU export or buffer release.
    """

    def __init__(self, attention, *, layer_1based, projection_positions,
                 target_positions, max_target_bytes):
        self._attention = attention
        self.layer_1based = layer_1based
        self.projection_positions = tuple(projection_positions)
        self.target_positions = tuple(target_positions)
        self.max_target_bytes = max_target_bytes
        self.failure: Optional[str] = None
        self.invocation_count = 0
        self._packet: Optional[ProjectedKVSlice] = None
        self._handle = None
        self._entered = False
        self._closed = False

    def _fail(self, error):
        self._packet = None
        self.failure = type(error).__name__ + ": " + str(error)

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("capture scope cannot be reused")
        self._entered = True
        try:
            if type(self.layer_1based) is not int or self.layer_1based < 1:
                raise ValueError("capture layer must be 1-based")
            _positions(self.projection_positions, "projection_positions")
            _positions(self.target_positions, "target_positions")
            if not set(self.target_positions) <= set(self.projection_positions):
                raise ValueError("target rows were not fully projected")
            if type(self.max_target_bytes) is not int or self.max_target_bytes < 0:
                raise ValueError("capture requires an explicit nonnegative target byte budget")
            projection = getattr(self._attention, "qkv_proj", None)
            if not callable(getattr(projection, "register_forward_hook", None)):
                raise ValueError("native attention has no hookable qkv_proj")
            self._handle = projection.register_forward_hook(self._hook)
        except (ValueError, MemoryError) as error:
            self._fail(error)
        return self

    def _hook(self, module, inputs, output):
        # Returning None leaves the original native projection output intact.
        self.invocation_count += 1
        if self.failure is not None:
            return None
        try:
            if self.invocation_count != 1:
                raise ValueError("capture expects exactly one actual QKV invocation")
            self._packet = self._copy_target(output)
        except (ValueError, MemoryError) as error:
            self._fail(error)
        return None

    def _copy_target(self, output):
        import torch
        if isinstance(output, tuple):
            if len(output) != 2:
                raise ValueError("native QKV tuple must be (projection, bias)")
            output = output[0]
        if not torch.is_tensor(output) or output.ndim != 2 or output.dtype != torch.bfloat16:
            raise ValueError("native QKV must be a BF16 2D projection")
        attention = self._attention
        geometry = tuple(getattr(attention, field, None) for field in
                         ("q_size", "kv_size", "num_heads", "num_kv_heads", "head_dim"))
        if any(type(n) is not int or n <= 0 for n in geometry):
            raise ValueError("native QKV geometry must be explicitly known")
        q_size, kv_size, q_heads, kv_heads, head_dim = geometry
        if (q_size != q_heads * head_dim or kv_size != kv_heads * head_dim
                or q_heads % kv_heads != 0):
            raise ValueError("native QKV geometry/GQA mapping differs")
        if tuple(output.shape) != (len(self.projection_positions), q_size + 2 * kv_size):
            raise ValueError("native QKV row layout or flattened geometry differs")
        target_count = len(self.target_positions)
        required = 2 * target_count * kv_size * output.element_size()
        if required > self.max_target_bytes:
            raise MemoryError("target K/V slices exceed capture reservation")
        source_rows = {p: i for i, p in enumerate(self.projection_positions)}
        # Only these two compact allocations are owned by the hook. Basic
        # views and copy_ avoid an index_select/full-QKV intermediate and do not
        # transfer position arrays to GPU. Consecutive rows form one copy run.
        shape = (target_count, kv_heads, head_dim)
        with torch.no_grad():
            key = torch.empty(shape, dtype=output.dtype, device=output.device)
            value = torch.empty(shape, dtype=output.dtype, device=output.device)
            row_indices = [source_rows[p] for p in self.target_positions]
            first = 0
            while first < target_count:
                last = first + 1
                while last < target_count and row_indices[last] == row_indices[last - 1] + 1:
                    last += 1
                begin, end = row_indices[first], row_indices[last - 1] + 1
                key[first:last].copy_(output[begin:end, q_size:q_size + kv_size].reshape(
                    last - first, kv_heads, head_dim))
                value[first:last].copy_(output[begin:end, q_size + kv_size:].reshape(
                    last - first, kv_heads, head_dim))
                first = last
        return ProjectedKVSlice(self.layer_1based, self.projection_positions,
                                self.target_positions, key, value)

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._handle is not None:
                self._handle.remove()
        finally:
            self._handle = None
            self._attention = None
            self._closed = True
            if exc_type is not None:
                self._packet = None
                self.failure = "BLOCK_EXECUTION_FAILED: " + exc_type.__name__
            elif self.failure is None and self.invocation_count != 1:
                self._fail(ValueError("capture requires one actual QKV invocation"))
        return False

    def packet_after_block(self):
        if not self._closed:
            raise RuntimeError("capture packet is available only after the block scope exits")
        return None if self.failure is not None else self._packet

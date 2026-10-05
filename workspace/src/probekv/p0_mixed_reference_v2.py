"""Independent full-query, prescribed-upstream-KV P0 numerical reference.

This diagnostic is deliberately not a production execution ledger or a Source
publisher. It runs each ordinary model block exactly once on all prompt rows,
replacing only the declared historical upstream K/V before the caller's normal
RoPE/attention. Full target rows are freshly projected at every layer. The
caller owns Source leases, model execution, authorization and output evidence.
Importing this module imports neither Torch nor any CUDA/backend package.
"""
from __future__ import annotations

import hashlib
from functools import partial

from .v8_schema10_execution import digest_json


def _range(values, name, token_count):
    rows = tuple(values)
    if (not rows or any(type(p) is not int or p < 0 or p >= token_count for p in rows)
            or rows != tuple(range(rows[0], rows[0] + len(rows)))):
        raise ValueError(name + ' must be a nonempty contiguous ordered prompt range')
    return rows


def _version(tensor):
    # vLLM may prepare tensors inside inference_mode. The caller's physical
    # lease/digest guard remains mandatory; an optional PyTorch counter must
    # not be mistaken for cryptographic integrity verification.
    try:
        return tensor._version
    except RuntimeError as error:
        if 'Inference tensors do not track version counter' not in str(error):
            raise
        return None


class ExplicitMixedReferenceHooks:
    """Single-use scoped reference; ``target_layers`` requires successful exit.

    ``source_layers`` contains pre-RoPE BF16 K/V, already prepared/leased by the
    caller, in the same row order as ``source_positions``. All layers are
    required, even before ``first_reuse_layer``. Repair maps name every reused
    layer and contain absolute Source positions (not target positions).

    Budgets cover owned target CPU storage and peak additional device tensors;
    they do not claim to reserve caller-owned model/Source/projection memory.
    The outer runner must obtain real reservations before entering this scope.
    """

    def __init__(self, inner_model, *, token_count, target_positions,
                 source_positions, repair_positions_by_layer, first_reuse_layer,
                 source_layers, max_target_host_bytes, max_transient_device_bytes,
                 check_deadline=None):
        self._model = inner_model
        self.token_count = token_count
        self.target_positions = tuple(target_positions)
        self.source_positions = tuple(source_positions)
        self.repair_positions_by_layer = dict(repair_positions_by_layer)
        self.first_reuse_layer = first_reuse_layer
        self._source_layers = tuple(source_layers)
        self.max_target_host_bytes = max_target_host_bytes
        self.max_transient_device_bytes = max_transient_device_bytes
        self._check_deadline = check_deadline
        self._device_references_released = False
        self._handles = []
        self._entered = self._closed = self._completed = False
        self._active_layer = None
        self._pending = None
        self._targets = []
        self._trace = []
        self._projection_counts = []
        self._source_versions = []
        self._geometry = []
        self._replacement_positions = []
        self._host_bytes = self._peak_device_bytes = 0
        self.failure = None
        self.cleanup_failure = None
        self._target_digest = None

    def _validate(self):
        import torch
        if type(self.token_count) is not int or self.token_count < 2:
            raise ValueError('explicit full prompt token_count required')
        if self._check_deadline is not None and not callable(self._check_deadline):
            raise ValueError('check_deadline must be callable or None')
        self.target_positions = _range(self.target_positions, 'target_positions', self.token_count)
        self.source_positions = _range(self.source_positions, 'source_positions', self.token_count)
        if self.source_positions[-1] >= self.target_positions[0]:
            raise ValueError('historical Source must be strictly causal upstream of target')
        blocks = tuple(getattr(self._model, 'layers', ()))
        if (not blocks or len(self._source_layers) != len(blocks)
                or type(self.first_reuse_layer) is not int
                or not 1 <= self.first_reuse_layer <= len(blocks)):
            raise ValueError('complete model/Source layers and legal first reuse layer required')
        if (any(type(k) is not int for k in self.repair_positions_by_layer)
                or set(self.repair_positions_by_layer) != set(range(self.first_reuse_layer, len(blocks)+1))):
            raise ValueError('repair recipe must explicitly name every reused layer only')
        for name in ('max_target_host_bytes', 'max_transient_device_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(name + ' must be an explicit positive byte budget')
        previous = set(self.source_positions)
        for layer in range(1, len(blocks)+1):
            if layer < self.first_reuse_layer:
                replacement = ()
            else:
                repair = tuple(self.repair_positions_by_layer[layer])
                if (any(type(p) is not int for p in repair) or repair != tuple(sorted(set(repair)))
                        or not set(repair) <= previous):
                    raise ValueError('repair positions must be ordered Source subsets without reentry')
                previous = set(repair)
                self.repair_positions_by_layer[layer] = repair
                replacement = tuple(p for p in self.source_positions if p not in previous)
            self._replacement_positions.append(replacement)
        if not any(self._replacement_positions):
            raise ValueError('mixed reference requires nonzero historical K/V injection')
        device = None
        for layer, (block, pair) in enumerate(zip(blocks, self._source_layers), 1):
            attn = getattr(block, 'self_attn', None)
            geom = tuple(getattr(attn, f, None) for f in
                         ('q_size', 'kv_size', 'num_heads', 'num_kv_heads', 'head_dim'))
            if any(type(n) is not int or n <= 0 for n in geom):
                raise ValueError('explicit native QKV geometry required')
            q, kv, nh, nkh, hd = geom
            if q != nh*hd or kv != nkh*hd or nh % nkh:
                raise ValueError('QKV geometry/GQA head mapping mismatch')
            if (not callable(getattr(block, 'register_forward_pre_hook', None))
                    or not callable(getattr(block, 'register_forward_hook', None))
                    or not callable(getattr(getattr(attn, 'qkv_proj', None), 'register_forward_hook', None))):
                raise ValueError('hookable native block and qkv_proj required')
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError('each Source layer must provide K and V')
            shape = (len(self.source_positions), nkh, hd)
            for tensor in pair:
                if (not torch.is_tensor(tensor) or tensor.dtype != torch.bfloat16
                        or tuple(tensor.shape) != shape or not tensor.is_contiguous()):
                    raise ValueError('Source K/V must be contiguous BF16 with exact row/head geometry')
                if device is None:
                    device = tensor.device
                if tensor.device != device:
                    raise ValueError('all prepared Source layers must share the projection device')
            # Conservative peak: returned QKV clone, compact pre-RoPE target
            # K/V, and finite-test boolean scratch. No GPU index arrays are used.
            target_bytes = 2*len(self.target_positions)*kv*2
            peak = self.token_count*(q+2*kv)*2 + target_bytes + max(
                len(self.source_positions)*kv, len(self.target_positions)*kv) + 1
            self._host_bytes += target_bytes
            self._peak_device_bytes = max(self._peak_device_bytes, peak)
            self._geometry.append(geom)
        if self._host_bytes > self.max_target_host_bytes:
            raise MemoryError('target-only host capture exceeds explicit reservation')
        if self._peak_device_bytes > self.max_transient_device_bytes:
            raise MemoryError('reference QKV/target transient scratch exceeds explicit reservation')
        # Do not issue any finite-test tensor operations until all allocations
        # have been checked against the explicit bound.
        for pair in self._source_layers:
            if any(not bool(torch.isfinite(t).all().item()) for t in pair):
                raise ValueError('nonfinite Source K/V')
            self._source_versions.append(tuple(_version(t) for t in pair))
        self._projection_counts = [0]*len(blocks)
        return blocks

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError('mixed reference hook scope cannot be reused')
        self._entered = True
        try:
            blocks = self._validate()
            for layer, block in enumerate(blocks, 1):
                self._handles.append(block.register_forward_pre_hook(partial(self._before_block, layer)))
                self._handles.append(block.self_attn.qkv_proj.register_forward_hook(partial(self._qkv, layer)))
                self._handles.append(block.register_forward_hook(partial(self._after_block, layer)))
        except BaseException as error:
            self.failure = type(error).__name__ + ': ' + str(error)
            try:
                self._remove_handles()
            finally:
                self._closed = True
            raise
        return self

    def _before_block(self, layer, module, args):
        if self._check_deadline is not None:
            self._check_deadline()
        if (self._closed or self._active_layer is not None or self._pending is not None
                or layer != len(self._trace)+1):
            raise ValueError('mixed reference requires one ordered full-prefill block per layer, no decode')
        self._active_layer = layer

    def _qkv(self, layer, module, args, output):
        import torch
        if self._active_layer != layer or self._pending is not None:
            raise ValueError('QKV projection outside its unique active reference block')
        self._projection_counts[layer-1] += 1
        if self._projection_counts[layer-1] != 1:
            raise ValueError('duplicate QKV invocation in reference block')
        original = output[0] if isinstance(output, tuple) and output else output
        q, kv, nh, nkh, hd = self._geometry[layer-1]
        source_k, source_v = self._source_layers[layer-1]
        if (not torch.is_tensor(original) or original.dtype != torch.bfloat16
                or tuple(original.shape) != (self.token_count, q+2*kv)
                or original.device != source_k.device):
            raise ValueError('QKV must project all prompt rows in BF16 on the prepared Source device')
        if tuple(_version(t) for t in (source_k, source_v)) != self._source_versions[layer-1]:
            raise ValueError('leased historical Source mutated during reference execution')
        start, stop = self.target_positions[0], self.target_positions[-1]+1
        with torch.no_grad():
            mixed = original.clone()
            # Copy target BEFORE the caller can apply in-place RoPE. Only this
            # compact target buffer lives until the completed-block fence.
            target_k = original[start:stop, q:q+kv].reshape(-1, nkh, hd).clone()
            target_v = original[start:stop, q+kv:].reshape(-1, nkh, hd).clone()
            for absolute in self._replacement_positions[layer-1]:
                index = absolute-self.source_positions[0]
                mixed[absolute, q:q+kv].copy_(source_k[index].reshape(kv))
                mixed[absolute, q+kv:].copy_(source_v[index].reshape(kv))
        self._pending = (layer, target_k, target_v)
        if isinstance(output, tuple):
            if hasattr(output, '_fields'):
                return type(output)(mixed, *output[1:])
            return (mixed,) + output[1:]
        return mixed

    def _after_block(self, layer, module, args, output):
        import torch
        if (self._active_layer != layer or self._pending is None
                or self._pending[0] != layer or self._projection_counts[layer-1] != 1):
            raise ValueError('completed reference block lacks its unique target projection')
        _, key, value = self._pending
        if key.device.type == 'cuda':
            # Diagnostic correctness fence, not an online timing optimization.
            # A synchronous target-only D2H is insufficient to establish that
            # later attention/MLP work on this stream has completed.
            torch.cuda.current_stream(key.device).synchronize()
        if not bool(torch.isfinite(key).all().item()) or not bool(torch.isfinite(value).all().item()):
            raise ValueError('nonfinite current target K/V')
        with torch.no_grad():
            host_pair = tuple(t.to(device='cpu', copy=True).contiguous() for t in (key, value))
        self._targets.append(host_pair)
        self._trace.append(dict(layer_1based=layer, projected_rows=self.token_count,
            attention_query_rows=self.token_count,
            target_positions=list(self.target_positions),
            historical_kv_positions=list(self._replacement_positions[layer-1]),
            repair_positions=list(self.repair_positions_by_layer.get(layer, self.source_positions)),
            target_full_projection=True, completed_block=True))
        self._pending = None
        self._active_layer = None

    def _remove_handles(self):
        handles, self._handles = self._handles, []
        first_error = None
        for handle in reversed(handles):
            try:
                handle.remove()
            except BaseException as error:
                first_error = first_error or error
        if first_error is not None:
            self.cleanup_failure = type(first_error).__name__ + ': ' + str(first_error)
            raise first_error

    def _finish(self):
        import torch
        if (self._active_layer is not None or self._pending is not None
                or len(self._trace) != len(self._source_layers)
                or any(n != 1 for n in self._projection_counts)
                or any(tuple(_version(t) for t in pair) != versions
                       for pair, versions in zip(self._source_layers, self._source_versions))):
            raise ValueError('incomplete/duplicate reference forward or mutated Source')
        digest = hashlib.sha256()
        for layer, pair in enumerate(self._targets, 1):
            digest.update(str((layer, tuple(pair[0].shape), 'BF16 pre-RoPE')).encode('ascii'))
            for tensor in pair:
                # Zero-copy view over owned CPU target data; no parent KV or
                # full-request tensor is copied or hashed by this scope.
                digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast('B'))
        self._target_digest = digest.hexdigest()
        self._completed = True

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is not None:
                self.failure = 'REFERENCE_EXECUTION_FAILED: ' + exc_type.__name__
            else:
                try:
                    self._finish()
                except BaseException as error:
                    self.failure = type(error).__name__ + ': ' + str(error)
                    raise
        finally:
            try:
                self._remove_handles()
            except BaseException:
                self.failure = self.failure or 'HOOK_CLEANUP_FAILED'
                # Keep the original model exception, if any, as the primary
                # failure. Cleanup failure is separately visible in the audit.
                if exc_type is None:
                    raise
            finally:
                self._closed = True
                self._active_layer = None
                # Model/Source and any failed in-flight target packet remain
                # held until the caller fences and explicitly releases them.
                # Removing Python hooks alone is not CUDA completion evidence.
                if self.failure is not None:
                    self._completed = False
                    self._targets.clear()
                    self._target_digest = None
        return False

    def release_device_references(self, *, caller_fenced):
        """Release only after scope exit and the caller's actual completion fence.

        This does not invent a completion event, drop a physical lease, or
        release the caller's HBM reservation. The owning runner performs those
        operations after its fence; an explicit false/missing fence is rejected.
        """
        if not self._closed or caller_fenced is not True:
            raise RuntimeError('exited reference scope and explicit caller completion fence required')
        self._model = None
        self._source_layers = ()
        self._pending = None
        self._check_deadline = None
        self._device_references_released = True
        if not self._completed or self.failure is not None:
            self._targets.clear()

    @property
    def target_layers(self):
        if not self._closed or not self._completed or self.failure is not None:
            raise RuntimeError('target layers require successfully completed reference scope')
        return tuple(self._targets)

    def audit(self):
        recipe = dict(kind='independent_explicit_mixed_full_query_reference',
            token_count=self.token_count, target_positions=list(self.target_positions),
            source_positions=list(self.source_positions), first_reuse_layer=self.first_reuse_layer,
            repair_positions_by_layer={str(k):list(v) for k,v in self.repair_positions_by_layer.items()},
            geometry=[list(g) for g in self._geometry])
        return dict(kind='p0_mixed_reference_hooks', completed=self._closed and self._completed,
            failure=self.failure, cleanup_failure=self.cleanup_failure,
            device_references_released=self._device_references_released,
            recipe=recipe, recipe_sha256=digest_json(recipe),
            executed_layers=list(self._trace), projection_counts=list(self._projection_counts),
            historical_kv_rows_replaced=sum(len(r['historical_kv_positions']) for r in self._trace),
            target_logical_digest=self._target_digest, target_owned_host_bytes=self._host_bytes if self._completed else 0,
            transient_device_bytes_upper=self._peak_device_bytes,
            parent_owned_kv_bytes=0, prefix_shadow_bytes=0,
            source_version_counters_available=bool(self._source_versions) and all(
                version is not None for pair in self._source_versions for version in pair),
            caller_source_lease_and_integrity_guard_required=True,
            reference_scope='fixed_upstream_recipe_only_not_dense_equivalence',
            publication_allowed=False, native_runtime_qualified=False, P1_execution_allowed=False,
            paper_evidence=False)

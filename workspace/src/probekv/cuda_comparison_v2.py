"""Explicit P0 CUDA comparison workspace; no authorization/quality certificate.

Only target K SelectionStates transfer here. Current QKV observation may retain
views in the existing engine, so its reservation lives until request close,
not merely until the scalar score has reached the CPU. No new CUDA stream,
Prefix shadow, full Source load, scorer or repair algorithm is introduced.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import time

from .source_manifest_v2 import request_input_digest
from .v8_schema6_hbm import HBMReservationKind, UnifiedHBMReservationManager


def comparison_workspace_estimate(*, active_rows, hidden_width, query_width, kv_width, target_rows):
    """Conservative tensor/sort admission estimate, not a CUDA allocator cap.

    Includes whole active-request cloned projection and retained QKV views,
    target current K, FP32 reductions, one historical K and stable-sort scratch.
    Hardware allocator/workspace overhead still needs actual P0 measurement.
    """
    values=(active_rows,hidden_width,query_width,kv_width,target_rows)
    if any(type(v) is not int or v<=0 for v in values) or target_rows>active_rows:
        raise ValueError('positive real execution geometry required')
    return 64*(active_rows*(hidden_width+query_width+2*kv_width)+target_rows*kv_width+target_rows)+16*1024**2


class ComparisonReservationV2:
    """CPU-testable request ownership; fenced release is deliberately explicit."""
    def __init__(self, manager, request_id, capacity_bytes):
        if (type(manager) is not UnifiedHBMReservationManager or not request_id
                or type(capacity_bytes) is not int or capacity_bytes<=0):
            raise ValueError('shared HBM manager, request and bounded bytes required')
        self.manager,self.request_id,self.capacity_bytes=manager,request_id,capacity_bytes
        self.reservation=None
        self.closed=self.quarantined=False

    def acquire(self, required_bytes):
        if self.closed or self.quarantined: raise RuntimeError('comparison workspace closed/quarantined')
        if type(required_bytes) is not int or not 0<required_bytes<=self.capacity_bytes:
            raise MemoryError('comparison shape exceeds explicit workspace budget')
        if self.reservation is None:
            self.reservation=self.manager.reserve_batch(owner_request_id=self.request_id,
                rows=(('v2_current_k_comparison',self.capacity_bytes,HBMReservationKind.SELECTION_WORKSPACE),))[0]
        return self.validate()

    def validate(self):
        r=self.reservation
        if r is None:raise ValueError('comparison reservation missing')
        if (self.manager.reservations.get(r.reservation_id) is not r or r.released
                or r.owner_request_id!=self.request_id or r.kind is not HBMReservationKind.SELECTION_WORKSPACE
                or r.bytes!=self.capacity_bytes):
            raise ValueError('comparison reservation owner/kind/size changed')
        return r

    def close(self, *, fence, clear_observations):
        if self.closed:return
        if self.quarantined:raise RuntimeError('comparison workspace quarantined; explicit recovery required')
        try:
            if self.reservation is not None:self.validate()
            fence()
            clear_observations()
        except BaseException:
            self.quarantined=True
            raise
        if self.reservation is not None:self.manager.release(self.reservation.reservation_id)
        self.closed=True


class CudaComparisonWorkspaceV2:
    """Opt-in actual torch CUDA path. Construction never starts model work.

    CPU tests mock CUDA explicitly; they must not be used as CUDA evidence.
    Timings below are diagnostic envelopes, not additive TTFT components.
    """
    def __init__(self, context, *, capacity_bytes):
        from .v8_schema10_native_adapter import NativeRequestContext
        import torch
        if not isinstance(context,NativeRequestContext) or context.engine is None:
            raise ValueError('an already-open native prefill context is required')
        if context.closed or context.finished or getattr(context,'comparison_workspace_v2',None) is not None:
            raise ValueError('request already closed or has a comparison workspace')
        hidden=context.engine.session.hidden_states
        if (not torch.is_tensor(hidden) or hidden.device.type!='cuda' or hidden.dtype!=torch.bfloat16
                or hidden.ndim!=2 or context.adapter.torch is not torch):
            raise ValueError('real native BF16 CUDA hidden state required')
        self.context,self.torch,self.device=context,torch,hidden.device
        self.request_id=context.request['request_id']
        self.input_digest=request_input_digest(context.request['token_ids'],tuple(range(len(context.request['token_ids']))))
        self.lease=ComparisonReservationV2(context.adapter.hbm,self.request_id,capacity_bytes)
        self.events=[];self._active=False;self._failed=False
        context.comparison_workspace_v2=self

    def _check(self, context):
        if (context is not self.context or context.closed or context.finished or self._failed
                or context.request['request_id']!=self.request_id
                or context.comparison_workspace_v2 is not self
                or context.adapter.hbm is not self.lease.manager
                or self.lease.closed or self.lease.quarantined
                or request_input_digest(context.request['token_ids'],tuple(range(len(context.request['token_ids']))))!=self.input_digest):
            raise ValueError('stale, failed or cross-request CUDA workspace')

    def required_bytes(self, segment_id):
        c=self.context;s=c.engine.session
        a=c.adapter.inner.layers[c.current_completed_depth].self_attn
        hidden=s.hidden_states
        if hidden.device!=self.device or hidden.dtype!=self.torch.bfloat16:
            raise ValueError('native hidden state device/dtype changed')
        return comparison_workspace_estimate(active_rows=len(s.active_positions),hidden_width=hidden.shape[-1],
            query_width=int(a.q_size),kv_width=int(a.kv_size),target_rows=len(c.segments[segment_id]['positions']))

    @contextmanager
    def operation(self, context, segment_id):
        self._check(context)
        if self._active:raise RuntimeError('nested comparison operation forbidden')
        required=self.required_bytes(segment_id)
        reservation=self.lease.acquire(required)  # BEFORE projection/normalization/H2D.
        self._active=True
        self._h2d_bytes=0;self._digest_d2h_bytes=0;self._digest_host_ms=0.
        started=time.perf_counter_ns();begin=end=None;passed=False
        try:
            with self.torch.cuda.device(self.device):
                begin=self.torch.cuda.Event(enable_timing=True)
                end=self.torch.cuda.Event(enable_timing=True)
                begin.record(self.torch.cuda.current_stream(self.device))
                yield self
                passed=True
        except BaseException:
            self._failed=True
            raise
        finally:
            cuda_ms=None
            try:
                # Failure may have enqueued work before raising. Fence the
                # device on that path before permitting normal request cleanup.
                if passed and end is not None:
                    end.record(self.torch.cuda.current_stream(self.device));end.synchronize()
                    cuda_ms=float(begin.elapsed_time(end))
                    if not math.isfinite(cuda_ms) or cuda_ms<0:
                        cuda_ms=None
                        raise RuntimeError('invalid CUDA event interval')
                else:self.torch.cuda.synchronize(self.device)
            except BaseException:
                self._failed=True;self.lease.quarantined=True
                raise
            finally:
                self._active=False
                self.events.append(dict(segment_id=segment_id,completed_depth=context.current_completed_depth,
                    reservation_id=reservation.reservation_id,reserved_bytes=reservation.bytes,
                    estimated_tensor_workspace_bytes=required,selection_state_h2d_bytes=self._h2d_bytes,
                    current_k_digest_d2h_bytes=self._digest_d2h_bytes,current_k_digest_host_ms=self._digest_host_ms,
                    host_ms=(time.perf_counter_ns()-started)/1e6,cuda_envelope_ms=cuda_ms,
                    outcome='COMPLETED' if passed and not self._failed else 'FAILED',
                    timing_scope='diagnostic_comparison_envelope_not_additive_ttft',
                    full_source_kv_bytes=0,native_runtime_qualified=False))

    def validate_current(self, current):
        if not self._active or current.device!=self.device:
            raise ValueError('current K does not belong to the active CUDA operation')

    def source_to_device(self, key):
        if not self._active or key.device.type!='cpu' or key.dtype!=self.torch.bfloat16:
            raise ValueError('independent CPU BF16 K SelectionState required')
        self._h2d_bytes+=key.numel()*key.element_size()
        return key.to(self.device,non_blocking=False)

    def current_digest(self, current):
        from .v8_schema10_storage import tensor_digest
        self.validate_current(current)
        started=time.perf_counter_ns()
        result=tensor_digest((current,))  # Small target K only; diagnostic D2H is charged.
        self._digest_d2h_bytes+=current.numel()*current.element_size()
        self._digest_host_ms+=(time.perf_counter_ns()-started)/1e6
        return result

    def close(self):
        if self._active:raise RuntimeError('cannot close active comparison')
        def clear():
            c=self.context
            c._observation.clear()
            if c.engine is not None:c.engine.session._clear_observation()
        self.lease.close(fence=lambda:self.torch.cuda.synchronize(self.device),clear_observations=clear)

    def audit(self):
        return dict(device=str(self.device),events=list(self.events),request_id=self.request_id,
            quarantined=self.lease.quarantined,closed=self.lease.closed,
            native_runtime_qualified=False,formal_profile_frozen=False,
            algorithm='unchanged_legacy_k_trim',diagnostic_only=True)

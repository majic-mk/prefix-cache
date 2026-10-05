"""Bounded independent mixed reference, never a production admission bypass.

One preregistered G0 upstream Source; full native query rows, frozen historical
K/V injection, complete target rows. The caller owns input authorization and a
visible store snapshot. This reference never supplies the natural Source pool.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
import time

from .native_p0_request_v2 import P0RequestFailure
from .p0_decode_evidence_v2 import DecodeInputRecorderV2
from .source_manifest_v2 import request_input_digest
from .source_store_v2 import TargetSourceStoreV2
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest, tensor_digest
from .v8_schema6_hbm import HBMReservationKind


def validate_mixed_reference_job(job, *, total_layers=None, _operation='explicit_mixed_reference'):
    """No model/tensor loads. Explicit masks are diagnostics, not a new scorer."""
    q = job['request']
    if (_operation not in ('explicit_mixed_reference', 'mixed_sparse_control')
            or job.get('operation') != _operation
            or 'sources_by_segment' in job or 'comparison_profile' in job
            or q.get('capture_logits') is not True or 'teacher_token_ids' not in q
            or q.get('native_dense_continuation') or q.get('capture_original_full_prefill')
            or q.get('publish_exact_prefix_shadow') or q.get('use_gpu_hot_cache')
            or q.get('retain_gpu_hot_cache') or q.get('prefetch_window', 0) != 0
            or q.get('correctness_repair_ratio', .15) != .15):
        raise ValueError('explicit mixed teacher diagnostic required; no online Source decisions or Prefix capture')
    DecodeInputRecorderV2(q, cached_prefix_tokens=0)
    segments = {s['segment_id']: s for s in q['segments']}
    if (len(segments) != len(q['segments']) or job.get('target_id') not in segments
            or job.get('upstream_segment_id') not in segments
            or job['target_id'] == job['upstream_segment_id']):
        raise ValueError('distinct complete target and upstream Segment required')
    for sid in (job['upstream_segment_id'], job['target_id']):
        segment = segments[sid]; positions = segment['positions']
        if (not isinstance(positions, list) or not positions
                or any(type(p) is not int or not 0 <= p < len(q['token_ids']) for p in positions)
                or positions != list(range(positions[0], positions[-1]+1))
                or list(segment['token_ids']) != [q['token_ids'][p] for p in positions]):
            raise ValueError('original contiguous token/absolute-position identity required')
    upstream = segments[job['upstream_segment_id']]['positions']
    target = segments[job['target_id']]['positions']
    if upstream[-1] >= target[0]:
        raise ValueError('historical injection must be strictly upstream of the full target')
    source = job.get('source', {})
    if (set(source) != {'source_id', 'artifact_digest'}
            or type(source['source_id']) is not str or not source['source_id']
            or not re.fullmatch('[0-9a-f]{64}', str(source['artifact_digest']))):
        raise ValueError('one immutable already-published Source/digest required')
    target_execution = job.get('target_execution', 'FULL_ALL_LAYERS')
    target_source = job.get('target_source')
    if target_execution not in ('FULL_ALL_LAYERS', 'R1_ALL_LAYERS'):
        raise ValueError('explicit FULL or target-r1 diagnostic execution required')
    if target_execution == 'R1_ALL_LAYERS':
        if (not isinstance(target_source, dict)
                or set(target_source) != {'source_id', 'artifact_digest'}
                or type(target_source['source_id']) is not str or not target_source['source_id']
                or type(target_source['artifact_digest']) is not str
                or not re.fullmatch('[0-9a-f]{64}', target_source['artifact_digest'])):
            raise ValueError('target-r1 requires its own immutable already-published Source/digest')
    elif 'target_source' in job:
        raise ValueError('ordinary full target must not silently acquire a target Source')
    boundary = job.get('first_reuse_layer')
    masks = job.get('repair_positions_by_layer')
    if (type(boundary) is not int or boundary < 1 or not isinstance(masks, dict) or not masks
            or any(type(k) is not str or not re.fullmatch('[1-9][0-9]*', k) for k in masks)):
        raise ValueError('explicit positive boundary and layer-indexed mask recipe required')
    layers = sorted(int(k) for k in masks)
    last = total_layers if total_layers is not None else layers[-1]
    if type(last) is not int or layers != list(range(boundary, last+1)):
        raise ValueError('every layer from boundary to model end must have a frozen mask')
    endpoint = job.get('upstream_repair_endpoint')
    if 'upstream_repair_endpoint' in job and (endpoint != 'R0_DIAGNOSTIC'
            or target_execution != 'FULL_ALL_LAYERS'):
        raise ValueError('only explicit upstream R0 with full target is an endpoint diagnostic')
    count = 0 if endpoint == 'R0_DIAGNOSTIC' else math.ceil(.15 * len(upstream))
    if count >= len(upstream):
        raise ValueError('mixed reference requires nonzero historical K/V replacement')
    previous = set(upstream)
    for layer in layers:
        rows = masks[str(layer)]
        if (not isinstance(rows, list) or any(type(p) is not int for p in rows)
                or len(rows) != count or rows != sorted(set(rows))
                or not set(rows) <= previous):
            raise ValueError('frozen fixed15 or explicit R0 absolute repair rows and no reentry required')
        previous = set(rows)
    return dict(upstream_positions=tuple(upstream), target_positions=tuple(target),
                repair_positions_by_layer={int(k):tuple(v) for k,v in masks.items()},
                target_execution=target_execution)


class _ReferenceResources:
    """Retained by the request when a fence fails; never unconditional unpin."""
    def __init__(self, context, lease, *, store, snapshot, bindings):
        self.context, self.lease = context, lease
        self.store, self.snapshot = store, snapshot
        self.bindings = deepcopy(bindings)
        self.acquired = False
        self.reservation = None
        self.source_layers = []
        self.target_r1_row = None
        self.target_r1_lease = None
        self.target_r1_acquired = False
        self.target_r1_layers = []
        self.hooks = None
        self.closed = self.quarantined = False

    def authorize(self, *, source_id, segment_id, bytes_required):
        """The diagnostic uses v2 leases, never the legacy production pool.

        This checks ownership only; it cannot authorize online reuse or Source
        publication. Both manual preparation and the resident loader call it.
        """
        from .p0_diagnostic_context_v2 import diagnostic_context_v2
        c, manager = self.context, self.context.adapter.hbm
        marker = diagnostic_context_v2(c)
        with self.store.lock:
            self.store._snapshot(self.snapshot)
            expected = self.bindings.get(segment_id)
            r = self.reservation
            if (self.closed or self.quarantined or c.closed or marker is None
                    or c.p0_diagnostic_resources_v2 is not self
                    or self.snapshot.request_id != c.request['request_id']
                    or expected is None or expected['source_id'] != source_id
                    or not self.acquired
                    or (self.target_r1_row is not None
                        and source_id == self.target_r1_row['source_id']
                        and not self.target_r1_acquired)
                    or not self.store._lease_counts.get(source_id)
                    or self.store._visible_row(self.snapshot, source_id) != expected
                    or type(bytes_required) is not int
                    or bytes_required != expected['target_kv_bytes']
                    or r is None or r.released
                    or manager.reservations.get(r.reservation_id) is not r
                    or r.owner_request_id != c.request['request_id']
                    or r.kind != HBMReservationKind.COMMITTED_EXECUTION
                    or r.bytes < sum(row['target_kv_bytes'] for row in self.bindings.values())):
                raise ValueError('diagnostic transfer lacks bound v2 Source lease/HBM ownership')

    def close_after_fence(self):
        if self.closed:
            return
        if self.quarantined:
            raise RuntimeError('mixed diagnostic resources quarantined; explicit recovery required')
        # Hooks must already be detached even if native forward raised.
        if self.hooks is not None:
            if self.hooks.cleanup_failure is not None:
                self.quarantined = True
                raise RuntimeError('mixed reference hook removal failed; adapter must remain quarantined')
            self.hooks.release_device_references(caller_fenced=True)
        # Both physical inputs remain protected until the caller has fenced.
        # A cleanup failure quarantines the reservation instead of making HBM
        # available while a remaining Source lease has uncertain state.
        try:
            if self.target_r1_acquired:
                self.target_r1_lease.__exit__(None, None, None)
                self.target_r1_acquired = False
            if self.acquired:
                self.lease.__exit__(None, None, None)
                self.acquired = False
        except BaseException:
            self.quarantined = True
            raise
        self.source_layers.clear()
        self.target_r1_layers.clear()
        if self.reservation is not None:
            self.context.adapter.hbm.release(self.reservation.reservation_id)
        self.closed = True


def execute_explicit_mixed_reference(context, store, snapshot, job, *,
                                     host_reference_bytes, cuda_reference_bytes):
    return _execute_mixed_control(context, store, snapshot, job,
        host_reference_bytes=host_reference_bytes, cuda_reference_bytes=cuda_reference_bytes)


def _execute_mixed_control(context, store, snapshot, job, *,
                           host_reference_bytes, cuda_reference_bytes, sparse=False):
    """Shared identity/resource lifecycle for independent reference and sparse arms.

The reference uses normal full-query blocks; the sparse arm uses the existing
resumable engine. Neither bypasses a production planner nor publishes targets.
Source/destination hashing is diagnostic. Target CPU arrays survive successful
close only until the raw evidence writer.
"""
    from .v8_schema10_native_adapter import NativeRequestContext
    from .p0_diagnostic_context_v2 import bind_p0_diagnostic_context
    from .p0_mixed_reference_v2 import ExplicitMixedReferenceHooks
    if type(context) is not NativeRequestContext or type(store) is not TargetSourceStoreV2:
        raise ValueError('native context and verified v2 Source store required')
    c, a = context, context.adapter
    operation = 'mixed_sparse_control' if sparse else 'explicit_mixed_reference'
    audit_field = 'execution' if sparse else 'reference'
    plan = validate_mixed_reference_job(job, total_layers=a.spec.num_layers, _operation=operation)
    if (c.request != job['request'] or c.engine is not None or c.finished or c.closed
            or c.cached_prefix_tokens or c.probe_fallback_reason
            or c.prepared or c.frozen or c.committed or c.source_capture_v2 is not None
            or c.source_consumption_v2 is not None or c.capture_reservation is not None
            or c.repair_ratio != .15 or snapshot.request_id != c.request['request_id']
            or type(c.arrival_ns) is not int or not 0 < c.arrival_ns <= time.perf_counter_ns()
            or any(type(v) is not int or v <= 0 for v in (host_reference_bytes, cuda_reference_bytes))):
        raise ValueError('fresh native no-Prefix context and explicit reference resource bounds required')
    if (store.config['model_signature'] != a.provenance['model_signature']
            or store.config['tokenizer_hash'] != a.provenance['tokenizer_hash']):
        raise ValueError('reference store/model/tokenizer identity mismatch')
    with store.lock:
        row = deepcopy(store._visible_row(snapshot, job['source']['source_id']))
        parent = store.parent_metadata(snapshot, row['source_id'])
    attn = a.inner.layers[0].self_attn
    source_tokens = tuple(c.request['token_ids'][p] for p in plan['upstream_positions'])
    if (row['artifact_digest'] != job['source']['artifact_digest'] or row['origin'] != 'EXACT_CONTEXT'
            or row['generation'] != 0 or parent.generation != 0
            or tuple(row['token_ids']) != source_tokens or row['content_key'] != store.content_key(source_tokens)
            or row['layer_count'] != a.spec.num_layers
            or tuple(row['shape']) != (len(source_tokens), attn.num_kv_heads, attn.head_dim)):
        raise ValueError('fixed G0 Source token/model/layer/Artifact identity mismatch')
    target_r1_row = None
    if plan['target_execution'] == 'R1_ALL_LAYERS':
        with store.lock:
            target_r1_row = deepcopy(store._visible_row(snapshot, job['target_source']['source_id']))
            target_parent = store.parent_metadata(snapshot, target_r1_row['source_id'])
        target_tokens = tuple(c.request['token_ids'][p] for p in plan['target_positions'])
        if (target_r1_row['artifact_digest'] != job['target_source']['artifact_digest']
                or target_r1_row['origin'] != 'EXACT_CONTEXT'
                or target_r1_row['generation'] != 0 or target_parent.generation != 0
                or tuple(target_r1_row['token_ids']) != target_tokens
                or target_r1_row['content_key'] != store.content_key(target_tokens)
                or target_r1_row['layer_count'] != a.spec.num_layers
                or tuple(target_r1_row['shape']) != (len(target_tokens), attn.num_kv_heads, attn.head_dim)):
            raise ValueError('target-r1 G0 Source token/model/layer/Artifact identity mismatch')
    ids, positions = c._prepared_inputs[:2]
    torch = a.torch
    expected = tuple(range(len(c.request['token_ids'])))
    if (tuple(ids.detach().cpu().tolist()) != tuple(c.request['token_ids'])
            or tuple(positions.detach().cpu().tolist()) != expected):
        raise ValueError('native input/position tensors differ from the frozen recipe')
    target_bytes = 4 * a.spec.num_layers * len(plan['target_positions']) * attn.num_kv_heads * attn.head_dim
    source_bytes = row['target_kv_bytes'] + (target_r1_row['target_kv_bytes'] if target_r1_row else 0)
    layer_bytes = source_bytes // a.spec.num_layers
    # Headroom includes source read/hash CPU temporaries separately from target
    # capture. It is not permission to retain a whole parent prompt allocation.
    host_transient = 3 * layer_bytes
    if (host_reference_bytes < target_bytes + host_transient
            or cuda_reference_bytes <= source_bytes):
        raise MemoryError('explicit diagnostic Source/target buffers exceed frozen bounds')
    recipe = dict(kind='explicit_mixed_sparse_control' if sparse else 'explicit_mixed_full_query_reference',
        request_id=c.request['request_id'],
        input_digest=request_input_digest(c.request['token_ids'], expected),
        model_signature=a.provenance['model_signature'], target_id=job['target_id'],
        target_positions=list(plan['target_positions']), upstream_segment_id=job['upstream_segment_id'],
        source_id=row['source_id'], artifact_digest=row['artifact_digest'], source_generation=0,
        source_birth_positions=row['birth_target_positions'], source_current_positions=list(plan['upstream_positions']),
        first_reuse_layer=job['first_reuse_layer'], repair_positions_by_layer=job['repair_positions_by_layer'],
        num_layers=a.spec.num_layers, prefix_tokens=0, target_execution=plan['target_execution'],
        production_publication_allowed=False)
    if 'upstream_repair_endpoint' in job:
        recipe['upstream_repair_endpoint'] = job['upstream_repair_endpoint']
    if target_r1_row is not None:
        recipe['target_source'] = dict(source_id=target_r1_row['source_id'],
            artifact_digest=target_r1_row['artifact_digest'], source_generation=0,
            source_birth_positions=target_r1_row['birth_target_positions'],
            source_current_positions=list(plan['target_positions']))
    bind_p0_diagnostic_context(c, recipe_sha256=digest_json(recipe), kind=operation)
    bindings = {job['upstream_segment_id']: row}
    if target_r1_row is not None:
        bindings[job['target_id']] = target_r1_row
    owner = _ReferenceResources(c, store.leased_target(snapshot, row['source_id']),
                               store=store, snapshot=snapshot, bindings=bindings)
    if target_r1_row is not None:
        owner.target_r1_row = target_r1_row
        owner.target_r1_lease = store.leased_target(snapshot, target_r1_row['source_id'])
    c.p0_diagnostic_resources_v2 = owner
    c.p0_reference_target_layers_v2 = ()
    report = dict(kind='p0_mixed_sparse_control' if sparse else 'p0_explicit_mixed_reference',
        status='RUNNING', request_id=c.request['request_id'],
        recipe=recipe, recipe_sha256=digest_json(recipe), answer=None, reference=None, cleanup=None,
        publication=dict(status='DIAGNOSTIC_NOT_PUBLISHED'), extra_forward_count=0,
        production_reuse_commit_observed=False, native_runtime_qualified=False,
        P1_execution_allowed=False, paper_evidence=False, diagnostic_costs={})
    error = None
    try:
        a.check_deadline()
        layers = owner.lease.__enter__(); owner.acquired = True
        target_r1_host_layers = None
        if target_r1_row is not None:
            target_r1_host_layers = owner.target_r1_lease.__enter__()
            owner.target_r1_acquired = True
        started = time.perf_counter_ns()
        path = store._path(row['kv_file'])
        if file_digest(path) != row['kv_file_digest']:
            raise ValueError('Source file digest changed before reference')
        before = tensor_digest(t for pair in layers for t in pair)
        if before != row['artifact_digest']:
            raise ValueError('Source logical digest changed before reference')
        report['diagnostic_costs']['source_before_hash_host_ms'] = (time.perf_counter_ns()-started)/1e6
        if target_r1_row is not None:
            started = time.perf_counter_ns()
            target_path = store._path(target_r1_row['kv_file'])
            if file_digest(target_path) != target_r1_row['kv_file_digest']:
                raise ValueError('target-r1 Source file digest changed before reference')
            target_before = tensor_digest(t for pair in target_r1_host_layers for t in pair)
            if target_before != target_r1_row['artifact_digest']:
                raise ValueError('target-r1 Source logical digest changed before reference')
            report['diagnostic_costs']['target_source_before_hash_host_ms'] = (time.perf_counter_ns()-started)/1e6
        owner.reservation = a.hbm.reserve_batch(owner_request_id=c.request['request_id'],
            rows=(('p0_explicit_mixed_reference', cuda_reference_bytes, HBMReservationKind.COMMITTED_EXECUTION),))[0]
        for sid, bound in bindings.items():
            owner.authorize(source_id=bound['source_id'], segment_id=sid,
                            bytes_required=bound['target_kv_bytes'])
        started = time.perf_counter_ns()
        # Retain partial allocations on the owner before the next transfer, so
        # an exception cannot free an in-flight tensor outside the final fence.
        for pair in layers:
            a.check_deadline()
            for t in pair:
                if t.dtype != torch.bfloat16 or not bool(torch.isfinite(t).all()):
                    raise ValueError('finite BF16 historical Source required')
            owner.source_layers.append([])
            for t in pair:
                owner.source_layers[-1].append(t.to(device=ids.device, copy=True))
        if target_r1_host_layers is not None:
            for pair in target_r1_host_layers:
                a.check_deadline()
                for t in pair:
                    if t.dtype != torch.bfloat16 or not bool(torch.isfinite(t).all()):
                        raise ValueError('finite BF16 target-r1 historical Source required')
                owner.target_r1_layers.append([])
                for t in pair:
                    owner.target_r1_layers[-1].append(t.to(device=ids.device, copy=True))
        c.synchronize()
        report['diagnostic_costs']['source_prepare_host_ms'] = (time.perf_counter_ns()-started)/1e6
        started = time.perf_counter_ns()
        destination = tensor_digest(t for pair in owner.source_layers for t in pair)
        if destination != before:
            raise ValueError('reference destination differs from leased Source')
        report['diagnostic_costs']['destination_hash_D2H_host_ms'] = (time.perf_counter_ns()-started)/1e6
        if target_r1_row is not None:
            started = time.perf_counter_ns()
            target_destination = tensor_digest(t for pair in owner.target_r1_layers for t in pair)
            if target_destination != target_before:
                raise ValueError('target-r1 destination differs from leased Source')
            report['diagnostic_costs']['target_destination_hash_D2H_host_ms'] = (time.perf_counter_ns()-started)/1e6
        a.inner.cache_fuse_metadata.update(check=False, collect=False, probekv_cfo_collector=None,
            probekv_resumable=False, reuse_active=False, dense_full_repair_endpoint=False, exact_prefix_tokens=0)
        started = time.perf_counter_ns()
        if sparse:
            from .p0_mixed_sparse_v2 import run_sparse_prefill
            report['extra_forward_count'] = 1
            hidden, execution, targets = run_sparse_prefill(c, owner, plan, row, job,
                host_target_bytes=host_reference_bytes-host_transient,
                transient_device_bytes=cuda_reference_bytes-source_bytes)
            report[audit_field] = execution
            c.p0_reference_target_layers_v2 = targets
        else:
            owner.hooks = ExplicitMixedReferenceHooks(a.inner, token_count=len(expected),
                target_positions=plan['target_positions'], source_positions=plan['upstream_positions'],
                repair_positions_by_layer=plan['repair_positions_by_layer'], first_reuse_layer=job['first_reuse_layer'],
                source_layers=owner.source_layers, max_target_host_bytes=host_reference_bytes-host_transient,
                max_transient_device_bytes=cuda_reference_bytes-source_bytes,
                check_deadline=a.check_deadline)
            with owner.hooks:
                report['extra_forward_count'] = 1
                hidden = a.outer(input_ids=ids, positions=positions, kv_caches=a.kv, attn_metadata=c.attention)
            report[audit_field] = owner.hooks.audit()
            c.p0_reference_target_layers_v2 = owner.hooks.target_layers
        report['diagnostic_costs']['reference_prefill_capture_host_ms'] = (time.perf_counter_ns()-started)/1e6
        if len(c.p0_reference_target_layers_v2) != a.spec.num_layers:
            raise RuntimeError('reference target capture is missing layers')
        first = []
        # Hooks are detached before the common endpoint invokes teacher decode.
        report['answer'] = c.finish_from_prefill_hidden(hidden, lambda:first.append(time.perf_counter_ns()))
        if not c.finished or len(first) != 1:
            raise RuntimeError('completed teacher answer and one first-token endpoint required')
        report['timing'] = dict(arrival_ns=c.arrival_ns, first_token_ns=first[0],
            ttft_ms=(first[0]-c.arrival_ns)/1e6, answer_complete_ns=time.perf_counter_ns(),
            scope='diagnostic_reference_not_production_TTFT')
        c.synchronize()
        started = time.perf_counter_ns()
        after = tensor_digest(t for pair in layers for t in pair)
        destination_after = tensor_digest(t for pair in owner.source_layers for t in pair)
        if (file_digest(path) != row['kv_file_digest'] or after != before or destination_after != before):
            raise ValueError('historical Source or diagnostic destination mutated')
        report['diagnostic_costs']['source_destination_after_hash_host_ms'] = (time.perf_counter_ns()-started)/1e6
        report['integrity'] = dict(source_before=before, destination=destination,
                                   source_after=after, destination_after=destination_after, passed=True)
        if target_r1_row is not None:
            started = time.perf_counter_ns()
            target_after = tensor_digest(t for pair in target_r1_host_layers for t in pair)
            target_destination_after = tensor_digest(t for pair in owner.target_r1_layers for t in pair)
            if (file_digest(target_path) != target_r1_row['kv_file_digest']
                    or target_after != target_before or target_destination_after != target_before):
                raise ValueError('historical target-r1 Source or diagnostic destination mutated')
            report['diagnostic_costs']['target_source_destination_after_hash_host_ms'] = (time.perf_counter_ns()-started)/1e6
            report['target_source_integrity'] = dict(source_before=target_before, destination=target_destination,
                source_after=target_after, destination_after=target_destination_after, passed=True)
        report['status'] = 'COMPLETED'
    except BaseException as exc:
        error = exc
        report.update(status='FAILED', failure=dict(type=type(exc).__name__, detail=str(exc)))
        if owner.hooks is not None:
            report[audit_field] = owner.hooks.audit()
    finally:
        try:
            c.close()
            report['cleanup'] = dict(passed=True, failures=[])
        except BaseException as exc:
            error = error or exc
            report['cleanup'] = dict(passed=False, failures=[dict(type=type(exc).__name__, detail=str(exc))])
        if error is not None:
            c.p0_reference_target_layers_v2 = ()
        if owner.hooks is not None:
            report[audit_field] = owner.hooks.audit()
        report['service_end_ns'] = time.perf_counter_ns()
        report['total_service_ms'] = (report['service_end_ns']-c.arrival_ns)/1e6
    if error is not None:
        report['status'] = 'FAILED'
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            error.p0_audit = report
            raise error
        raise P0RequestFailure(report, error) from error
    return report

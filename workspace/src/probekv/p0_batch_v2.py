"""Sequential, manifest-bounded native P0 batch. No implicit retry or rental."""
from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import re
import signal
import threading
import time

from .native_p0_operation_v2 import P0ComparisonActionV2
from .native_p0_request_v2 import execute_p0_request, P0RequestFailure
from .p0_evidence_v2 import P0EvidenceWriter
from .p0_exact_control_v2 import validate_exact_control_job, execute_exact_capture_control
from .p0_exact_pair_v2 import validate_exact_pairs, evaluate_exact_pairs
from .p0_mixed_control_v2 import validate_mixed_reference_job, execute_explicit_mixed_reference
from .p0_mixed_pair_v2 import validate_mixed_pairs, evaluate_mixed_pairs
from .source_comparison_v2 import ComparisonProfileBindingV2, runtime_binding_digest, scorer_digest
from .source_manifest_v2 import request_input_digest
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


class _ActionDeadlineExceeded(BaseException):
    """Cancellation, not an optional candidate publication failure.

    Native request drivers catch BaseException to fence/clean and retain audit;
    per-candidate 'except Exception' must not swallow this batch stop.
    """


class _ActionWallDeadline:
    """Cover Python publication as well as model execution; disarm on failure.

    Native P0 uses a dedicated Linux main process. Its launcher must retain
    the outer absolute-deadline watchdog for blocked C/CUDA calls. CPU fixtures
    perform the elapsed check only and cannot provide GPU deadline evidence.
    """
    def __init__(self, seconds, *, native):
        self.seconds = seconds
        self.native = native
        self.started = None
        self.armed = False

    def start(self):
        if not _positive(self.seconds):
            raise ValueError('finite action wall deadline required')
        if self.native:
            if (not hasattr(signal, 'setitimer') or
                    threading.current_thread() is not threading.main_thread()):
                raise RuntimeError('native P0 requires main-thread POSIX action watchdog')
            if any(signal.getitimer(signal.ITIMER_REAL)):
                raise RuntimeError('refuse to replace another active process timer')
            self.previous = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, self._expired)
            self.armed = True
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        self.started = time.perf_counter()
        return self

    def _expired(self, signum, frame):
        self.close()
        raise _ActionDeadlineExceeded('P0 action wall deadline exceeded (including publication)')

    def check(self):
        if self.started is not None and time.perf_counter()-self.started >= self.seconds:
            self.close()
            raise _ActionDeadlineExceeded('P0 action wall deadline exceeded (including publication)')

    def close(self):
        if self.armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.previous)
            self.armed = False


def _authority_duration_seconds(authority):
    """Explicit time-only approval never fabricates a tariff or zero bill.

    Historical manifests remain metered by default. The additional mode needs
    explicit operator-managed billing, null monetary fields and a finite time
    cap; it cannot be inferred from missing metered fields.
    """
    if not _positive(authority.get('maximum_gpu_hours')):
        raise ValueError('explicit finite positive GPU time cap required')
    mode = authority.get('billing_mode', 'metered_caps')
    seconds = authority['maximum_gpu_hours'] * 3600
    if mode == 'operator_managed_time_cap':
        if (authority.get('operator_managed_billing') is not True
                or 'unit_price_per_hour' not in authority or authority['unit_price_per_hour'] is not None
                or 'maximum_cost' not in authority or authority['maximum_cost'] is not None):
            raise ValueError('explicit operator billing and null unknown monetary values required')
    elif mode == 'metered_caps':
        if any(not _positive(authority.get(k)) for k in ('unit_price_per_hour', 'maximum_cost')):
            raise ValueError('current positive price and cost cap required')
        seconds = min(seconds, authority['maximum_cost']/authority['unit_price_per_hour']*3600)
    else:
        raise ValueError('unknown P0 billing authority mode')
    if not math.isfinite(seconds):
        raise ValueError('unbounded P0 authorization duration')
    return seconds


def validate_p0_batch(manifest, *, actual_binding, now_unix):
    """Pure preflight; a reference string records authorization, not payment."""
    unsigned = {k:v for k,v in manifest.items() if k != 'manifest_sha256'}
    if digest_json(unsigned) != manifest.get('manifest_sha256'):
        raise ValueError('P0 manifest digest mismatch')
    if (manifest.get('kind') != 'bounded_native_p0_batch_v2' or manifest.get('phase') != 'P0'
            or manifest.get('locked_test_accessed') is not False
            or manifest.get('automatic_rental_allowed') is not False):
        raise ValueError('explicit P0-only, no locked test/no rental contract required')
    if not re.fullmatch('[0-9a-f]{64}',str(manifest.get('native_manifest_sha256',''))):
        raise ValueError('bound native environment manifest required')
    required = {'code_commit','runtime_digest','patch_sha256','model_signature','tokenizer_hash',
                'gpu_uuid','instance_id','input_manifest_sha256','initial_pool_sha256','initial_registry_sha256'}
    binding = manifest['binding']
    if (set(binding) != required or any(type(v) is not str or not v for v in binding.values())
            or binding != actual_binding):
        raise ValueError('code/model/GPU/input/pool binding mismatch')
    if binding['runtime_digest'] != runtime_binding_digest():
        raise ValueError('stale P0 runtime digest')
    authority = manifest['authority']
    if (not isinstance(authority.get('approval_reference'), str) or not authority['approval_reference'].strip()
            or authority.get('instance_id') != binding['instance_id']
            or authority.get('gpu_uuid') != binding['gpu_uuid']
            or any(not _positive(authority.get(k)) for k in
                   ('maximum_gpu_hours','starts_at_unix','expires_at_unix'))
            or not authority['starts_at_unix'] <= now_unix < authority['expires_at_unix']):
        raise ValueError('current scoped GPU authorization/window/price/time/cost cap missing')
    authorized_seconds = _authority_duration_seconds(authority)
    limits = manifest['limits']
    registry = manifest['registry_budget']
    if (set(registry) != {'max_bytes','max_manifest_bytes'}
            or any(type(registry.get(k)) is not int or registry[k] <= 0 for k in ('max_bytes','max_manifest_bytes'))
            or registry['max_manifest_bytes'] > registry['max_bytes']):
        raise ValueError('explicit bounded shared manifest registry required')
    if any(not _positive(limits.get(k)) for k in ('initialization_upper_seconds','cleanup_seconds')):
        raise ValueError('initialization and cleanup reserves required')
    for k in ('host_capture_bytes','host_comparison_bytes','cuda_comparison_bytes'):
        if type(limits.get(k)) is not int or limits[k] <= 0:
            raise ValueError('explicit integer resource limits required')
    jobs = manifest['jobs']
    if (not isinstance(jobs, list) or not jobs or type(limits.get('maximum_actions')) is not int
            or len(jobs) != limits['maximum_actions']):
        raise ValueError('exact finite action count required')
    seen, requests = {}, set()
    for job in jobs:
        action_id = job['action_id']; q = job['request']
        if (not isinstance(action_id,str) or not re.fullmatch('[A-Za-z0-9_-]{1,80}', action_id)
                or action_id in seen or type(q.get('request_id')) is not str or not q['request_id']
                or q['request_id'] in requests or not _positive(job.get('upper_seconds'))):
            raise ValueError('unique bound action/request IDs and upper duration required')
        if job.get('input_origin') not in ('controlled_provenance_diagnostic','frozen_development'):
            raise ValueError('explicit non-locked input origin required')
        if (job.get('request_sha256') != digest_json(q) or not q.get('token_ids')
                or any(type(t) is not int or t<0 for t in q['token_ids'])
                or q.get('capture_original_full_prefill') or q.get('publish_exact_prefix_shadow')
                or q.get('use_gpu_hot_cache') or q.get('retain_gpu_hot_cache')
                or q.get('prefetch_window',0) != 0 or q.get('correctness_repair_ratio',.15) != .15):
            raise ValueError('P0 request identity or fixed15/target-only execution contract differs')
        if type(q.get('max_new_tokens')) is not int or q['max_new_tokens'] < 1:
            raise ValueError('explicit bounded generation required')
        operation = job.get('operation', 'source_request')
        if operation == 'exact_capture_control':
            validate_exact_control_job(job)
            # Diagnostic capture never supplies the pool, including via a later
            # birth_action reference to the private capture candidate.
            seen[action_id] = set(); requests.add(q['request_id'])
            continue
        if operation in ('explicit_mixed_reference', 'mixed_sparse_control'):
            validate_mixed_reference_job(job, _operation=operation)
            if any(type(limits.get(k)) is not int or limits[k] <= 0 for k in
                   ('mixed_reference_host_bytes', 'mixed_reference_cuda_bytes')):
                raise ValueError('explicit mixed reference host/CUDA resource bounds required')
            # This extra reference forward is diagnostic, never a natural
            # Source birth even when every target row is recomputed.
            seen[action_id] = set(); requests.add(q['request_id'])
            continue
        if operation == 'controlled_lineage_birth':
            from .p0_lineage_control_v2 import validate_lineage_job
            validate_lineage_job(job)
        elif operation != 'source_request':
            raise ValueError('unsupported P0 operation; no implicit diagnostic dispatch')
        segments = {s['segment_id'] for s in q['segments']}
        sources = job['sources_by_segment']
        if not sources or not set(sources) <= segments:
            raise ValueError('actual comparison/capture targets required')
        for refs in sources.values():
            if not isinstance(refs,list) or len(refs)>4 or len({digest_json(r) for r in refs}) != len(refs):
                raise ValueError('0..4 distinct frozen Source references required')
            for ref in refs:
                if set(ref) == {'source_id'}:
                    if type(ref['source_id']) is not str or not ref['source_id']:
                        raise ValueError('explicit historical Source ID required')
                elif set(ref) == {'birth_action_id','segment_id'}:
                    if ref['birth_action_id'] not in seen or ref['segment_id'] not in seen[ref['birth_action_id']]:
                        raise ValueError('future/unplanned Source publication reference')
                else:
                    raise ValueError('only existing Source or earlier publication references allowed')
        p = ComparisonProfileBindingV2(**job['comparison_profile'])
        if (p.model_signature != binding['model_signature'] or p.runtime_digest != binding['runtime_digest']
                or p.scoring_function_digest != scorer_digest()):
            raise ValueError('comparison binding differs from actual P0 runtime')
        if operation == 'controlled_lineage_birth':
            from .p0_lineage_control_v2 import lineage_targets
            targets, generations = lineage_targets(job)
            seen[action_id] = {t for t in targets if generations[t] <= 1}
        else:
            seen[action_id] = set(sources)
        requests.add(q['request_id'])
    validate_exact_pairs(manifest)
    validate_mixed_pairs(manifest)
    inputs = [dict(action_id=j['action_id'], request_sha256=j['request_sha256'], input_origin=j['input_origin']) for j in jobs]
    if digest_json(inputs) != binding['input_manifest_sha256']:
        raise ValueError('input manifest is not bound to exact request sequence')
    available = min(authorized_seconds,
                    authority['expires_at_unix']-now_unix)
    needed = limits['initialization_upper_seconds'] + sum(j['upper_seconds'] for j in jobs) + limits['cleanup_seconds']
    if needed > available:
        raise ValueError('bounded action plan plus initialization/cleanup exceeds current authority')
    return dict(status='INPUTS_VALIDATED', available_seconds=available, planned_upper_seconds=needed,
                native_numerics='NOT_RUN', P1_execution_allowed=False)


def _native_origin(adapter):
    import torch
    from .v8_schema10_native_adapter import NativeOnlineAdapter
    if not isinstance(adapter, NativeOnlineAdapter) or adapter.torch is not torch or not torch.cuda.is_available():
        raise ValueError('actual native CUDA adapter required')
    return 'real_cuda_execution'


def run_p0_native_batch(adapter, store, manifest, *, actual_binding, output, session_started_ns):
    """Caller validated environment before loading model; time includes that load.

    Output must be new. There is deliberately no automatic resume: emitted
    successful results do not prove a restorable native/physical pool state.
    """
    validate_p0_batch(manifest, actual_binding=actual_binding, now_unix=time.time())
    if (type(session_started_ns) is not int or not 0 < session_started_ns <= time.perf_counter_ns()
            or file_digest(store.root/'catalog.json') != actual_binding['initial_pool_sha256']
            or file_digest(store.registry._root/'registry.json') != actual_binding['initial_registry_sha256']
            or store.config['model_signature'] != actual_binding['model_signature']
            or store.config['tokenizer_hash'] != actual_binding['tokenizer_hash']):
        raise ValueError('initial pool or session/model identity differs')
    if any(j['comparison_profile']['provenance_policy'] != store.config['policy']
           for j in manifest['jobs'] if j.get('operation', 'source_request') in
           ('source_request', 'controlled_lineage_birth')):
        raise ValueError('batch provenance policy differs from existing isolated pool')
    origin = _native_origin(adapter)
    if adapter.active is not None or adapter.hbm.active_reserved_bytes:
        raise ValueError('P0 batch needs a quiescent adapter/HBM allocator')
    if (adapter.provenance['model_signature'] != actual_binding['model_signature']
            or adapter.provenance['tokenizer_hash'] != actual_binding['tokenizer_hash']):
        raise ValueError('actual native adapter model/tokenizer differs from store/manifest')
    writer = P0EvidenceWriter(output, binding=actual_binding, manifest=manifest)
    authority, limits = manifest['authority'], manifest['limits']
    old_deadline = adapter.deadline
    completed, failed, published = [], [], {}
    stop = 'COMPLETED'
    try:
        for job in manifest['jobs']:
            elapsed = (time.perf_counter_ns()-session_started_ns)/1e9
            remaining = min(_authority_duration_seconds(authority)-elapsed,
                authority['expires_at_unix']-time.time())
            if remaining < job['upper_seconds']+limits['cleanup_seconds']:
                stop = 'BUDGET_STOP'; break
            action_id = job['action_id']; snapshot = context = None
            partial = None
            action_timer = _ActionWallDeadline(job['upper_seconds'], native=origin=='real_cuda_execution')
            writer.append('action_started',action_id, dict(request_sha256=job['request_sha256'],
                elapsed_seconds=elapsed, remaining_authorized_seconds=remaining))
            try:
                action_timer.start()
                # Prefix reset is explicit diagnostic preparation, included in
                # action/total cost; never nonce-edit input or copy CUDA pointers.
                adapter.deadline = min(old_deadline, time.perf_counter()+job['upper_seconds'])
                adapter.reset()
                control = job.get('operation') == 'exact_capture_control'
                mixed = job.get('operation') in ('explicit_mixed_reference', 'mixed_sparse_control')
                if not control:
                    snapshot = store.begin_request(job['request']['request_id'])
                if not control and not mixed:
                    resolved = {}
                    for sid, refs in job['sources_by_segment'].items():
                        resolved[sid] = tuple(r['source_id'] if 'source_id' in r else
                            published[(r['birth_action_id'],r['segment_id'])] for r in refs)
                    profile = ComparisonProfileBindingV2(**job['comparison_profile'])
                started = time.perf_counter_ns()
                with adapter.open_request(job['request'], arrival_ns=started) as context:
                    try:
                        if control:
                            report = execute_exact_capture_control(context, job,
                                authorization_domain=store.config['authorization_domain'],
                                host_capture_bytes=limits['host_capture_bytes'],
                                registry_budget=manifest['registry_budget'])
                        elif mixed:
                            if job['operation'] == 'mixed_sparse_control':
                                from .p0_mixed_sparse_v2 import execute_mixed_sparse_control
                                operation = execute_mixed_sparse_control
                            else:
                                operation = execute_explicit_mixed_reference
                            report = operation(context, store, snapshot, job,
                                host_reference_bytes=limits['mixed_reference_host_bytes'],
                                cuda_reference_bytes=limits['mixed_reference_cuda_bytes'])
                        else:
                            actions = tuple(P0ComparisonActionV2(job['request']['request_id'],sid,snapshot.snapshot_id,
                                request_input_digest(job['request']['token_ids'],tuple(range(len(job['request']['token_ids'])))),
                                profile.completed_depth, ids, profile.binding_digest,
                                limits['host_comparison_bytes'],limits['cuda_comparison_bytes']) for sid,ids in resolved.items())
                            if job.get('operation') == 'controlled_lineage_birth':
                                from .p0_lineage_control_v2 import execute_lineage_birth
                                report = execute_lineage_birth(context,store,snapshot,profile,actions,job,
                                    host_capture_bytes=limits['host_capture_bytes'])
                            else:
                                report = execute_p0_request(context,store,snapshot,profile,actions,
                                                            host_capture_bytes=limits['host_capture_bytes'])
                    except BaseException as exc:
                        action_timer.close()
                        # Native context-manager cleanup may itself raise. Keep
                        # the original driver audit even if it replaces exc.
                        partial = exc.audit if isinstance(exc,P0RequestFailure) else getattr(exc,'p0_audit',None)
                        raise
                    logits = getattr(context,'logit_trace',())
                    # Persistence/consumer checks can fail after a successful
                    # forward. Preserve that execution audit in the failure
                    # event even if no complete record file can be written.
                    partial = report
                if mixed:
                    # The driver fences and closes physical resources; only
                    # then may the immutable visibility snapshot be released.
                    if not context.closed:
                        raise RuntimeError('mixed reference context did not close after device fence')
                    store.end_request(snapshot)
                target_layers = getattr(context, 'p0_reference_target_layers_v2', None)
                try:
                    if mixed:
                        target = next(s for s in job['request']['segments'] if s['segment_id'] == job['target_id'])
                        attn = adapter.inner.layers[0].self_attn
                        shape = (len(target['positions']), attn.num_kv_heads, attn.head_dim)
                        if (not isinstance(target_layers, (tuple, list))
                                or len(target_layers) != adapter.spec.num_layers
                                or any(not isinstance(pair, (tuple, list)) or len(pair) != 2
                                       or any(tuple(t.shape) != shape for t in pair)
                                       for pair in target_layers)):
                            raise ValueError('completed mixed reference requires all actual target layers and model geometry')
                    record = writer.write_action(action_id,audit=report,logits=logits,origin=origin,
                        **({'target_layers': target_layers} if mixed else {}))
                finally:
                    if mixed:
                        context.p0_reference_target_layers_v2 = ()
                        # The hook audit keeps no device ownership here; release
                        # its redundant host tuple after evidence persistence.
                        owner = getattr(context, 'p0_diagnostic_resources_v2', None)
                        if owner is not None and owner.closed:
                            owner.hooks = None
                        target_layers = None
                action_timer.check()
                action_timer.close()
                if report['status'] != 'COMPLETED':
                    failed.append(action_id); stop = 'RECOVERY_REQUIRED'; break
                for sid,row in (report.get('publication') or {}).get('targets',{}).items():
                    if row.get('status') == 'PUBLISHED':
                        published[(action_id,sid)] = row['source_id']
                completed.append(action_id)
                writer.append('action_completed',action_id,dict(numerical_verdict='NOT_EVALUATED',
                    evidence_origin=record['evidence_origin']))
            except BaseException as exc:
                action_timer.close()
                snapshot_cleanup_error = None
                if snapshot is not None and snapshot.snapshot_id in store._snapshots:
                    # Never unpin a snapshot after an unsuccessful device fence.
                    if context is None or getattr(context,'closed',False):
                        try: store.end_request(snapshot)
                        except Exception as cleanup_exc:
                            snapshot_cleanup_error = type(cleanup_exc).__name__+':'+str(cleanup_exc)
                partial = partial or (exc.audit if isinstance(exc,P0RequestFailure) else getattr(exc,'p0_audit',None))
                writer.append('action_failed',action_id,dict(error_type=type(exc).__name__,detail=str(exc),
                    partial_request_audit=partial, resume_allowed=False,
                    snapshot_cleanup_error=snapshot_cleanup_error,
                    snapshot_retained=bool(snapshot is not None and snapshot.snapshot_id in store._snapshots)))
                failed.append(action_id); stop = 'FAILED'
                if isinstance(exc,(KeyboardInterrupt,SystemExit)):
                    raise
                break
            finally:
                action_timer.close()
    finally:
        adapter.deadline = old_deadline
        pairs = evaluate_exact_pairs(output, manifest, completed)
        for pair in pairs:
            writer.append('exact_capture_pair_evaluated', None, pair)
        mixed_pairs = evaluate_mixed_pairs(output, manifest, completed)
        for pair in mixed_pairs:
            writer.append('mixed_reference_pair_evaluated', None, pair)
        if stop == 'COMPLETED' and any(p['status'] == 'FAILED' for p in pairs + mixed_pairs):
            stop = 'NUMERICAL_PAIR_FAILED'
        elapsed = (time.perf_counter_ns()-session_started_ns)/1e9
        attempted = set(completed+failed)
        summary = writer.finalize(dict(status=stop, completed_action_ids=completed, failed_action_ids=failed,
            pending_action_ids=[j['action_id'] for j in manifest['jobs'] if j['action_id'] not in attempted],
            elapsed_seconds=elapsed,
            estimated_usage_cost=(None if authority.get('billing_mode') == 'operator_managed_time_cap'
                                  else elapsed/3600*authority['unit_price_per_hour']),
            billing_mode=authority.get('billing_mode', 'metered_caps'),
            cost_semantics=('unknown tariff; operator manages instance billing; time cap still enforced'
                            if authority.get('billing_mode') == 'operator_managed_time_cap'
                            else 'elapsed usage estimate, not provider invoice'),
            evidence_origin=origin, numerical_verdict='NOT_EVALUATED',
            exact_capture_pairs=pairs, mixed_reference_pairs=mixed_pairs, automatic_resume_allowed=False))
    return summary

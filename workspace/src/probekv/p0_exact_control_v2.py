"""T20 exact capture on/off on the real resumable engine, never pool supply.

The caller owns native block allocation and authorization. These diagnostic
actions have no lookup, selector, lease, admission bypass or publication. Both
arms execute the same full-row layer path. GPU numerics remain a later test.
"""
from __future__ import annotations

from dataclasses import asdict
import time

from .native_p0_request_v2 import P0RequestFailure
from .source_manifest_v2 import RequestManifestRegistry, request_input_digest
from .v8_schema10_execution import digest_json


def validate_exact_control_job(job):
    q = job['request']
    targets = job.get('target_ids')
    if (job.get('operation') != 'exact_capture_control'
            or type(job.get('capture_enabled')) is not bool
            or not isinstance(targets, list) or not targets
            or any(type(s) is not str or not s for s in targets)
            or len(set(targets)) != len(targets)
            or not set(targets) <= {s['segment_id'] for s in q['segments']}
            or 'sources_by_segment' in job or 'comparison_profile' in job):
        raise ValueError('exact control requires explicit capture toggle/targets, no Source decisions')
    if (q.get('capture_logits') is not True
            or type(q.get('max_new_tokens')) is not int or q['max_new_tokens'] < 1
            or not isinstance(q.get('teacher_token_ids'), list)
            or len(q['teacher_token_ids']) != q['max_new_tokens']-1
            or any(type(t) is not int or t < 0 for t in q['teacher_token_ids'])
            or q.get('native_dense_continuation', False)
            or q.get('capture_original_full_prefill') or q.get('publish_exact_prefix_shadow')
            or q.get('use_gpu_hot_cache') or q.get('retain_gpu_hot_cache')
            or q.get('prefetch_window', 0) != 0
            or q.get('correctness_repair_ratio', .15) != .15):
        raise ValueError('matched teacher/full-row engine path required; r1/mixed are separate diagnostics')


def _full_row_recipe(context, target_ids):
    session = context.engine.session
    tokens = tuple(context.request['token_ids']); rows = tuple(range(len(tokens)))
    layers = [row for row in session.layer_audit if 'layer' in row]
    total = context.adapter.spec.num_layers
    if (session.current_layer != total or tuple(session.token_ids) != tokens
            or session.model_signature != context.adapter.provenance['model_signature']
            or tuple(session.absolute_positions) != rows or session.exact_prefix_tokens != 0
            or session.commits or context.committed or context.prepared or context.frozen
            or context.engine.tickets or session.source_handles
            or len(layers) != total or [r['layer'] for r in layers] != list(range(1,total+1))
            or any(tuple(r['active_before']) != rows or tuple(r['active_after']) != rows for r in layers)):
        raise ValueError('exact control did not execute every full-row block exactly once without Source injection')
    recipe = dict(kind='exact_full_row_resumable_control',
        input_digest=request_input_digest(tokens, rows),
        model_signature=context.adapter.provenance['model_signature'], num_layers=total,
        prefix_tokens=0, source_imports=[],
        target_positions={sid:list(context.segments[sid]['positions']) for sid in target_ids},
        layers=[dict(layer=r['layer'], active_before=list(r['active_before']),
                     active_after=list(r['active_after']), union_mask_digest=r['union_mask_digest']) for r in layers])
    return dict(recipe=recipe, recipe_sha256=digest_json(recipe))


def execute_exact_capture_control(context, job, *, authorization_domain,
                                  host_capture_bytes, registry_budget):
    """One actual forward; capture candidates stay private and are discarded."""
    from .v8_schema10_native_adapter import NativeRequestContext
    validate_exact_control_job(job)
    if (type(context) is not NativeRequestContext or context.request != job['request']
            or context.current_completed_depth != 0 or context.finished or context.closed
            or context.cached_prefix_tokens or context.probe_fallback_reason
            or context.prepared or context.committed or context.frozen
            or context.source_capture_v2 is not None or context.source_consumption_v2 is not None
            or context.capture_reservation is not None
            or type(context.arrival_ns) is not int or not 0 < context.arrival_ns <= time.perf_counter_ns()
            or context.repair_ratio != .15
            or not authorization_domain or type(host_capture_bytes) is not int or host_capture_bytes <= 0):
        raise ValueError('fresh native d0/no-Prefix/no-Source context and explicit capture budget required')
    if (set(registry_budget) != {'max_bytes', 'max_manifest_bytes'}
            or any(type(v) is not int or v <= 0 for v in registry_budget.values())
            or registry_budget['max_manifest_bytes'] > registry_budget['max_bytes']):
        raise ValueError('bounded private in-memory registry required; no persistent destination')
    # An in-memory private registry, NOT store.registry: diagnostic capture
    # cannot supply a later batch action or mutate the natural pool's history.
    registry = RequestManifestRegistry(**registry_budget)
    audit = dict(kind='p0_exact_capture_control', status='RUNNING', request_id=context.request['request_id'],
        capture_enabled=job['capture_enabled'], answer=None, execution=None, capture=None, cleanup=None,
        publication=dict(status='DIAGNOSTIC_NOT_PUBLISHED'), extra_forward_count=0,
        native_runtime_qualified=False, P1_execution_allowed=False, paper_evidence=False)
    error = None
    try:
        context.adapter.check_deadline()
        if job['capture_enabled']:
            audit['capture_configuration'] = context.configure_target_capture_v2(
                target_ids=tuple(job['target_ids']), registry=registry,
                authorization_domain=authorization_domain, host_budget_bytes=host_capture_bytes)
        # Also begin the resumable engine for capture-off: comparing a native
        # monolithic dense arm against a hooked resumable arm confounds T20.
        context._begin()
        first = []
        audit['answer'] = context.finish(lambda:first.append(time.perf_counter_ns()))
        end = time.perf_counter_ns()
        if not context.finished or len(first) != 1 or not context.arrival_ns <= first[0] <= end:
            raise RuntimeError('one completed native answer and actual first-token endpoint required')
        audit['execution'] = _full_row_recipe(context, job['target_ids'])
        audit['timing'] = dict(arrival_ns=context.arrival_ns, first_token_ns=first[0],
            answer_complete_ns=end, ttft_ms=(first[0]-context.arrival_ns)/1e6,
            answer_service_ms=(end-context.arrival_ns)/1e6)
        if job['capture_enabled']:
            capture = context.source_capture_v2
            candidates = context.target_candidates_v2
            audit['capture'] = capture.audit()
            if (not capture._finalized or capture._failed or set(candidates) != set(job['target_ids'])
                    or capture.registry is not registry):
                raise ValueError('capture-on diagnostic did not obtain every target; no replacement forward')
            audit['captured_targets'] = {}
            for sid,candidate in candidates.items():
                proof = candidate['proof']
                storage = candidate['storage_audit']
                if (candidate['origin'] != 'EXACT_CONTEXT' or candidate['generation'] != 0
                        or proof.generation != 0 or len(candidate['layers']) != context.adapter.spec.num_layers
                        or storage['parent_owned_kv_bytes'] != 0 or storage['prefix_shadow_bytes'] != 0):
                    raise ValueError('capture-on target not complete exact/independently owned')
                audit['captured_targets'][sid] = dict(proof=asdict(proof),
                    artifact_digest=candidate['artifact_digest'], storage_audit=storage,
                    publication_state=candidate['publication_state'])
        elif context.target_candidates_v2 or context.source_capture_v2 is not None:
            raise ValueError('capture-off arm unexpectedly captured targets')
        audit['status'] = 'COMPLETED'
    except BaseException as exc:
        error = exc
        audit.update(status='FAILED', failure=dict(type=type(exc).__name__,detail=str(exc)))
    finally:
        try:
            context.close()
            # Unlike a production publication owner, a diagnostic caller must
            # not retain candidate tensor storage after the fenced action.
            context.target_candidates_v2.clear()
            audit['cleanup'] = dict(passed=True, failures=[])
        except BaseException as exc:
            audit['cleanup'] = dict(passed=False, failures=[dict(type=type(exc).__name__,detail=str(exc))])
            error = error or exc
        audit['service_end_ns'] = time.perf_counter_ns()
        audit['total_service_ms'] = (audit['service_end_ns']-context.arrival_ns)/1e6
    if error is not None:
        audit['status'] = 'FAILED'
        if isinstance(error,(KeyboardInterrupt,SystemExit)):
            error.p0_audit = audit
            raise error
        raise P0RequestFailure(audit,error) from error
    return audit

"""P1 isolated Source-build dispatch over the qualified P0 native primitives.

This is not a launcher or a GPU authorization. The caller owns preflight,
deadline enforcement, evidence persistence and the native request manager.
Only Source births are executable here; QA arms remain separate operations.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import hashlib

from .p1_input_consumer_v2 import resolve_input_graph
from .source_comparison_v2 import ComparisonProfileBindingV2, runtime_binding_digest, scorer_digest
from .source_manifest_v2 import request_input_digest
from .v8_schema10_execution import digest_json


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def dispatch_binding():
    # New execution code is bound separately, NOT claimed to have the old P0
    # runtime's numerical qualification just because its dependencies match.
    return dict(base_runtime_digest=runtime_binding_digest(),
                build_dispatch_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def _seal(value, field):
    value[field] = digest_json(value)
    return value


def _check_seal(value, field, expected=None):
    actual = digest_json({k:v for k,v in value.items() if k != field})
    _require(value.get(field) == actual and (expected is None or actual == expected),
             'frozen '+field+' differs')


def _birth_request(build):
    q = deepcopy(build['request'])
    for segment in q['segments']:
        segment.pop('content_key', None)
    q.update(request_id=q['request_id']+':P1build:'+digest_json(build['build_id'])[:16],
             max_new_tokens=1, prefetch_window=0)
    from .p0_launch_recipe_v2 import _request
    _request(q)
    return q


def bind_request_content_keys(request, store):
    """Only rebind exact content identities to the actual isolated namespace.

    The compiled request remains immutable. No token, position, question,
    sampling setting or Source choice is changed at this execution boundary.
    """
    q=deepcopy(request)
    for segment in q['segments']:
        positions=segment['positions'];tokens=segment['token_ids']
        _require([q['token_ids'][p] for p in positions]==tokens,'content-key binding token/position mismatch')
        key=store.content_key(tokens)
        _require(segment.get('content_key',key)==key,'request contains foreign store content key')
        segment['content_key']=key
    return q


def compile_p1_build(documents, *, identity, build_id, comparison_profile, resources):
    """Re-resolve frozen inputs; compile exactly one registered birth.

    Generation stops after one token: no teacher QA, extra full forward or
    publication of reference logits. Token IDs and target positions are exact.
    Every birth uses an isolated store; the M1 store may contain only its parent.
    """
    graph = resolve_input_graph(documents, **identity)
    rows = [b for b in graph['source_builds'] if b['build_id'] == build_id]
    _require(len(rows) == 1, 'one frozen build ID required')
    b = rows[0]
    p = ComparisonProfileBindingV2(**comparison_profile)
    _require(p.model_signature == graph['model_signature'] and p.completed_depth == 2
             and p.provenance_policy == 'ALLOW_MIXED_G1'
             and p.runtime_digest == runtime_binding_digest()
             and p.scoring_function_digest == scorer_digest(), 'current d2 model/runtime/profile required')
    _require(set(resources) == {'host_capture_bytes','host_comparison_bytes','cuda_comparison_bytes'}
             and all(type(x) is int and x > 0 for x in resources.values()), 'explicit resource bounds required')
    q = deepcopy(b['request'])
    _require(not any(q.get(k) for k in ('teacher_token_ids','capture_logits','capture_original_full_prefill',
        'publish_exact_prefix_shadow','use_gpu_hot_cache','retain_gpu_hot_cache','native_dense_continuation'))
        and q.get('correctness_repair_ratio', .15) == .15, 'incompatible birth execution override')
    target = 'U' if b['role'] == 'G0_parent' else 'C'
    original_keys = {s['segment_id']:s.get('content_key') for s in q['segments']}
    q = _birth_request(b)
    _require(any(s['segment_id'] == target for s in q['segments']), 'birth target missing')
    parent = next((r for r in graph['source_builds'] if r['build_id'] in b['depends_on']), None)
    result = dict(kind='P1_source_build_operation_v2', build_id=build_id, stage=b['stage'],
        group_id=b['group_id'], role=b['role'], target_id=target,
        expected_generation=1 if b['role']=='mixed_M1' else 0,
        input_graph_sha256=graph['graph_sha256'], original_request_sha256=b['request_sha256'],
        request=q, request_sha256=digest_json(q), comparison_profile=asdict(p), resources=deepcopy(resources),
        dispatch_binding=dispatch_binding(), maximum_seconds=b['maximum_seconds'],
        original_content_keys=original_keys, original_max_new_tokens=b['request'].get('max_new_tokens'),
        original_prefetch_window=b['request'].get('prefetch_window',0),
        parent_build_id=None if parent is None else parent['build_id'],
        parent_original_request_sha256=None if parent is None else parent['request_sha256'],
        parent_birth_request=None if parent is None else _birth_request(parent),
        isolated_store_required=True, prompt_tokens_modified=False, extra_forward_allowed=False,
        authority_granted=False, GPU_execution_allowed=False, QA_evaluated=False, paper_evidence=False)
    return _seal(result, 'operation_sha256')


def validate_build_context(context, store, snapshot, operation, *, expected_operation_sha256, parent_receipt=None):
    """No layer execution or snapshot creation on validation failure."""
    from .native_p0_operation_v2 import P0ComparisonActionV2, validate_p0_comparison_action
    _check_seal(operation, 'operation_sha256', expected_operation_sha256)
    o = operation; q = bind_request_content_keys(o['request'],store); target = o['target_id']
    _require(o['kind']=='P1_source_build_operation_v2' and o['dispatch_binding']==dispatch_binding()
             and o['authority_granted'] is False and o['GPU_execution_allowed'] is False
             and o['isolated_store_required'] is True and o['extra_forward_allowed'] is False,
             'current isolated build operation required, not execution authority')
    _require(context.request == q and digest_json(o['request'])==o['request_sha256']
             and context.current_completed_depth==0 and q['max_new_tokens']==1
             and context.adapter.costs is None, 'fresh matched birth and no online economics required')
    p = ComparisonProfileBindingV2(**o['comparison_profile'])
    sources = {target: ()}
    job = dict(operation='source_request', request=q, request_sha256=o['request_sha256'],
               comparison_profile=asdict(p), sources_by_segment={target: []})
    with store.lock:
        store._snapshot(snapshot)
        _require(len(store._snapshots)==1 and not store._plans and not any(store._lease_counts.values()),
                 'single quiescent birth snapshot required')
        if o['role']=='mixed_M1':
            _require(parent_receipt is not None, 'actual parent build receipt required')
            _check_seal(parent_receipt, 'receipt_sha256')
            r = parent_receipt
            _require(r['build_id']==o['parent_build_id'] and r['role']=='G0_parent'
                     and r['input_graph_sha256']==o['input_graph_sha256']
                     and r['original_request_sha256']==o['parent_original_request_sha256']
                     and r['dispatch_binding']==o['dispatch_binding'] and r['generation']==0
                     and r['cleanup_passed'] is True, 'parent receipt identity differs')
            sid = r['source_id']
            _require(set(store._catalog['rows'])=={sid}, 'mixed build pool must contain only its verified parent')
            row = store._visible_row(snapshot, sid)
            _require(row['artifact_digest']==r['artifact_digest'] and row['generation']==0
                     and row['origin']=='EXACT_CONTEXT' and row['manifest_reference']==r['manifest_reference']
                     and row['birth_request_id']==r['birth_request_id'], 'actual parent backing differs from receipt')
            # A self-hashed receipt alone cannot prove which prompt produced
            # the backing. Rebind actual stored manifest to frozen parent input.
            from .source_manifest_v2 import ManifestReference
            pq = o['parent_birth_request']
            manifest, _ = store._manifest_record(ManifestReference(**row['manifest_reference']),
                row['token_ids'],row['birth_target_positions'])
            pt = next(s for s in pq['segments'] if s['segment_id']=='U')
            _require(row['birth_request_id']==pq['request_id'] and row['token_ids']==pt['token_ids']
                     and row['birth_target_positions']==pt['positions']
                     and manifest['token_ids']==pq['token_ids']
                     and manifest['absolute_positions']==list(range(len(pq['token_ids'])))
                     and manifest['input_digest']==request_input_digest(pq['token_ids'],tuple(range(len(pq['token_ids'])))),
                     'parent full historical context differs from frozen birth')
            store._verify_row(row, full=True)
            sources = {'U': (sid,), target: ()}
            job.update(operation='controlled_lineage_birth', sources_by_segment={'U':[{'source_id':sid}],target:[]},
                upstream_segment_id='U', target_id=target, expected_parent_generation=0, expected_target_generation=1,
                input_origin='controlled_provenance_diagnostic', economic_admission_evaluated=False,
                production_reuse_commit_observed=False, parent_preparation_mode='preregistered_source_correctness_only')
        else:
            _require(parent_receipt is None and not store._catalog['rows']
                     and o['role'] in ('historical_exact','independent_S0','exact_E','G0_parent')
                     and o['expected_generation']==0, 'exact birth requires empty independent pool')
    actions = tuple(P0ComparisonActionV2(q['request_id'],sid,snapshot.snapshot_id,
        request_input_digest(q['token_ids'],tuple(range(len(q['token_ids'])))), p.completed_depth,
        ids,p.binding_digest,o['resources']['host_comparison_bytes'],o['resources']['cuda_comparison_bytes'])
        for sid,ids in sources.items())
    for action in actions:
        validate_p0_comparison_action(context, store, snapshot, p, action)
    if o['role']=='mixed_M1':
        from .p0_lineage_control_v2 import validate_lineage_job
        validate_lineage_job(job, total_layers=context.adapter.spec.num_layers)
    return p, actions, job


def execute_p1_source_build(context, store, snapshot, operation, *, expected_operation_sha256, parent_receipt=None):
    """One real native prefill, then checked publication; no callback PASS.

    Existing primitives own close/fence/end_snapshot once execution begins.
    Failure never schedules another forward; outer runner preserves the audit.
    This wrapper must itself pass new GPU qualification before P1 launch.
    """
    p, actions, job = validate_build_context(context,store,snapshot,operation,
        expected_operation_sha256=expected_operation_sha256,parent_receipt=parent_receipt)
    from .native_p0_request_v2 import execute_p0_request, P0RequestFailure
    from .p0_lineage_control_v2 import execute_lineage_birth
    limit = operation['resources']['host_capture_bytes']
    if operation['role']=='mixed_M1':
        audit = execute_lineage_birth(context,store,snapshot,p,actions,job,host_capture_bytes=limit)
    else:
        audit = execute_p0_request(context,store,snapshot,p,actions,host_capture_bytes=limit)
    try:
        _require(audit['status']=='COMPLETED' and audit['cleanup']['passed'] is True
                 and audit['extra_forward_count']==0, 'birth execution/cleanup incomplete')
        event = audit['publication']['targets'][operation['target_id']]
        _require(event['publication_performed'] is True, 'registered build did not publish; no automatic rerun')
        with store.lock:
            row = store._catalog['rows'][event['source_id']]
            target = next(s for s in operation['request']['segments'] if s['segment_id']==operation['target_id'])
            _require(row['birth_request_id']==operation['request']['request_id']
                     and row['token_ids']==target['token_ids'] and row['birth_target_positions']==target['positions']
                     and row['generation']==operation['expected_generation']
                     and row['layer_count']==context.adapter.spec.num_layers
                     and row['parent_owned_kv_bytes']==row['prefix_shadow_bytes']==0,
                     'actual published birth identity/geometry differs')
            store._verify_row(row, full=True)
            receipt = dict(kind='P1_source_build_receipt_v2', build_id=operation['build_id'],role=operation['role'],
                input_graph_sha256=operation['input_graph_sha256'],original_request_sha256=operation['original_request_sha256'],
                operation_sha256=operation['operation_sha256'],dispatch_binding=operation['dispatch_binding'],
                source_id=row['source_id'],artifact_digest=row['artifact_digest'],generation=row['generation'],
                origin=row['origin'],birth_request_id=row['birth_request_id'],manifest_reference=deepcopy(row['manifest_reference']),
                cleanup_passed=True,publication_audit_sha256=digest_json(audit),QA_evaluated=False,paper_evidence=False)
        return dict(audit=audit,receipt=_seal(receipt,'receipt_sha256'))
    except Exception as exc:
        raise P0RequestFailure(audit,exc) from exc

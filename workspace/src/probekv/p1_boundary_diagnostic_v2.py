"""Independent, bounded d2/layer3 diagnostic; legacy P1 remains layer9.

No production selector, economic admission, Source creation or threshold fit.
The existing runtime is reused without changing its audited file hashes.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import math
import json
import time

from .p1_build_dispatch_v2 import _require, _seal, _check_seal, bind_request_content_keys
from .p1_qa_dispatch_v2 import verify_qa_source
from .p1_action_runner_v2 import runner_binding
from .native_consumption_v2 import NativeV2ConsumptionSession
from .source_comparison_v2 import ComparisonProfileBindingV2, ComparisonSessionV2
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest


MODES = ('dense_teacher', 'source_teacher_r1', 'dense_qa', 'source_greedy_r1', 'source_qa')


def boundary_binding():
    return dict(runner_binding(), boundary_dispatch_sha256=file_digest(Path(__file__)))


def compile_boundary_operation(legacy, *, mode):
    """Caller must independently verify legacy membership in the frozen graph."""
    _check_seal(legacy, 'operation_sha256')
    _require(mode in MODES and legacy['kind'] == 'P1_fixed_source_QA_operation_v2'
             and legacy['stage'] == 'P1-E' and legacy['comparison_profile']['completed_depth'] == 8,
             'qualified exact legacy P1-E action required')
    dense = mode.startswith('dense')
    _require((legacy['arm'] == 'dense') == dense, 'mode must retain actual frozen Source/dense arm')
    if mode in ('source_teacher_r1', 'source_greedy_r1'):
        _require(legacy['arm'] == 'historical_1', 'r1 sentinel uses prescribed first historical Source')
    q = deepcopy(legacy['request'])
    _require(q['max_new_tokens'] == 32 and len(q['token_ids']) >= 31
             and q.get('prefetch_window', 0) == 0 and q.get('partition_role') == 'fit'
             and not any(q.get(k) for k in ('teacher_token_ids', 'capture_logits', 'correctness_repair_ratio',
                 'native_dense_continuation', 'force_nonpaper_measurement_admission')),
             'unchanged exposed fit request, no hidden execution overrides')
    q['request_id'] += ':layer3:' + mode
    ratio = 1.0 if mode.endswith('r1') else .15
    if ratio == 1.0:
        q['correctness_repair_ratio'] = ratio
    if 'teacher' in mode:
        # Teacher diagnostics feed exactly 31 declared tokens; QA-only answer
        # stop markers cannot terminate or classify these numerical probes.
        q.pop('answer_boundary_contract', None)
        q.update(capture_logits=True, teacher_token_ids=list(q['token_ids'][-31:]))
    from .v8_schema10_native_adapter import validate_native_sampling_request
    validate_native_sampling_request(q)
    profile = dict(legacy['comparison_profile'], completed_depth=2)
    # This binding is diagnostic, not a requalified residual threshold.
    ComparisonProfileBindingV2(**profile)
    o = deepcopy(legacy)
    o.pop('operation_sha256')
    o.update(kind='layer3_boundary_diagnostic_operation_v2', stage='BOUNDARY_L3', mode=mode,
        request=q, request_sha256=digest_json(q), legacy_operation_sha256=legacy['operation_sha256'],
        first_reuse_layer=None if dense else 3, completed_depth=2,
        repair_ratio=None if dense else ratio, comparison_profile=profile,
        dispatch_binding=boundary_binding(), diagnostic_only=True,
        boundary_only_exploration=True, full_P4_execution_allowed=False)
    return _seal(o, 'operation_sha256')


def verify_boundary_operation(operation, legacy):
    _check_seal(operation, 'operation_sha256')
    _require(operation == compile_boundary_operation(legacy, mode=operation['mode']),
             'boundary operation changed beyond the registered recipe')


class BoundarySourceSession(NativeV2ConsumptionSession):
    def prepare_prescribed(self, receipt, operation, source_id):
        o = operation; c = self.context
        _check_seal(o, 'operation_sha256')
        _require(o['kind'] == 'layer3_boundary_diagnostic_operation_v2'
            and o['dispatch_binding'] == boundary_binding() and o['mode'].startswith('source')
            and c.request == bind_request_content_keys(o['request'], self.store)
            and c.current_completed_depth == 2 and o['comparison_profile'] == asdict(self.comparison.profile)
            and receipt.segment_id == 'C' and receipt.winner_source_id == source_id
            and tuple(receipt.eligible_ids) == tuple(receipt.available_ids) == tuple(receipt.compared_ids) == (source_id,)
            and o['publication_allowed'] is False and o['economic_admission_evaluated'] is False,
            'exact prescribed layer3 diagnostic and unique real observation required')
        return self._prepare_bound(receipt, controlled_recipe=dict(recipe_digest=o['operation_sha256'],
            source_id=source_id, residual_admission_evaluated=False, production_execution_allowed=False,
            experiment_kind='layer3_boundary_diagnostic'))


def commit_boundary_diagnostic(c, o):
    from .v8_schema6_hbm import HBMReservationKind
    _require(c.current_completed_depth == 2 and set(c.prepared) == {'C'} and not c.committed
             and o['first_reuse_layer'] == 3 and o['repair_ratio'] == c.repair_ratio
             and o['mode'].startswith('source') and isinstance(c.source_consumption_v2, BoundarySourceSession),
             'layer3 diagnostic commit cannot be used by production')
    c.source_consumption_v2.assert_can_commit('C', controlled_recipe=True)
    positions = tuple(c.segments['C']['positions']); support = tuple(c.supports['C'][3])
    _require(len(support) == math.ceil(len(positions)*o['repair_ratio'])
             and support == tuple(sorted(set(support))) and set(support) <= set(positions),
             'wrong absolute support/count')
    c.engine.commit_ready_segment(segment_id='C', boundary=3, segment_positions=positions,
        repair_positions=support, scheduler_boundary=3)
    c.adapter.hbm.promote(c.replica_reservations['C'].reservation_id,
        expected=HBMReservationKind.WINNER_PREFETCH, target=HBMReservationKind.COMMITTED_EXECUTION)
    c.committed['C'] = 3; c.generation += 1


def validate_layer_recipe(layers, *, prompt_count, target_positions, support, total_layers, source):
    """Full QKV projection at layer3 is not claimed to be skipped computation."""
    rows = list(range(prompt_count)); target = set(target_positions)
    reduced = sorted((set(rows)-target) | set(support)) if source else rows
    _require(len(layers) == total_layers, 'missing or duplicate model layers')
    for number, row in enumerate(layers, 1):
        before = rows if number <= 3 else reduced
        after = rows if number < 3 else reduced
        _require(row['layer'] == number and list(row['active_before']) == before
                 and list(row['active_after']) == after,
                 'layer order/active rows differ at layer %d' % number)
    return dict(layers_once=True, first_two_layers_full=True, consumer_layer_1based=3,
                block3_full_projection_accounted=True, non_target_rows_preserved=True)


def execute_boundary_diagnostic(context, store, snapshot, operation, *, source_receipt=None):
    from .native_p0_operation_v2 import P0ComparisonActionV2, validate_p0_comparison_action
    from .native_p0_request_v2 import P0RequestFailure
    from .source_manifest_v2 import request_input_digest
    o = operation; c = context
    _check_seal(o, 'operation_sha256')
    _require(o['dispatch_binding'] == boundary_binding() and o['kind'] == 'layer3_boundary_diagnostic_operation_v2'
        and c.request == bind_request_content_keys(o['request'], store) and c.current_completed_depth == 0
        and not c.cached_prefix_tokens and not c.prepared and not c.committed and not c.frozen
        and not c.finished and not c.closed and getattr(c, 'source_capture_v2', None) is None
        and getattr(c, 'source_consumption_v2', None) is None
        and c.repair_ratio == (o['repair_ratio'] or .15)
        and c.adapter.native_repair_metric == 'normalized_kv_deviation'
        and 2 in c.adapter.depths and c.adapter.costs is None,
        'fresh qualified layer3 diagnostic context required')
    profile = ComparisonProfileBindingV2(**o['comparison_profile'])
    _require(profile.completed_depth == 2 and profile.model_signature == store.config['model_signature']
             == c.adapter.provenance['model_signature'], 'model/depth differs')
    row = None
    if o['mode'].startswith('source'):
        _require(source_receipt is not None, 'missing Source cannot become dense')
        row = verify_qa_source(store, snapshot, o, source_receipt)
        validate_p0_comparison_action(c, store, snapshot, profile, P0ComparisonActionV2(
            c.request['request_id'], 'C', snapshot.snapshot_id,
            request_input_digest(c.request['token_ids'], tuple(range(len(c.request['token_ids'])))), 2,
            (row['source_id'],), profile.binding_digest,
            o['resources']['host_comparison_bytes'], o['resources']['cuda_comparison_bytes']))
    else:
        _require(source_receipt is None, 'dense arm cannot consume Source')
    audit = dict(kind='layer3_boundary_diagnostic_execution_v2', status='RUNNING',
        request_id=c.request['request_id'], operation_sha256=o['operation_sha256'], mode=o['mode'],
        source_id=None if row is None else row['source_id'], answer=None, cleanup=None,
        source_selection_evaluated=False, economic_admission_evaluated=False,
        production_reuse_commit_observed=False, source_publication_allowed=False,
        extra_forward_count=0, paper_evidence=False)
    issuer = None; error = None
    try:
        c.adapter.check_deadline(); c.advance_to_depth(2)
        if row is not None:
            workspace = c.configure_cuda_comparison_v2(capacity_bytes=o['resources']['cuda_comparison_bytes'])
            issuer = ComparisonSessionV2(store, snapshot, profile,
                workspace_bytes=o['resources']['host_comparison_bytes'], cuda_workspace=workspace)
            bridge = BoundarySourceSession(c, issuer); c.source_consumption_v2 = bridge
            receipt = issuer.compare_native(c, 'C'); audit['observation'] = asdict(receipt)
            bridge.prepare_prescribed(receipt, o, row['source_id'])
            c.finish_selection(c.frozen, c.prepared)
            ready, mask_digest = c.ready_for_final_commit(c.prepared)
            _require(ready == {'C': 3} and c.current_completed_depth == 2, 'boundary moved before commit')
            commit_boundary_diagnostic(c, o)
            audit.update(repair_support=deepcopy(c.supports), mask_digest=mask_digest,
                         committed_boundaries=dict(c.committed))
        first = []; audit['answer'] = c.finish(lambda: first.append(time.perf_counter_ns()))
        _require(c.finished and len(first) == 1 and first[0] >= c.arrival_ns, 'missing first token')
        layers = [dict(layer=r['layer'], active_before=list(r['active_before']), active_after=list(r['active_after']),
                       union_mask_digest=r['union_mask_digest']) for r in c.engine.session.layer_audit if 'layer' in r]
        audit['layers'] = layers
        audit['execution_checks'] = validate_layer_recipe(layers, prompt_count=len(c.request['token_ids']),
            target_positions=c.segments['C']['positions'], support=c.supports['C'][3] if row else (),
            total_layers=c.adapter.spec.num_layers, source=row is not None)
        if row:
            _require(c.committed == {'C': 3} and c.engine.session.commits['C'].source_id == row['source_id'],
                     'prescribed Source did not execute')
            ticket = c.prepared['C']
            # Qualification, never hot online hashing. Completes all layer copies first.
            _require(ticket.fully_ready() and ticket.per_request_full_digest_verified
                and ticket.expected_artifact_digest == ticket.source_digest_before
                == ticket.source_digest_after == ticket.destination_digest == row['artifact_digest'],
                'qualification Source/destination integrity incomplete')
            with store.lock:
                store._verify_row(store._visible_row(snapshot, row['source_id']), full=True)
            audit['integrity'] = {k:getattr(ticket,k) for k in (
                'expected_artifact_digest', 'source_digest_before', 'source_digest_after',
                'destination_digest', 'per_request_full_digest_verified')}
        audit.update(status='COMPLETED', ttft_ms=(first[0]-c.arrival_ns)/1e6,
            timing_scope='request_open_to_first_token_including_diagnostic_preparation')
    except BaseException as exc:
        error = exc; audit.update(status='FAILED', failure=dict(type=type(exc).__name__, detail=str(exc)))
    finally:
        failures = []; fenced = False
        if getattr(c, 'source_consumption_v2', None) is not None:
            try: audit['consumption'] = c.source_consumption_v2.audit()
            except BaseException as exc: failures.append(str(exc)); error = error or exc
        try: c.close(); fenced = True
        except BaseException as exc: failures.append(str(exc)); error = error or exc
        if issuer is not None:
            try: issuer.close()
            except BaseException as exc: failures.append(str(exc)); error = error or exc
        if fenced:
            try: store.end_request(snapshot)
            except BaseException as exc: failures.append(str(exc)); error = error or exc
        audit['cleanup'] = dict(passed=not failures, failures=failures, context_fenced=fenced)
    if error:
        audit['status'] = 'FAILED'; raise P0RequestFailure(audit, error) from error
    return audit


def read_boundary_evidence(directory):
    """Verify append-only raw event, request and code binding before a gate."""
    from .p0_evidence_v2 import read_p0_events
    root = Path(directory)
    read = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
    result = read('result.json'); manifest = read('manifest.json')
    _check_seal(manifest, 'manifest_sha256')
    _require(manifest['boundary_binding'] == boundary_binding(), 'boundary implementation changed')
    _require(result['status'] == 'COMPLETED' and result['evidence_origin'] == 'real_cuda_execution'
             and file_digest(root/'actions.jsonl') == result['raw_event_sha256'], 'real completed raw evidence required')
    events = read_p0_events(root/'actions.jsonl', binding=manifest['binding'])
    _require(len(events) == result['event_count'] and events[-1]['event_sha256'] == result['final_event_sha256']
             and events[-1]['kind'] == 'batch_stopped' and events[-1]['payload']['status'] == 'COMPLETED',
             'incomplete event chain')
    records = [e for e in events if e['kind'] == 'action_recorded' and e['action_id'] == 'action']
    _require(len(records) == 1 and file_digest(root/'action/record.json') == records[0]['payload']['record']['sha256'],
             'raw record differs')
    record = read('action/record.json'); audit = read('action/request.json')
    _require(record['request']['file'] == 'request.json' and record['evidence_origin'] == 'real_cuda_execution'
             and record['execution_completed'] is True and file_digest(root/'action/request.json') == record['request']['sha256']
             and audit['status'] == 'COMPLETED' and audit['cleanup']['passed'] is True,
             'raw action/cleanup differs')
    op = read('operation.json'); _check_seal(op, 'operation_sha256', manifest['operation_sha256'])
    _require(audit['operation_sha256'] == op['operation_sha256'] and op['dispatch_binding'] == boundary_binding()
             and audit['request_id'] == op['request']['request_id'], 'actual operation differs')
    positions = next(s['positions'] for s in op['request']['segments'] if s['segment_id'] == 'C')
    support = audit.get('repair_support', {}).get('C', {}).get('3', [])
    validate_layer_recipe(audit['layers'], prompt_count=len(op['request']['token_ids']), target_positions=positions,
        support=support, total_layers=manifest['num_layers'], source=op['mode'].startswith('source'))
    if op['mode'].startswith('source'):
        integrity = audit['integrity']
        _require(audit['committed_boundaries'] == {'C':3} and len(support) == math.ceil(len(positions)*op['repair_ratio'])
                 and integrity['per_request_full_digest_verified'] is True
                 and len({integrity[k] for k in ('expected_artifact_digest', 'source_digest_before',
                                                'source_digest_after', 'destination_digest')}) == 1,
                 'actual support/Source integrity failed')
    return dict(root=str(root), operation=op, audit=audit, manifest=manifest,
        result_sha256=file_digest(root/'result.json'), record_sha256=file_digest(root/'action/record.json'),
        manifest_sha256=file_digest(root/'manifest.json'))


def evaluate_boundary_gate(directories):
    """Recompute strict numerics from files, not a caller-supplied passed flag."""
    from .p0_evidence_v2 import compare_p0_logit_actions
    _require(set(directories) in ({'dense_teacher','source_teacher_r1'},
        {'dense_teacher','source_teacher_r1','dense_qa','source_greedy_r1'}), 'exact bounded sentinel set required')
    rows = {mode:read_boundary_evidence(p) for mode,p in directories.items()}
    for mode,row in rows.items():
        _require(row['operation']['mode'] == mode, 'sentinel role differs')
    left = rows['dense_teacher']; right = rows['source_teacher_r1']
    _require(left['operation']['original_request_sha256'] == right['operation']['original_request_sha256'],
             'numerical comparison uses different frozen requests')
    numerical = compare_p0_logit_actions(Path(left['root'])/'action', Path(right['root'])/'action',
        record_hashes=(left['record_sha256'],right['record_sha256']),
        manifest_hashes=(left['manifest_sha256'],right['manifest_sha256']),
        relative_l2_limit=1e-4, minimum_positions=32)
    passed = numerical['numeric_passed'] and numerical['predicted_token_ids_identical']
    greedy = None
    if 'dense_qa' in rows:
        a,b = rows['dense_qa'], rows['source_greedy_r1']
        _require(a['operation']['original_request_sha256'] == b['operation']['original_request_sha256']
            == left['operation']['original_request_sha256']
            and b['audit']['source_id'] == right['audit']['source_id'], 'greedy comparison conditions differ')
        for k in ('code_commit','runtime_digest','patch_sha256','model_signature','tokenizer_hash','gpu_uuid','instance_id'):
            _require(a['manifest']['binding'][k] == b['manifest']['binding'][k] == left['manifest']['binding'][k],
                     'sentinel environment differs')
        greedy = a['audit']['answer']['token_ids'] == b['audit']['answer']['token_ids']
        passed = passed and greedy
    return dict(status='PASSED' if passed else 'FAILED', teacher=numerical, greedy_token_ids_identical=greedy,
        fixed15_boundary_QA_allowed=bool(passed and greedy is True),
        evidence={k:dict(directory=v['root'],result_sha256=v['result_sha256']) for k,v in rows.items()},
        P4_execution_allowed=False, production_reuse_commit_observed=False, paper_evidence=False)

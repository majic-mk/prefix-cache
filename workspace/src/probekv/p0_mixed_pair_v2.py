"""Preregistered T21 mixed-reference comparison, never global qualification.

The normal full-query reference and the independent sparse executor must use
the same frozen upstream KV, masks and target. Whole-request dense is not an
implementation reference for this test. All numeric limits are explicit.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from .p0_evidence_v2 import compare_p0_logit_actions, read_p0_events, read_p0_target_kv
from .p0_mixed_control_v2 import validate_mixed_reference_job
from .source_manifest_v2 import request_input_digest
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest


_FIELDS = {'pair_id', 'reference_action_id', 'candidate_action_id', 'relative_l2_limit',
           'minimum_positions', 'require_predicted_token_ids_equal',
           'target_relative_l2_limit', 'target_absolute_max_limit', 'target_read_bytes'}


def _safe(value):
    return type(value) is str and re.fullmatch('[A-Za-z0-9_-]{1,80}', value)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value)


def _positions(value, expected):
    # Python list equality would otherwise accept False as absolute position 0.
    return type(value) is list and all(type(p) is int for p in value) and value == expected


def validate_mixed_pairs(manifest):
    pairs = manifest.get('mixed_reference_pairs', [])
    if not isinstance(pairs, list):
        raise ValueError('mixed_reference_pairs must be an explicit list')
    if not pairs:
        return []
    jobs = manifest['jobs']
    if len({j['action_id'] for j in jobs}) != len(jobs):
        raise ValueError('mixed pair jobs require unique action IDs')
    jobs = {j['action_id']: j for j in jobs}
    seen = set()
    for pair in pairs:
        if (type(pair) is not dict or set(pair) != _FIELDS
                or any(not _safe(pair[k]) for k in ('pair_id', 'reference_action_id', 'candidate_action_id'))
                or pair['pair_id'] in seen or pair['reference_action_id'] == pair['candidate_action_id']
                or type(pair['require_predicted_token_ids_equal']) is not bool
                or type(pair['minimum_positions']) is not int or pair['minimum_positions'] < 1
                or type(pair['target_read_bytes']) is not int or pair['target_read_bytes'] < 1
                or any(type(pair[k]) not in (int, float) or not math.isfinite(pair[k]) or pair[k] < 0
                       for k in ('relative_l2_limit', 'target_relative_l2_limit', 'target_absolute_max_limit'))):
            raise ValueError('strict pair identities, numeric policy and target reader budget required')
        seen.add(pair['pair_id'])
        arms = []
        for field, operation in (('reference_action_id', 'explicit_mixed_reference'),
                                 ('candidate_action_id', 'mixed_sparse_control')):
            if pair[field] not in jobs:
                raise ValueError('mixed pair references an unplanned action')
            job = jobs[pair[field]]
            validate_mixed_reference_job(job, _operation=operation)
            if (digest_json(job['request']) != job.get('request_sha256')
                    or job['request']['max_new_tokens'] < pair['minimum_positions']):
                raise ValueError('bound request digest and adequate teacher positions required')
            arms.append(job)
        if (any(arms[0][k] != arms[1][k] for k in
                ('source', 'target_id', 'upstream_segment_id', 'first_reuse_layer', 'repair_positions_by_layer'))
                or arms[0].get('target_execution', 'FULL_ALL_LAYERS') != arms[1].get('target_execution', 'FULL_ALL_LAYERS')
                or arms[0].get('target_source') != arms[1].get('target_source')
                or arms[0].get('upstream_repair_endpoint') != arms[1].get('upstream_repair_endpoint')
                or {k:v for k,v in arms[0]['request'].items() if k != 'request_id'}
                   != {k:v for k,v in arms[1]['request'].items() if k != 'request_id'}):
            raise ValueError('mixed arms must share Source, target, masks, boundary and request conditioning')
    return pairs


def _recipe(audit, job, binding, descriptor, *, sparse):
    request = job['request']
    full = list(range(len(request['token_ids'])))
    segments = {s['segment_id']: s for s in request['segments']}
    upstream = segments[job['upstream_segment_id']]['positions']
    target = segments[job['target_id']]['positions']
    recipe = audit['recipe']
    r1 = job.get('target_execution', 'FULL_ALL_LAYERS') == 'R1_ALL_LAYERS'
    fields = {'kind', 'request_id', 'input_digest', 'model_signature', 'target_id', 'target_positions',
              'upstream_segment_id', 'source_id', 'artifact_digest', 'source_generation',
              'source_birth_positions', 'source_current_positions', 'first_reuse_layer',
              'repair_positions_by_layer', 'num_layers', 'prefix_tokens', 'target_execution',
              'production_publication_allowed'}
    if r1:
        fields.add('target_source')
    if 'upstream_repair_endpoint' in job:
        fields.add('upstream_repair_endpoint')
        if recipe.get('upstream_repair_endpoint') != job['upstream_repair_endpoint']:
            raise ValueError('upstream endpoint differs from frozen recipe')
    operation = 'mixed_sparse_control' if sparse else 'explicit_mixed_reference'
    expected_kind = 'explicit_mixed_sparse_control' if sparse else 'explicit_mixed_full_query_reference'
    if (set(recipe) != fields or digest_json(recipe) != audit['recipe_sha256']
            or recipe['kind'] != expected_kind or audit.get('kind') != 'p0_' + operation
            or recipe['request_id'] != request['request_id']
            or recipe['input_digest'] != request_input_digest(request['token_ids'], full)
            or recipe['model_signature'] != binding['model_signature']
            or recipe['target_id'] != job['target_id'] or recipe['target_positions'] != target
            or recipe['upstream_segment_id'] != job['upstream_segment_id']
            or recipe['source_current_positions'] != upstream
            or recipe['source_id'] != job['source']['source_id']
            or recipe['artifact_digest'] != job['source']['artifact_digest']
            or type(recipe['source_generation']) is not int or recipe['source_generation'] != 0
            or type(recipe['prefix_tokens']) is not int or recipe['prefix_tokens'] != 0
            or recipe['target_execution'] != ('R1_ALL_LAYERS' if r1 else 'FULL_ALL_LAYERS')
            or recipe['production_publication_allowed'] is not False
            or recipe['first_reuse_layer'] != job['first_reuse_layer']
            or recipe['repair_positions_by_layer'] != job['repair_positions_by_layer']):
        raise ValueError('mixed actual execution recipe differs from frozen Source/input/mask identity')
    birth = recipe['source_birth_positions']
    total = recipe['num_layers']
    if (type(total) is not int or total < 1 or not isinstance(birth, list) or len(birth) != len(upstream)
            or any(type(p) is not int or p < 0 for p in birth)
            or birth != list(range(birth[0], birth[0]+len(birth)))
            or descriptor['layer_count'] != total or descriptor['shape'][0] != len(target)):
        raise ValueError('actual target/model layers or Source birth geometry missing')
    validate_mixed_reference_job(job, total_layers=total, _operation=operation)
    if r1:
        source = recipe['target_source']
        source_fields = {'source_id', 'artifact_digest', 'source_generation',
                         'source_birth_positions', 'source_current_positions'}
        birth_target = source.get('source_birth_positions')
        if (set(source) != source_fields
                or source['source_id'] != job['target_source']['source_id']
                or source['artifact_digest'] != job['target_source']['artifact_digest']
                or type(source['source_generation']) is not int or source['source_generation'] != 0
                or not _positions(source['source_current_positions'], target)
                or type(birth_target) is not list or len(birth_target) != len(target)
                or any(type(p) is not int or p < 0 for p in birth_target)
                or birth_target != list(range(birth_target[0], birth_target[0]+len(target)))):
            raise ValueError('target-r1 explicit Source identity/geometry differs')
        guard = audit.get('target_source_integrity', {})
        if guard.get('passed') is not True or any(guard.get(k) != source['artifact_digest'] for k in
                ('source_before', 'destination', 'source_after', 'destination_after')):
            raise ValueError('target-r1 Source/destination integrity not verified')
    if (audit.get('publication') != {'status': 'DIAGNOSTIC_NOT_PUBLISHED'}
            or type(audit.get('extra_forward_count')) is not int or audit['extra_forward_count'] != 1
            or audit.get('production_reuse_commit_observed') is not False
            or audit['answer'].get('whole_request_origin') != 'p0_' + operation):
        raise ValueError('diagnostic identity, cost count or no-publication contract missing')
    marker = audit['answer']['p0_diagnostic_context_v2']
    if (marker.get('kind') != operation or marker.get('request_id') != request['request_id']
            or marker.get('input_digest') != recipe['input_digest']
            or marker.get('model_signature') != recipe['model_signature']
            or marker.get('recipe_sha256') != audit['recipe_sha256']
            or any(marker.get(k) is not False for k in
                   ('grants_execution_authority', 'exact_prefix_publication_allowed', 'source_publication_allowed'))):
        raise ValueError('typed restrictive diagnostic endpoint proof is missing or stale')
    integrity = audit['integrity']
    if (integrity.get('passed') is not True or any(integrity.get(k) != recipe['artifact_digest'] for k in
            ('source_before', 'destination', 'source_after', 'destination_after'))):
        raise ValueError('Source/destination before/after integrity mismatch')
    trace = audit['execution' if sparse else 'reference']
    if (trace.get('completed') is not True or trace.get('failure') is not None
            or trace.get('cleanup_failure') is not None or trace.get('device_references_released') is not True
            or trace.get('publication_allowed') is not False
            or any(type(trace.get(k)) is not int or trace[k] != 0
                   for k in ('parent_owned_kv_bytes', 'prefix_shadow_bytes'))
            or not _sha(trace.get('target_logical_digest'))):
        raise ValueError('complete fenced diagnostic trace/target ownership proof required')
    if (type(trace.get('target_owned_host_bytes')) is not int
            or trace['target_owned_host_bytes'] != descriptor['total_tensor_bytes']):
        raise ValueError('owned target bytes differ from raw layer payload')
    if sparse:
        if (trace.get('kind') != 'p0_actual_cacheblend_sparse_execution'
                or trace.get('projection_counts') != [1]*total
                or any(type(n) is not int for n in trace['projection_counts'])
                or type(trace.get('geometry')) is not list
                or len(trace['geometry']) != total
                or trace.get('target_r1_endpoint_exercised') is not r1):
            raise ValueError('actual sparse geometry/projection/r1 endpoint proof missing')
        for g in trace['geometry']:
            if (type(g) is not list or len(g) != 5 or any(type(n) is not int or n < 1 for n in g)
                    or g[0] != g[2]*g[4] or g[1] != g[3]*g[4] or g[2] % g[3]
                    or g[3:] != descriptor['shape'][1:]):
                raise ValueError('sparse GQA geometry differs from raw target heads/dimension')
    if not sparse:
        hook = trace.get('recipe')
        expected_hook = dict(kind='independent_explicit_mixed_full_query_reference',
            token_count=len(full), target_positions=target, source_positions=upstream,
            first_reuse_layer=recipe['first_reuse_layer'],
            repair_positions_by_layer=recipe['repair_positions_by_layer'])
        if (type(hook) is not dict or set(hook) != set(expected_hook) | {'geometry'}
                or digest_json(hook) != trace.get('recipe_sha256')
                or digest_json({k:hook[k] for k in expected_hook}) != digest_json(expected_hook)
                or trace.get('kind') != 'p0_mixed_reference_hooks'
                or trace.get('projection_counts') != [1]*total
                or any(type(n) is not int for n in trace['projection_counts'])):
            raise ValueError('actual reference hook recipe/hash/projection counts differ')
        geometry = hook['geometry']
        if type(geometry) is not list or len(geometry) != total:
            raise ValueError('reference geometry required for every layer')
        for g in geometry:
            if (type(g) is not list or len(g) != 5 or any(type(n) is not int or n < 1 for n in g)
                    or g[0] != g[2]*g[4] or g[1] != g[3]*g[4] or g[2] % g[3]
                    or g[3:] != descriptor['shape'][1:]):
                raise ValueError('reference GQA geometry differs from raw target heads/dimension')
    layers = trace['executed_layers']
    if not isinstance(layers, list) or len(layers) != total:
        raise ValueError('one completed target execution record per actual model layer required')
    previous = full
    replaced = 0
    for index, layer in enumerate(layers, 1):
        repair = upstream if index < recipe['first_reuse_layer'] else recipe['repair_positions_by_layer'][str(index)]
        historical = [p for p in upstream if p not in set(repair)]
        active = [p for p in full if p not in set(historical)]
        if (type(layer.get('layer_1based')) is not int or layer['layer_1based'] != index
                or not _positions(layer.get('target_positions'), target)
                or not _positions(layer.get('historical_kv_positions'), historical)
                or not _positions(layer.get('repair_positions'), repair) or layer.get('target_full_projection') is not True
                or layer.get('completed_block') is not True):
            raise ValueError('target rows/injected Source mask were changed or omitted in a layer')
        if sparse:
            if (not _positions(layer.get('projected_positions'), previous)
                    or not _positions(layer.get('attention_query_positions'), active)):
                raise ValueError('actual sparse projection/query rows differ from no-reentry execution recipe')
            if (type(layer.get('actual_qkv_invocations')) is not int or layer['actual_qkv_invocations'] != 1
                    or type(layer.get('actual_attention_output_rows')) is not int
                    or layer['actual_attention_output_rows'] != len(active)):
                raise ValueError('actual sparse calls/attention outputs were not observed')
            status = layer.get('native_runtime_status', {})
            expected_status = 0 if index < recipe['first_reuse_layer'] else (1 if index == recipe['first_reuse_layer'] else 2)
            if (type(status.get('status')) is not int or status['status'] != expected_status
                    or status.get('dense_full_repair') is not False):
                raise ValueError('native status does not prove sparse mixed upstream')
            if r1:
                active_r1 = index >= recipe['first_reuse_layer']
                if (any(layer.get(k) is not active_r1 for k in ('target_r1_commit_active',
                        'target_r1_source_installed', 'target_r1_current_kv_writeback_verified'))
                        or not _positions(layer.get('target_r1_repair_positions'), target if active_r1 else [])):
                    raise ValueError('actual target-r1 Source install/commit/current K/V writeback missing')
            previous = active
        elif (type(layer.get('projected_rows')) is not int or layer['projected_rows'] != len(full)
                or type(layer.get('attention_query_rows')) is not int or layer['attention_query_rows'] != len(full)):
            raise ValueError('independent reference must execute normal full-query blocks')
        replaced += len(historical)
    if (not replaced or type(trace.get('historical_kv_rows_replaced')) is not int
            or trace['historical_kv_rows_replaced'] != replaced):
        raise ValueError('no nonzero observed historical KV injection; cannot qualify two dense arms')
    return {k:v for k,v in recipe.items() if k not in ('kind', 'request_id')}, trace


def _target_digest(layers):
    import torch
    digest = hashlib.sha256()
    for index, pair in enumerate(layers, 1):
        digest.update(str((index, tuple(pair[0].shape), 'BF16 pre-RoPE')).encode('ascii'))
        for tensor in pair:
            digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast('B'))
    return digest.hexdigest()


def _target_numerics(left, right, policy):
    import numpy as np
    reports = []
    for layer, (pa, pb) in enumerate(zip(left, right), 1):
        for name, ta, tb in zip(('K', 'V'), pa, pb):
            a, b = ta.float().numpy().astype(np.float64), tb.float().numpy().astype(np.float64)
            difference = a-b
            norm, distance = float(np.linalg.norm(a)), float(np.linalg.norm(difference))
            relative = distance/norm if norm else (0. if distance == 0 else None)
            absolute = float(np.max(np.abs(difference)))
            passed = (relative is not None and relative <= policy['target_relative_l2_limit']
                      and absolute <= policy['target_absolute_max_limit'])
            reports.append(dict(layer_1based=layer, tensor=name, relative_l2=relative,
                                absolute_max=absolute, passed=passed))
            del a, b, difference
    return dict(numeric_passed=all(r['passed'] for r in reports), layers=reports,
                scope='per_layer_per_tensor_all_target_rows',
                relative_l2_limit=policy['target_relative_l2_limit'],
                absolute_max_limit=policy['target_absolute_max_limit'])


def evaluate_mixed_pairs(output, manifest, completed_action_ids):
    pairs = validate_mixed_pairs(manifest)
    if not pairs:
        return []
    output = Path(output)
    jobs = {j['action_id']: j for j in manifest['jobs']}
    completed = set(completed_action_ids)
    results = []
    for pair in pairs:
        ids = (pair['reference_action_id'], pair['candidate_action_id'])
        r1 = jobs[ids[1]].get('target_execution', 'FULL_ALL_LAYERS') == 'R1_ALL_LAYERS'
        r0 = jobs[ids[1]].get('upstream_repair_endpoint') == 'R0_DIAGNOSTIC'
        report = {**pair, 'test_id': 'T21', 'evidence_scope': 'fixed_upstream_mixed_target_full_only',
                  'native_runtime_qualified': False, 'gpu_runtime_qualified': False, 'P0_qualified': False,
                  'P1_execution_allowed': False, 'paper_evidence': False, 'r1_endpoint_qualified': False,
                  'real_cuda_T21_passed': False, 'real_cuda_target_r1_passed': False}
        if r1:
            report.update(test_id='T31', evidence_scope='fixed_upstream_mixed_target_r1_only')
        if r0:
            report.update(test_id='T31', evidence_scope='upstream_r0_full_target_reference_only')
        if not set(ids) <= completed:
            results.append({**report, 'status': 'PENDING', 'pending_action_ids': [i for i in ids if i not in completed]})
            continue
        try:
            manifest_path = output/'manifest.json'
            if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
                raise ValueError('saved manifest differs from preregistered mixed pair')
            manifest_sha = file_digest(manifest_path)
            events = read_p0_events(output/'actions.jsonl', binding=manifest['binding'])
            record_hashes = []
            for action_id in ids:
                matches = [e for e in events if e['kind'] == 'action_recorded' and e['action_id'] == action_id]
                if len(matches) != 1 or matches[0]['payload']['record']['file'] != 'record.json':
                    raise ValueError('one chain-bound raw record per mixed arm required')
                record_hashes.append(matches[0]['payload']['record']['sha256'])
            numerical = compare_p0_logit_actions(output/ids[0], output/ids[1],
                record_hashes=record_hashes, manifest_hashes=(manifest_sha, manifest_sha),
                relative_l2_limit=pair['relative_l2_limit'], minimum_positions=pair['minimum_positions'])
            audits = [json.loads((output/i/'request.json').read_text(encoding='utf-8')) for i in ids]
            records = [json.loads((output/i/'record.json').read_text(encoding='utf-8')) for i in ids]
            descriptors = [r['target_kv'] for r in records]
            if (any(type(d) is not dict for d in descriptors)
                    or descriptors[0]['shape'] != descriptors[1]['shape']
                    or descriptors[0]['layer_count'] != descriptors[1]['layer_count']):
                raise ValueError('matched raw target model geometry required')
            # Bound both retained BF16 arms plus conservative one-layer FP64
            # conversion/difference/norm scratch, not merely one reader call.
            shape = descriptors[0]['shape']
            if (type(shape) is not list or len(shape) != 3
                    or any(type(n) is not int or n < 1 for n in shape)
                    or type(descriptors[0]['layer_count']) is not int or descriptors[0]['layer_count'] < 1):
                raise ValueError('positive target geometry required before allocating comparison arrays')
            payload = math.prod(shape)*4*descriptors[0]['layer_count']
            needed = 2*payload + 48*math.prod(shape)
            if needed > pair['target_read_bytes']:
                raise MemoryError('mixed target pair exceeds explicit aggregate reader/scratch budget')
            verified = [_recipe(a, jobs[i], manifest['binding'], d, sparse=(n == 1))
                        for n, (a, i, d) in enumerate(zip(audits, ids, descriptors))]
            if verified[0][0] != verified[1][0]:
                raise ValueError('mixed reference and sparse actual upstream recipes differ')
            targets = [read_p0_target_kv(output/i, expected_record_sha256=h, max_bytes=payload)
                       for i, h in zip(ids, record_hashes)]
            if any(_target_digest(t) != v[1]['target_logical_digest'] for t, v in zip(targets, verified)):
                raise ValueError('raw target arrays differ from actual captured target digest')
            target_numerical = _target_numerics(*targets, pair)
            passed = (numerical['numeric_passed'] and target_numerical['numeric_passed']
                      and (not pair['require_predicted_token_ids_equal'] or numerical['predicted_token_ids_identical']))
            real = numerical['evidence_origins'] == ['real_cuda_execution']*2
            results.append({**report, 'status': ('PASSED' if real else 'CPU_ONLY') if passed else 'FAILED',
                'scoped_checks_passed': passed, 'real_cuda_T21_passed': passed and real and not r1 and not r0,
                'real_cuda_upstream_r0_passed': passed and real and r0,
                'real_cuda_target_r1_passed': passed and real and r1,
                'recipe_alignment_verified': True, 'numerical': numerical, 'target_numerical': target_numerical,
                'record_sha256': record_hashes, 'manifest_sha256': manifest_sha,
                'failure': None if passed else 'NUMERIC_OR_PREDICTED_TOKEN_POLICY_FAILED'})
        except (ValueError, KeyError, TypeError, OSError, IndexError, MemoryError) as exc:
            results.append({**report, 'status': 'FAILED', 'scoped_checks_passed': False,
                            'real_cuda_T21_passed': False,
                            'failure': {'type': type(exc).__name__, 'detail': str(exc)}})
    return results

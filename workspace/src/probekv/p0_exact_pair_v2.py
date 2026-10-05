"""Preregistered T20 exact capture on/off comparison, not P0/P1 qualification.

Validation is side-effect free. Evaluation returns FAILED for corrupt evidence,
PENDING without opening unfinished arms, CPU_ONLY for valid fixture evidence,
and PASSED only for the scoped real-CUDA T20 comparison. No files are written.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from .p0_decode_evidence_v2 import DecodeInputRecorderV2
from .p0_evidence_v2 import compare_p0_logit_actions, read_p0_events
from .p0_exact_control_v2 import validate_exact_control_job
from .source_manifest_v2 import request_input_digest
from .v8_schema10_execution import digest_json
from .v8_schema10_storage import file_digest


_PAIR_FIELDS = {'pair_id', 'reference_action_id', 'candidate_action_id',
                'relative_l2_limit', 'minimum_positions', 'require_predicted_token_ids_equal'}


def _safe(value):
    return isinstance(value, str) and re.fullmatch('[A-Za-z0-9_-]{1,80}', value)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def validate_exact_pairs(manifest):
    """Validate declared pairs only; no default BF16 numerical tolerance."""
    pairs = manifest.get('exact_capture_pairs', [])
    if not isinstance(pairs, list):
        raise ValueError('exact_capture_pairs must be an explicit list')
    if not pairs:
        return []
    jobs = manifest['jobs']
    ids = [j['action_id'] for j in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError('pair jobs must have unique action identities')
    by_id = {j['action_id']: j for j in jobs}
    seen = set()
    for pair in pairs:
        if (not isinstance(pair, dict) or set(pair) != _PAIR_FIELDS
                or any(not _safe(pair[k]) for k in ('pair_id', 'reference_action_id', 'candidate_action_id'))
                or pair['pair_id'] in seen
                or pair['reference_action_id'] == pair['candidate_action_id']
                or type(pair['require_predicted_token_ids_equal']) is not bool
                or type(pair['relative_l2_limit']) not in (int, float)
                or not math.isfinite(pair['relative_l2_limit']) or pair['relative_l2_limit'] < 0
                or type(pair['minimum_positions']) is not int or pair['minimum_positions'] < 1):
            raise ValueError('strict safe pair IDs and explicit finite numerical policy required')
        seen.add(pair['pair_id'])
        arms = []
        for key, capture in (('reference_action_id', False), ('candidate_action_id', True)):
            if pair[key] not in by_id:
                raise ValueError('pair references an unplanned action')
            job = by_id[pair[key]]
            validate_exact_control_job(job)
            request = job['request']
            DecodeInputRecorderV2(request, cached_prefix_tokens=0)
            if (job['capture_enabled'] is not capture
                    or pair['minimum_positions'] > request['max_new_tokens']
                    or digest_json(request) != job.get('request_sha256')):
                raise ValueError('pair capture roles, request hash or logit count differ')
            positions = {s['segment_id']: s['positions'] for s in request['segments']}
            if len(positions) != len(request['segments']):
                raise ValueError('duplicate target segment identity')
            for sid in job['target_ids']:
                pos = positions[sid]
                if (not isinstance(pos, list) or not pos
                        or any(type(p) is not int or not 0 <= p < len(request['token_ids']) for p in pos)
                        or pos != list(range(pos[0], pos[-1] + 1))):
                    raise ValueError('target positions must be original contiguous prompt rows')
            arms.append(job)
        if (arms[0]['target_ids'] != arms[1]['target_ids']
                or {k:v for k,v in arms[0]['request'].items() if k != 'request_id'}
                   != {k:v for k,v in arms[1]['request'].items() if k != 'request_id'}):
            raise ValueError('capture pair changes request conditions beyond request_id')
    return pairs


def _recipe(audit, job, binding):
    execution = audit['execution']
    recipe = execution['recipe']
    request = job['request']
    rows = list(range(len(request['token_ids'])))
    targets = {s['segment_id']: s['positions'] for s in request['segments'] if s['segment_id'] in job['target_ids']}
    total = recipe.get('num_layers')
    expected_fields = {'kind', 'input_digest', 'model_signature', 'num_layers', 'prefix_tokens',
                       'source_imports', 'target_positions', 'layers'}
    if (set(execution) != {'recipe', 'recipe_sha256'} or set(recipe) != expected_fields
            or digest_json(recipe) != execution['recipe_sha256']
            or recipe['kind'] != 'exact_full_row_resumable_control'
            or recipe['input_digest'] != request_input_digest(tuple(request['token_ids']), tuple(rows))
            or recipe['model_signature'] != binding['model_signature']
            or type(total) is not int or total < 1
            or type(recipe['prefix_tokens']) is not int or recipe['prefix_tokens'] != 0
            or recipe['source_imports'] != [] or recipe['target_positions'] != targets
            or not isinstance(recipe['layers'], list) or len(recipe['layers']) != total):
        raise ValueError('exact full-row execution recipe identity/geometry mismatch')
    for index, layer in enumerate(recipe['layers'], 1):
        if (set(layer) != {'layer', 'active_before', 'active_after', 'union_mask_digest'}
                or type(layer['layer']) is not int or layer['layer'] != index
                or layer['active_before'] != rows or layer['active_after'] != rows
                or any(type(p) is not int for p in layer['active_before'] + layer['active_after'])
                or layer['union_mask_digest'] != hashlib.sha256(
                    json.dumps(rows, separators=(',', ':')).encode('ascii')).hexdigest()):
            raise ValueError('exact control omitted/reordered a layer or prompt row')
    if (audit.get('kind') != 'p0_exact_capture_control'
            or audit.get('capture_enabled') is not job['capture_enabled']
            or audit.get('publication') != {'status': 'DIAGNOSTIC_NOT_PUBLISHED'}
            or type(audit.get('extra_forward_count')) is not int or audit['extra_forward_count'] != 0
            or audit['answer']['decode_input_trace_v2']['cached_prefix_tokens'] != 0):
        raise ValueError('exact control used Prefix, publication or an extra forward')
    if not job['capture_enabled']:
        if audit.get('capture') is not None or audit.get('captured_targets'):
            raise ValueError('capture-off action captured targets')
        return recipe
    capture = audit['capture']
    captured = audit['captured_targets']
    if (capture.get('kind') != 'actual_layer_target_capture_v2'
            or type(capture.get('layers_recorded')) is not int or capture['layers_recorded'] != total
            or type(capture.get('layers_submitted')) is not int or capture['layers_submitted'] != total
            or set(capture['target_reservations']) != set(targets)
            or capture.get('rejected_targets') != {}
            or capture.get('prefix_shadow_created') is not False
            or capture.get('publication_performed') is not False
            or type(capture.get('extra_forward_count')) is not int or capture['extra_forward_count'] != 0
            or set(captured) != set(targets)):
        raise ValueError('target capture incomplete, rejected, shadow-coupled or published')
    events = capture['events']
    target_count = sum(len(p) for p in targets.values())
    if not isinstance(events, list) or len(events) != total:
        raise ValueError('one actual target capture event per completed layer required')
    for index, event in enumerate(events, 1):
        if (event.get('event') != 'target_capture_after_layer' or event.get('layer') != index
                or event.get('projected_rows') != len(rows) or event.get('effective_current_rows') != len(rows)
                or event.get('captured_rows') != target_count):
            raise ValueError('capture did not preserve complete current rows at every layer')
    for sid, positions in targets.items():
        candidate = captured[sid]
        proof, storage = candidate['proof'], candidate['storage_audit']
        if (proof.get('origin') != 'EXACT_CONTEXT' or type(proof.get('generation')) is not int or proof['generation'] != 0
                or proof.get('request_id') != request['request_id']
                or proof.get('input_digest') != recipe['input_digest']
                or proof.get('model_signature') != binding['model_signature']
                or proof.get('positions') != positions
                or type(proof.get('expected_layers')) is not int or proof['expected_layers'] != total
                or not _sha(proof.get('ledger_digest')) or not _sha(candidate.get('artifact_digest'))
                or candidate.get('publication_state') != 'VALIDATED_CANDIDATE_NOT_PUBLISHED'
                or any(type(storage.get(k)) is not int or storage[k] != 0
                       for k in ('parent_owned_kv_bytes', 'prefix_shadow_bytes'))
                or type(storage.get('target_kv_bytes')) is not int or storage['target_kv_bytes'] <= 0
                or type(storage.get('selection_state_bytes')) is not int or storage['selection_state_bytes'] < 0):
            raise ValueError('captured target proof is not complete G0/target-owned/no-parent/no-shadow')
    return recipe


def evaluate_exact_pairs(output, manifest, completed_action_ids):
    """Return per-pair reports; damaged completed evidence fails without erasing it."""
    pairs = validate_exact_pairs(manifest)
    if not pairs:
        return []
    output = Path(output)
    jobs = {j['action_id']: j for j in manifest['jobs']}
    completed = set(completed_action_ids)
    results = []
    for pair in pairs:
        ids = [pair['reference_action_id'], pair['candidate_action_id']]
        report = {**pair, 'test_id': 'T20', 'evidence_scope': 'exact_capture_on_off_only',
                  'native_runtime_qualified': False, 'gpu_runtime_qualified': False,
                  'P0_qualified': False, 'P1_execution_allowed': False, 'paper_evidence': False}
        if not set(ids) <= completed:
            results.append({**report, 'status': 'PENDING', 'pending_action_ids': [i for i in ids if i not in completed]})
            continue
        try:
            manifest_path = output / 'manifest.json'
            if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
                raise ValueError('saved manifest differs from preregistered pair')
            manifest_sha = file_digest(manifest_path)
            events = read_p0_events(output / 'actions.jsonl', binding=manifest['binding'])
            record_hashes = []
            for action_id in ids:
                matched = [e for e in events if e['kind'] == 'action_recorded' and e['action_id'] == action_id]
                if len(matched) != 1 or matched[0]['payload']['record']['file'] != 'record.json':
                    raise ValueError('exactly one chain-bound raw record required per arm')
                record_hashes.append(matched[0]['payload']['record']['sha256'])
            numerical = compare_p0_logit_actions(
                output / ids[0], output / ids[1], record_hashes=tuple(record_hashes),
                manifest_hashes=(manifest_sha, manifest_sha),
                relative_l2_limit=pair['relative_l2_limit'], minimum_positions=pair['minimum_positions'])
            audits = [json.loads((output / i / 'request.json').read_text(encoding='utf-8')) for i in ids]
            recipes = [_recipe(audit, jobs[i], manifest['binding']) for audit, i in zip(audits, ids)]
            if recipes[0] != recipes[1]:
                raise ValueError('capture changes actual full-row execution recipe')
            passed = numerical['numeric_passed'] and (not pair['require_predicted_token_ids_equal']
                                                       or numerical['predicted_token_ids_identical'])
            is_cuda = numerical['evidence_origins'] == ['real_cuda_execution', 'real_cuda_execution']
            results.append({**report, 'status': ('PASSED' if is_cuda else 'CPU_ONLY') if passed else 'FAILED',
                'scoped_checks_passed': passed, 'real_cuda_T20_passed': passed and is_cuda,
                'recipe_alignment_verified': True, 'numerical': numerical,
                'record_sha256': record_hashes, 'manifest_sha256': manifest_sha,
                'failure': None if passed else 'NUMERIC_OR_PREDICTED_TOKEN_POLICY_FAILED'})
        except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
            results.append({**report, 'status': 'FAILED', 'scoped_checks_passed': False,
                            'real_cuda_T20_passed': False,
                            'failure': {'type': type(exc).__name__, 'detail': str(exc)}})
    return results

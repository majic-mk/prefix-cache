"""Read-only P0 evidence consumer. Never authorizes a GPU or P1 experiment.

Verify records and numerical arrays, not a caller's passed=true summary. The
manifest byte digest is an external frozen input. Different batches retain
their own runtime identity; old evidence is never relabelled as a new run.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .p0_evidence_v2 import read_p0_events
from .p0_exact_pair_v2 import evaluate_exact_pairs
from .p0_mixed_pair_v2 import evaluate_mixed_pairs
from .p0_lineage_control_v2 import lineage_targets, validate_lineage_job
from .v8_schema10_execution import digest_json


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def _read(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('missing or oversized P0 metadata')
    return json.loads(path.read_text(encoding='utf-8'))


def _inside(root, name):
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in ('.', '..'):
        raise ValueError('unsafe evidence filename/action id')
    path = (root / name).resolve()
    if path.parent != root.resolve():
        raise ValueError('evidence symlink escapes batch')
    return path


def verify_batch(root, *, manifest_sha256, expected_binding):
    """Fail closed on omitted/duplicated actions, corrupt records or CPU origin."""
    root = Path(root)
    if not isinstance(manifest_sha256, str) or not re.fullmatch('[0-9a-f]{64}', manifest_sha256):
        raise ValueError('external manifest SHA256 required')
    if file_sha(root / 'manifest.json') != manifest_sha256:
        raise ValueError('frozen manifest bytes changed')
    manifest = _read(root / 'manifest.json')
    if (manifest.get('manifest_sha256') != digest_json({k:v for k,v in manifest.items() if k != 'manifest_sha256'})
            or not expected_binding or manifest.get('binding') != expected_binding):
        raise ValueError('manifest/runtime binding mismatch')
    result = _read(root / 'result.json')
    planned = [job['action_id'] for job in manifest['jobs']]
    if (not planned or len(set(planned)) != len(planned)
            or result.get('status') != 'COMPLETED' or result.get('failed_action_ids') != []
            or result.get('pending_action_ids') != [] or result.get('completed_action_ids') != planned):
        raise ValueError('incomplete/duplicate/unplanned P0 actions')
    if result.get('raw_event_sha256') != file_sha(root / 'actions.jsonl'):
        raise ValueError('event file digest differs')
    events = read_p0_events(root / 'actions.jsonl', binding=expected_binding)
    if (not events or result.get('event_count') != len(events)
            or result.get('final_event_sha256') != events[-1]['event_sha256']):
        raise ValueError('event chain/result endpoint mismatch')
    records = {}; recorded = []
    for event in events:
        if event['kind'] == 'action_failed':
            raise ValueError('failed event cannot be qualified')
        if event['kind'] != 'action_recorded':
            continue
        aid = event['action_id']
        if aid not in planned or aid in records:
            raise ValueError('duplicate or unplanned action record')
        folder = _inside(root, aid)
        desc = event['payload']['record']; path = _inside(folder, desc['file'])
        if file_sha(path) != desc['sha256']:
            raise ValueError('record digest differs')
        record = _read(path)
        if record.get('evidence_origin') != 'real_cuda_execution':
            raise ValueError('CPU/simulated evidence is not GPU qualification')
        path = _inside(folder, record['request']['file'])
        if file_sha(path) != record['request']['sha256']:
            raise ValueError('request audit digest differs')
        audit = _read(path)
        if audit.get('cleanup', {}).get('passed') is not True:
            raise ValueError('cleanup missing or failed')
        records[aid] = audit; recorded.append(aid)
    if recorded != planned:
        raise ValueError('recorded order/completeness differs from frozen jobs')
    pairs = evaluate_exact_pairs(root, manifest, planned) + evaluate_mixed_pairs(root, manifest, planned)
    if any(pair['status'] != 'PASSED' for pair in pairs):
        raise ValueError('raw numerical pair did not pass')
    return dict(manifest=manifest, result=result, audits=records, numerical_pairs=pairs,
                manifest_file_sha256=manifest_sha256, events_sha256=result['raw_event_sha256'])


def verify_lineage_scope(batch):
    """Derive observed lineage cases from bound per-action proofs.

    This verifies persisted structural assertions, not answer quality or a
    numerical dense equivalence claim for mixed-context execution.
    """
    rows = []
    jobs = {job['action_id']: job for job in batch['manifest']['jobs']}
    for aid, audit in batch['audits'].items():
        job = jobs[aid]
        if job.get('operation') != 'controlled_lineage_birth':
            continue
        validate_lineage_job(job)
        _, expected = lineage_targets(job)
        # Historical single-target evidence has a distinct, explicit layout.
        # Do not interpret missing fields in a multi-target record as success.
        if 'target_ids' not in job:
            sid = job['target_id']
            proofs = {sid:audit['target_proof']}
            visibility = {sid:audit['birth_snapshot_sees_child'] if expected[sid] <= 1 else None}
            rejection = {sid:audit.get('g2_rejection_reason')}
        else:
            proofs = audit['target_proofs']
            visibility = audit['visibility_by_target']
            rejection = audit['rejection_by_target']
        if (audit.get('controlled_recipe_commit_observed') is not True
                or audit.get('economic_admission_evaluated') is not False
                or audit.get('production_reuse_commit_observed') is not False
                or any(visibility.get(sid) is not False for sid,g in expected.items() if g <= 1)):
            raise ValueError('controlled lineage evidence misstates admission/visibility')
        targets = {s['segment_id']:s for s in job['request']['segments']}
        if set(proofs) != set(expected):
            raise ValueError('target proof set differs')
        for sid, generation in expected.items():
            proof = proofs[sid]
            if (type(generation) is not int or proof['generation'] != generation
                    or proof['positions'] != targets[sid]['positions']
                    or proof['origin'] != ('EXACT_CONTEXT' if generation == 0 else 'MIXED_CONTEXT_FULL_SEGMENT')):
                raise ValueError('target lineage/positions differ from recipe')
            published = audit['publication']['targets'][sid]['publication_performed']
            if published is not (generation <= 1):
                raise ValueError('G0/G1 publication or G2 rejection differs')
            if generation > 1 and not rejection.get(sid):
                raise ValueError('G2 rejection reason missing')
        rows.append(dict(action_id=aid, expected_generation=expected,
                         targets=sorted(expected), target_tokens={sid:len(targets[sid]['positions']) for sid in expected}))
    return rows


def assess_numerical_correctness(batch_specs):
    """Aggregate checked E/M pairs; no data, QA or full-P0 success implied."""
    reports = []; blockers = []; exact = mixed = r1 = ordinary_mixed = 0
    common = None
    for spec in batch_specs:
        try:
            batch = verify_batch(spec['root'], manifest_sha256=spec['manifest_sha256'],
                                 expected_binding=spec['binding'])
            identity = {k:batch['manifest']['binding'][k] for k in
                        ('model_signature', 'tokenizer_hash', 'patch_sha256')}
            if common is not None and common != identity:
                raise ValueError('cannot merge numerical qualification across models/patches')
            common = identity
            for pair in batch['numerical_pairs']:
                if pair['minimum_positions'] < 32 or pair['relative_l2_limit'] > 1e-4 or pair['require_predicted_token_ids_equal'] is not True:
                    raise ValueError('pair weakens frozen numerical policy')
                if pair['test_id'] == 'T20':
                    exact += 1
                else:
                    if pair['target_relative_l2_limit'] > 1e-4 or pair['target_absolute_max_limit'] > 1e-3:
                        raise ValueError('mixed pair weakens frozen KV policy')
                    mixed += 1
                    if pair.get('real_cuda_T21_passed') is True:
                        ordinary_mixed += 1
                    # The raw pair verifier derives T31 from the validated
                    # target-r1 recipe, not the user-visible pair name.
                    if pair.get('real_cuda_target_r1_passed') is True:
                        r1 += 1
            reports.append(dict(root=str(spec['root']), binding=batch['manifest']['binding'],
                manifest_sha256=batch['manifest_file_sha256'], events_sha256=batch['events_sha256'],
                numerical_pairs=batch['numerical_pairs'], lineage=verify_lineage_scope(batch)))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            blockers.append(dict(root=str(spec.get('root')), reason=str(exc)))
    if not exact: blockers.append(dict(reason='EXACT_RAW_NUMERICAL_PAIR_MISSING'))
    if not ordinary_mixed: blockers.append(dict(reason='MIXED_RAW_NUMERICAL_PAIR_MISSING'))
    if not r1: blockers.append(dict(reason='MIXED_R1_RAW_NUMERICAL_PAIR_MISSING'))
    return dict(kind='p0_raw_acceptance_v2', status='PASS' if not blockers else 'BLOCKED',
        scoped_numerical_correctness_verified=not blockers, batches=reports, blockers=blockers,
        counts=dict(exact_pairs=exact, mixed_pairs=mixed, mixed_r1_pairs=r1),
        full_P0_complete=False, P1_execution_allowed=False, GPU_execution_allowed=False,
        QA_evaluated=False, paper_evidence=False, locked_test_accessed=False)

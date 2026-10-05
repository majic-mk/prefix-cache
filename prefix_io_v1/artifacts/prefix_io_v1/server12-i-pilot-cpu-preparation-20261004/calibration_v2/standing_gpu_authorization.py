"""Bind the user's existing standing grant; never request or enlarge it.

This module uses only project metadata. It neither probes nor launches a GPU.
The original GPU guard still owns timeout, sessions and cumulative accounting.
"""
from pathlib import Path
import hashlib
import json
import math

AMENDMENT = 'artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004/PROJECT_GPU_AUTHORIZATION_AMENDMENT.json'
AMENDMENT_SHA = '8acab9a6d1e67d8c5237f2ff1736cff6e9248c7d359c944e80648c7cba5895ec'
INSTRUCTION = '授权gpu，只要有gpu就用，以后无需授权'
BASE_PERMISSION = 'experiments/prefix_io_v1/configs/permissions.yaml'
BASE_PERMISSION_SHA = '795d24f9955379653e34ffbbfd7614ad9ef96ad0ac9a208c6b4baefd994fac50'


def require(value, reason):
    if not value:
        raise ValueError('STANDING_AUTHORIZATION_REJECTED: ' + reason)


def read_reference(root, relative, expected_sha):
    root = Path(root).resolve(strict=True)
    require(type(relative) is str and relative and '\\' not in relative and ':' not in relative
            and not relative.startswith('/') and all(x not in ('', '.', '..') for x in relative.split('/')),
            'safe project reference')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink metadata')
    require(path.is_file() and path.stat().st_size <= 1024**2, 'bounded regular metadata')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'immutable authorization drift')
    return raw, dict(path=relative, bytes=len(raw), sha256=expected_sha)


def pairs(values):
    result = {}
    for key, value in values:
        require(key not in result, 'duplicate key')
        result[key] = value
    return result


def validate_amendment(document, root, base_ref):
    require(type(document) is dict, 'authorization object')
    expected = dict(schema='c5_project_persistent_gpu_authorization_amendment_v1',
                    root=str(Path(root).resolve()), issuer='human_user', authorization=True,
                    human_instruction=INSTRUCTION, original_max_gpu_seconds=28800,
                    original_budget_unchanged=True, source='direct_current_user_instruction',
                    base_permission_ref=base_ref,
                    GPU_authority_scope='current_frozen_project_experiments_with_existing_stage_gates',
                    future_gpu_approval_questions_required=False)
    for key, value in expected.items():
        require(type(document.get(key)) is type(value) and document[key] == value, key)
    return document


def standing_grant(root):
    raw, reference = read_reference(root, AMENDMENT, AMENDMENT_SHA)
    _, base_ref = read_reference(root, BASE_PERMISSION, BASE_PERMISSION_SHA)
    document = json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda x: require(False, 'nonfinite JSON'))
    return validate_amendment(document, root, base_ref), reference


def validate_bound_grant(root, grant):
    document, reference = standing_grant(root)
    require(type(grant) is dict, 'bound grant object')
    require(grant.get('standing_authorization_ref') == reference, 'standing provenance')
    require(grant.get('binding_origin') == 'derived_from_existing_human_standing_authorization', 'binding origin')
    require(grant.get('human_instruction') == document['human_instruction'], 'human instruction')
    require(grant.get('per_round_reapproval_required') is False, 'no renewed approval')
    return reference


def bind_standing_grant(root, context, *, label, gpu_uuid, revision_ref):
    document, reference = standing_grant(root)
    require(type(context) is dict and context.get('origin') == 'live_site_readonly_context', 'real context required')
    require(context.get('label') == label and context.get('root') == str(Path(root).resolve()), 'context scope')
    require(context.get('revision_lock_ref') == revision_ref, 'context source lock')
    require(context.get('seconds_limit') == 1200 and context.get('reserved_seconds') == 1220, 'bounded calibration')
    require(type(gpu_uuid) is str and context.get('resources', {}).get('gpu_uuid') == gpu_uuid, 'context UUID')
    return dict(schema='c5_gpu_entry_explicit_human_grant_v1', issuer=document['issuer'], authorization=True,
                human_instruction=document['human_instruction'], context_ref=None, gpu_uuid=gpu_uuid,
                label=label, revision_lock_ref=revision_ref, max_attempts=1, max_processes=6,
                seconds_limit=1200, reserved_seconds=1220, instruction_is_gpu_execution_authorization=True,
                standing_authorization_ref=reference, per_round_reapproval_required=False,
                binding_origin='derived_from_existing_human_standing_authorization')


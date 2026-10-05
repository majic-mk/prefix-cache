"""Read-only, stage-scoped input readiness, never GPU or P1 qualification.

Controlled numerical P0 is allowed to use declared diagnostic inputs; it does
not need a P1 natural-data cohort. Conversely, a completed controlled P0 batch
does not qualify the cohort, historical Sources, or S0/M1 construction recipes.
This helper checks real files and their bindings without SSH, model execution,
payments, rental, or writes. The server entry must still recheck actual imports,
GPU identity, authority window and resources immediately before execution.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from .v8_schema10_execution import digest_json


STAGES = ('CONTROLLED_P0_E', 'CONTROLLED_P0_M', 'P1_E', 'P1_M')
_MAX_METADATA_BYTES = 32 * 1024 * 1024
_BASE_REQUIRED = (
    'src/probekv/p0_batch_v2.py', 'src/probekv/p0_evidence_v2.py',
    'src/probekv/source_comparison_v2.py',
    'scripts/server/run_decoupled_v2_p0.py',
)
_STAGE_REQUIRED = {
    'CONTROLLED_P0_E': ('src/probekv/p0_exact_control_v2.py',
                        'src/probekv/p0_exact_pair_v2.py'),
    'CONTROLLED_P0_M': ('src/probekv/p0_mixed_control_v2.py',
                        'src/probekv/p0_mixed_reference_v2.py',
                        'src/probekv/p0_mixed_pair_v2.py',
                        'src/probekv/p0_mixed_sparse_v2.py'),
}


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value)


def _bytes_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root, relative):
    name = Path(relative)
    if (not isinstance(relative, str) or not relative or name.is_absolute()
            or '..' in name.parts):
        raise ValueError('unsafe evidence-relative path')
    target = (root / name).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        raise ValueError('evidence file escapes its root')
    return target


def _read_ref(ref, label):
    if (not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}
            or not isinstance(ref['path'], str) or not ref['path']
            or not _sha(ref['sha256'])):
        raise ValueError(label + ': explicit path and SHA256 required')
    path = Path(ref['path'])
    if not path.is_file() or path.stat().st_size > _MAX_METADATA_BYTES:
        raise ValueError(label + ': metadata missing or above bounded size')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise ValueError(label + ': file digest mismatch')
    return json.loads(data), path.resolve()


def _verify_cpu_artifacts(workspace, refs, stage):
    """Re-derive success from tests/logs/commands and the actual current files."""
    loaded = {name: _read_ref(refs.get(name), name) for name in
              ('cpu_report', 'worktree_files', 'test_results', 'commands')}
    report = loaded['cpu_report'][0]
    files = loaded['worktree_files'][0]
    tests = loaded['test_results'][0]
    commands, command_path = loaded['commands']
    if any(not isinstance(value, dict) for value in (report, files, commands)):
        raise ValueError('CPU report/worktree/command metadata must be objects')
    if (not isinstance(files.get('files'), dict) or not files['files']
            or hashlib.sha256(json.dumps(files['files'], sort_keys=True,
                ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() != files.get('digest')
            or report.get('worktree_digest') != files['digest']
            or report.get('base_commit') != files.get('base_commit')):
        raise ValueError('CPU worktree report/digest binding differs')
    for name, expected in files['files'].items():
        path = _inside(workspace, name)
        if not _sha(expected) or not path.is_file() or _bytes_digest(path) != expected:
            raise ValueError('current worktree differs from CPU validation: ' + name)
    for name in _BASE_REQUIRED + _STAGE_REQUIRED[stage]:
        if name not in files['files']:
            raise ValueError('CPU evidence omitted required implementation: ' + name)
    if not isinstance(tests, list) or not tests:
        raise ValueError('actual nonempty test records required, not passed=true')
    ids, states, logs = set(), [], {}
    for row in tests:
        tid = row.get('test_id')
        if not isinstance(tid, str) or not tid or tid in ids:
            raise ValueError('duplicate/missing CPU test identity')
        ids.add(tid)
        if (row.get('runtime_worktree_digest') != files['digest']
                or row.get('runtime_commit') != report['base_commit']):
            raise ValueError('CPU test provenance differs')
        state = row.get('observed')
        if state not in ('PASS', 'SKIP'):
            raise ValueError('CPU regression has failed or unknown test')
        if state == 'SKIP' and (tid !=
                'test_schema10_native_stage_progress.PreflightResourceTests.test_cpu_cannot_produce_real_preflight_rows'
                or row.get('detail') != 'CPU fail-closed test'):
            raise ValueError('unqualified new CPU skip')
        path, expected = row.get('evidence_path'), row.get('evidence_hash')
        if not isinstance(path, str) or not _sha(expected):
            raise ValueError('CPU test raw log missing')
        previous = logs.setdefault(path, expected)
        if previous != expected:
            raise ValueError('CPU records disagree on raw log digest')
        states.append(state)
    for path, expected in logs.items():
        if not Path(path).is_file() or _bytes_digest(path) != expected:
            raise ValueError('CPU test raw log digest differs')
    if (report.get('status') != 'CPU_CHECKS_PASS_NATIVE_PENDING'
            or type(report.get('tests_run')) is not int or report['tests_run'] != len(tests)
            or report.get('passed') != states.count('PASS')
            or report.get('skipped') != states.count('SKIP')
            or report.get('errors') != 0 or report.get('failures') != 0
            or report.get('locked_test_accessed') is not False):
        raise ValueError('CPU summary does not match raw test records')
    seen = set()
    for row in commands.get('repository_checks', []):
        log = _inside(command_path.parent, row.get('log', ''))
        name = log.name
        if name in seen or row.get('returncode') != 0 or row.get('passed') is not True:
            raise ValueError('missing/failed/duplicate repository check')
        if not _sha(row.get('log_sha256')) or _bytes_digest(log) != row['log_sha256']:
            raise ValueError('repository command log digest differs')
        seen.add(name)
    if not {'compileall.log', 'contract_validator.log', 'diff_check.log'} <= seen:
        raise ValueError('compile/contract/diff check evidence incomplete')
    return dict(worktree_digest=files['digest'], code_commit=report['base_commit'],
                tests_run=len(tests), skipped=states.count('SKIP'))


def _validate_pair_scope(manifest, stage):
    if stage == 'CONTROLLED_P0_E':
        from .p0_exact_pair_v2 import validate_exact_pairs
        pairs = validate_exact_pairs(manifest)
    else:
        from .p0_mixed_pair_v2 import validate_mixed_pairs
        pairs = validate_mixed_pairs(manifest)
    if not pairs:
        raise ValueError('controlled P0 requires complete preregistered numerical pairs')
    # This launch path is not a way to promote arbitrary train/calibration rows.
    if any(job.get('input_origin') != 'controlled_provenance_diagnostic'
           for job in manifest['jobs']):
        raise ValueError('controlled P0 launch accepts isolated diagnostic inputs only')
    return len(pairs)


def validate_p0_model_geometry(manifest, runtime):
    """Reject impossible frozen jobs before loading weights or querying a GPU.

    The caller first validates the native attachment. Config bytes are also
    rebound here to that actual model audit; no layers/vocabulary are inferred
    from the supplied action's own last mask or from illustrative defaults.
    """
    from .model_adapters import SCHEMA6_MODEL_SPECS
    from .v8_schema10_native_factory import verified_model_asset_path
    from .p0_mixed_control_v2 import validate_mixed_reference_job
    key = runtime.get('model_key')
    if key not in SCHEMA6_MODEL_SPECS:
        raise ValueError('P0 model geometry requires a known native adapter')
    spec = SCHEMA6_MODEL_SPECS[key]
    audit, _ = _read_ref(dict(path=runtime['model_audit_path'],
                              sha256=runtime['model_audit_sha256']), 'model_audit')
    config_path = verified_model_asset_path(runtime['model_path'], 'config.json')
    expected = audit.get('files', {}).get('config.json')
    if not _sha(expected) or _bytes_digest(config_path) != expected:
        raise ValueError('actual model config is not bound to the verified audit')
    if config_path.stat().st_size > _MAX_METADATA_BYTES:
        raise ValueError('model config exceeds bounded metadata size')
    config = json.loads(config_path.read_text(encoding='utf-8'))
    layers, vocab = config.get('num_hidden_layers'), config.get('vocab_size')
    maximum, supported = runtime.get('max_model_len'), config.get('max_position_embeddings')
    if type(layers) is not int or layers != spec.num_layers:
        raise ValueError('model config layer count differs from frozen adapter')
    if (type(vocab) is not int or vocab < 1 or type(maximum) is not int or maximum < 1
            or type(supported) is not int or supported < maximum):
        raise ValueError('explicit supported model vocabulary/context geometry required')
    jobs = manifest.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('nonempty concrete P0 jobs required for model geometry')
    shapes = []
    for job in jobs:
        request = job['request']
        tokens = request.get('token_ids'); count = request.get('max_new_tokens')
        if (not isinstance(tokens, list) or not tokens
                or any(type(token) is not int or not 0 <= token < vocab for token in tokens)
                or type(count) is not int or count < 1
                or len(tokens) + count > maximum):
            raise ValueError('P0 prompt/generation exceeds actual vocabulary or context bound')
        if 'teacher_token_ids' in request:
            teacher = request['teacher_token_ids']
            if (not isinstance(teacher, list) or len(teacher) != count - 1
                    or any(type(token) is not int or not 0 <= token < vocab for token in teacher)):
                raise ValueError('teacher sequence exceeds actual vocabulary or generation bound')
        operation = job.get('operation', 'source_request')
        if operation in ('explicit_mixed_reference', 'mixed_sparse_control'):
            validate_mixed_reference_job(job, total_layers=layers, _operation=operation)
        elif operation in ('source_request', 'controlled_lineage_birth'):
            if operation == 'controlled_lineage_birth':
                from .p0_lineage_control_v2 import validate_lineage_job
                validate_lineage_job(job, total_layers=layers)
            depth = job['comparison_profile']['completed_depth']
            if type(depth) is not int or not 1 <= depth < layers:
                raise ValueError('comparison completed depth has no legal d+1 projection')
        elif operation != 'exact_capture_control':
            raise ValueError('unknown P0 operation for model geometry')
        shapes.append(dict(action_id=job['action_id'], prompt_tokens=len(tokens),
                           maximum_generated_tokens=count, operation=operation))
    return dict(kind='p0_model_geometry_preflight_v2', model_key=key,
        model_config_sha256=expected, num_layers=layers, vocab_size=vocab,
        max_model_len=maximum, jobs=shapes, model_loaded=False, gpu_observed=False)


def assess_stage_readiness(stage, *, workspace, evidence, now_unix):
    """Assess concrete evidence references without authorizing any GPU action.

    `evidence` names {path, sha256} records. P0 uses cpu_report/worktree_files/
    test_results/commands, batch_manifest/native_manifest, patch_audit/
    patch_manifest, initial_pool/initial_registry. `installed_runtime_root` and
    `instance_id` are explicit current operator inputs, never historical defaults.

    An INPUTS_VERIFIED result means only that the fixed server entry may perform
    its final hardware/import/authority recheck. P1 is fail-closed until its
    independent real-model, data and Source consumers are implemented; merely
    presenting hand-authored passed=true artifacts cannot unlock it.
    """
    if stage not in STAGES:
        raise ValueError('unknown stage; explicit P0/P1 branch required')
    if type(now_unix) not in (int, float) or not math.isfinite(now_unix):
        raise ValueError('explicit finite current time required')
    if not isinstance(evidence, dict):
        raise ValueError('named evidence references required')
    controlled_stage = 'CONTROLLED_P0_' + stage[-1]
    blockers, checked, values = [], {}, {}

    def check(code, action):
        try:
            value = action()
            checked[code] = True
            return value
        except (ValueError, KeyError, TypeError, AttributeError, OSError, ImportError) as exc:
            checked[code] = False
            blockers.append(dict(code=code, detail=str(exc)))
            return None

    cpu = check('CPU_VALIDATION', lambda: _verify_cpu_artifacts(
        Path(workspace).resolve(), evidence, controlled_stage))
    for name in ('batch_manifest', 'native_manifest', 'patch_audit', 'patch_manifest',
                 'initial_pool', 'initial_registry'):
        values[name] = check(name.upper(), lambda name=name: _read_ref(evidence.get(name), name))
    manifest = values['batch_manifest'][0] if values['batch_manifest'] else None
    native = values['native_manifest'][0] if values['native_manifest'] else None
    runtime = None
    if manifest is not None:
        check('COMPLETE_NUMERICAL_PAIR_RECIPE', lambda: _validate_pair_scope(manifest, controlled_stage))
    if native is not None:
        from .v8_schema10_native_factory import validate_native_attachment
        runtime = check('NATIVE_ASSETS', lambda: validate_native_attachment(native, allow_unmeasured=True))
    if manifest is not None and runtime is not None:
        check('MODEL_GEOMETRY', lambda: validate_p0_model_geometry(manifest, runtime))
    if runtime is not None:
        from .v8_schema10_native_factory import verify_installed_runtime_sources
        root = evidence.get('installed_runtime_root')
        if not isinstance(root, str) or not root:
            blockers.append(dict(code='INSTALLED_RUNTIME_FILES', detail='explicit patched import root missing'))
        else:
            check('INSTALLED_RUNTIME_FILES', lambda: verify_installed_runtime_sources(runtime, root))
    if values['patch_audit'] and values['patch_manifest']:
        from .cacheblend_patch import validate_native_patch_audit
        check('PATCH_CHAIN', lambda: validate_native_patch_audit(
            values['patch_audit'][0], values['patch_manifest'][1]))
    if manifest is not None and native is not None and runtime is not None and cpu is not None:
        def validate_bindings():
            from .p0_batch_v2 import validate_p0_batch
            from .source_comparison_v2 import runtime_binding_digest
            from .source_store_v2 import parse_target_catalog_v2
            if not values['initial_pool'] or not values['initial_registry'] or not values['patch_audit']:
                raise ValueError('initial pool/registry/patch evidence missing')
            if (manifest['native_manifest_sha256'] != evidence['native_manifest']['sha256']
                    or values['patch_audit'][0]['cacheblend_patch_sha256'] != native['binding']['patch_sha256']):
                raise ValueError('batch/native/patch cross-file binding differs')
            source = runtime['source_provenance']
            if (source['code_commit'] != cpu['code_commit']
                    or native['binding']['code_commit'] != cpu['code_commit']):
                raise ValueError('CPU/runtime code revision differs')
            pool = parse_target_catalog_v2(values['initial_pool'][0])
            if (pool['config']['model_signature'] != source['model_signature']
                    or pool['config']['tokenizer_hash'] != source['tokenizer_hash']):
                raise ValueError('pool and model/tokenizer identities differ')
            actual = dict(code_commit=cpu['code_commit'], runtime_digest=runtime_binding_digest(),
                patch_sha256=native['binding']['patch_sha256'], model_signature=source['model_signature'],
                tokenizer_hash=source['tokenizer_hash'], instance_id=evidence.get('instance_id'),
                gpu_uuid=runtime['cost_provenance']['gpu'],
                input_manifest_sha256=manifest['binding']['input_manifest_sha256'],
                initial_pool_sha256=evidence['initial_pool']['sha256'],
                initial_registry_sha256=evidence['initial_registry']['sha256'])
            return validate_p0_batch(manifest, actual_binding=actual, now_unix=now_unix)
        check('SCOPED_AUTHORITY_AND_BATCH_BINDINGS', validate_bindings)
    else:
        blockers.append(dict(code='SCOPED_AUTHORITY_AND_BATCH_BINDINGS',
                             detail='concrete CPU/native/batch prerequisites incomplete'))
    if stage.startswith('P1_'):
        # Do not invent a consumer for P1 data/qualification while only a P0
        # diagnostic runner exists. Material presence is recorded, not trusted.
        names = ('p0_real_model_evidence', 'qualified_cohort', 'source_construction_manifest',
                 's0_recipe' if stage == 'P1_E' else 'paired_m1_birth_recipe')
        for name in names:
            check(name.upper(), lambda name=name: _read_ref(evidence.get(name), name))
        blockers.append(dict(code='P1_QUALIFICATION_CONSUMER_NOT_IMPLEMENTED',
            detail='Need independent P0-' + stage[-1] + ' raw numerical verification, lineage-isolated '
                   'original cases, historical Source proofs and bounded S0/E/M1 construction; file presence is not PASS'))
    ready = not blockers
    return dict(kind='p0_stage_input_readiness_v2', stage=stage,
        status='INPUTS_VERIFIED_RUNTIME_RECHECK_REQUIRED' if ready else 'BLOCKED',
        controlled_p0_inputs_ready=ready and stage.startswith('CONTROLLED_P0_'),
        local_cpu_inputs_verified=cpu is not None,
        checks=checked, blockers=blockers, local_validation=cpu,
        requires_natural_p1_cohort=stage.startswith('P1_'),
        actual_gpu_observed=False, server_recheck_required=True,
        gpu_execution_allowed=False, automatic_rental_allowed=False,
        P0_E_qualified=False, P0_M_qualified=False,
        P1_E_execution_allowed=False, P1_M_execution_allowed=False,
        paper_evidence=False, locked_test_accessed=False)

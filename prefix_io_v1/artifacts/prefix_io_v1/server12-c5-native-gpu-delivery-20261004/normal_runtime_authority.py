"""Append per-arm live metadata from the user's persistent project GPU grant.

No GPU job, model, native backend, SDK compilation or library loading is started.
The frozen normal controller remains the sole scope/launch/guard verifier. This
helper does not modify that controller, its common source lock, or base permission.
"""
from copy import deepcopy
from hashlib import sha256
import argparse
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
CALIBRATION = 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004'
AMENDMENT = 'artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004/PROJECT_GPU_AUTHORIZATION_AMENDMENT.json'
AMENDMENT_SHA = '8acab9a6d1e67d8c5237f2ff1736cff6e9248c7d359c944e80648c7cba5895ec'
HUMAN_INSTRUCTION = '授权gpu，只要有gpu就用，以后无需授权'
COMMON_SHA = 'aaa67309fab4c6dc7fe07853a31c3a99cdef727408da9fab604201bf95416d4d'
BASE = 'experiments/prefix_io_v1/configs/permissions.yaml'
BASE_REF = dict(path=BASE, bytes=1307,
    sha256='795d24f9955379653e34ffbbfd7614ad9ef96ad0ac9a208c6b4baefd994fac50')
CONTROL_REF = dict(path=D + '/control_p4_single_file.py', bytes=55526,
    sha256='1bdc4428d8eaf9bb57d38b31717a08de2daea98edec1d0b74de634f30c2f0510')
RESOURCE_REF = dict(path=CALIBRATION + '/gpu_entry_binding.py', bytes=33971,
    sha256='9a434ce4c9a47088df0582d32640174a71d69271b3c2569975d1ddcbde9e23fd')
FORBIDDEN = ('torch', 'vllm', 'py_kvcache')
IMPORT_ATTEMPTS = []


class CPUOnlyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in FORBIDDEN):
            IMPORT_ATTEMPTS.append(fullname)
            raise RuntimeError('authority metadata must not import a GPU/model backend: ' + fullname)
        return None


def require(value, reason):
    if not value:
        raise ValueError('NORMAL_AUTHORITY_METADATA_REJECTED: ' + reason)


def safe(root, relative):
    require(type(relative) is str and relative and ':' not in relative and '\\' not in relative
        and not relative.startswith('/') and all(p not in ('', '.', '..') for p in relative.split('/')),
        'project-relative actual evidence path')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'source/evidence symlink refused')
    require(path.resolve().is_relative_to(root), 'source/evidence outside project')
    return path


def ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), 'actual file required: ' + relative)
    with path.open('rb') as stream:
        digest = sha256()
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return dict(path=relative, bytes=path.stat().st_size, sha256=digest.hexdigest())


def checked(root, expected):
    require(type(expected) is dict and set(expected) == {'path', 'bytes', 'sha256'} and
        type(expected['bytes']) is int and ref(root, expected['path']) == expected, 'actual source/evidence pin drift')
    return expected


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, 'nonfinite JSON'))


def load(root, expected, name):
    checked(root, expected)
    path = safe(root, expected['path'])
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
        checked(root, expected)
        return module
    finally:
        sys.modules.pop(name, None)


def write_new(root, relative, value):
    path = safe(root, relative)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return ref(root, relative)


def validate_amendment(document, root):
    expected = dict(schema='c5_project_persistent_gpu_authorization_amendment_v1',
        root=str(root), issuer='human_user', authorization=True, human_instruction=HUMAN_INSTRUCTION,
        original_max_gpu_seconds=28800, original_budget_unchanged=True,
        source='direct_current_user_instruction', base_permission_ref=BASE_REF,
        GPU_authority_scope='current_frozen_project_experiments_with_existing_stage_gates')
    require(type(document) is dict and all(type(document.get(key)) is type(value) and
        document.get(key) == value for key, value in expected.items()),
        'the exact current direct persistent user instruction and original budget must be pinned')
    return document


def issue(root, mode):
    root = Path(root).resolve(strict=True)
    require(not root.is_symlink() and mode in ('off', 'shadow', 'on'), 'actual project and finite arm required')
    require(not IMPORT_ATTEMPTS and not any(name == base or name.startswith(base + '.')
        for name in sys.modules for base in FORBIDDEN), 'CPU-only clean process required')
    amendment_ref = ref(root, AMENDMENT)
    require(amendment_ref['sha256'] == AMENDMENT_SHA and amendment_ref['bytes'] == 1114,
        'persistent authorization provenance changed')
    validate_amendment(strict_json(safe(root, AMENDMENT).read_bytes()), root)
    checked(root, BASE_REF)
    C = load(root, CONTROL_REF, '_server12_frozen_normal_authority_controller')
    require(C.D == D and C.CALIBRATION == CALIBRATION and C.BASE_PERMISSION == BASE,
        'frozen normal controller source domain changed')
    common_ref = C.ref(C.LOCK, root)
    require(common_ref['sha256'] == COMMON_SHA, 'actual frozen normal common source closure changed')
    _lock, refs = C.common_rows(root, verify_bytes=True)
    require(refs[CONTROL_REF['path']] == CONTROL_REF and refs[BASE] == BASE_REF and
        refs[RESOURCE_REF['path']] == RESOURCE_REF, 'actual pinned shared sources/permission required')
    config_relative = C.config_path(mode)
    config_ref = C.ref(config_relative, root)
    config = C.read(config_relative, root)
    C.validate_configuration_document(root, config, config_relative)
    prior = C.verify_previous_qualification(root, config, refs)
    receipt = C.load_receipt(root)
    require(receipt.binding_ref.mapping() == refs[C.BINDING] and receipt.signature[1] == config['gpu_uuid'],
        'real typed current-domain calibration and configured GPU required')
    base = C.permission_document(root, BASE)
    require(C.number(base.get('max_gpu_hours'), 'original cumulative hours', 0.01) <= 8 and
        all(base.get(key) is False for key in C.DENIED if key != 'allow_model_downloads'),
        'original budget and non-GPU system/delete/payment/rental prohibitions must be retained')
    require(base.get('approved_experiment_root') == str(C.safe('experiments/prefix_io_v1/runs', root)),
        'actual original guard evidence root')
    ledger_before_ref = C.ref(C.LEDGER, root)
    ledger = C.read(C.LEDGER, root)
    require(ledger.get('active_reservation') is None, 'finish and drain previous original guard first')
    used = C.number(ledger.get('gpu_wall_seconds'), 'actual cumulative GPU usage')
    maximum = min(28800, base['max_gpu_hours'] * 3600)
    require(used + 320 <= maximum, 'per-arm reserve exceeds remaining original budget')
    paths = [D + '/LIVE_CONTEXT_' + mode + '.json', D + '/HUMAN_GPU_GRANT_' + mode + '.json',
        D + '/EFFECTIVE_GPU_PERMISSION_' + mode + '.json', C.authority_path(mode)]
    require(not any(C.safe(path, root).exists() for path in paths) and
        not C.safe('experiments/prefix_io_v1/runs/' + C.label(mode), root).exists() and
        not C.safe(D + '/LAUNCH_INTENT_' + mode + '.json', root).exists(),
        'append-only metadata and one actual attempt per arm; existing state must be reviewed')
    B = load(root, RESOURCE_REF, '_server12_original_real_resource_probe')
    resources = B.probe_resources(root)
    require(B._resource_shape(resources) == config['gpu_uuid'], 'actual live site/device differs from frozen normal domain')
    sdk_ref = refs[C.SITE_SDK]
    sdk = load(root, sdk_ref, '_server12_original_real_sdk_asset_audit')
    _helper, sdk_assets, sdk_rows = sdk.load_site_assets(root, refs)
    require(type(sdk_assets.get('driver')) is dict and all(C.exact(sdk_assets['driver'].get(key), value)
        for key, value in C.HOST_DRIVER_REF.items()), 'real current typed host driver drift')
    observed = time.time()
    self_ref = C.ref(Path(__file__).resolve().relative_to(root).as_posix(), root)
    for reference in (amendment_ref, BASE_REF, CONTROL_REF, RESOURCE_REF, config_ref, common_ref, sdk_ref):
        C.verify_reference(reference, root)
    require(C.ref(C.LEDGER, root) == ledger_before_ref, 'original ledger changed during metadata preparation')
    context = dict(schema='c5_normal_live_context_v1', origin='live_site_readonly_context',
        root=str(root), mode=mode, label=C.label(mode), gpu_uuid=config['gpu_uuid'],
        config_ref=config_ref, source_lock_ref=common_ref, binding_ref=refs[C.BINDING],
        previous_qualification_ref=config['previous_qualification_ref'], seconds_limit=300, reserved_seconds=320,
        resources=resources, observed_unix=observed, valid_until_unix=observed + 3600,
        project_authorization_amendment_ref=amendment_ref, metadata_helper_ref=self_ref,
        current_sdk_asset_audit=dict(adapter_ref=sdk_ref, helper_ref=sdk_rows[sdk.HELPER],
            inventory_ref=sdk_rows[sdk.INVENTORY], compiler_proof_ref=sdk_rows[sdk.PROOF],
            driver_ref=sdk_assets['driver'], actual_full_byte_sha256_verified=True,
            GPU_operations=0, shared_object_loaded=False),
        ledger_before_ref=ledger_before_ref, ledger_gpu_wall_seconds_before=used,
        GPU_operations=0, native_execution_qualification=False)
    C.validate_live_interval(context, require_current=True)
    context_ref = write_new(root, paths[0], context)
    grant_expected = dict(schema='c5_normal_explicit_human_grant_v1', issuer='human_user', authorization=True,
        root=str(root), mode=mode, label=C.label(mode), gpu_uuid=config['gpu_uuid'],
        context_ref=context_ref, config_ref=config_ref, source_lock_ref=common_ref,
        binding_ref=refs[C.BINDING], previous_qualification_ref=config['previous_qualification_ref'],
        seconds_limit=300, reserved_seconds=320, max_attempts=1, max_processes=1,
        instruction_is_gpu_execution_authorization=True)
    grant = dict(grant_expected, human_instruction=HUMAN_INSTRUCTION,
        project_authorization_amendment_ref=amendment_ref, metadata_helper_ref=self_ref,
        reused_project_authorization=True, current_mode_requires_all_original_stage_gates=True)
    grant_ref = write_new(root, paths[1], grant)
    permission_binding = dict(grant_expected)
    for key in ('issuer', 'authorization', 'schema', 'instruction_is_gpu_execution_authorization'):
        permission_binding.pop(key)
    permission_binding['human_grant_ref'] = grant_ref
    permission = deepcopy(base)
    # Narrow the baseline's optional download/auxiliary capabilities for this
    # particular finite arm; preserve the actual base bytes, roots and budget.
    permission.update({key: False for key in C.DENIED})
    permission.update(allow_gpu_runs=True, approved_gpu_ids=[config['gpu_uuid']],
        approved_auxiliary_storage=None,
        normal_mode_binding=permission_binding, project_authorization_amendment_ref=amendment_ref,
        metadata_helper_ref=self_ref)
    permission_ref = write_new(root, paths[2], permission)
    authority = dict(schema='c5_normal_mode_site_authority_v1', status='LIVE_HUMAN_BOUND',
        origin='live_site_binding', synthetic=False, root=str(root), mode=mode, label=C.label(mode),
        gpu_uuid=config['gpu_uuid'], config_ref=config_ref, source_lock_ref=common_ref,
        binding_ref=refs[C.BINDING], previous_qualification_ref=config['previous_qualification_ref'],
        seconds_limit=300, reserved_seconds=320, max_attempts=1, max_processes=1,
        context_ref=context_ref, human_grant_ref=grant_ref, effective_permission_ref=permission_ref,
        base_permission_ref=BASE_REF, guard_source_ref=refs[C.GUARD])
    authority_ref = write_new(root, paths[3], authority)
    require(C.load_authority(root, config) == authority, 'frozen controller rejected actual generated authority')
    require(C.ref(C.LEDGER, root) == ledger_before_ref, 'CPU metadata must not consume or change GPU budget')
    checked(root, self_ref)
    checked(root, amendment_ref)
    require(not IMPORT_ATTEMPTS and not any(name == base or name.startswith(base + '.')
        for name in sys.modules for base in FORBIDDEN), 'no model/GPU backend imports allowed')
    return dict(status='PASS_PERSISTENT_PROJECT_GRANT_LIVE_NORMAL_METADATA', mode=mode,
        label=C.label(mode), gpu_uuid=config['gpu_uuid'], config_ref=config_ref,
        source_lock_ref=common_ref, binding_ref=refs[C.BINDING], context_ref=context_ref,
        human_grant_ref=grant_ref, effective_permission_ref=permission_ref, authority_ref=authority_ref,
        project_authorization_amendment_ref=amendment_ref, metadata_helper_ref=self_ref,
        previous_qualification_reexecuted=(prior is not None), GPU_operations=0,
        actual_model_processes_started=0, actual_GPU_jobs_started=0, gpu_launch_performed=False,
        original_gpu_seconds_limit=maximum, actual_gpu_seconds_used=used,
        remaining_original_gpu_seconds=maximum-used, next_job_reserve_seconds=320,
        original_gpu_ledger_unchanged=True, native_execution_verified=False,
        normal_runtime_condition_qualified=False, full_runtime_cost_qualified=False,
        strategy_effect_verified=False, P4_completed=False, forbidden_import_attempts=IMPORT_ATTEMPTS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--mode', required=True, choices=('off', 'shadow', 'on'))
    args = parser.parse_args(argv)
    sys.meta_path.insert(0, CPUOnlyImports())
    try:
        print(json.dumps(issue(args.root, args.mode), sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps(dict(status='NORMAL_AUTHORITY_METADATA_REJECTED', reason=str(exc),
            error_type=type(exc).__name__, GPU_operations=0, actual_GPU_jobs_started=0,
            native_execution_verified=False, normal_runtime_condition_qualified=False,
            strategy_effect_verified=False, forbidden_import_attempts=IMPORT_ATTEMPTS),
            sort_keys=True, ensure_ascii=False, allow_nan=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())

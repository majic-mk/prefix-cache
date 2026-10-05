"""CPU preparation and separate, fail-closed authority for finite normal arms.

Freeze and prepare import no model/GPU code and do not authorize execution.
Only a fresh human/site/source/mode grant can reach the unchanged original
GPU budget guard. Calibration receipts are replayed through their public API.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'
BASE_NORMAL = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
BASE_NORMAL_LOCK_SHA = 'aaa67309fab4c6dc7fe07853a31c3a99cdef727408da9fab604201bf95416d4d'
PROTOCOL = D + '/PROTOCOL.json'
AMENDMENT = 'artifacts/prefix_io_v1/server12-c5-native-gpu-delivery-20261004/PROJECT_GPU_AUTHORIZATION_AMENDMENT.json'
AMENDMENT_SHA = '8acab9a6d1e67d8c5237f2ff1736cff6e9248c7d359c944e80648c7cba5895ec'
V6 = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004'
C = V6 + '/common_candidate'
CALIBRATION = 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004'
SITE_SDK = CALIBRATION + '/site_sdk_binding.py'
SITE_SDK_SHA = '31cb4973e9021f7849f703c2db93961bd2c03d3dd35b5b672c72da25334ffc8b'
HOST_DRIVER_REF = dict(path='/usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05', bytes=96276264,
    sha256='f27223c58d4c0d2ead3c2d747eb30a530c449f89c42e16b23e9bef04f3e6dc2e')
CALIBRATION_CANONICAL = CALIBRATION + '/p4_single_file_receipt.py'
CALIBRATION_CANONICAL_SHA = '8deb840a759249f339015634bc58adb95e986e7c787ff1b7f71b698fa4ec7a0d'
G_RECEIPT = C + '/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py'
RECEIPT = BASE_NORMAL + '/p4_single_file_receipt.py'
HISTORY = BASE_NORMAL + '/historical_calibration_binding.py'
LEDGER_SNAPSHOT = BASE_NORMAL + '/SERVER12_LEDGER_SNAPSHOT.json'
CALIBRATION_BINDING = CALIBRATION + '/NATIVE_SINGLE_FILE_BINDING.json'
CALIBRATION_BINDING_SHA = 'd79be29c8139a1cd42577b54bdd33e00bedcaa1fc6dbf5100df682b32ca3f42b'
G_CANONICAL_SHA = '50cc763a191a66cb002ad1e6091f35fa2af9d413b37ca251af0487686ab73a82'
CANONICAL_SHA = '3ef359a623d02d9ed220c7e44a0d5fbfa3b4bcbe0b598b8523b800d5aff8a73e'
GUARD_SHA = '3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
GUARD = 'experiments/prefix_io_v1/scripts/run_gpu_stage.py'
BASE_PERMISSION = 'experiments/prefix_io_v1/configs/permissions.yaml'
LOCK = D + '/COMMON_SOURCE_LOCK.json'
BINDING = BASE_NORMAL + '/NORMAL_RUNTIME_BINDING.json'
MANIFEST = 'artifacts/prefix_io_v1/server11-native-cost-v1-20261003/SSD_INPUT_MANIFEST.json'
STORAGE = 'experiments/prefix_io_v1/runs/server11-native-cost-01-private-storage'
PURPOSE = 'FIXED_NATIVE_COST_REPEATABILITY_DIAGNOSTIC_ONLY'
SCOPE = 'server12_c5_normal_repeatability_v1'
MODES = ('off',)
DENIED = ('allow_model_downloads', 'allow_driver_or_system_changes',
    'allow_shared_data_deletion', 'allow_payment', 'allow_new_cloud_rental')
UUID_RE = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
LOCK_SCHEMA = 'c5_repeatability_common_source_lock_v1'


def require(value, message):
    if not value:
        raise ValueError('NORMAL_ENTRY_REJECTED: ' + message)


def exact(value, expected):
    return type(value) is type(expected) and value == expected


def project(root=None):
    return Path(ROOT if root is None else root).resolve(strict=True)


def safe(relative, root=None):
    root = project(root)
    require(type(relative) is str and relative and '\\' not in relative and ':' not in relative
        and not relative.startswith('/') and all(part not in ('', '.', '..') for part in relative.split('/')),
        'explicit project-relative POSIX path required')
    path = root / relative
    cursor = path
    while cursor != root:
        require(not cursor.is_symlink(), 'symlink evidence refused: ' + relative)
        cursor = cursor.parent
    require(path.resolve().is_relative_to(root), 'outside original workspace')
    return path


def ref(relative, root=None):
    path = safe(relative, root)
    require(path.is_file(), 'regular reference file required: ' + relative)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=relative, bytes=path.stat().st_size, sha256=digest)


def reference_shape(value, root=None):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'} and
        type(value['bytes']) is int and value['bytes'] >= 0 and type(value['sha256']) is str and
        re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None, 'exact typed immutable reference')
    safe(value['path'], root)
    return value


def verify_reference(value, root=None):
    reference_shape(value, root)
    require(ref(value['path'], root) == value, 'reference changed: ' + value['path'])
    return value


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(relative, root=None):
    path = safe(relative, root)
    require(path.is_file() and path.stat().st_size <= 32 * 1024**2, 'bounded JSON evidence')
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, 'nonfinite JSON: ' + value))


def put(name, value, root=None):
    relative = D + '/' + name
    with safe(relative, root).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return ref(relative, root)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_module(root, reference, name):
    verify_reference(reference, root)
    path = safe(reference['path'], root)
    require(path.suffix == '.py' and 0 < path.stat().st_size <= 4 * 1024**2, 'nonempty bounded source module')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == reference['sha256'], 'source drift before compile')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    require(name not in sys.modules, 'fresh CPU source module')
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), 'exec', dont_inherit=True), module.__dict__)
        verify_reference(reference, root)
        return module
    finally:
        sys.modules.pop(name, None)


def rows_map(rows, root, *, verify_bytes=False):
    require(type(rows) is list and 1 <= len(rows) <= 10000, 'bounded nonempty source closure')
    result = {}
    for row in rows:
        reference_shape(row, root)
        require(row['path'] not in result, 'duplicate source path')
        if verify_bytes:
            verify_reference(row, root)
        result[row['path']] = row
    return result


def metadata_reference(value, root=None):
    """Normalize real project metadata; only the exact current host pin is external.

    Recognition alone proves no host bytes or native execution. Freeze separately
    calls the pinned original SDK asset auditor before using the external marker.
    The original JSON value remains unchanged in its full-byte frozen document.
    """
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'} and
        type(value.get('path')) is str and type(value.get('bytes')) is int and value['bytes'] >= 0 and
        type(value.get('sha256')) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None,
        'exact typed immutable metadata reference')
    path = value['path']
    if path == HOST_DRIVER_REF['path']:
        require(all(exact(value[key], item) for key, item in HOST_DRIVER_REF.items()),
            'only the exact current pinned host driver metadata is supported')
        return None
    if path.startswith('/') or Path(path).is_absolute():
        parts = path.replace('\\', '/').split('/')
        require(all(part not in ('', '.', '..') for part in (parts[1:] if path.startswith('/') else parts)),
            'canonical lexical absolute metadata path required')
        absolute = Path(path)
        require(absolute.is_absolute(), 'unknown external metadata reference')
        try:
            relative = absolute.relative_to(project(root)).as_posix()
        except ValueError:
            require(False, 'unknown external metadata reference')
        actual = safe(relative, root)
        require(actual == absolute and actual.resolve(strict=True) == absolute,
            'absolute project metadata must retain its exact canonical path')
        normalized = dict(value, path=relative)
        verify_reference(normalized, root)
        return normalized
    return reference_shape(value, root)


def number(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, 'finite number: ' + name)
    return value


def validate_live_interval(context, launch_unix=None, *, require_current=False):
    number(context.get('observed_unix'), 'live context timestamp', 1)
    number(context.get('valid_until_unix'), 'live context expiry', 1)
    require(context['observed_unix'] <= time.time() + 5 and
        context['valid_until_unix'] > context['observed_unix'] and
        context['valid_until_unix'] - context['observed_unix'] <= 3600,
        'bounded genuine site context validity')
    if require_current:
        require(context['valid_until_unix'] > time.time(), 'live site context expired')
    if launch_unix is not None:
        number(launch_unix, 'actual launch timestamp', 1)
        require(context['observed_unix'] <= launch_unix <= context['valid_until_unix'],
            'launch outside authorized live context interval')


def permission_document(root, relative):
    raw = safe(relative, root).read_bytes()
    require(0 < len(raw) <= 1024**2, 'bounded permission document')
    if relative.endswith('.json'):
        return read(relative, root)
    import yaml
    value = yaml.safe_load(raw.decode('utf-8'))
    require(type(value) is dict, 'permission mapping')
    return value


def closure_digest(refs):
    return hashlib.sha256(json.dumps([refs[path] for path in sorted(refs)], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def suffix(index):
    require(type(index) is int and 0 <= index < 3, 'fixed diagnostic index 0..2')
    return 'off%02d' % (index + 1)


def label(index):
    return 'server12-c5-native-repeat-' + suffix(index)


def config_path(index):
    return D + '/CONFIG_' + suffix(index) + '.json'


def authority_path(index):
    return D + '/AUTHORITY_' + suffix(index) + '.json'


def result_path(index):
    return D + '/DIAGNOSTIC_' + suffix(index) + '.json'


def proof_path(index, phase):
    require(phase in ('before', 'launch', 'after'), 'explicit proof phase')
    return D + '/SOURCE_' + suffix(index) + '_' + phase.upper() + '.json'


def config_relative(root, path):
    root = project(root)
    if isinstance(path, Path) and not path.is_absolute():
        path = path.as_posix()
    elif isinstance(path, Path) or (type(path) is str and Path(path).is_absolute()):
        try:
            path = Path(path).relative_to(root).as_posix()
        except ValueError:
            require(False, 'configuration outside project')
    require(path in [config_path(index) for index in range(3)], 'fixed diagnostic configuration')
    safe(path, root)
    return path


def generated_mode_path(relative):
    return relative.startswith(D + '/') and relative.removeprefix(D + '/').startswith(
        ('CONFIG_', 'AUTHORITY_', 'SCOPE_', 'SOURCE_', 'LIVE_CONTEXT_', 'LEDGER_BEFORE_',
         'HUMAN_GPU_GRANT_', 'EFFECTIVE_GPU_PERMISSION_', 'DIAGNOSTIC_', 'LAUNCH_', 'GPU_GUARD_'))


def common_rows(root, *, verify_bytes=False):
    lock = read(LOCK, root)
    require(lock.get('schema') == LOCK_SCHEMA and lock.get('scope') == SCOPE and
        exact(lock.get('GPU_operations'), 0) and lock.get('normal_runtime_qualified') is False and
        lock.get('gpu_authority_issued') is False, 'fixed diagnostic common lock')
    refs = rows_map(lock.get('files'), root, verify_bytes=verify_bytes)
    require(LOCK not in refs and LEDGER not in refs and all(not generated_mode_path(path) for path in refs),
        'live mode metadata and ledger outside immutable shared closure')
    required = (BINDING, RECEIPT, HISTORY, LEDGER_SNAPSHOT, GUARD, BASE_PERMISSION, MANIFEST, SITE_SDK,
        G_RECEIPT, CALIBRATION_CANONICAL, PROTOCOL, D + '/control_p4_single_file.py',
        D + '/run_p4_single_file_experiment.py', D + '/process_entry_context.py',
        D + '/verify_repeatability.py', D + '/notification_runtime_adapter.py',
        D + '/observation_cpu_cost.py', C + '/native_full_step_collector.py', C + '/single_file_runtime_binding.py')
    for path in required:
        require(path in refs, 'required real source absent: ' + path)
    require(refs[RECEIPT]['sha256'] == CANONICAL_SHA and refs[GUARD]['sha256'] == GUARD_SHA and
        refs[G_RECEIPT]['sha256'] == G_CANONICAL_SHA and
        refs[CALIBRATION_CANONICAL]['sha256'] == CALIBRATION_CANONICAL_SHA and
        refs[SITE_SDK]['sha256'] == SITE_SDK_SHA, 'unchanged actual public/guard/SDK sources')
    base_ref = lock.get('base_normal_lock_ref')
    reference_shape(base_ref, root); verify_reference(base_ref, root)
    require(base_ref['path'] == BASE_NORMAL + '/COMMON_SOURCE_LOCK.json' and
        base_ref['sha256'] == BASE_NORMAL_LOCK_SHA, 'exact original normal closure ancestry')
    base = read(base_ref['path'], root)
    require(all(refs.get(row['path']) == row for row in base['files']), 'retain every original normal leaf')
    host = lock.get('current_host_asset_verification')
    require(host == base.get('current_host_asset_verification') and host.get('driver_ref') == HOST_DRIVER_REF,
        'unchanged original fully audited current host metadata')
    verify_reference(refs[PROTOCOL], root)
    protocol = read(PROTOCOL, root)
    fixed = dict(schema='c5_fixed_workload_repeatability_protocol_v1', scope=SCOPE,
        repetitions=3, seed=2829, prompt_first_token=28100, cost_upper_ns=16238752, step_budget_ns=13171328)
    require(all(exact(protocol.get(key), value) for key, value in fixed.items()), 'frozen finite protocol; no refit')
    return lock, refs


def canonical_module(root, refs):
    reference = refs[RECEIPT]; verify_reference(reference, root)
    require(reference['sha256'] == CANONICAL_SHA, 'unchanged actual public receipt source')
    name = 'prefix_io_control.p4_single_file_receipt'
    module = sys.modules.get(name)
    if module is None:
        path = safe(RECEIPT, root)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), module.__dict__)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    require(module.__name__ == name and Path(module.__file__).resolve() == safe(RECEIPT, root).resolve() and
        module.ExactSingleFileReceipt.__module__ == name and
        Path(module.load_verified_single_file.__code__.co_filename).resolve() == safe(RECEIPT, root).resolve(),
        'one actual source-bound public class and loader')
    verify_reference(reference, root)
    return module


def load_receipt(root=None, binding_relative=None):
    root = project(root)
    require(binding_relative in (None, BINDING), 'only existing real normal binding')
    _, refs = common_rows(root)
    module = canonical_module(root, refs)
    receipt = module.load_verified_single_file(root, BINDING)
    require(type(receipt) is module.ExactSingleFileReceipt and receipt.condition_only is True and
        receipt.production_qualified is False and receipt.binding_ref.mapping() == refs[BINDING] and
        type(receipt.cost_upper_ns) is int and receipt.cost_upper_ns == 16238752 and
        type(receipt.step_budget_ns) is int and receipt.step_budget_ns == 13171328,
        'actual public typed receipt; frozen numerical gate unchanged')
    return receipt


def expected_config(root, index, gpu_uuid, previous=None):
    suffix(index)
    require(previous is None and type(gpu_uuid) is str and UUID_RE.fullmatch(gpu_uuid) is not None,
        'off-only diagnostic with real GPU; no predecessor promotion')
    return dict(root=str(project(root)), label=label(index), gpu_uuid=gpu_uuid, purpose=PURPOSE,
        seconds_limit=300, reserved_seconds=320, active_ledger=LEDGER, storage=STORAGE,
        out='experiments/prefix_io_v1/runs/' + label(index) + '/details', overlay_relative=C + '/source',
        collector_relative=C + '/native_full_step_collector.py',
        runtime_binding_relative=C + '/single_file_runtime_binding.py', source_lock=LOCK,
        input_manifest=MANIFEST, binding_relative=BINDING, mode='off', diagnostic_index=index,
        protocol_ref=ref(PROTOCOL, root), previous_qualification_ref=None, authority_relative=authority_path(index))


def validate_configuration_document(root, config, relative):
    require(type(config) is dict and type(config.get('diagnostic_index')) is int, 'explicit typed diagnostic index')
    index = config['diagnostic_index']
    expected = expected_config(root, index, config.get('gpu_uuid'))
    require(relative == config_path(index) and set(config) == set(expected) and
        all(exact(config.get(key), value) for key, value in expected.items()), 'fixed finite configuration')
    return index


def verify_previous_qualification(root, config, refs, _depth=0):
    require(config['mode'] == 'off' and config['previous_qualification_ref'] is None and
        type(config['diagnostic_index']) is int and 0 <= config['diagnostic_index'] < 3,
        'repeatability does not inherit or issue a normal qualification')
    return None


def prepare_all(root=None):
    root = project(root); _, refs = common_rows(root, verify_bytes=True)
    receipt = load_receipt(root)
    rows = []
    for index in range(3):
        config = expected_config(root, index, receipt.signature[1])
        validate_configuration_document(root, config, config_path(index))
        rows.append(put('CONFIG_' + suffix(index) + '.json', config, root))
    return dict(status='PASS_FIXED_DIAGNOSTIC_CONFIG_CPU_PREPARATION', config_refs=rows,
        source_lock_ref=ref(LOCK, root), binding_ref=refs[BINDING], public_loader_calls=1,
        GPU_operations=0, gpu_launch_allowed=False, authority_issued=False, runtime_condition_qualified=False)


def prepare(index, root=None):
    root = project(root); suffix(index); _, refs = common_rows(root, verify_bytes=True)
    receipt = load_receipt(root); config = expected_config(root, index, receipt.signature[1])
    row = put('CONFIG_' + suffix(index) + '.json', config, root)
    return dict(status='PASS_FIXED_DIAGNOSTIC_CONFIG_CPU_PREPARATION', config_ref=row,
        source_lock_ref=ref(LOCK, root), binding_ref=refs[BINDING], GPU_operations=0, gpu_launch_allowed=False)


def load_authority(root, config):
    index = config['diagnostic_index']; tag = suffix(index)
    authority = read(authority_path(index), root)
    fixed = dict(schema='c5_repeatability_site_authority_v1', status='LIVE_HUMAN_BOUND',
        origin='live_site_binding', synthetic=False, root=str(project(root)), mode='off', diagnostic_index=index,
        protocol_ref=config['protocol_ref'], label=label(index), gpu_uuid=config['gpu_uuid'],
        config_ref=ref(config_path(index), root), source_lock_ref=ref(LOCK, root), binding_ref=ref(BINDING, root),
        previous_qualification_ref=None, seconds_limit=300, reserved_seconds=320, max_attempts=1, max_processes=1)
    keys = set(fixed) | {'context_ref', 'human_grant_ref', 'effective_permission_ref', 'base_permission_ref', 'guard_source_ref'}
    require(type(authority) is dict and set(authority) == keys and
        all(exact(authority.get(key), value) for key, value in fixed.items()), 'fresh actual diagnostic site authority')
    paths = dict(context_ref=D + '/LIVE_CONTEXT_' + tag + '.json',
        human_grant_ref=D + '/HUMAN_GPU_GRANT_' + tag + '.json',
        effective_permission_ref=D + '/EFFECTIVE_GPU_PERMISSION_' + tag + '.json',
        base_permission_ref=BASE_PERMISSION, guard_source_ref=GUARD)
    for key, path in paths.items():
        reference_shape(authority[key], root)
        require(authority[key]['path'] == path, 'fresh diagnostic reference path')
        verify_reference(authority[key], root)
    require(authority['guard_source_ref']['sha256'] == GUARD_SHA, 'original guard unchanged')
    context = read(authority['context_ref']['path'], root)
    context_expected = dict(schema='c5_repeatability_live_context_v1', origin='live_site_readonly_context',
        root=fixed['root'], mode='off', diagnostic_index=index, protocol_ref=config['protocol_ref'],
        label=label(index), gpu_uuid=config['gpu_uuid'], config_ref=fixed['config_ref'],
        source_lock_ref=fixed['source_lock_ref'], binding_ref=fixed['binding_ref'],
        previous_qualification_ref=None, seconds_limit=300, reserved_seconds=320)
    require(all(exact(context.get(key), value) for key, value in context_expected.items()), 'actual site context mismatch')
    validate_live_interval(context)
    resources = context.get('resources'); require(type(resources) is dict, 'actual readable site resources')
    for key, minimum in (('cpu_cores', 1), ('cpu_affinity_count', 1), ('available_memory_bytes', 32*1024**3),
        ('gpu_free_mib', 28000), ('primary_free_bytes', 8*1024**3+128*1024**2)):
        number(resources.get(key), key, minimum)
    require(resources.get('gpu_uuid') == config['gpu_uuid'] and resources.get('cpu_counters_readable') is True,
        'actual UUID and CPU counters')
    grant = read(authority['human_grant_ref']['path'], root)
    grant_expected = dict(schema='c5_repeatability_explicit_human_grant_v1', issuer='human_user', authorization=True,
        root=fixed['root'], mode='off', diagnostic_index=index, protocol_ref=config['protocol_ref'],
        label=label(index), gpu_uuid=config['gpu_uuid'], context_ref=authority['context_ref'],
        config_ref=fixed['config_ref'], source_lock_ref=fixed['source_lock_ref'], binding_ref=fixed['binding_ref'],
        previous_qualification_ref=None, seconds_limit=300, reserved_seconds=320, max_attempts=1, max_processes=1,
        instruction_is_gpu_execution_authorization=True)
    require(all(exact(grant.get(key), value) for key, value in grant_expected.items()) and
        grant.get('human_instruction') == '授权gpu，只要有gpu就用，以后无需授权', 'actual persistent direct GPU grant')
    amendment_ref = grant.get('project_authorization_amendment_ref')
    verify_reference(amendment_ref, root)
    require(amendment_ref['path'] == AMENDMENT and amendment_ref['sha256'] == AMENDMENT_SHA,
        'actual unchanged persistent amendment')
    amendment = read(AMENDMENT, root)
    require(amendment.get('issuer') == 'human_user' and amendment.get('authorization') is True and
        amendment.get('original_max_gpu_seconds') == 28800 and amendment.get('original_budget_unchanged') is True and
        amendment.get('human_instruction') == grant['human_instruction'], 'persistent grant cannot grow scope/budget')
    permission = permission_document(root, authority['effective_permission_ref']['path'])
    base = permission_document(root, BASE_PERMISSION)
    require(permission.get('allow_gpu_runs') is True and permission.get('approved_gpu_ids') == [config['gpu_uuid']] and
        number(permission.get('max_gpu_hours'), 'effective budget', 0.01) <= min(8, number(base.get('max_gpu_hours'), 'base budget', 0.01)),
        'original GPU UUID/budget retained')
    require(all(permission.get(key) == base.get(key) for key in ('approved_experiment_root', 'approved_dependency_root')) and
        permission.get('approved_experiment_root') == str(safe('experiments/prefix_io_v1/runs', root)) and
        all(permission.get(key) is False for key in DENIED) and permission.get('approved_auxiliary_storage') is None,
        'original roots; no system/download/delete/rental scope expansion')
    permission_binding = dict(grant_expected)
    for key in ('schema', 'issuer', 'authorization', 'instruction_is_gpu_execution_authorization'):
        permission_binding.pop(key)
    permission_binding['human_grant_ref'] = authority['human_grant_ref']
    require(permission.get('normal_mode_binding') == permission_binding and
        all(exact(permission['normal_mode_binding'].get(key), value) for key, value in permission_binding.items()),
        'effective permission binds actual protocol/index/config/source/grant')
    return authority


def verify_source_proof(root, config, authority, phase, refs):
    index = config['diagnostic_index']; document = read(proof_path(index, phase), root)
    expected = dict(schema='c5_repeatability_full_source_verification_v1', phase=phase,
        diagnostic_index=index, protocol_ref=config['protocol_ref'], source_lock_ref=ref(LOCK, root),
        config_ref=ref(config_path(index), root), authority_ref=ref(authority_path(index), root),
        binding_ref=ref(BINDING, root), human_grant_ref=authority['human_grant_ref'],
        permission_ref=authority['effective_permission_ref'], context_ref=authority['context_ref'],
        previous_qualification_ref=None, files_verified=len(refs), source_rows_sha256=closure_digest(refs), failed=[], GPU_operations=0)
    require(all(exact(document.get(key), value) for key, value in expected.items()), 'independent complete actual source proof')
    return document


def relevant_source_refs(refs):
    fixed = {BINDING, PROTOCOL, BASE_PERMISSION, GUARD, SITE_SDK, RECEIPT, HISTORY, LEDGER_SNAPSHOT}
    selected = {path: row for path, row in refs.items() if path.endswith('.py') or path in fixed}
    require(all(path in selected for path in fixed), 'complete actual relevant source pins')
    return selected


def verify_metadata_configuration(root, config_file):
    root = project(root); relative = config_relative(root, config_file)
    config = read(relative, root); validate_configuration_document(root, config, relative)
    _, refs = common_rows(root)
    for row in relevant_source_refs(refs).values():
        verify_reference(row, root)
    authority = load_authority(root, config)
    verify_source_proof(root, config, authority, 'before', refs)
    return config, refs, authority


def verify_configuration_with_receipt(root, config_file):
    config, refs, authority = verify_metadata_configuration(root, config_file)
    receipt = load_receipt(root)
    require(receipt.signature[1] == config['gpu_uuid'] and receipt.binding_ref.mapping() == refs[BINDING],
        'actual public finite condition matches diagnostic GPU/source binding')
    return config, refs, authority, receipt


def validate_context_type(root, context, refs):
    require(type(context).__module__ in sys.modules, 'actual process context module')
    module = sys.modules[type(context).__module__]
    require(type(context) is getattr(module, 'ProcessEntryContext', None) and
        Path(module.__file__).resolve() == safe(D + '/process_entry_context.py', root).resolve() and
        Path(context.configuration.__func__.__code__.co_filename).resolve() == safe(D + '/process_entry_context.py', root).resolve(),
        'only exact independently frozen process context class')
    verify_reference(refs[D + '/process_entry_context.py'], root)


def verify_configuration(root, config_file, *, _context=None):
    if _context is None:
        return verify_configuration_with_receipt(root, config_file)[:3]
    _, refs = common_rows(root); validate_context_type(root, _context, refs)
    return _context.configuration(root, config_file, controller=sys.modules[__name__])


def native_command(index):
    return ['.venv/bin/python', '-B', D + '/run_p4_single_file_experiment.py', '--execute', '--config', config_path(index)]


def guard_command(index):
    return ['.venv/bin/python', '-B', GUARD, '--permissions-path',
        D + '/EFFECTIVE_GPU_PERMISSION_' + suffix(index) + '.json', '--label', label(index), '--seconds', '300', '--'] + native_command(index)


def scope(root, index):
    root = project(root); config = read(config_path(index), root)
    validate_configuration_document(root, config, config_path(index))
    _, refs = common_rows(root, verify_bytes=True)
    authority = load_authority(root, config)
    validate_live_interval(read(authority['context_ref']['path'], root), require_current=True)
    receipt = load_receipt(root)
    require(receipt.signature[1] == config['gpu_uuid'] and receipt.binding_ref.mapping() == refs[BINDING],
        'actual finite public condition and current site before independent source proof')
    result = dict(schema='c5_repeatability_guarded_scope_v1', mode='off', diagnostic_index=index,
        protocol_ref=config['protocol_ref'], label=label(index), config_ref=ref(config_path(index), root),
        source_lock_ref=ref(LOCK, root), binding_ref=refs[BINDING], authority_ref=ref(authority_path(index), root),
        context_ref=authority['context_ref'], human_grant_ref=authority['human_grant_ref'],
        permission_ref=authority['effective_permission_ref'], previous_qualification_ref=None,
        command=guard_command(index), primary_free_floor_bytes=8*1024**3, primary_reserve_bytes=128*1024**2,
        seconds_limit=300, reserved_seconds=320, GPU_operations=0, launch_is_not_success=True)
    return put('SCOPE_' + suffix(index) + '.json', result, root)


def verify_scope(root, config, authority):
    index = config['diagnostic_index']; document = read(D + '/SCOPE_' + suffix(index) + '.json', root)
    expected = dict(schema='c5_repeatability_guarded_scope_v1', mode='off', diagnostic_index=index,
        protocol_ref=config['protocol_ref'], label=label(index), config_ref=ref(config_path(index), root),
        source_lock_ref=ref(LOCK, root), binding_ref=ref(BINDING, root), authority_ref=ref(authority_path(index), root),
        context_ref=authority['context_ref'], human_grant_ref=authority['human_grant_ref'],
        permission_ref=authority['effective_permission_ref'], previous_qualification_ref=None,
        command=guard_command(index), primary_free_floor_bytes=8*1024**3, primary_reserve_bytes=128*1024**2,
        seconds_limit=300, reserved_seconds=320, GPU_operations=0, launch_is_not_success=True)
    require(document == expected and all(exact(document.get(key), value) for key, value in expected.items()), 'fixed independent diagnostic scope')
    return document


def check_sources(index, phase, root=None):
    root = project(root); suffix(index)
    config = read(config_path(index), root); validate_configuration_document(root, config, config_path(index))
    authority = load_authority(root, config); verify_scope(root, config, authority)
    _, refs = common_rows(root); failed = []
    for row in refs.values():
        try:
            verify_reference(row, root)
        except (ValueError, OSError) as exc:
            failed.append(dict(path=row['path'], reason=str(exc)))
    for row in [authority[key] for key in ('context_ref', 'human_grant_ref', 'effective_permission_ref',
        'base_permission_ref', 'guard_source_ref')] + [ref(config_path(index), root), ref(authority_path(index), root)]:
        verify_reference(row, root)
    value = dict(schema='c5_repeatability_full_source_verification_v1', phase=phase,
        diagnostic_index=index, protocol_ref=config['protocol_ref'], source_lock_ref=ref(LOCK, root),
        config_ref=ref(config_path(index), root), authority_ref=ref(authority_path(index), root),
        binding_ref=ref(BINDING, root), human_grant_ref=authority['human_grant_ref'],
        permission_ref=authority['effective_permission_ref'], context_ref=authority['context_ref'],
        previous_qualification_ref=None, files_verified=len(refs)-len(failed), source_rows_sha256=closure_digest(refs),
        failed=failed, actual_verification_utc=now(), GPU_operations=0)
    row = put(Path(proof_path(index, phase)).name, value, root)
    require(not failed, 'complete source closure failed')
    return row


def verify_launch_intent(root, config, authority, refs):
    index = config['diagnostic_index']; verify_scope(root, config, authority)
    verify_source_proof(root, config, authority, 'launch', refs)
    intent = read(D + '/LAUNCH_INTENT_' + suffix(index) + '.json', root)
    expected = dict(command=guard_command(index), scope_ref=ref(D + '/SCOPE_' + suffix(index) + '.json', root),
        config_ref=ref(config_path(index), root), authority_ref=ref(authority_path(index), root),
        source_lock_ref=ref(LOCK, root), binding_ref=ref(BINDING, root), human_grant_ref=authority['human_grant_ref'],
        permission_ref=authority['effective_permission_ref'], source_before_ref=ref(proof_path(index, 'before'), root),
        launch_source_ref=ref(proof_path(index, 'launch'), root), ledger_before_ref=ref(D + '/LEDGER_BEFORE_' + suffix(index) + '.json', root))
    require(all(exact(intent.get(key), value) for key, value in expected.items()), 'actual pinned prelaunch intent')
    validate_live_interval(read(authority['context_ref']['path'], root), intent.get('launch_unix'))
    before = read(intent['ledger_before_ref']['path'], root)
    require(before.get('active_reservation') is None and type(before.get('events')) is list and
        exact(before.get('gpu_wall_seconds'), intent.get('ledger_gpu_wall_seconds_before')), 'actual idle prelaunch ledger snapshot')
    number(before.get('gpu_wall_seconds'), 'immutable prelaunch usage')
    return intent


def _verify_active_guard_metadata(root, config, authority):
    _, refs = common_rows(root); verify_launch_intent(root, config, authority, refs)
    index = config['diagnostic_index']
    expected = dict(label=config['label'], gpu_uuid=config['gpu_uuid'], seconds_limit=300, reserved_seconds=320,
        command=native_command(index), permissions=authority['effective_permission_ref'],
        evidence=str(safe('experiments/prefix_io_v1/runs/' + config['label'], root)))
    ledger = read(LEDGER, root); active = ledger.get('active_reservation')
    require(type(active) is dict and type(active.get('id')) is str and active['id'] and
        all(exact(active.get(key), value) for key, value in expected.items()), 'actual original active reservation binding')
    original_id = active['id']; deadline = time.monotonic() + 1.0
    while active.get('session_id') is None and time.monotonic() < deadline:
        time.sleep(0.01); ledger = read(LEDGER, root); active = ledger.get('active_reservation')
        require(type(active) is dict and active.get('id') == original_id and
            all(exact(active.get(key), value) for key, value in expected.items()), 'original reservation changed during publication')
    sid = active.get('session_id'); runner = active.get('runner_pid')
    require(type(sid) is int and sid > 0 and type(runner) is int and runner > 0 and os.name == 'posix' and
        os.getsid(0) == sid and os.getpgid(0) == sid and type(active.get('process_group')) is int and active.get('process_group') == sid and
        os.environ.get('CUDA_VISIBLE_DEVICES') == config['gpu_uuid'] and active.get('state') != 'cleanup_unresolved',
        'actual original process group/session/GPU')
    tokens = (Path('/proc') / str(runner) / 'cmdline').read_bytes().decode().rstrip('\x00').split('\x00')
    require(GUARD in tokens and authority['effective_permission_ref']['path'] in tokens and config['label'] in tokens,
        'actual bound original guard process')
    number(ledger.get('gpu_wall_seconds'), 'used cumulative GPU seconds')
    require(ledger['gpu_wall_seconds'] + 320 <= 28800, 'original cumulative budget exhausted')
    intent = verify_launch_intent(root, config, authority, refs)
    before = read(intent['ledger_before_ref']['path'], root)
    require(ledger['events'] == before['events'] and exact(ledger['gpu_wall_seconds'], before['gpu_wall_seconds']),
        'current live ledger prefix/accounting changed inside one original job')
    return dict(active, ledger_gpu_seconds_used=ledger['gpu_wall_seconds'])


def verify_active_guard(root, config, authority, *, _context=None):
    actual, refs, actual_authority = verify_configuration(root, config_path(config['diagnostic_index']), _context=_context)
    require(actual == config and actual_authority == authority, 'current actual caller config/authority')
    active = _verify_active_guard_metadata(root, config, authority)
    if _context is not None:
        validate_context_type(root, _context, refs); _context.validate_guard(active)
    return active


def verify_completed_guard(root, config, authority, guard_ref=None):
    actual, refs, actual_authority = verify_configuration(root, config_path(config['diagnostic_index']))
    require(actual == config and actual_authority == authority, 'completed caller config/authority')
    intent = verify_launch_intent(root, config, authority, refs)
    verify_source_proof(root, config, authority, 'after', refs)
    relative = 'experiments/prefix_io_v1/runs/' + config['label'] + '/result.json'
    require(guard_ref is None or guard_ref == ref(relative, root), 'fixed actual completed guard evidence')
    event = read(relative, root)
    expected = dict(label=config['label'], gpu_uuid=config['gpu_uuid'], command=native_command(config['diagnostic_index']),
        permissions=authority['effective_permission_ref'], evidence=str(safe(relative.rsplit('/', 1)[0], root)),
        exit=0, child_exit=0, timed_out=False, interrupted_signal=None, error=None, gpu_job_attempted=True,
        session_drained=True, session_members_before_cleanup=[], session_members_after_cleanup=[])
    require(all(exact(event.get(key), value) for key, value in expected.items()), 'actual completed original guard/session drain')
    require(type(event.get('reservation_id')) is str and event['reservation_id'] and type(event.get('session_id')) is int
        and event['session_id'] > 0 and number(event.get('elapsed_seconds'), 'real bounded elapsed', 0.000001) <= 320,
        'real completed reservation/session/time')
    ledger = read(LEDGER, root)
    require(ledger.get('active_reservation') is None and type(ledger.get('events')) is list and
        [row for row in ledger['events'] if row.get('reservation_id') == event['reservation_id']] == [event],
        'unique actual completed event; independent post-job verification is idle')
    before = read(intent['ledger_before_ref']['path'], root); prefix = before['events']; index = len(prefix)
    require(ledger['events'][:index] == prefix and len(ledger['events']) > index and ledger['events'][index] == event,
        'unchanged actual prelaunch prefix and first appended event')
    completed = dict(before, events=prefix + [event], active_reservation=None,
        gpu_wall_seconds=before['gpu_wall_seconds'] + event['elapsed_seconds'])
    history = load_module(root, refs[HISTORY], '_repeatability_completed_history_' + str(time.monotonic_ns()))
    history.verify_ledger_extension(completed, ledger, event, ledger_before_seconds=before['gpu_wall_seconds'], allow_active=False)
    return event


def after(index, root=None):
    root = project(root); config, _, authority = verify_configuration(root, config_path(index))
    require(read(LEDGER, root).get('active_reservation') is None, 'wait for actual guard completion')
    event = read('experiments/prefix_io_v1/runs/' + label(index) + '/result.json', root)
    require(event.get('session_drained') is True and event.get('session_members_after_cleanup') == [], 'actual session not drained')
    row = check_sources(index, 'after', root); verify_completed_guard(root, config, authority)
    return row


def launch(index, root=None):
    root = project(root); tag = suffix(index)
    config, refs, authority = verify_configuration(root, config_path(index)); scope_value = verify_scope(root, config, authority)
    validate_live_interval(read(authority['context_ref']['path'], root), require_current=True)
    ledger = read(LEDGER, root); number(ledger.get('gpu_wall_seconds'), 'real used GPU seconds')
    require(ledger.get('active_reservation') is None and ledger['gpu_wall_seconds'] + 320 <= 28800, 'original budget/idle gate')
    run = 'experiments/prefix_io_v1/runs/' + label(index)
    require(not safe(run, root).exists() and not safe(D + '/LAUNCH_INTENT_' + tag + '.json', root).exists(), 'single attempt, no retry')
    require(os.name == 'posix', 'original Linux guard only')
    stat = os.statvfs(root); free = stat.f_bavail * stat.f_frsize
    require(free >= scope_value['primary_free_floor_bytes'] + scope_value['primary_reserve_bytes'], 'storage floor/reserve')
    launch_source = check_sources(index, 'launch', root)
    telemetry = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10).stdout
    rows = [line.split(',') for line in telemetry.strip().splitlines()]
    require(len(rows) == 1 and rows[0][0].strip() == config['gpu_uuid'] and int(rows[0][2]) >= 28000, 'same free current GPU')
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10).stdout
    require(not processes.strip(), 'other actual GPU process')
    raw = safe(LEDGER, root).read_bytes(); snapshot = json.loads(raw, object_pairs_hook=pairs)
    require(snapshot == ledger and read(LEDGER, root) == ledger, 'actual ledger changed before launch')
    with safe(D + '/LEDGER_BEFORE_' + tag + '.json', root).open('xb') as stream:
        stream.write(raw)
    overrides = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    command = guard_command(index)
    put('LAUNCH_INTENT_' + tag + '.json', dict(command=command, environment_overrides=overrides,
        scope_ref=ref(D + '/SCOPE_' + tag + '.json', root), config_ref=ref(config_path(index), root),
        authority_ref=ref(authority_path(index), root), source_lock_ref=ref(LOCK, root), binding_ref=ref(BINDING, root),
        human_grant_ref=authority['human_grant_ref'], permission_ref=authority['effective_permission_ref'],
        source_before_ref=ref(proof_path(index, 'before'), root), launch_source_ref=launch_source,
        ledger_before_ref=ref(D + '/LEDGER_BEFORE_' + tag + '.json', root), ledger_gpu_wall_seconds_before=ledger['gpu_wall_seconds'],
        primary_free_bytes=free, gpu_telemetry=telemetry, compute_processes=processes, launch_unix=time.time(), utc=now()), root)
    with safe(D + '/GPU_GUARD_' + tag + '_STDOUT.log', root).open('xb') as out, safe(D + '/GPU_GUARD_' + tag + '_STDERR.log', root).open('xb') as err:
        env = dict(os.environ); env.update(overrides)
        proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True, close_fds=True)
    return put('LAUNCH_RECEIPT_' + tag + '.json', dict(guard_pid=proc.pid, command=command, utc=now(), detached_original_guard=True, launch_is_not_success=True), root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('action', choices=('prepare_all', 'prepare', 'scope', 'before', 'launch', 'after'))
    parser.add_argument('--index', type=int, choices=range(3))
    args = parser.parse_args(argv)
    if args.action == 'prepare_all':
        result = prepare_all(args.root)
    else:
        require(args.index is not None, '--index required')
        if args.action == 'scope':
            result = scope(args.root, args.index)
        elif args.action == 'before':
            result = check_sources(args.index, 'before', args.root)
        else:
            result = {'prepare': prepare, 'launch': launch, 'after': after}[args.action](args.index, args.root)
    print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0


if __name__ == '__main__':
    raise SystemExit(main())

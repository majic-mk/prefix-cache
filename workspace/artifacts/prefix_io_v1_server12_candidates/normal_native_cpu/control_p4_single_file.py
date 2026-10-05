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
D = 'artifacts/prefix_io_v1/server12-c5-normal-native-cpu-20261004'
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
RECEIPT = D + '/p4_single_file_receipt.py'
HISTORY = D + '/historical_calibration_binding.py'
LEDGER_SNAPSHOT = D + '/SERVER12_LEDGER_SNAPSHOT.json'
CALIBRATION_BINDING = CALIBRATION + '/NATIVE_SINGLE_FILE_BINDING.json'
CALIBRATION_BINDING_SHA = 'd79be29c8139a1cd42577b54bdd33e00bedcaa1fc6dbf5100df682b32ca3f42b'
G_CANONICAL_SHA = '50cc763a191a66cb002ad1e6091f35fa2af9d413b37ca251af0487686ab73a82'
CANONICAL_SHA = '3ef359a623d02d9ed220c7e44a0d5fbfa3b4bcbe0b598b8523b800d5aff8a73e'
GUARD_SHA = '3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
GUARD = 'experiments/prefix_io_v1/scripts/run_gpu_stage.py'
BASE_PERMISSION = 'experiments/prefix_io_v1/configs/permissions.yaml'
LOCK = D + '/COMMON_SOURCE_LOCK.json'
BINDING = D + '/NORMAL_RUNTIME_BINDING.json'
MANIFEST = 'artifacts/prefix_io_v1/server11-native-cost-v1-20261003/SSD_INPUT_MANIFEST.json'
STORAGE = 'experiments/prefix_io_v1/runs/server11-native-cost-01-private-storage'
PURPOSE = 'FINITE_NORMAL_NOTIFICATION_LIFECYCLE_DIAGNOSTIC_ONLY'
SCOPE = 'server12_c5_normal_native_runtime_v1'
MODES = ('off', 'shadow', 'on')
DENIED = ('allow_model_downloads', 'allow_driver_or_system_changes',
    'allow_shared_data_deletion', 'allow_payment', 'allow_new_cloud_rental')
UUID_RE = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
LOCK_SCHEMA = 'c5_normal_common_source_lock_v1'


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


def label(mode):
    require(mode in MODES, 'known arm required')
    return 'server12-c5-native-normal-' + mode + '01'


def config_path(mode):
    label(mode)
    return D + '/CONFIG_' + mode + '.json'


def result_path(mode):
    label(mode)
    return D + '/QUALIFICATION_' + mode + '.json'


def authority_path(mode):
    label(mode)
    return D + '/AUTHORITY_' + mode + '.json'


def config_relative(root, path):
    root = project(root)
    if isinstance(path, Path) and not path.is_absolute():
        path = path.as_posix()
    elif isinstance(path, Path) or (type(path) is str and Path(path).is_absolute()):
        try:
            path = Path(path).relative_to(root).as_posix()
        except ValueError:
            require(False, 'configuration outside project')
    require(path in [config_path(mode) for mode in MODES], 'fixed normal per-arm configuration')
    safe(path, root)
    return path


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


def load_receipt(root=None, binding_relative=None):
    root = project(root)
    binding_relative = BINDING if binding_relative is None else binding_relative
    require(binding_relative in (BINDING, CALIBRATION_BINDING), 'explicit real native binding required')
    canonical_ref = ref(RECEIPT, root)
    require(type(CANONICAL_SHA) is str and canonical_ref['sha256'] == CANONICAL_SHA,
        'actual pinned public historical-compatible canonical source')
    require(ref(G_RECEIPT, root)['sha256'] == G_CANONICAL_SHA, 'unchanged actual G canonical ancestry')
    require(ref(CALIBRATION_CANONICAL, root)['sha256'] == CALIBRATION_CANONICAL_SHA,
        'unchanged actual server12 calibration public canonical ancestry')
    if binding_relative == CALIBRATION_BINDING:
        require(ref(binding_relative, root)['sha256'] == CALIBRATION_BINDING_SHA,
            'completed calibration binding provenance changed')
    module = load_module(root, canonical_ref, '_normal_public_canonical_' + str(time.monotonic_ns()))
    receipt = module.load_verified_single_file(root, binding_relative)
    require(type(receipt) is module.ExactSingleFileReceipt and receipt.condition_only is True and
        receipt.production_qualified is False, 'real public typed conditional receipt required')
    require(type(receipt.cost_upper_ns) is int and receipt.cost_upper_ns > 0 and
        type(receipt.step_budget_ns) is int and receipt.step_budget_ns > 0 and
        type(receipt.signature) is tuple and len(receipt.signature) == 9 and
        UUID_RE.fullmatch(receipt.signature[1]) is not None, 'actual native finite cost/device signature')
    verify_reference(receipt.binding_ref.mapping(), root)
    return receipt


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


def generated_mode_path(relative):
    if not relative.startswith(D + '/'):
        return False
    name = relative.removeprefix(D + '/')
    return name.startswith(('CONFIG_', 'AUTHORITY_', 'SCOPE_', 'SOURCE_', 'LIVE_CONTEXT_', 'LEDGER_BEFORE_',
        'HUMAN_GPU_GRANT_', 'EFFECTIVE_GPU_PERMISSION_', 'QUALIFICATION_', 'LAUNCH_', 'GPU_GUARD_'))


def common_rows(root, *, verify_bytes=False):
    lock = read(LOCK, root)
    require(lock.get('schema') == LOCK_SCHEMA and lock.get('scope') == SCOPE and
        exact(lock.get('GPU_operations'), 0) and lock.get('normal_runtime_qualified') is False and
        lock.get('gpu_authority_issued') is False, 'normal common source lock schema')
    rows = rows_map(lock.get('files'), root, verify_bytes=verify_bytes)
    require(LOCK not in rows and LEDGER not in rows and all(not generated_mode_path(path) for path in rows),
        'configuration/grant/prerequisite/proofs must remain outside shared common lock')
    for path in (BINDING, RECEIPT, G_RECEIPT, CALIBRATION_CANONICAL, SITE_SDK,
        HISTORY, LEDGER_SNAPSHOT, GUARD, BASE_PERMISSION, MANIFEST,
        D + '/control_p4_single_file.py', D + '/run_p4_single_file_experiment.py',
        D + '/verify_p4_single_file.py', D + '/combined_runtime_contract.py',
        D + '/notification_runtime_adapter.py', CALIBRATION + '/native_conditional_cost.py',
        D + '/native_conditional_cost.py', D + '/prepare_and_verify_native_cost.py',
        C + '/native_full_step_collector.py', C + '/single_file_runtime_binding.py'):
        require(path in rows, 'required actual shared source missing: ' + path)
    require(rows[GUARD]['sha256'] == GUARD_SHA and type(CANONICAL_SHA) is str and
        rows[RECEIPT]['sha256'] == CANONICAL_SHA and rows[G_RECEIPT]['sha256'] == G_CANONICAL_SHA and
        rows[CALIBRATION_CANONICAL]['sha256'] == CALIBRATION_CANONICAL_SHA and rows[SITE_SDK]['sha256'] == SITE_SDK_SHA,
        'original guard, G/server12 canonical ancestry and new public canonical pins')
    host = lock.get('current_host_asset_verification')
    require(type(host) is dict and host.get('driver_ref') == HOST_DRIVER_REF and
        host.get('sdk_adapter_ref') == rows[SITE_SDK] and host.get('actual_byte_sha256_verified') is True and
        host.get('GPU_operations') == 0 and type(host.get('GPU_operations')) is int and
        host.get('native_execution_qualification') is False and host.get('shared_object_loaded') is False,
        'explicit original full-byte SDK/current host CPU audit metadata required')
    require(metadata_reference(host['driver_ref'], root) is None, 'exact typed external current host pin')
    for key in ('sdk_helper_ref', 'inventory_ref', 'compiler_proof_ref', 'existing_rebind_ref'):
        reference_shape(host.get(key), root)
        require(rows.get(host[key]['path']) == host[key], 'host SDK evidence outside frozen project closure')
    return lock, rows


def nested_references(value):
    if type(value) is dict:
        if set(value) == {'path', 'bytes', 'sha256'}:
            yield value
        else:
            for child in value.values():
                yield from nested_references(child)
    elif type(value) is list:
        for child in value:
            yield from nested_references(child)


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


def freeze_common(expected_calibration_lock_sha256, root=None):
    root = project(root)
    require(type(expected_calibration_lock_sha256) is str and
        re.fullmatch('[0-9a-f]{64}', expected_calibration_lock_sha256) is not None,
        'explicit expected calibration source lock SHA-256')
    require(not safe(LOCK, root).exists() and not safe(BINDING, root).exists(),
        'append-only freeze; inspect existing files instead of overwriting')
    require(ref(CALIBRATION_BINDING, root)['sha256'] == CALIBRATION_BINDING_SHA,
        'completed actual calibration binding provenance')
    domain = read(CALIBRATION_BINDING, root)
    calibration = domain.get('calibration')
    require(type(calibration) is dict and set(calibration) == {'plan_ref', 'measurements_ref', 'guard_ref'},
        'actual calibration references required')
    for row in calibration.values():
        verify_reference(row, root)
    plan = read(calibration['plan_ref']['path'], root)
    verify_reference(plan['source_lock_ref'], root)
    require(plan['source_lock_ref']['sha256'] == expected_calibration_lock_sha256,
        'calibration lock differs from explicitly expected source')
    calibration_lock = read(plan['source_lock_ref']['path'], root)
    locked = rows_map(calibration_lock.get('files'), root, verify_bytes=True)
    verified_calibration_leaves = frozenset(locked)
    require(LEDGER not in locked, 'mutable cumulative ledger cannot be frozen as a calibration asset')
    require(ref(GUARD, root)['sha256'] == GUARD_SHA and locked.get(GUARD) == ref(GUARD, root),
        'original budget guard unchanged')
    require(locked.get(SITE_SDK) == ref(SITE_SDK, root) and locked[SITE_SDK]['sha256'] == SITE_SDK_SHA,
        'same source-pinned real current site SDK adapter')
    sdk = load_module(root, locked[SITE_SDK], '_normal_site_asset_audit_' + str(time.monotonic_ns()))
    require(type(sdk.DRIVER) is dict and set(sdk.DRIVER) == set(HOST_DRIVER_REF) and
        all(exact(sdk.DRIVER[key], value) for key, value in HOST_DRIVER_REF.items()),
        'current trusted adapter driver pin differs from normal metadata allowlist')
    _sdk_helper, actual_assets, sdk_rows = sdk.load_site_assets(root, locked)
    require(type(actual_assets.get('driver')) is dict and
        all(exact(actual_assets['driver'].get(key), value) for key, value in HOST_DRIVER_REF.items()),
        'original full-byte SDK audit returned a different real driver')
    host_verification = dict(schema='c5_normal_current_host_asset_cpu_verification_v1',
        sdk_adapter_ref=locked[SITE_SDK], sdk_helper_ref=sdk_rows[sdk.HELPER],
        inventory_ref=sdk_rows[sdk.INVENTORY], compiler_proof_ref=sdk_rows[sdk.PROOF],
        existing_rebind_ref=sdk_rows[sdk.RESULT], driver_ref=dict(HOST_DRIVER_REF),
        audit_method='pinned_unchanged_helper_load_audited_assets_complete_byte_sha256',
        actual_byte_sha256_verified=True, shared_object_loaded=False, GPU_operations=0,
        native_execution_qualification=False)
    # Public replay follows source validation. No private issuer or dict promotion.
    normal_sources = [ref(path.relative_to(root).as_posix(), root)
        for path in sorted(safe(D, root).glob('*.py'))]
    normal_overlay = normal_sources + [ref(LEDGER_SNAPSHOT, root)]
    contract = load_module(root, ref(D + '/combined_runtime_contract.py', root),
        '_normal_source_groups_' + str(time.monotonic_ns()))
    groups = contract.runtime_source_groups(root, plan, locked, normal_overlay)
    binding = dict(schema_version=1, scope='p4_single_file_native_binding_v1', calibration=calibration,
        runtime_common_refs=groups['runtime_common_refs'], runtime_overlay_refs=groups['runtime_overlay_refs'],
        kernel_mode=domain['kernel_mode'])
    binding_ref = put('NORMAL_RUNTIME_BINDING.json', binding, root)
    receipt = load_receipt(root, BINDING)
    # The new public canonical API independently reconstructs real raw evidence,
    # original math and the unchanged A-only budget. This must never invoke the
    # old G loader against an already-extended mutable ledger.
    require(receipt.calibration_source_lock_sha256 == expected_calibration_lock_sha256,
        'normal binding must preserve actual calibration source domain')
    merged = dict(locked)
    def add(row):
        reference_shape(row, root)
        require(row['path'] != LEDGER, 'mutable cumulative ledger outside immutable common closure')
        require(row['path'] not in merged or merged[row['path']] == row,
            'conflicting source/evidence reference')
        if row['path'] in merged:
            return
        verify_reference(row, root)
        merged[row['path']] = row
    for row in normal_overlay + [binding_ref, ref(CALIBRATION_BINDING, root), plan['source_lock_ref']]:
        add(row)
    # The completed G verifier also reads these independently preregistered files.
    # Freeze the immutable documents themselves; a launch-intent ledger snapshot
    # describes prior state and must not freeze the cumulative live ledger.
    for name in ('GPU_LAUNCH_INTENT.json', 'SIX_PROCESS_AUTHORIZED_SCOPE.json',
        'PLAN_REFERENCE.json', 'SOURCE_LAUNCH_VERIFICATION.json'):
        add(ref(CALIBRATION + '/' + name, root))
    # Capture real consumed raw/child/proof evidence as well as calibration assets.
    documents = [domain, plan, read(calibration['measurements_ref']['path'], root)]
    pending = [row for document in documents for row in nested_references(document)]
    inspected = set()
    excluded = []
    host_metadata = []
    absolute_project_metadata = []
    frozen_leaf_json = []
    documents_read = 0
    while pending:
        require(len(pending) <= 50000 and len(merged) <= 10000 and documents_read <= 1024,
            'bounded immutable calibration provenance traversal')
        raw_row = pending.pop()
        row = metadata_reference(raw_row, root)
        if row is None:
            if raw_row not in host_metadata:
                host_metadata.append(raw_row)
            continue  # Actual bytes were already verified by the original SDK helper above.
        if row != raw_row and raw_row not in absolute_project_metadata:
            absolute_project_metadata.append(dict(original_ref=raw_row, project_ref=row))
        if row['path'] == LEDGER:
            # Historical LIVE_BINDING_CONTEXT ledger_before_ref is an as-of
            # snapshot digest. It does not assert present ledger bytes and is not
            # consumed as a file by the calibration verifier. Retain its origin
            # explicitly while the original guard verifies the live ledger.
            if row not in excluded:
                excluded.append(row)
            continue
        add(row)
        if row['path'] in inspected:
            continue
        inspected.add(row['path'])
        if row['path'] in verified_calibration_leaves:
            if row['path'].endswith('.json'):
                frozen_leaf_json.append(row)
            continue  # Every original locked leaf was separately byte/SHA checked, before SDK audit.
        if row['path'].endswith('.json') and row['bytes'] <= 32 * 1024**2 and row['path'] != plan['source_lock_ref']['path']:
            documents_read += 1
            pending.extend(nested_references(read(row['path'], root)))
    require(all(not generated_mode_path(path) for path in merged) and LOCK not in merged and LEDGER not in merged,
        'one immutable common lock; no per-mode authority cycles')
    lock_ref = put('COMMON_SOURCE_LOCK.json', dict(schema=LOCK_SCHEMA, scope=SCOPE,
        calibration_source_lock_ref=plan['source_lock_ref'], calibration_binding_ref=ref(CALIBRATION_BINDING, root),
        normal_binding_ref=binding_ref, files=[merged[path] for path in sorted(merged)],
        historical_mutable_references_excluded=excluded,
        historical_mutable_reference_note='as-of ledger digests are retained without claiming current byte verification; original guard checks live ledger',
        current_host_asset_verification=host_verification,
        current_external_host_metadata_refs=host_metadata,
        absolute_project_metadata_normalized=absolute_project_metadata,
        preregistered_calibration_leaf_json_not_recursively_expanded=frozen_leaf_json,
        provenance_traversal='each original calibration leaf fully byte/SHA verified; newly discovered raw parent/child/proof JSON fully traversed; immutable old asset JSON is retained as verified leaf, not reinterpreted as current hardware',
        GPU_operations=0, gpu_authority_issued=False, normal_runtime_qualified=False,
        origin='actual_completed_calibration_and_new_normal_source_closure', created_utc=now()), root)
    return dict(status='PASS_REAL_NORMAL_BINDING_CPU_PREPARATION', source_lock_ref=lock_ref,
        binding_ref=binding_ref, calibration_source_lock_sha256=expected_calibration_lock_sha256,
        cost_upper_ns=receipt.cost_upper_ns, step_budget_ns=receipt.step_budget_ns,
        GPU_operations=0, gpu_launch_allowed=False, normal_runtime_qualified=False,
        on_observation_cost_measured=False, production_qualified=False)


def expected_config(root, mode, gpu_uuid, previous):
    require(type(gpu_uuid) is str and UUID_RE.fullmatch(gpu_uuid) is not None, 'real calibration GPU UUID')
    return dict(root=str(project(root)), label=label(mode), gpu_uuid=gpu_uuid, purpose=PURPOSE,
        seconds_limit=300, reserved_seconds=320, active_ledger=LEDGER, storage=STORAGE,
        out='experiments/prefix_io_v1/runs/' + label(mode) + '/details', overlay_relative=C + '/source',
        collector_relative=C + '/native_full_step_collector.py',
        runtime_binding_relative=C + '/single_file_runtime_binding.py', source_lock=LOCK,
        input_manifest=MANIFEST, binding_relative=BINDING, mode=mode,
        previous_qualification_ref=previous, authority_relative=authority_path(mode))


def validate_configuration_document(root, config, relative):
    require(type(config) is dict and config.get('mode') in MODES, 'explicit finite normal mode')
    mode = config['mode']
    expected = expected_config(root, mode, config.get('gpu_uuid'), config.get('previous_qualification_ref'))
    require(relative == config_path(mode) and set(config) == set(expected) and
        all(exact(config[key], value) for key, value in expected.items()), 'fixed exact normal config fields')
    if mode == 'off':
        require(config['previous_qualification_ref'] is None, 'off has no predecessor')
    else:
        reference_shape(config['previous_qualification_ref'], root)
        require(config['previous_qualification_ref']['path'] == result_path('off' if mode == 'shadow' else 'shadow'),
            'ordered independently qualified predecessor path')
    return mode


def verify_previous_qualification(root, config, refs, _depth=0):
    require(type(_depth) is int and 0 <= _depth <= 2, 'bounded predecessor chain')
    mode = config['mode']; previous = config['previous_qualification_ref']
    if mode == 'off':
        require(previous is None, 'off is first')
        return None
    verify_reference(previous, root)
    require(previous['path'] == result_path('off' if mode == 'shadow' else 'shadow'), 'ordered prior report')
    report = read(previous['path'], root)
    require(report.get('scope') == SCOPE and report.get('mode') == ('off' if mode == 'shadow' else 'shadow') and
        report.get('native_execution_verified') is True and report.get('runtime_condition_qualified') is True and
        report.get('permits_next_mode') == mode and report.get('source_lock_ref') == ref(LOCK, root) and
        report.get('binding_ref') == refs[BINDING], 'same-source independently qualified predecessor')
    evidence = report.get('evidence_refs')
    require(type(evidence) is dict and set(evidence) == {'config', 'raw_result', 'actual_guard', 'source_before', 'source_after'},
        'complete prior raw evidence closure')
    for row in evidence.values():
        verify_reference(row, root)
    prior_config = read(evidence['config']['path'], root)
    require(prior_config.get('source_lock') == LOCK and prior_config.get('binding_relative') == BINDING and
        prior_config.get('gpu_uuid') == config['gpu_uuid'], 'same normal lock, binding and GPU')
    verifier_ref = refs[D + '/verify_p4_single_file.py']
    verifier = load_module(root, verifier_ref, '_normal_prior_verifier_' + str(time.monotonic_ns()))
    replayed = verifier.verify_runtime(root, config_ref=evidence['config'], result_ref=evidence['raw_result'],
        guard_ref=evidence['actual_guard'], before_ref=evidence['source_before'],
        after_ref=evidence['source_after'], _depth=_depth + 1)
    require(replayed == report, 'predecessor must replay actual raw/guard/source evidence, not flags')
    return replayed


def prepare(mode, root=None):
    root = project(root); label(mode)
    _, refs = common_rows(root, verify_bytes=True)
    receipt = load_receipt(root)
    require(refs[BINDING] == receipt.binding_ref.mapping(), 'real receipt inside frozen normal closure')
    previous = None if mode == 'off' else ref(result_path('off' if mode == 'shadow' else 'shadow'), root)
    config = expected_config(root, mode, receipt.signature[1], previous)
    verify_previous_qualification(root, config, refs)
    row = put('CONFIG_' + mode + '.json', config, root)
    return dict(status='PASS_NORMAL_CONFIG_CPU_PREPARATION', mode=mode, config_ref=row,
        source_lock_ref=ref(LOCK, root), binding_ref=refs[BINDING], GPU_operations=0,
        gpu_launch_allowed=False, authority_issued=False, normal_runtime_qualified=False)


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


def load_authority(root, config):
    mode = config['mode']; relative = authority_path(mode)
    authority = read(relative, root)
    fixed = dict(schema='c5_normal_mode_site_authority_v1', status='LIVE_HUMAN_BOUND',
        origin='live_site_binding', synthetic=False, root=str(project(root)), mode=mode,
        label=label(mode), gpu_uuid=config['gpu_uuid'], config_ref=ref(config_path(mode), root),
        source_lock_ref=ref(LOCK, root), binding_ref=ref(BINDING, root),
        previous_qualification_ref=config['previous_qualification_ref'], seconds_limit=300,
        reserved_seconds=320, max_attempts=1, max_processes=1)
    keys = set(fixed) | {'context_ref', 'human_grant_ref', 'effective_permission_ref', 'base_permission_ref', 'guard_source_ref'}
    require(type(authority) is dict and set(authority) == keys and
        all(exact(authority.get(key), value) for key, value in fixed.items()),
        'fresh separate normal mode site authority required')
    required_references = (('context_ref', D + '/LIVE_CONTEXT_' + mode + '.json'),
        ('human_grant_ref', D + '/HUMAN_GPU_GRANT_' + mode + '.json'),
        ('effective_permission_ref', D + '/EFFECTIVE_GPU_PERMISSION_' + mode + '.json'),
        ('base_permission_ref', BASE_PERMISSION), ('guard_source_ref', GUARD))
    for key, path in required_references:
        reference_shape(authority[key], root)
        require(authority[key]['path'] == path, 'fresh normal reference path: ' + key)
    for key, _ in required_references:
        verify_reference(authority[key], root)
    require(authority['guard_source_ref']['sha256'] == GUARD_SHA, 'original guard source changed')
    if config['previous_qualification_ref'] is not None:
        verify_reference(config['previous_qualification_ref'], root)
    context = read(authority['context_ref']['path'], root)
    context_expected = dict(schema='c5_normal_live_context_v1', origin='live_site_readonly_context',
        root=fixed['root'], mode=mode, label=label(mode), gpu_uuid=config['gpu_uuid'],
        config_ref=fixed['config_ref'], source_lock_ref=fixed['source_lock_ref'], binding_ref=fixed['binding_ref'],
        previous_qualification_ref=config['previous_qualification_ref'], seconds_limit=300, reserved_seconds=320)
    require(all(exact(context.get(key), value) for key, value in context_expected.items()), 'current mode context mismatch')
    validate_live_interval(context)
    resources = context.get('resources'); require(type(resources) is dict, 'actual resource context')
    for key, minimum in (('cpu_cores', 1), ('cpu_affinity_count', 1), ('available_memory_bytes', 32 * 1024**3),
        ('gpu_free_mib', 28000), ('primary_free_bytes', 8 * 1024**3 + 128 * 1024**2)):
        number(resources.get(key), key, minimum)
    require(resources.get('gpu_uuid') == config['gpu_uuid'] and resources.get('cpu_counters_readable') is True,
        'actual GPU/resource UUID and CPU counters')
    grant = read(authority['human_grant_ref']['path'], root)
    grant_expected = dict(schema='c5_normal_explicit_human_grant_v1', issuer='human_user', authorization=True,
        root=fixed['root'], mode=mode, label=label(mode), gpu_uuid=config['gpu_uuid'],
        context_ref=authority['context_ref'], config_ref=fixed['config_ref'], source_lock_ref=fixed['source_lock_ref'],
        binding_ref=fixed['binding_ref'], previous_qualification_ref=config['previous_qualification_ref'],
        seconds_limit=300, reserved_seconds=320, max_attempts=1, max_processes=1,
        instruction_is_gpu_execution_authorization=True)
    require(all(exact(grant.get(key), value) for key, value in grant_expected.items()) and
        type(grant.get('human_instruction')) is str and grant['human_instruction'].strip() and
        grant['human_instruction'].strip() not in ('继续', '继续直至可以上gpu验证', '继续直至可以上GPU验证'),
        'new direct normal mode/source/context human grant required')
    permission = permission_document(root, authority['effective_permission_ref']['path'])
    base = permission_document(root, BASE_PERMISSION)
    require(permission.get('allow_gpu_runs') is True and permission.get('approved_gpu_ids') == [config['gpu_uuid']],
        'effective normal GPU permission UUID')
    require(number(permission.get('max_gpu_hours'), 'effective budget', 0.01) <=
        min(8, number(base.get('max_gpu_hours'), 'original budget', 0.01)), 'original cumulative budget cannot grow')
    require(all(permission.get(key) == base.get(key) for key in ('approved_experiment_root', 'approved_dependency_root')) and
        permission.get('approved_experiment_root') == str(safe('experiments/prefix_io_v1/runs', root)),
        'original permission roots preserved')
    require(all(permission.get(key) is False for key in DENIED) and
        permission.get('approved_auxiliary_storage') is None, 'forbidden operations remain prohibited')
    permission_binding = dict(grant_expected)
    permission_binding.pop('issuer'); permission_binding.pop('authorization'); permission_binding.pop('schema')
    permission_binding.pop('instruction_is_gpu_execution_authorization')
    permission_binding['human_grant_ref'] = authority['human_grant_ref']
    require(permission.get('normal_mode_binding') == permission_binding and
        all(exact(permission['normal_mode_binding'].get(key), value) for key, value in permission_binding.items()),
        'effective permission separately binds exact normal grant/context/source/mode')
    return authority


def closure_digest(refs):
    return hashlib.sha256(json.dumps([refs[path] for path in sorted(refs)], sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify_source_proof(root, config, authority, phase, refs):
    require(phase in ('before', 'launch', 'after'), 'source verification phase')
    document = read(D + '/SOURCE_' + config['mode'] + '_' + phase.upper() + '.json', root)
    expected = dict(schema='c5_normal_full_source_verification_v1', phase=phase,
        source_lock_ref=ref(LOCK, root), config_ref=ref(config_path(config['mode']), root),
        authority_ref=ref(authority_path(config['mode']), root), binding_ref=ref(BINDING, root),
        human_grant_ref=authority['human_grant_ref'], permission_ref=authority['effective_permission_ref'],
        context_ref=authority['context_ref'], previous_qualification_ref=config['previous_qualification_ref'],
        files_verified=len(refs), source_rows_sha256=closure_digest(refs), failed=[], GPU_operations=0)
    require(all(exact(document.get(key), value) for key, value in expected.items()),
        'complete current common and per-mode source proof required')
    return document


def verify_configuration(root, config_file):
    root = project(root); relative = config_relative(root, config_file)
    config = read(relative, root); validate_configuration_document(root, config, relative)
    _, refs = common_rows(root)
    for path in (D + '/control_p4_single_file.py', D + '/run_p4_single_file_experiment.py',
        D + '/verify_p4_single_file.py', D + '/combined_runtime_contract.py',
        SITE_SDK, RECEIPT, BINDING, GUARD, BASE_PERMISSION):
        verify_reference(refs[path], root)
    authority = load_authority(root, config)
    verify_source_proof(root, config, authority, 'before', refs)
    receipt = load_receipt(root)
    require(receipt.signature[1] == config['gpu_uuid'] and receipt.binding_ref.mapping() == refs[BINDING],
        'actual canonical binding and calibration GPU match normal site')
    return config, refs, authority


def native_command(mode):
    return ['.venv/bin/python', '-B', D + '/run_p4_single_file_experiment.py',
        '--execute', '--config', config_path(mode)]


def guard_command(mode):
    return ['.venv/bin/python', '-B', GUARD, '--permissions-path',
        D + '/EFFECTIVE_GPU_PERMISSION_' + mode + '.json', '--label', label(mode), '--seconds', '300', '--'] + native_command(mode)


def scope(root, mode):
    root = project(root); label(mode)
    config = read(config_path(mode), root); validate_configuration_document(root, config, config_path(mode))
    _, refs = common_rows(root, verify_bytes=True)
    authority = load_authority(root, config)
    validate_live_interval(read(authority['context_ref']['path'], root), require_current=True)
    verify_previous_qualification(root, config, refs)
    receipt = load_receipt(root)
    require(receipt.signature[1] == config['gpu_uuid'] and receipt.binding_ref.mapping() == refs[BINDING],
        'real canonical normal condition and site')
    result = dict(schema='c5_normal_guarded_scope_v1', mode=mode, label=label(mode),
        config_ref=ref(config_path(mode), root), source_lock_ref=ref(LOCK, root), binding_ref=refs[BINDING],
        authority_ref=ref(authority_path(mode), root), context_ref=authority['context_ref'],
        human_grant_ref=authority['human_grant_ref'], permission_ref=authority['effective_permission_ref'],
        previous_qualification_ref=config['previous_qualification_ref'], command=guard_command(mode),
        primary_free_floor_bytes=8 * 1024**3, primary_reserve_bytes=128 * 1024**2,
        seconds_limit=300, reserved_seconds=320, GPU_operations=0, launch_is_not_success=True)
    return put('SCOPE_' + mode + '.json', result, root)


def verify_scope(root, config, authority):
    mode = config['mode']; document = read(D + '/SCOPE_' + mode + '.json', root)
    expected = dict(schema='c5_normal_guarded_scope_v1', mode=mode, label=label(mode),
        config_ref=ref(config_path(mode), root), source_lock_ref=ref(LOCK, root), binding_ref=ref(BINDING, root),
        authority_ref=ref(authority_path(mode), root), context_ref=authority['context_ref'],
        human_grant_ref=authority['human_grant_ref'], permission_ref=authority['effective_permission_ref'],
        previous_qualification_ref=config['previous_qualification_ref'], command=guard_command(mode),
        primary_free_floor_bytes=8 * 1024**3, primary_reserve_bytes=128 * 1024**2,
        seconds_limit=300, reserved_seconds=320, GPU_operations=0, launch_is_not_success=True)
    require(document == expected and all(exact(document.get(key), value) for key, value in expected.items()), 'immutable separate normal scope')
    return document


def check_sources(mode, phase, root=None):
    root = project(root); label(mode)
    config = read(config_path(mode), root); validate_configuration_document(root, config, config_path(mode))
    authority = load_authority(root, config); verify_scope(root, config, authority)
    _, refs = common_rows(root)
    failed = []
    for row in refs.values():
        try:
            verify_reference(row, root)
        except (ValueError, OSError) as exc:
            failed.append(dict(path=row['path'], reason=str(exc)))
    # Per-mode refs remain outside common lock but are checked on every proof.
    for row in [authority[key] for key in ('context_ref', 'human_grant_ref', 'effective_permission_ref',
        'base_permission_ref', 'guard_source_ref')] + [ref(config_path(mode), root), ref(authority_path(mode), root)]:
        verify_reference(row, root)
    value = dict(schema='c5_normal_full_source_verification_v1', phase=phase,
        source_lock_ref=ref(LOCK, root), config_ref=ref(config_path(mode), root),
        authority_ref=ref(authority_path(mode), root), binding_ref=ref(BINDING, root),
        human_grant_ref=authority['human_grant_ref'], permission_ref=authority['effective_permission_ref'],
        context_ref=authority['context_ref'], previous_qualification_ref=config['previous_qualification_ref'],
        files_verified=len(refs) - len(failed), source_rows_sha256=closure_digest(refs), failed=failed,
        actual_verification_utc=now(), GPU_operations=0)
    row = put('SOURCE_' + mode + '_' + phase.upper() + '.json', value, root)
    require(not failed, 'source closure failed')
    return row


def verify_launch_intent(root, config, authority, refs):
    mode = config['mode']; verify_scope(root, config, authority)
    verify_source_proof(root, config, authority, 'launch', refs)
    intent = read(D + '/LAUNCH_INTENT_' + mode + '.json', root)
    expected = dict(command=guard_command(mode), scope_ref=ref(D + '/SCOPE_' + mode + '.json', root),
        config_ref=ref(config_path(mode), root), authority_ref=ref(authority_path(mode), root),
        source_lock_ref=ref(LOCK, root), binding_ref=ref(BINDING, root),
        human_grant_ref=authority['human_grant_ref'], permission_ref=authority['effective_permission_ref'],
        source_before_ref=ref(D + '/SOURCE_' + mode + '_BEFORE.json', root),
        launch_source_ref=ref(D + '/SOURCE_' + mode + '_LAUNCH.json', root),
        ledger_before_ref=ref(D + '/LEDGER_BEFORE_' + mode + '.json', root))
    require(all(exact(intent.get(key), value) for key, value in expected.items()), 'actual frozen prelaunch normal intent')
    context = read(authority['context_ref']['path'], root)
    validate_live_interval(context, intent.get('launch_unix'))
    before = read(intent['ledger_before_ref']['path'], root)
    require(before.get('active_reservation') is None and type(before.get('events')) is list and
        exact(before.get('gpu_wall_seconds'), intent.get('ledger_gpu_wall_seconds_before')),
        'actual immutable idle prelaunch ledger snapshot')
    number(before.get('gpu_wall_seconds'), 'immutable prelaunch usage')
    return intent


def verify_active_guard(root, config, authority):
    actual, refs, actual_authority = verify_configuration(root, config_path(config['mode']))
    require(actual == config and actual_authority == authority, 'active caller config/authority mismatch')
    verify_launch_intent(root, config, authority, refs)
    expected = dict(label=config['label'], gpu_uuid=config['gpu_uuid'], seconds_limit=300,
        reserved_seconds=320, command=native_command(config['mode']),
        permissions=authority['effective_permission_ref'],
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
    require(type(sid) is int and sid > 0 and type(runner) is int and runner > 0 and
        os.name == 'posix' and os.getsid(0) == sid and os.getpgid(0) == sid and
        active.get('process_group') == sid and os.environ.get('CUDA_VISIBLE_DEVICES') == config['gpu_uuid'] and
        active.get('state') != 'cleanup_unresolved', 'actual original process group/session/GPU')
    tokens = (Path('/proc') / str(runner) / 'cmdline').read_bytes().decode().rstrip('\x00').split('\x00')
    require(GUARD in tokens and authority['effective_permission_ref']['path'] in tokens and config['label'] in tokens,
        'actual bound original guard process')
    number(ledger.get('gpu_wall_seconds'), 'used cumulative GPU seconds')
    require(ledger['gpu_wall_seconds'] + 320 <= 28800, 'original eight-hour cumulative budget exhausted')
    return dict(active, ledger_gpu_seconds_used=ledger['gpu_wall_seconds'])


def verify_completed_guard(root, config, authority, guard_ref=None):
    actual, refs, actual_authority = verify_configuration(root, config_path(config['mode']))
    require(config == actual and authority == actual_authority, 'completed caller config/authority mismatch')
    intent = verify_launch_intent(root, config, authority, refs)
    verify_source_proof(root, config, authority, 'after', refs)
    relative = 'experiments/prefix_io_v1/runs/' + config['label'] + '/result.json'
    require(guard_ref is None or guard_ref == ref(relative, root), 'fixed completed original guard evidence')
    event = read(relative, root)
    expected = dict(label=config['label'], gpu_uuid=config['gpu_uuid'], command=native_command(config['mode']),
        permissions=authority['effective_permission_ref'], evidence=str(safe(relative.rsplit('/', 1)[0], root)),
        exit=0, child_exit=0, timed_out=False, interrupted_signal=None, error=None,
        gpu_job_attempted=True, session_drained=True, session_members_before_cleanup=[], session_members_after_cleanup=[])
    require(all(exact(event.get(key), value) for key, value in expected.items()), 'normal original guard completion and session drain')
    require(type(event.get('reservation_id')) is str and event['reservation_id'] and
        type(event.get('session_id')) is int and event['session_id'] > 0, 'real completed reservation/session')
    require(number(event.get('elapsed_seconds'), 'actual GPU elapsed', 0.000001) <= 320, 'bounded normal GPU elapsed')
    ledger = read(LEDGER, root)
    require(type(ledger.get('events')) is list and
        [row for row in ledger['events'] if row.get('reservation_id') == event['reservation_id']] == [event],
        'unique completed original ledger event')
    before = read(intent['ledger_before_ref']['path'], root)
    prefix = before['events']; index = len(prefix)
    require(ledger['events'][:index] == prefix and len(ledger['events']) > index and
        ledger['events'][index] == event, 'exact actual prelaunch prefix and first appended completed event')
    # This container reconstructs the after-state from the actual immutable
    # before bytes and actual completed event already found in the live ledger.
    # It is neither a manufactured event nor a GPU permission document.
    completed = dict(before)
    completed.update(events=prefix + [event], active_reservation=None,
        gpu_wall_seconds=before['gpu_wall_seconds'] + event['elapsed_seconds'])
    history = load_module(root, refs[HISTORY], '_normal_guard_history_' + str(time.monotonic_ns()))
    history.verify_ledger_extension(completed, ledger, event,
        ledger_before_seconds=before['gpu_wall_seconds'], allow_active=True)
    active = ledger.get('active_reservation')
    if active is not None:
        require(type(active) is dict and type(active.get('id')) is str and active['id'] and
            active['id'] != event['reservation_id'], 'historical completion cannot be an unfinished reservation')
        next_mode = next((mode for mode in MODES if active.get('label') == label(mode)), None)
        require(next_mode is not None and MODES.index(next_mode) > MODES.index(config['mode']),
            'only separately authorized successor may be active during prior replay')
        next_config = read(config_path(next_mode), root)
        validate_configuration_document(root, next_config, config_path(next_mode))
        next_authority = load_authority(root, next_config)
        verify_source_proof(root, next_config, next_authority, 'before', refs)
        verify_launch_intent(root, next_config, next_authority, refs)
        active_expected = dict(label=label(next_mode), gpu_uuid=next_config['gpu_uuid'],
            seconds_limit=300, reserved_seconds=320, command=native_command(next_mode),
            permissions=next_authority['effective_permission_ref'],
            evidence=str(safe('experiments/prefix_io_v1/runs/' + label(next_mode), root)))
        require(all(exact(active.get(key), value) for key, value in active_expected.items()) and
            active.get('state') != 'cleanup_unresolved', 'active successor original reservation/source/permission mismatch')
        sid = active.get('session_id'); runner = active.get('runner_pid')
        require(type(sid) is int and sid > 0 and type(runner) is int and runner > 0 and
            active.get('process_group') == sid, 'actual active successor original session')
        tokens = (Path('/proc') / str(runner) / 'cmdline').read_bytes().decode().rstrip('\x00').split('\x00')
        require(GUARD in tokens and next_authority['effective_permission_ref']['path'] in tokens and
            label(next_mode) in tokens, 'actual successor original guard process')
    return event


def launch(mode, root=None):
    root = project(root); label(mode)
    # No GPU probe or process launch is reachable without all fresh authority gates.
    config, refs, authority = verify_configuration(root, config_path(mode))
    scope_value = verify_scope(root, config, authority)
    validate_live_interval(read(authority['context_ref']['path'], root), require_current=True)
    verify_previous_qualification(root, config, refs)
    ledger = read(LEDGER, root)
    number(ledger.get('gpu_wall_seconds'), 'used GPU budget')
    require(ledger.get('active_reservation') is None and ledger['gpu_wall_seconds'] + 320 <= 28800,
        'original cumulative GPU budget unavailable')
    run = 'experiments/prefix_io_v1/runs/' + label(mode)
    require(not safe(run, root).exists() and not safe(D + '/LAUNCH_INTENT_' + mode + '.json', root).exists(),
        'one attempt only; existing invocation must be inspected')
    require(os.name == 'posix', 'original Linux guard required')
    stat = os.statvfs(root); free = stat.f_bavail * stat.f_frsize
    require(free >= scope_value['primary_free_floor_bytes'] + scope_value['primary_reserve_bytes'], 'storage floor/reserve unavailable')
    before = ref(D + '/SOURCE_' + mode + '_BEFORE.json', root)
    launch_source = check_sources(mode, 'launch', root)
    telemetry = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=10).stdout
    rows = [line.split(',') for line in telemetry.strip().splitlines()]
    require(len(rows) == 1 and rows[0][0].strip() == config['gpu_uuid'] and int(rows[0][2]) >= 28000,
        'same expected free GPU unavailable')
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=10).stdout
    require(not processes.strip(), 'other GPU process; do not interfere')
    # Preserve the exact real prelaunch ledger bytes outside the common lock.
    ledger_raw = safe(LEDGER, root).read_bytes()
    actual_snapshot = json.loads(ledger_raw, object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, 'nonfinite original ledger snapshot'))
    latest_ledger = read(LEDGER, root)
    require(actual_snapshot == ledger and latest_ledger == ledger and latest_ledger.get('active_reservation') is None,
        'original ledger changed during authorized prelaunch checks')
    snapshot_relative = D + '/LEDGER_BEFORE_' + mode + '.json'
    with safe(snapshot_relative, root).open('xb') as stream:
        stream.write(ledger_raw)
    ledger_before_ref = ref(snapshot_relative, root)
    command = guard_command(mode)
    overrides = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    put('LAUNCH_INTENT_' + mode + '.json', dict(command=command, environment_overrides=overrides,
        scope_ref=ref(D + '/SCOPE_' + mode + '.json', root), config_ref=ref(config_path(mode), root),
        authority_ref=ref(authority_path(mode), root), source_lock_ref=ref(LOCK, root), binding_ref=ref(BINDING, root),
        human_grant_ref=authority['human_grant_ref'], permission_ref=authority['effective_permission_ref'],
        source_before_ref=before, launch_source_ref=launch_source, ledger_before_ref=ledger_before_ref,
        ledger_gpu_wall_seconds_before=ledger['gpu_wall_seconds'], primary_free_bytes=free,
        gpu_telemetry=telemetry, compute_processes=processes, launch_unix=time.time(), utc=now()), root)
    with safe(D + '/GPU_GUARD_' + mode + '_STDOUT.log', root).open('xb') as out, safe(D + '/GPU_GUARD_' + mode + '_STDERR.log', root).open('xb') as err:
        env = dict(os.environ); env.update(overrides)
        proc = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
            stdout=out, stderr=err, start_new_session=True, close_fds=True)
    return put('LAUNCH_RECEIPT_' + mode + '.json', dict(guard_pid=proc.pid, command=command,
        utc=now(), detached_original_guard=True, launch_is_not_success=True), root)


def after(mode, root=None):
    root = project(root); label(mode)
    config, _, authority = verify_configuration(root, config_path(mode))
    ledger = read(LEDGER, root)
    require(ledger.get('active_reservation') is None, 'wait for original guard completion')
    event = read('experiments/prefix_io_v1/runs/' + label(mode) + '/result.json', root)
    require(event.get('session_drained') is True and event.get('session_members_after_cleanup') == [],
        'original session remains alive')
    row = check_sources(mode, 'after', root)
    verify_completed_guard(root, config, authority)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('action', choices=('freeze', 'prepare', 'scope', 'before', 'launch', 'after'))
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--calibration-lock-sha256')
    args = parser.parse_args(argv)
    if args.action == 'freeze':
        result = freeze_common(args.calibration_lock_sha256, args.root)
    else:
        require(args.mode is not None, '--mode required')
        if args.action == 'scope':
            result = scope(args.root, args.mode)
        elif args.action == 'before':
            result = check_sources(args.mode, 'before', args.root)
        else:
            result = {'prepare': prepare, 'launch': launch, 'after': after}[args.action](args.mode, args.root)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

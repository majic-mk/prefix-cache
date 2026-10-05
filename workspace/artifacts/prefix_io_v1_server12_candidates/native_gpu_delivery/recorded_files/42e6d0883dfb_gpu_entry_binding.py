"""Strict site binding for one C5 native common-cost job; imports no GPU/model code.

CPU preparation does not issue a grant. The two immutable source locks avoid a
self-reference: revision bytes first, then site files including configuration.
The original run_gpu_stage.py alone owns GPU budget, timeout and session cleanup.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import time

DIR = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004'
LABEL = 'server11-c5-native-common-cost-gpu03'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
BASE_PERMISSION = 'experiments/prefix_io_v1/configs/permissions.yaml'
GUARD = 'experiments/prefix_io_v1/scripts/run_gpu_stage.py'
GUARD_SHA = '3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a'
ANCESTOR = 'artifacts/prefix_io_v1/server11-native-cost-v6-20261003/gpu-source-lock-native-cost.json'
ANCESTOR_SHA = '53fe06db2572b6d07b5dfdaac31ca46aed62fe0710bbb63e006a4d7d089d8e22'
MANIFEST = 'artifacts/prefix_io_v1/server11-native-cost-v1-20261003/SSD_INPUT_MANIFEST.json'
STORAGE = 'experiments/prefix_io_v1/runs/server11-native-cost-01-private-storage'
PURPOSE = 'CURRENT_CONTEXT_NATIVE_FULL_STEP_SSD_READ_DIAGNOSTIC_ONLY'
RUN = 'experiments/prefix_io_v1/runs/' + LABEL
SITE_LOCK = DIR + '/SITE_SOURCE_LOCK.json'
REVISION_LOCK = DIR + '/REVISION_SOURCE_LOCK.json'
BINDING = DIR + '/GPU_ENTRY_BINDING.json'
CONFIG = DIR + '/NATIVE_COST_CONFIG.json'
CONTEXT = DIR + '/LIVE_BINDING_CONTEXT.json'
GRANT = DIR + '/HUMAN_GPU_GRANT.json'
PERMISSION = DIR + '/EFFECTIVE_GPU_PERMISSION.json'
UUID_RE = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')
DENIED = ('allow_model_downloads', 'allow_driver_or_system_changes',
          'allow_shared_data_deletion', 'allow_payment', 'allow_new_cloud_rental')


def reject(message):
    raise RuntimeError('GPU_ENTRY_REJECTED: ' + message)


def exact(value, expected):
    return type(value) is type(expected) and value == expected


def config_relative(root, path):
    if isinstance(path, Path) and not path.is_absolute():
        # argparse(type=Path) preserves a normal relative CLI path as a Path.
        # It must be checked as project-relative, not relative_to(absolute root).
        path = path.as_posix()
    elif isinstance(path, Path) or (isinstance(path, str) and Path(path).is_absolute()):
        actual = Path(path)
        try:
            path = actual.relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            reject('configuration outside project')
    if path != CONFIG:
        reject('only the new live site configuration is accepted')
    safe(root, path)
    return path


def safe(root, relative):
    root = Path(root).resolve(strict=True)
    if (not isinstance(relative, str) or not relative or '\\' in relative
            or ':' in relative or relative.startswith('/')
            or any(p in ('', '.', '..') for p in relative.split('/'))):
        reject('project-relative POSIX path required')
    path = root / relative
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            reject('symlink rejected: ' + relative)
        cursor = cursor.parent
    if not path.resolve().is_relative_to(root):
        reject('outside project')
    return path


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            reject('duplicate JSON key: ' + key)
        value[key] = item
    return value


def parse(raw):
    return json.loads(raw, object_pairs_hook=_pairs,
                      parse_constant=lambda value: reject('nonfinite JSON value'))


def read(root, relative):
    return parse(safe(root, relative).read_bytes())


def ref(root, relative):
    path = safe(root, relative)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return {'path': relative, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def checked_ref(root, row):
    if (not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}
            or isinstance(row['bytes'], bool) or not isinstance(row['bytes'], int)
            or row['bytes'] < 0 or not isinstance(row['sha256'], str)
            or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
        reject('invalid immutable reference')
    if ref(root, row['path']) != row:
        reject('bytes/source drift: ' + row['path'])
    return row


def checked_rows(root, rows, *, verify_bytes=True):
    if not isinstance(rows, list) or not rows:
        reject('nonempty source closure required')
    result = {}
    for row in rows:
        if verify_bytes:
            checked_ref(root, row)
        elif (not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}
              or isinstance(row['bytes'], bool) or not isinstance(row['bytes'], int)
              or row['bytes'] < 0 or not isinstance(row['sha256'], str)
              or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            reject('invalid closure reference')
        else:
            safe(root, row['path'])
        if row['path'] in result:
            reject('duplicate source path')
        result[row['path']] = row
    return result


def number(value, name, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < minimum):
        reject('invalid number: ' + name)
    return value


def permission_document(root, relative):
    raw = safe(root, relative).read_bytes()
    try:
        value = parse(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Real launch uses the installed, already frozen PyYAML. CPU tests use
        # JSON (also valid YAML), and no optional package is imported for rejects.
        import yaml
        class StrictLoader(yaml.SafeLoader):
            pass
        def mapping(loader, node, deep=False):
            return _pairs([(loader.construct_object(k, deep=deep),
                            loader.construct_object(v, deep=deep)) for k, v in node.value])
        StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        value = yaml.load(raw.decode('utf-8'), Loader=StrictLoader)
    if not isinstance(value, dict):
        reject('permission mapping required')
    return value


def _revision(root, row, *, full=False):
    checked_ref(root, row)
    if row['path'] != REVISION_LOCK:
        reject('unexpected revision lock')
    lock = read(root, row['path'])
    if lock.get('schema') != 'c5_gpu_entry_revision_asset_lock_v1':
        reject('revision lock schema')
    ancestor = lock.get('baseline_source_lock')
    checked_ref(root, ancestor)
    if ancestor['path'] != ANCESTOR or ancestor['sha256'] != ANCESTOR_SHA:
        reject('source ancestry mismatch; old authorization is never inherited')
    original = read(root, ANCESTOR)
    if len(original.get('files', [])) != 4655:
        reject('incomplete original asset ancestry')
    rows = checked_rows(root, lock.get('files'), verify_bytes=full)
    for original_row in original['files']:
        if rows.get(original_row['path']) != original_row:
            reject('original source omitted or changed')
    required = [GUARD, BASE_PERMISSION, MANIFEST, ANCESTOR,
                DIR + '/gpu_entry_binding.py', DIR + '/control_native_cost_job.py',
                DIR + '/run_native_cost_experiment.py',
                DIR + '/native_conditional_cost.py', DIR + '/prepare_and_verify_native_cost.py',
                DIR + '/common_candidate/native_full_step_collector.py',
                DIR + '/common_candidate/single_file_runtime_binding.py',
                DIR + '/NATIVE_SOURCE_INHERITANCE.json',
                DIR + '/common_candidate/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py']
    for relative in required:
        if relative not in rows:
            reject('revision closure missing: ' + relative)
        if not full:
            checked_ref(root, rows[relative])
    if rows[GUARD]['sha256'] != GUARD_SHA:
        reject('original budget guard changed')
    inheritance = read(root, DIR + '/NATIVE_SOURCE_INHERITANCE.json')
    if (inheritance.get('schema') != 'c5_gpu_entry_native_source_inheritance_v1'
            or len(inheritance.get('unchanged_common_files', [])) != 63):
        reject('full unchanged common source inheritance required')
    for original_row in inheritance['unchanged_common_files']:
        common_ref = dict(path=DIR + '/common_candidate/' + original_row['path'],
                          bytes=original_row['bytes'], sha256=original_row['sha256'])
        if rows.get(common_ref['path']) != common_ref:
            reject('unchanged common candidate source omitted or changed')
    manifest = read(root, MANIFEST)
    files = [item for group in manifest.get('groups', []) for item in group.get('files', [])]
    if len(files) != 24 or manifest.get('file_count') != 24:
        reject('production KV input count changed')
    for item in files:
        data_ref = dict(path=STORAGE + '/' + item['path'], bytes=item['bytes'], sha256=item['sha256'])
        if item['bytes'] != 917504 or rows.get(data_ref['path']) != data_ref:
            reject('KV input source closure missing or changed')
    for item in manifest.get('source_refs', []):
        if rows.get(item['path']) != item:
            reject('KV producer provenance missing')
    if not full:
        # These are the actual new entry/collector sources, not a path cache.
        # Full large-asset bytes are audited by the immutable before/after proofs.
        for relative, source in rows.items():
            if relative.startswith(DIR + '/'):
                checked_ref(root, source)
    return rows


def _resource_shape(resource):
    if (not isinstance(resource, dict)
            or resource.get('origin') != 'live_linux_readonly_resource_probe'
            or resource.get('status') != 'LIVE_RESOURCE_READY'
            or resource.get('nvidia_nodes_visible') is not True
            or resource.get('compute_processes') != []):
        reject('actual resource observation unavailable')
    gpu = resource.get('gpu_uuid')
    if not isinstance(gpu, str) or not UUID_RE.fullmatch(gpu):
        reject('one actual GPU UUID required')
    number(resource.get('cpu_cores'), 'CPU cores', 1)
    number(resource.get('cpu_affinity_count'), 'CPU affinity', 1)
    number(resource.get('available_memory_bytes'), 'available memory', 32 * 1024 ** 3)
    number(resource.get('gpu_free_mib'), 'free VRAM', 28000)
    number(resource.get('primary_free_bytes'), 'PRIMARY free storage', 10 * 1024 ** 3)
    if resource.get('cpu_counters_readable') is not True:
        reject('CPU quota/counters unknown')
    return gpu


def load_binding(root, binding_relative):
    if binding_relative != BINDING:
        reject('wrong revision binding path')
    value = read(root, binding_relative)
    if value.get('schema') != 'c5_gpu_entry_site_binding_v1' or value.get('status') != 'LIVE_HUMAN_BOUND':
        reject('UNBOUND/CPU preparation has no execution authority')
    if value.get('origin') != 'live_site_binding' or value.get('synthetic') is not False:
        reject('synthetic or unknown binding origin')
    if value.get('root') != str(Path(root).resolve()) or value.get('label') != LABEL:
        reject('target root/label mismatch')
    if value.get('site_lock_relative') != SITE_LOCK or value.get('config_relative') != CONFIG:
        reject('site closure/config path mismatch')
    rows = _revision(root, value.get('revision_lock_ref'))
    for key, relative in [('context_ref', CONTEXT), ('human_grant_ref', GRANT),
                          ('effective_permission_ref', PERMISSION)]:
        checked_ref(root, value.get(key))
        if value[key]['path'] != relative:
            reject('wrong site reference: ' + key)
    context = read(root, CONTEXT)
    if (context.get('schema') != 'c5_gpu_entry_live_context_v1'
            or context.get('origin') != 'live_site_readonly_context'
            or context.get('revision_lock_ref') != value['revision_lock_ref']
            or context.get('label') != LABEL or context.get('root') != value['root']
            or context.get('seconds_limit') != 1200 or context.get('reserved_seconds') != 1220):
        reject('invalid live binding context')
    uuid = _resource_shape(context.get('resources'))
    if value.get('gpu_uuid') != uuid:
        reject('binding/context UUID mismatch')
    grant = read(root, GRANT)
    if (grant.get('schema') != 'c5_gpu_entry_explicit_human_grant_v1'
            or grant.get('issuer') != 'human_user' or grant.get('authorization') is not True
            or grant.get('context_ref') != value['context_ref']
            or grant.get('gpu_uuid') != uuid or grant.get('label') != LABEL
            or grant.get('revision_lock_ref') != value['revision_lock_ref']
            or not exact(grant.get('max_attempts'), 1) or not exact(grant.get('max_processes'), 6)
            or not exact(grant.get('seconds_limit'), 1200) or not exact(grant.get('reserved_seconds'), 1220)
            or not isinstance(grant.get('human_instruction'), str)
            or not grant['human_instruction'].strip()
            or grant.get('instruction_is_gpu_execution_authorization') is not True):
        reject('new direct human GPU execution grant required')
    if grant.get('human_instruction') in ('继续', '继续直至可以上gpu验证', '继续直至可以上GPU验证'):
        reject('preparation instruction is not GPU execution authorization')
    permission = permission_document(root, PERMISSION)
    base = permission_document(root, BASE_PERMISSION)
    if permission.get('allow_gpu_runs') is not True or permission.get('approved_gpu_ids') != [uuid]:
        reject('effective permission UUID mismatch')
    if number(permission.get('max_gpu_hours'), 'budget hours', 0.01) > min(8, number(base.get('max_gpu_hours'), 'base budget')):
        reject('cumulative GPU budget enlarged')
    for key in ('approved_experiment_root', 'approved_dependency_root'):
        if permission.get(key) != base.get(key):
            reject('effective permission root differs from base')
    if permission.get('approved_experiment_root') != str(safe(root, 'experiments/prefix_io_v1/runs')):
        reject('guard evidence path differs from bound run directory')
    if any(permission.get(key) is not False for key in DENIED) or permission.get('approved_auxiliary_storage') is not None:
        reject('forbidden permissions expanded')
    for key, expected in [('label', LABEL), ('revision_lock_ref', value['revision_lock_ref']),
                          ('context_ref', value['context_ref']), ('human_grant_ref', value['human_grant_ref']),
                          ('seconds_limit', 1200), ('reserved_seconds', 1220), ('max_attempts', 1), ('max_processes', 6)]:
        if not exact(permission.get('gpu_entry_binding', {}).get(key), expected):
            reject('effective permission job/source binding mismatch: ' + key)
    if rows[BASE_PERMISSION] != ref(root, BASE_PERMISSION):
        reject('base permission drift')
    return value


def verify_configuration(root, config_path):
    config_path = config_relative(root, config_path)
    config = read(root, config_path)
    binding_ref = config.get('gpu_entry_binding_ref')
    checked_ref(root, binding_ref)
    binding = load_binding(root, binding_ref['path'])
    expected = dict(root=str(Path(root).resolve()), label=LABEL, gpu_uuid=binding['gpu_uuid'],
                    purpose=PURPOSE, seconds_limit=1200, reserved_seconds=1220,
                    active_ledger=LEDGER, storage=STORAGE, out=RUN + '/details',
                    source_lock=SITE_LOCK, input_manifest=MANIFEST,
                    collector_relative=DIR + '/common_candidate/native_full_step_collector.py',
                    gpu_entry_binding_ref=binding_ref)
    if set(config) != set(expected) or any(not exact(config[key], value) for key, value in expected.items()):
        reject('configuration fields differ from frozen one-job contract')
    lock = read(root, SITE_LOCK)
    if lock.get('schema') != 'c5_gpu_entry_site_source_lock_v1' or lock.get('revision_lock_ref') != binding['revision_lock_ref']:
        reject('final site source lock required')
    refs = checked_rows(root, lock.get('files'), verify_bytes=False)
    revision = _revision(root, binding['revision_lock_ref'])
    required = dict(revision)
    for path in (REVISION_LOCK, CONFIG, BINDING, CONTEXT, GRANT, PERMISSION):
        required[path] = ref(root, path)
    if refs != required:
        reject('final site closure is incomplete or widened')
    verify_source_proof(root, 'SOURCE_BEFORE_VERIFICATION.json', config, binding, refs)
    return config, refs, binding


def closure_digest(refs):
    raw = json.dumps([refs[key] for key in sorted(refs)], sort_keys=True,
                     separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def verify_source_proof(root, filename, config, binding, refs):
    proof = read(root, DIR + '/' + filename)
    expected = dict(schema='c5_gpu_entry_full_source_verification_v1',
                    status='FULL_BYTE_SOURCE_CLOSURE_VERIFIED', label=LABEL,
                    gpu_uuid=binding['gpu_uuid'], source_lock_ref=ref(root, SITE_LOCK),
                    revision_lock_ref=binding['revision_lock_ref'], config_ref=ref(root, CONFIG),
                    binding_ref=ref(root, BINDING), context_ref=binding['context_ref'],
                    human_grant_ref=binding['human_grant_ref'],
                    permission_ref=binding['effective_permission_ref'],
                    source_count=len(refs), source_rows_sha256=closure_digest(refs), failed=[])
    expected['files_verified'] = len(refs)
    if any(not exact(proof.get(key), value) for key, value in expected.items()):
        reject('missing, partial, stale or foreign full byte source verification')
    if proof.get('phase') != filename.removeprefix('SOURCE_').removesuffix('_VERIFICATION.json').lower():
        reject('source verification phase mismatch')
    return proof


def native_command():
    return ['.venv/bin/python', '-B', DIR + '/run_native_cost_experiment.py', '--execute', '--config', CONFIG]


def guard_command():
    return ['.venv/bin/python', '-B', GUARD, '--permissions-path', PERMISSION,
            '--label', LABEL, '--seconds', '1200', '--'] + native_command()


def verify_launch_intent(root, binding):
    intent = read(root, DIR + '/GPU_LAUNCH_INTENT.json')
    if (intent.get('command') != guard_command()
            or intent.get('launch_source_ref') != ref(root, DIR + '/SOURCE_LAUNCH_VERIFICATION.json')
            or intent.get('site_lock_ref') != ref(root, SITE_LOCK)
            or intent.get('binding_ref') != ref(root, BINDING)
            or intent.get('human_grant_ref') != binding['human_grant_ref']):
        reject('launch intent/source binding mismatch')
    for key, relative in [('plan_ref', DIR + '/NATIVE_COST_PLAN.json'),
                          ('plan_ref_receipt_ref', DIR + '/PLAN_REFERENCE.json'),
                          ('scope_ref', DIR + '/SIX_PROCESS_AUTHORIZED_SCOPE.json')]:
        checked_ref(root, intent.get(key))
        if intent[key]['path'] != relative:
            reject('prelaunch plan/scope provenance missing')
    if read(root, intent['plan_ref_receipt_ref']['path']) != intent['plan_ref']:
        reject('prelaunch plan reference container mismatch')
    return intent


def verify_active_guard(root, config, binding):
    actual, _, actual_binding = verify_configuration(root, CONFIG)
    if actual != config or actual_binding != binding:
        reject('active caller configuration mismatch')
    refs = checked_rows(root, read(root, SITE_LOCK).get('files'), verify_bytes=False)
    verify_source_proof(root, 'SOURCE_LAUNCH_VERIFICATION.json', config, binding, refs)
    verify_launch_intent(root, binding)
    budget = read(root, LEDGER)
    reservation = budget.get('active_reservation')
    if not isinstance(reservation, dict):
        reject('original guard has not reserved this GPU job')
    expected = dict(label=LABEL, gpu_uuid=binding['gpu_uuid'], seconds_limit=1200,
                    reserved_seconds=1220, command=native_command(),
                    permissions=binding['effective_permission_ref'], evidence=str(safe(root, RUN)))
    if any(not exact(reservation.get(key), value) for key, value in expected.items()):
        reject('active original guard reservation mismatch')
    if not isinstance(reservation.get('id'), str) or not reservation['id']:
        reject('actual original guard reservation id missing')
    runner_pid = reservation.get('runner_pid')
    if isinstance(runner_pid, bool) or not isinstance(runner_pid, int) or runner_pid <= 0:
        reject('actual original guard process missing')
    if reservation.get('session_id') is None:
        original_id = reservation['id']
        deadline = time.monotonic() + 1.0
        while reservation.get('session_id') is None and time.monotonic() < deadline:
            time.sleep(0.01)
            budget = read(root, LEDGER)
            reservation = budget.get('active_reservation')
            if (not isinstance(reservation, dict) or reservation.get('id') != original_id
                    or any(not exact(reservation.get(key), value) for key, value in expected.items())):
                reject('guard reservation changed during bounded session publication wait')
    session = reservation.get('session_id')
    if (isinstance(session, bool) or not isinstance(session, int) or session <= 0
            or not hasattr(os, 'getsid') or os.getsid(0) != session):
        reject('native process is outside the original guard session')
    if reservation.get('process_group') != session or reservation.get('state') == 'cleanup_unresolved':
        reject('guard session is unresolved')
    if os.getpgid(0) != session or os.environ.get('CUDA_VISIBLE_DEVICES') != binding['gpu_uuid']:
        reject('original guard process group/visible GPU mismatch')
    cmdline = Path('/proc') / str(runner_pid) / 'cmdline'
    tokens = cmdline.read_bytes().decode('utf-8').rstrip('\x00').split('\x00')
    if GUARD not in tokens or PERMISSION not in tokens or LABEL not in tokens:
        reject('active reservation runner is not the original bound guard')
    number(budget.get('gpu_wall_seconds'), 'used budget')
    if budget['gpu_wall_seconds'] + 1220 > 28800:
        reject('original eight-hour GPU budget unavailable')
    return reservation


def verify_completed_guard(root, config, binding, guard_ref=None):
    actual, _, actual_binding = verify_configuration(root, CONFIG)
    if config != actual or binding != actual_binding:
        reject('completed caller configuration mismatch')
    refs = checked_rows(root, read(root, SITE_LOCK).get('files'), verify_bytes=False)
    verify_source_proof(root, 'SOURCE_AFTER_VERIFICATION.json', config, binding, refs)
    expected_ref = ref(root, RUN + '/result.json')
    if guard_ref is not None and guard_ref != expected_ref:
        reject('completed guard reference mismatch')
    event = read(root, expected_ref['path'])
    expected = dict(label=LABEL, gpu_uuid=binding['gpu_uuid'], command=native_command(),
                    permissions=binding['effective_permission_ref'], evidence=str(safe(root, RUN)),
                    exit=0, child_exit=0, timed_out=False, interrupted_signal=None, error=None,
                    gpu_job_attempted=True, session_drained=True,
                    session_members_before_cleanup=[], session_members_after_cleanup=[])
    if any(not exact(event.get(key), value) for key, value in expected.items()):
        reject('normal original guard completion required')
    elapsed = number(event.get('elapsed_seconds'), 'actual GPU elapsed', 0.000001)
    if elapsed > 1220:
        reject('guard elapsed exceeds bounded job reserve')
    if (not isinstance(event.get('reservation_id'), str) or not event['reservation_id']
            or isinstance(event.get('session_id'), bool) or not isinstance(event.get('session_id'), int)
            or event['session_id'] <= 0):
        reject('real guard reservation/session evidence missing')
    ledger = read(root, LEDGER)
    if ledger.get('active_reservation') is not None or not isinstance(ledger.get('events'), list):
        reject('guard/ledger not drained')
    matches = [row for row in ledger['events'] if row.get('reservation_id') == event['reservation_id']]
    if matches != [event]:
        reject('guard result is not the unique original ledger event')
    number(ledger.get('gpu_wall_seconds'), 'cumulative GPU usage')
    if ledger['gpu_wall_seconds'] < elapsed or ledger['gpu_wall_seconds'] > 28800:
        reject('original budget accounting mismatch')
    intent = verify_launch_intent(root, binding)
    number(intent.get('ledger_gpu_wall_seconds_before'), 'prelaunch ledger usage')
    if ledger['gpu_wall_seconds'] != intent['ledger_gpu_wall_seconds_before'] + elapsed:
        reject('original guard accounting delta mismatch')
    return event


def _read_text(path):
    return Path(path).read_text(encoding='utf-8').strip()


def nvidia_device_nodes(device_root=Path('/dev')):
    """Return allocated numeric NVIDIA character nodes with a real control node.

    Container allocation can retain the host ordinal, such as nvidia5. UVM,
    capability directories, symlinks and regular numeric files are not GPUs.
    This reads device metadata only; unique UUID/VRAM/process checks stay below.
    """
    device_root = Path(device_root)
    try:
        if not stat.S_ISCHR((device_root / 'nvidiactl').lstat().st_mode):
            return []
        nodes = []
        for path in device_root.iterdir():
            if re.fullmatch(r'nvidia[0-9]+', path.name):
                try:
                    if stat.S_ISCHR(path.lstat().st_mode):
                        nodes.append(str(path))
                except OSError:
                    continue
        return sorted(nodes)
    except OSError:
        return []


def probe_resources(root):
    """Read actual Linux resources; nvidia-smi telemetry never launches a model."""
    result = dict(origin='live_linux_readonly_resource_probe', status='RESOURCE_LIMITED',
                  nvidia_nodes_visible=False, nvidia_device_nodes=[], gpu_uuid=None, gpu_free_mib=None,
                  compute_processes=None, cpu_cores=None, cpu_affinity_count=None,
                  cpu_counters_readable=False, available_memory_bytes=None,
                  primary_free_bytes=None, errors=[])
    if os.name != 'posix' or not Path('/proc/self/cgroup').exists():
        result['errors'].append('Linux /proc resource observations unavailable')
        return result
    try:
        affinity = len(os.sched_getaffinity(0))
        result['cpu_affinity_count'] = affinity
        cgroups = _read_text('/proc/self/cgroup').splitlines()
        unified = [line.split(':', 2)[2] for line in cgroups if line.startswith('0::')]
        if len(unified) != 1:
            reject('readable cgroup-v2 quota required')
        base = Path('/sys/fs/cgroup') / unified[0].lstrip('/')
        if not base.is_dir():
            base = Path('/sys/fs/cgroup')
        quotas, limits, stats, quota_observed, memory_observed = [], [], [], False, False
        node = base
        while node.is_relative_to(Path('/sys/fs/cgroup')):
            if (node / 'cpu.max').exists():
                quota_observed = True
                quota, period = _read_text(node / 'cpu.max').split()
                period = int(period)
                if period <= 0 or (quota != 'max' and int(quota) <= 0):
                    reject('CPU quota invalid')
                if quota != 'max': quotas.append(int(quota) / period)
            if (node / 'cpu.stat').exists():
                counters = dict(line.split() for line in _read_text(node / 'cpu.stat').splitlines())
                if 'usage_usec' in counters and int(counters['usage_usec']) >= 0: stats.append(True)
            if (node / 'memory.max').exists():
                memory_observed = True
                limit = _read_text(node / 'memory.max')
                if limit != 'max':
                    limits.append(max(0, int(limit) - int(_read_text(node / 'memory.current'))))
            if node == Path('/sys/fs/cgroup'): break
            node = node.parent
        result['cpu_cores'] = min([affinity] + quotas)
        result['cpu_counters_readable'] = bool(stats) and quota_observed
        if not memory_observed:
            reject('container/host memory quota observation unavailable')
        info = dict(line.split(':', 1) for line in _read_text('/proc/meminfo').splitlines())
        host_available = int(info['MemAvailable'].split()[0]) * 1024
        result['available_memory_bytes'] = min([host_available] + limits)
        st = os.statvfs(root)
        result['primary_free_bytes'] = st.f_bavail * st.f_frsize
        result['nvidia_device_nodes'] = nvidia_device_nodes()
        result['nvidia_nodes_visible'] = bool(result['nvidia_device_nodes'])
        if not result['nvidia_nodes_visible']:
            result['errors'].append('no NVIDIA device nodes')
            return result
        query = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=10, check=True).stdout
        devices = [line.split(',') for line in query.strip().splitlines()]
        if len(devices) != 1 or len(devices[0]) != 2:
            reject('exactly one visible GPU required')
        result['gpu_uuid'] = devices[0][0].strip()
        result['gpu_free_mib'] = int(devices[0][1].strip())
        processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10, check=True).stdout
        result['compute_processes'] = [int(line.strip()) for line in processes.strip().splitlines() if line.strip()]
        result['status'] = 'LIVE_RESOURCE_READY'
        _resource_shape(result)
    except Exception as exc:
        result['status'] = 'RESOURCE_LIMITED'
        result['errors'].append(str(exc))
    return result


def check_launch(root, config, binding):
    actual, _, actual_binding = verify_configuration(root, CONFIG)
    if actual != config or actual_binding != binding:
        reject('launch config drift')
    resources = probe_resources(root)
    if _resource_shape(resources) != binding['gpu_uuid']:
        reject('current GPU differs from live binding')
    ledger = read(root, LEDGER)
    if ledger.get('active_reservation') is not None:
        reject('unfinished original GPU reservation')
    if number(ledger.get('gpu_wall_seconds'), 'cumulative GPU usage') + 1220 > 28800:
        reject('insufficient original cumulative budget')
    if any(safe(root, path).exists() for path in (RUN, DIR + '/GPU_LAUNCH_INTENT.json', DIR + '/GPU_LAUNCH_RECEIPT.json')):
        reject('label already used; no automatic retry')
    return resources, ledger

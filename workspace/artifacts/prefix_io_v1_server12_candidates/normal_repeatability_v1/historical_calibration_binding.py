"""Replay completed server12 calibration against its actual ledger snapshot.

This metadata adapter changes no original source or live ledger. It reuses the
complete pinned server12 guard checks before its live-ledger read, then validates the
actual as-of snapshot and the current ledger's append-only extension. New GPU
execution authority remains the separate normal controller's responsibility.
"""
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
import sys

G = 'artifacts/prefix_io_v1/server11-c5-gpu-entry-path-revision-20261004'
CALIBRATION = 'artifacts/prefix_io_v1/server12-c5-native-recalibration-20261004'
D = 'artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'
ORIGINAL_BINDING = CALIBRATION + '/gpu_entry_binding.py'
ORIGINAL_BYTES = 33971
ORIGINAL_SHA = '9a434ce4c9a47088df0582d32640174a71d69271b3c2569975d1ddcbde9e23fd'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
SNAPSHOT = D + '/SERVER12_LEDGER_SNAPSHOT.json'
SNAPSHOT_BYTES = 393479
SNAPSHOT_SHA = 'f7b749aff3744b65f146d7c7fa44cee813d2a7b9bd1b7c24968e150aa7216e77'
HISTORICAL_RESERVATION = 'bbe0d5e7cc084147985d4d413301e2ce'
LEGACY_FIRST_EVENT_SHA = '5641d122126438219b2636cc39957fb59f9114336000a97e9e9d49a688725abc'
MAX_SECONDS = 28800


def require(value, message):
    if not value:
        raise ValueError('HISTORICAL_CALIBRATION_REJECTED: ' + message)


def number(value, name, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
        'finite typed ' + name)
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def strict_pairs(items):
    value = {}
    for key, item in items:
        require(key not in value, 'duplicate JSON key')
        value[key] = item
    return value


def parse(raw):
    return json.loads(raw, object_pairs_hook=strict_pairs,
        parse_constant=lambda value: require(False, 'nonfinite JSON'))


def safe(root, relative):
    root = Path(root).resolve(strict=True)
    require(type(relative) is str and relative and '\\' not in relative and ':' not in relative
        and not relative.startswith('/') and all(x not in ('', '.', '..') for x in relative.split('/')),
        'project-relative source required')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'source/evidence symlink refused')
    require(path.resolve().is_relative_to(root), 'source/evidence outside project')
    return path


def pinned_bytes(root, relative, size, digest):
    path = safe(root, relative)
    require(path.is_file() and path.stat().st_size == size, 'actual pinned file size: ' + relative)
    raw = path.read_bytes()
    require(len(raw) == size and sha256(raw).hexdigest() == digest, 'actual pinned file SHA: ' + relative)
    return raw


def original_api(root):
    raw = pinned_bytes(root, ORIGINAL_BINDING, ORIGINAL_BYTES, ORIGINAL_SHA)
    name = '_normal_unchanged_G_metadata_' + str(id(raw))
    spec = importlib.util.spec_from_file_location(name, safe(root, ORIGINAL_BINDING))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(spec.origin), 'exec', dont_inherit=True), module.__dict__)
        pinned_bytes(root, ORIGINAL_BINDING, ORIGINAL_BYTES, ORIGINAL_SHA)
        return module
    finally:
        sys.modules.pop(name, None)


def completed_guard_prefix(api, root, config, binding, guard_ref=None):
    """Execute every original guard/source/authority check before ledger read.

    The namespace is a copy. Neither the original module nor its functions are
    monkeypatched. The only added statement returns its already checked event.
    """
    raw = pinned_bytes(root, ORIGINAL_BINDING, ORIGINAL_BYTES, ORIGINAL_SHA)
    function = next(node for node in ast.parse(raw).body
        if isinstance(node, ast.FunctionDef) and node.name == 'verify_completed_guard')
    stop = next(index for index, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'ledger')
    require(stop > 10, 'complete original pre-ledger validation prefix')
    function = deepcopy(function)
    function.name = '_historical_original_completed_prefix'
    function.body = function.body[:stop] + ast.parse('return event').body
    namespace = dict(api.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
        str(safe(root, ORIGINAL_BINDING)) + '::unchanged_completed_prefix', 'exec', dont_inherit=True), namespace)
    return namespace[function.name](root, config, binding, guard_ref)


def event_ids(events, *, allow_legacy_first):
    ids = set()
    for index, row in enumerate(events):
        require(type(row) is dict, 'actual ledger event mapping')
        number(row.get('elapsed_seconds'), 'event elapsed seconds')
        identity = row.get('reservation_id')
        if identity is None:
            # The frozen real ledger starts with one older smoke record without
            # an ID. Preserve that exact historical record; invent no new ID.
            require(allow_legacy_first and index == 0 and
                sha256(canonical(row).encode()).hexdigest() == LEGACY_FIRST_EVENT_SHA,
                'only the exact frozen legacy first event may omit reservation ID')
            continue
        require(type(identity) is str and identity and identity not in ids,
            'nonempty unique completed reservation IDs')
        ids.add(identity)
    return ids


def validate_active_reservation(active, completed_ids, remaining_seconds, *, allow_active):
    if active is None:
        return 'IDLE'
    require(allow_active is True and type(active) is dict, 'active reservation permitted only as new metadata')
    identity = active.get('id')
    require(type(identity) is str and identity and identity not in completed_ids,
        'completed historical reservation cannot still be active')
    reserved = number(active.get('reserved_seconds'), 'active reserved seconds', 1)
    limit = number(active.get('seconds_limit'), 'active execution limit', 1)
    require(limit <= reserved <= remaining_seconds and active.get('state') != 'cleanup_unresolved',
        'new active reservation fits original remaining budget and is resolved')
    return 'ACTIVE_NEW_RESERVATION_METADATA_ONLY'


def verify_ledger_extension(snapshot, current, historical_event, *, ledger_before_seconds, allow_active=True):
    """Pure checked history relation; fixtures never become native receipts.

    For another normal arm the controller can project its already verified
    before ledger plus the actual unique completed guard event. This helper
    validates that post-event prefix against the current real ledger. Only
    verify_completed_guard below additionally loads the fixed real server12 snapshot.
    """
    require(type(allow_active) is bool and type(snapshot) is dict and type(current) is dict
        and type(historical_event) is dict, 'explicit typed history inputs')
    before = number(ledger_before_seconds, 'historical prelaunch budget')
    require(snapshot.get('active_reservation') is None, 'actual completed snapshot idle')
    prior, events = snapshot.get('events'), current.get('events')
    require(type(prior) is list and 1 <= len(prior) <= 10000 and type(events) is list
        and len(prior) <= len(events) <= 10000, 'bounded complete ledger prefix')
    require(canonical(prior[-1]) == canonical(historical_event), 'snapshot ends at the actual historical guard')
    ids = event_ids(events, allow_legacy_first=True)
    identity = historical_event.get('reservation_id')
    require(type(identity) is str and identity and
        sum(row.get('reservation_id') == identity for row in prior) == 1,
        'historical guard has one real reservation identity')
    require(canonical(events[:len(prior)]) == canonical(prior), 'entire historical prefix changed or truncated')
    historical_wall = number(snapshot.get('gpu_wall_seconds'), 'historical used budget')
    require(historical_wall == before + number(historical_event.get('elapsed_seconds'), 'historical elapsed'),
        'actual historical guard budget delta')
    expected = historical_wall
    for row in events[len(prior):]:
        expected += number(row.get('elapsed_seconds'), 'appended guard elapsed')
    actual = number(current.get('gpu_wall_seconds'), 'current used budget')
    require(actual == expected and actual <= MAX_SECONDS, 'exact appended-event accounting and original eight-hour cap')
    active = validate_active_reservation(current.get('active_reservation'), ids, MAX_SECONDS-actual,
        allow_active=allow_active)
    return dict(status='PASS_HISTORICAL_LEDGER_EXTENSION_METADATA', snapshot_events=len(prior),
        current_events=len(events), appended_events=len(events)-len(prior), historical_gpu_seconds=historical_wall,
        current_gpu_seconds=actual, current_active_status=active, GPU_operations=0,
        new_gpu_authority_issued=False, normal_native_execution_qualified=False)


def verify_completed_guard(project, config, binding, guard_ref=None):
    root = Path(project).resolve(strict=True)
    api = original_api(root)
    event = completed_guard_prefix(api, root, config, binding, guard_ref)
    require(event.get('reservation_id') == HISTORICAL_RESERVATION, 'only actual completed server12 calibration')
    snapshot = parse(pinned_bytes(root, SNAPSHOT, SNAPSHOT_BYTES, SNAPSHOT_SHA))
    intent = api.verify_launch_intent(root, binding)
    current = api.read(root, LEDGER)
    verify_ledger_extension(snapshot, current, event,
        ledger_before_seconds=intent.get('ledger_gpu_wall_seconds_before'), allow_active=True)
    if current.get('active_reservation') is not None:
        verify_diagnostic_active(root, current['active_reservation'])
    pinned_bytes(root, ORIGINAL_BINDING, ORIGINAL_BYTES, ORIGINAL_SHA)
    pinned_bytes(root, SNAPSHOT, SNAPSHOT_BYTES, SNAPSHOT_SHA)
    return event


def checked_metadata_ref(root, row):
    """A real source/permission reference, never a receipt or authorization flag."""
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and
        type(row['bytes']) is int and 0 < row['bytes'] <= 32 * 1024**2 and
        type(row['sha256']) is str and len(row['sha256']) == 64 and
        all(c in '0123456789abcdef' for c in row['sha256']), 'exact nonempty metadata source reference')
    return pinned_bytes(root, row['path'], row['bytes'], row['sha256'])


def verify_diagnostic_active(root, active):
    """Authenticate the new diagnostic scope before allowing history replay.

    The original guard pins the effective permission bytes. That document pins
    the source lock and config, so the new controller is checked before import.
    Its public metadata API then verifies fresh authority, source proofs, actual
    guard/session/GPU and index. This does not authorize a process or manufacture
    a receipt. The API is deliberately receipt-free to avoid recursive issuance.
    """
    require(type(active) is dict and type(active.get('id')) is str and active['id'],
        'actual original active reservation required')
    labels = tuple('server12-c5-native-repeat-off' + str(i + 1).zfill(2) for i in range(3))
    require(active.get('label') in labels, 'unknown active diagnostic namespace')
    index = labels.index(active['label'])
    config_path = D + '/CONFIG_off' + str(index + 1).zfill(2) + '.json'
    permission_path = D + '/EFFECTIVE_GPU_PERMISSION_off' + str(index + 1).zfill(2) + '.json'
    permission_ref = active.get('permissions')
    require(type(permission_ref) is dict and permission_ref.get('path') == permission_path,
        'actual active diagnostic permission path')
    permission = parse(checked_metadata_ref(root, permission_ref))
    binding = permission.get('normal_mode_binding')
    require(type(binding) is dict and type(binding.get('diagnostic_index')) is int and
        binding['diagnostic_index'] == index and binding.get('mode') == 'off' and
        binding.get('label') == active['label'] and binding.get('gpu_uuid') == active.get('gpu_uuid'),
        'actual permission index/mode/label/GPU binding')
    config_ref = binding.get('config_ref')
    require(type(config_ref) is dict and config_ref.get('path') == config_path,
        'actual permission configuration reference')
    config = parse(checked_metadata_ref(root, config_ref))
    require(type(config.get('diagnostic_index')) is int and config['diagnostic_index'] == index and
        config.get('mode') == 'off' and config.get('label') == active['label'] and
        config.get('gpu_uuid') == active.get('gpu_uuid'), 'actual diagnostic config identity')
    source_ref = binding.get('source_lock_ref')
    require(type(source_ref) is dict and source_ref.get('path') == D + '/COMMON_SOURCE_LOCK.json' and
        config.get('source_lock') == source_ref['path'], 'permission pins the diagnostic common source lock')
    locked = parse(checked_metadata_ref(root, source_ref))
    rows = locked.get('files')
    require(type(rows) is list and 1 <= len(rows) <= 10000,
        'bounded real diagnostic source closure')
    controller_path = D + '/control_p4_single_file.py'
    candidates = [row for row in rows if type(row) is dict and row.get('path') == controller_path]
    require(len(candidates) == 1, 'one actual controller source in the permission-pinned closure')
    raw = checked_metadata_ref(root, candidates[0])
    name = '_repeatability_history_actual_controller_' + str(id(raw))
    spec = importlib.util.spec_from_file_location(name, safe(root, controller_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(spec.origin), 'exec', dont_inherit=True), module.__dict__)
        require(module.D == D and module.SCOPE == 'server12_c5_normal_repeatability_v1',
            'actual source-bound diagnostic controller scope')
        actual_config, refs, authority = module.verify_metadata_configuration(root, config_path)
        require(canonical(actual_config) == canonical(config) and refs.get(controller_path) == candidates[0],
            'public diagnostic metadata config/source agreement')
        verified = module._verify_active_guard_metadata(root, actual_config, authority)
        require(type(verified) is dict and verified.get('id') == active['id'] and
            all(canonical(verified.get(key)) == canonical(active.get(key)) for key in
                ('label', 'gpu_uuid', 'permissions', 'session_id', 'process_group', 'runner_pid')),
            'actual current original diagnostic reservation/session agreement')
        checked_metadata_ref(root, candidates[0])
        checked_metadata_ref(root, config_ref)
        checked_metadata_ref(root, source_ref)
        checked_metadata_ref(root, permission_ref)
        return dict(status='PASS_ACTUAL_DIAGNOSTIC_ACTIVE_HISTORY_METADATA', diagnostic_index=index,
            source_lock_ref=source_ref, config_ref=config_ref, controller_ref=candidates[0],
            reservation_id=active['id'], GPU_operations=0, new_gpu_authority_issued=False)
    finally:
        sys.modules.pop(name, None)


def verify_configuration(root, path):
    return original_api(root).verify_configuration(root, path)


def load_binding(root, relative):
    return original_api(root).load_binding(root, relative)


def read(root, relative):
    return original_api(root).read(root, relative)


def verify_active_guard(*_args, **_kwargs):
    require(False, 'historical calibration grants no new GPU execution authority')


def main(argv=None):
    print(json.dumps(dict(status='HISTORICAL_METADATA_API_ONLY', GPU_operations=0,
        new_gpu_authority_issued=False, normal_native_execution_qualified=False), sort_keys=True))
    return 2


if __name__ == '__main__':
    raise SystemExit(main())

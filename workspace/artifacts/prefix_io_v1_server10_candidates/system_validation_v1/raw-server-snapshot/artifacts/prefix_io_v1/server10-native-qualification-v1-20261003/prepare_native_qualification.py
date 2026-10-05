"""Append server10 native off/shadow configuration; CPU only, no launcher.

Original qualifier and GPU budget guard remain unchanged. The original entry
does NOT enforce finite job names/order from scope metadata: the outer operator
must execute only the emitted off then shadow commands, stop on failure, and
check durable guard/runtime closure before the second command.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
DELIVERY = 'artifacts/prefix_io_v1/server10-native-qualification-v1-20261003'
SELF = DELIVERY + '/prepare_native_qualification.py'
PLAN = DELIVERY + '/NATIVE_QUALIFICATION_CPU_PLAN.json'
LOCK = DELIVERY + '/gpu-source-lock-native-qualification.json'
SCOPE = DELIVERY + '/GPU_STAGE_AUTHORIZATION.json'
HUMAN = DELIVERY + '/HUMAN_AUTHORIZATION_RECORD.json'
PERMISSION = 'experiments/prefix_io_v1/configs/permissions.server10.reference.yaml'
GPU_UUID = 'GPU-4b4d17ec-95a3-4efd-2bc4-1613333e949f'
ANCESTOR = 'artifacts/prefix_io_v1/server10-reference-migration-v1-20261003/gpu-source-lock-server10-reference.json'
ANCESTOR_SHA = 'ae3bf8437a41c4c531bf211689b4f7f68af7f8dd558a42914f2eef733988aa54'
SCRIPT = 'experiments/prefix_io_v1/scripts/qualify_p4_native_gpu.py'
SCRIPT_PIN = dict(path=SCRIPT, bytes=29692, sha256='5d9604fc07d80495a49febfcd488cb2601f74a2bef107261c8e623a4d0a1c77d')
GUARD = 'experiments/prefix_io_v1/scripts/run_gpu_stage.py'
GUARD_PIN = dict(path=GUARD, bytes=13013, sha256='3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a')
PERMISSION_PIN = dict(path=PERMISSION, bytes=685, sha256='aa54897390dd18c71f4578d45c87b74a3d26a3c63daea2660bf752c41be3be50')
NAMES = dict(off='server10-p4-native-off-01', shadow='server10-p4-native-shadow-01')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def safe(relative):
    require(type(relative) is str and relative and '\\' not in relative
            and not Path(relative).is_absolute() and all(x not in ('', '.', '..') for x in relative.split('/')), 'safe relative path')
    path = ROOT / relative
    cursor = path
    while cursor != ROOT:
        require(not cursor.is_symlink(), 'no evidence symlink')
        cursor = cursor.parent
    return path


def ref(relative):
    path = safe(relative)
    require(path.is_file() and 0 < path.stat().st_size < 4 * 1024**2, 'small source/config file')
    raw = path.read_bytes()
    return dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def write(relative, value):
    with safe(relative).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def arguments(mode):
    require(mode in NAMES, 'off/shadow only')
    return ['.venv/bin/python', '-B', SCRIPT, '--mode', mode, '--name', NAMES[mode],
            '--scope-record', SCOPE, '--source-lock', LOCK, '--permissions-path', PERMISSION, '--launch']


def check_outer_ledger(ledger, mode):
    require(mode in NAMES and ledger.get('active_reservation') is None, 'no active job')
    require(type(ledger.get('gpu_wall_seconds')) in (float, int)
            and 0 <= ledger['gpu_wall_seconds'] <= 28800, 'original cumulative ledger')
    events = ledger.get('events')
    require(type(events) is list and all(type(e) is dict for e in events), 'durable guard events')
    selected = {key: [e for e in events if e.get('label') == name] for key, name in NAMES.items()}
    require(not selected[mode], 'no retry of a consumed native job')
    if mode == 'off':
        require(not selected['shadow'], 'off must precede shadow')
        require(ledger['gpu_wall_seconds'] + 400 <= 28800, 'two job reserve fits original eight hours')
    else:
        require(len(selected['off']) == 1, 'one actual prior off job required')
        event = selected['off'][0]
        for key, expected in (('exit', 0), ('child_exit', 0), ('timed_out', False), ('error', None),
                              ('session_drained', True), ('session_members_before_cleanup', []),
                              ('session_members_after_cleanup', [])):
            require(type(event.get(key)) is type(expected) and event[key] == expected,
                    'off guard closure required: ' + key)
        require(ledger['gpu_wall_seconds'] + 200 <= 28800, 'second reserve fits original eight hours')
    return selected


def load_original(rows):
    for pin in (SCRIPT_PIN, GUARD_PIN, PERMISSION_PIN):
        require(rows.get(pin['path']) == pin and ref(pin['path']) == pin, 'unchanged source/guard/effective permission pin')
    path = safe(SCRIPT)
    spec = importlib.util.spec_from_file_location('_server10_native_config_original', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare():
    require(Path(__file__).resolve() == safe(SELF).resolve(), 'server deployed CPU generator location')
    ancestor_ref = ref(ANCESTOR)
    require(ancestor_ref['sha256'] == ANCESTOR_SHA, 'actual frozen migration ancestor')
    ancestor = json.loads(safe(ANCESTOR).read_bytes())
    rows = {r['path']: r for r in ancestor['files']}
    require(len(rows) == len(ancestor['files']) == 4104, 'all 4104 migration ancestor refs preserved')
    original = load_original(rows)
    ledger = json.loads(safe('experiments/prefix_io_v1/gpu-budget-ledger.json').read_bytes())
    check_outer_ledger(ledger, 'off')
    for name in NAMES.values():
        require(not safe('experiments/prefix_io_v1/runs/' + name).exists(), 'unused exact native run names')
    plan = dict(schema_version=1, status='CPU_NATIVE_CONFIGURATION_NOT_GPU_RESULT', gpu_uuid=GPU_UUID,
                original_entry_ref=SCRIPT_PIN, original_guard_ref=GUARD_PIN, baseline_source_lock=ancestor_ref,
                qualification_context=original.qualification_context(), permitted_run_names=NAMES,
                allowed_modes=['off', 'shadow'], maximum_jobs=2, maximum_total_reserve_seconds=400,
                retries=0, shared_storage_reserve_per_job_bytes=128 * 1024**2, PRIMARY_floor_bytes=8 * 1024**3,
                cumulative_GPU_budget_seconds=28800, GPU_budget_reset=False,
                commands={mode: arguments(mode) for mode in NAMES},
                outer_operator_must_enforce_names_count_order_and_success=True,
                original_scope_parser_enforces_names_count_order=False,
                off_required_before_shadow=['guard exit/child exit zero', 'no timeout/error', 'OS session fully drained',
                    'PASS_REAL_GPU_P4_NATIVE_OFF', 'exact_content true', 'real_cuda and real_linux_aio true',
                    'all four native phases closed, drained, balanced and admitted parents retired'],
                model_loaded=False, full_P4_complete=False, effect_verified=False, GPU_operations=0)
    for relative in (PLAN, LOCK, HUMAN, SCOPE, DELIVERY + '/CPU_CONFIGURATION_RESULT.json'):
        require(not safe(relative).exists(), 'append-only native configuration')
    write(PLAN, plan)
    added = [ancestor_ref, ref(SELF), ref(PLAN)]
    require(not (set(rows) & {r['path'] for r in added}), 'new native metadata only')
    lock = dict(schema_version=1, status='CPU_NATIVE_SOURCE_CANDIDATE_NO_AUTHORITY', allow_gpu_runs=False,
                allow_gpu_initialization=False, ancestor_source_lock=ancestor_ref, original_rows_unchanged=True,
                files=copy.deepcopy(ancestor['files']) + added)
    write(LOCK, lock)
    lock_ref = ref(LOCK)
    scope = dict(schema_version=1, status='USER_AUTHORIZED_P4_GPU_NATIVE_QUALIFICATION',
                 allow_gpu_initialization=True, allow_gpu_runs=True, allowed_modes=['off', 'shadow'],
                 new_executor=False, allow_model_downloads=False, gpu_uuid=GPU_UUID,
                 base_permissions=PERMISSION_PIN, source_lock=LOCK, source_lock_sha256=lock_ref['sha256'],
                 qualification_context=original.qualification_context(), permitted_run_names=NAMES,
                 maximum_jobs=2, maximum_total_planned_reserve_seconds=400, retries=0,
                 required_order=['off', 'shadow'], shadow_requires_successful_off_and_shutdown=True)
    human = dict(schema_version=1, authorization_origin='direct_typed_user_message',
                 verbatim_user_request_excerpt='旧服务器已经关机，已经把代码克隆到新的服务器，接下来验证在新服务器进行gpu实验，完成系统验证',
                 user_designated_host='connect.westd.seetacloud.com', user_designated_port=12351,
                 credentials_excluded=True, copied_form_response=False,
                 interpretation='Bounded native off/shadow system qualification within the original cumulative GPU budget',
                 source_lock_ref=lock_ref, cpu_plan_ref=ref(PLAN), scope_configuration=copy.deepcopy(scope))
    write(HUMAN, human)
    scope['human_authorization_record'] = ref(HUMAN)
    write(SCOPE, scope)
    result = dict(status='CPU_NATIVE_CONFIGURATION_PREPARED_NO_GPU_OPERATION', source_count=len(lock['files']),
                  source_lock_ref=lock_ref, scope_ref=ref(SCOPE), cpu_plan_ref=ref(PLAN),
                  original_entry_unchanged=True, original_guard_unchanged=True, GPU_operations=0,
                  full_source_bytes_verified=False, original_launch_gates_still_required=True,
                  outer_operator_order_gate_still_required=True, commands=plan['commands'])
    write(DELIVERY + '/CPU_CONFIGURATION_RESULT.json', result)
    print(json.dumps(result))


def preflight(mode):
    lock_ref = ref(LOCK)
    lock = json.loads(safe(LOCK).read_bytes())
    rows = {r['path']: r for r in lock['files']}
    original = load_original(rows)
    scope = json.loads(safe(SCOPE).read_bytes())
    require(scope['source_lock_sha256'] == lock_ref['sha256'] and scope['permitted_run_names'] == NAMES,
            'same prepared native config')
    ledger = json.loads(safe('experiments/prefix_io_v1/gpu-budget-ledger.json').read_bytes())
    check_outer_ledger(ledger, mode)
    if mode == 'shadow':
        report = json.loads(safe('experiments/prefix_io_v1/runs/' + NAMES['off'] + '/details/result.json').read_bytes())
        require(report.get('status') == 'PASS_REAL_GPU_P4_NATIVE_OFF' and
                all(report.get(key) is True for key in ('exact_content', 'real_cuda', 'real_linux_aio')),
                'successful real off native byte qualification required')
    args = SimpleNamespace(mode=mode, name=NAMES[mode], scope_record=SCOPE, source_lock=LOCK, permissions_path=PERMISSION)
    # Only the original pure-CPU function; never invoke guard, main, or qualify.
    sys.path.insert(0, str(ROOT / 'experiments/prefix_io_v1/scripts'))
    try:
        gates, reason = original.launch_gates(ROOT, args)
    finally:
        sys.path.pop(0)
    require(gates is not None and reason == 'READY_FOR_AUTHORIZED_OLD_GUARD', 'actual original CPU launch gates')
    print(json.dumps(dict(status='PASS_ORIGINAL_NATIVE_CPU_LAUNCH_GATES_NO_GPU', mode=mode,
                          source_count=len(gates['source_refs']), source_lock_ref=lock_ref,
                          PRIMARY_free_bytes=gates['PRIMARY_free_bytes'], GPU_operations=0,
                          full_source_bytes_verified=True, guard_invoked=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--preflight', choices=['off', 'shadow'])
    args = parser.parse_args()
    prepare() if args.prepare else preflight(args.preflight)

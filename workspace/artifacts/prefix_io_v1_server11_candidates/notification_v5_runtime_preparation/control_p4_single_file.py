"""Append-only finite experiment preparation; unchanged original GPU guard."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


# This preparation module cannot grant native or GPU qualification.
def notification_adapter():
    name = "_server11_c5_cpu_notification_adapter"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("notification_runtime_adapter.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server11-c5-runtime-preparation-cpu-20261004'
C = 'artifacts/prefix_io_v1/server11-p4-notification-candidate-v5-cpu-20261003'
V6 = 'artifacts/prefix_io_v1/server11-native-cost-v6-20261003'
RECEIPT = C + '/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py'
GPU = 'GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9'
PERMISSION = 'experiments/prefix_io_v1/configs/permissions.server11.native-cost.yaml'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
GUARD = 'experiments/prefix_io_v1/scripts/run_gpu_stage.py'
LOCK = D + '/COMMON_SOURCE_LOCK.json'
BINDING = D + '/SINGLE_FILE_BINDING.json'
MODES = ('off', 'shadow', 'on')


def safe(relative):
    if (type(relative) is not str or not relative or Path(relative).is_absolute()
            or '\\' in relative or any(x in ('', '.', '..') for x in relative.split('/'))):
        raise ValueError('explicit project-relative path required')
    path = ROOT / relative
    cursor = path
    while cursor != ROOT:
        if cursor.is_symlink():
            raise ValueError('symlink evidence refused: ' + relative)
        cursor = cursor.parent
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('outside original workspace')
    return path


def ref(relative):
    path = safe(relative)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=relative, bytes=path.stat().st_size, sha256=digest)


def read(relative):
    return json.loads(safe(relative).read_bytes())


def put(name, value):
    relative = D + '/' + name
    with safe(relative).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return ref(relative)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def label(mode):
    if mode not in MODES:
        raise ValueError('known arm required')
    return 'server11-c5-notification-' + mode + '-preparation01'


def config_path(mode):
    return D + '/CONFIG_' + mode + '.json'


def result_path(mode):
    return D + '/QUALIFICATION_' + mode + '.json'


def verify_reference(value):
    if ref(value['path']) != value:
        raise ValueError('reference changed: ' + value['path'])


def load_receipt():
    notification_adapter().block_gpu_start()
    source = safe(RECEIPT)
    expected = ref(RECEIPT)
    spec = importlib.util.spec_from_file_location('_v4_scope_verified_receipt', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        exec(compile(source.read_bytes(), str(source), 'exec', dont_inherit=True), module.__dict__)
        value = module.load_verified_single_file(ROOT, BINDING)
        if ref(RECEIPT) != expected or value.binding_ref.mapping() != ref(BINDING):
            raise ValueError('receipt source or binding changed during verification')
        return value
    finally:
        sys.modules.pop(spec.name, None)


def freeze_common(expected_calibration_lock_sha256):
    notification_adapter().block_gpu_start()
    ancestor = V6 + '/gpu-source-lock-native-cost.json'
    if (type(expected_calibration_lock_sha256) is not str or
            len(expected_calibration_lock_sha256) != 64 or
            any(c not in '0123456789abcdef' for c in expected_calibration_lock_sha256) or
            ref(ancestor)['sha256'] != expected_calibration_lock_sha256):
        raise ValueError('actual native calibration ancestor mismatch')
    plan = read(V6 + '/NATIVE_COST_PLAN.json')
    if plan.get('source_lock_ref') != ref(ancestor):
        raise ValueError('actual v6 plan must bind the supplied calibration ancestor')
    old = {row['path']: row for row in read(ancestor)['files']}
    candidate = sorted(path.relative_to(ROOT).as_posix()
                       for path in [*safe(C).glob('*.py'), *safe(C+'/source').rglob('*.py')]
                       if '__pycache__' not in path.parts)
    runtime = sorted(path.relative_to(ROOT).as_posix() for path in safe(D).glob('*.py'))
    common_names = [plan[k]['path'] for k in ('model_plan_ref', 'model_config_ref', 'cuda_event_source_ref')]
    common_names += ['third_party/work/vllm-author-p4-02-cpu/vllm/v1/worker/gpu_model_runner.py']
    common_names += [plan['native_source_ref']['path']]
    if plan['native_source_ref'] != ref(C+'/source/third_party/work/py-kvcache-p4-02-cpu/py_kvcache/reactor.py'):
        raise ValueError('runtime and calibration must use actual same C4 reactor')
    for name in common_names:
        if old[name] != ref(name):
            raise ValueError('unchanged common calibration source required')
    binding = dict(schema_version=1, scope='p4_single_file_native_binding_v1',
        calibration=dict(plan_ref=ref(V6+'/NATIVE_COST_PLAN.json'),
            measurements_ref=ref(V6+'/NATIVE_MEASUREMENTS.json'),
            guard_ref=ref('experiments/prefix_io_v1/runs/server11-native-cost-six-window-06/result.json')),
        runtime_common_refs=[ref(name) for name in common_names],
        runtime_overlay_refs=[ref(name) for name in candidate+runtime if name not in common_names],
        kernel_mode='original_execute_sample_eager_triton_attention')
    binding_ref = put('SINGLE_FILE_BINDING.json', binding)
    receipt = load_receipt()  # Strictly replay six raw windows; no stored PASS trust.
    additions = candidate + runtime + [ancestor, BINDING, PERMISSION,
        V6+'/NATIVE_COST_PLAN.json', V6+'/NATIVE_MEASUREMENTS.json',
        V6+'/NATIVE_CONDITIONAL_COST_RESULT.json', V6+'/SOURCE_BEFORE_VERIFICATION.json',
        V6+'/SOURCE_AFTER_VERIFICATION.json']
    additions += [row['path'] for row in binding['calibration'].values()]
    for name in additions:
        value = ref(name)
        if name in old and old[name] != value:
            raise ValueError('old source must remain unchanged: ' + name)
        old[name] = value
    lock_ref = put('COMMON_SOURCE_LOCK.json', dict(schema_version=1,
        scope='finite_single_file_experiment_common_code_and_assets', files=list(old.values()),
        ancestor_ref=ref(ancestor), binding_ref=binding_ref,
        arm_configs_bound_separately_in_scope=True,
        production_qualified=False, strategy_effect_verified=False))
    print(json.dumps(dict(common_source_lock_ref=lock_ref, binding_ref=binding_ref,
                         source_count=len(old), frozen_cost_upper_ns=receipt.cost_upper_ns,
                         internal_step_budget_ns=receipt.step_budget_ns)))


def check_sources(mode, phase):
    scope = read(D + '/SCOPE_' + mode + '.json')
    verify_reference(scope['source_lock_ref'])
    verify_reference(scope['config_ref'])
    failed = []
    for row in read(LOCK)['files']:
        try:
            verify_reference(row)
        except Exception as exc:
            failed.append(dict(path=row['path'], reason=str(exc)))
    value = dict(phase=phase, source_lock_ref=ref(LOCK), config_ref=ref(config_path(mode)),
        files_verified=len(read(LOCK)['files'])-len(failed), failed=failed,
        actual_verification_utc=now(), GPU_operations=0)
    record = put('SOURCE_' + mode + '_' + phase.upper() + '.json', value)
    if failed:
        raise ValueError('source closure failed')
    return record


def prepare(mode):
    notification_adapter().block_gpu_start()
    previous = None
    if mode != 'off':
        previous_mode = MODES[MODES.index(mode)-1]
        previous = ref(result_path(previous_mode))
        receipt = read(previous['path'])
        if (receipt.get('scope') != 'server11_c5_notification_cpu_preparation_v1'
                or receipt.get('permits_next_mode') != mode or receipt.get('mode') != previous_mode
                or receipt.get('native_execution_verified') is not True
                or receipt.get('runtime_condition_qualified') is not True):
            raise ValueError('independent preceding qualification did not permit this arm')
    cost = load_receipt()
    config = dict(root=str(ROOT), label=label(mode), gpu_uuid=GPU,
        purpose='CPU_ONLY_C5_NOTIFICATION_WIRING_PREPARATION_GPU_BLOCKED',
        seconds_limit=300, reserved_seconds=320, active_ledger=LEDGER,
        storage='experiments/prefix_io_v1/runs/server11-native-cost-01-private-storage',
        out='experiments/prefix_io_v1/runs/'+label(mode)+'/details', source_lock=LOCK,
        input_manifest='artifacts/prefix_io_v1/server11-native-cost-v1-20261003/SSD_INPUT_MANIFEST.json',
        collector_relative=C+'/native_full_step_collector.py',
        runtime_binding_relative=C+'/single_file_runtime_binding.py',
        mode=mode, binding_relative=BINDING, overlay_relative=C+'/source',
        previous_qualification_ref=previous)
    config_ref = put('CONFIG_' + mode + '.json', config)
    scope_ref = put('SCOPE_' + mode + '.json', dict(
        origin='direct_human_instruction_bounded_execution', credentials_omitted=True,
        user_instruction='继续；在新服务器完成剩余操作，沿用原8小时累计GPU预算',
        mode=mode, label=label(mode), gpu_uuid=GPU, config_ref=config_ref,
        source_lock_ref=ref(LOCK), binding_ref=ref(BINDING), permission_ref=ref(PERMISSION),
        previous_qualification_ref=previous, maximum_gpu_jobs=1, maximum_model_processes=1,
        seconds_limit=300, reserved_seconds=320, cumulative_gpu_limit_seconds=28800,
        maximum_output_tokens=258, warmup_output_tokens=128, measured_output_tokens=128,
        foreground_prompt_tokens=129, frozen_first_prompt_token=28100, seed=2829,
        measured_ssd_operations=1, measured_ssd_physical_bytes=917504,
        internal_step_budget_ns=cost.step_budget_ns, frozen_cost_upper_ns=cost.cost_upper_ns,
        cost_scope='strict_replayed_v6_calibration_only_upper_and_A_only_budget',
        primary_reserve_bytes=2*1024**3, primary_free_floor_bytes=8*1024**3,
        no_new_model_downloads=True, no_system_or_driver_changes=True,
        no_existing_data_deletion=True, general_cost_production_qualification=False,
        resource_release_credit=False, formal_SLOs=None, strategy_effect_verified=False,
        fail_stops_progression=True, recorded_utc=now()))
    print(json.dumps(dict(config_ref=config_ref, scope_ref=scope_ref)))


def launch(mode):
    notification_adapter().block_gpu_start()
    scope = read(D+'/SCOPE_'+mode+'.json')
    for key in ('config_ref', 'source_lock_ref', 'binding_ref', 'permission_ref'):
        verify_reference(scope[key])
    if scope['previous_qualification_ref'] is not None:
        verify_reference(scope['previous_qualification_ref'])
    ledger = read(LEDGER)
    if ledger['active_reservation'] is not None or ledger['gpu_wall_seconds']+320 > 28800:
        raise ValueError('original cumulative GPU budget unavailable')
    run = 'experiments/prefix_io_v1/runs/'+label(mode)
    if safe(run).exists() or safe(D+'/LAUNCH_INTENT_'+mode+'.json').exists():
        raise ValueError('existing invocation must be inspected, never blindly retried')
    stat = os.statvfs(ROOT)
    free = stat.f_bavail*stat.f_frsize
    if free < scope['primary_free_floor_bytes']+scope['primary_reserve_bytes']:
        raise ValueError('storage floor/reserve unavailable')
    telemetry = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.free','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=10).stdout
    rows = [line.split(',') for line in telemetry.strip().splitlines()]
    if len(rows)!=1 or rows[0][0].strip()!=GPU or int(rows[0][2])<28000:
        raise ValueError('expected free GPU unavailable')
    processes = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=10).stdout
    if processes.strip():
        raise ValueError('other GPU process; do not interfere')
    before = check_sources(mode, 'before')
    command = ['.venv/bin/python','-B',GUARD,'--permissions-path',PERMISSION,
        '--label',label(mode),'--seconds','300','--','.venv/bin/python','-B',
        D+'/run_p4_single_file_experiment.py','--execute','--config',config_path(mode)]
    overrides = dict(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='0')
    put('LAUNCH_INTENT_'+mode+'.json',dict(command=command,environment_overrides=overrides,
        scope_ref=ref(D+'/SCOPE_'+mode+'.json'),config_ref=ref(config_path(mode)),
        source_before_ref=before,ledger_before_ref=ref(LEDGER),
        ledger_gpu_wall_seconds_before=ledger['gpu_wall_seconds'],primary_free_bytes=free,
        gpu_telemetry=telemetry,compute_processes=processes,utc=now()))
    with safe(D+'/GPU_GUARD_'+mode+'_STDOUT.log').open('xb') as out, safe(D+'/GPU_GUARD_'+mode+'_STDERR.log').open('xb') as err:
        env = dict(os.environ); env.update(overrides)
        proc = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=out,stderr=err,start_new_session=True,close_fds=True)
    print(json.dumps(put('LAUNCH_RECEIPT_'+mode+'.json',dict(guard_pid=proc.pid,command=command,
        utc=now(),detached_original_guard=True,launch_is_not_success=True))))


def after(mode):
    notification_adapter().block_gpu_start()
    if read(LEDGER)['active_reservation'] is not None:
        raise ValueError('wait for original guard completion')
    result = read('experiments/prefix_io_v1/runs/'+label(mode)+'/result.json')
    if not result['session_drained'] or result['session_members_after_cleanup']:
        raise ValueError('original session remains alive')
    print(json.dumps(check_sources(mode,'after')))


def main():
    print(json.dumps(notification_adapter().blocked_document(), sort_keys=True))
    return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','prepare','launch','after'))
    parser.add_argument('--mode',choices=MODES)
    parser.add_argument('--calibration-lock-sha256')
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze_common(args.calibration_lock_sha256)
    elif args.mode is None:
        parser.error('--mode required')
    else:
        {'prepare':prepare,'launch':launch,'after':after}[args.action](args.mode)


if __name__ == '__main__':
    raise SystemExit(main())

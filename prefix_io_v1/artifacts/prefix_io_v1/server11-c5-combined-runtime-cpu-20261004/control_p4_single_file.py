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
D = 'artifacts/prefix_io_v1/server11-c5-combined-runtime-cpu-20261004'
C = 'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004/common_candidate'
V6 = 'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004'
RECEIPT = C + '/source/third_party/work/prefix-io-p4-02-cpu/src/prefix_io_control/p4_single_file_receipt.py'
GPU = None
PERMISSION = None
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
GUARD = 'experiments/prefix_io_v1/scripts/run_gpu_stage.py'
LOCK = None
BINDING = None
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
    return 'server11-c5-combined-notification-' + mode + '-preparation01'


def config_path(mode):
    return D + '/CONFIG_' + mode + '.json'


def result_path(mode):
    return D + '/QUALIFICATION_' + mode + '.json'


def verify_reference(value):
    if ref(value['path']) != value:
        raise ValueError('reference changed: ' + value['path'])


def load_receipt():
    notification_adapter().block_gpu_start()


def freeze_common(expected_calibration_lock_sha256):
    notification_adapter().block_gpu_start()


def check_sources(mode, phase):
    notification_adapter().block_gpu_start()
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

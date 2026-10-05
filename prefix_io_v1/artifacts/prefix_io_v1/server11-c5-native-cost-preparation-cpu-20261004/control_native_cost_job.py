"""CPU source closure and detached invocation of the unchanged original guard.

No CUDA import, model code, retry, budget reset, cleanup, or source mutation.
The original guard alone owns execution timing, session cleanup and accounting.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

CPU_PREPARATION_BLOCKED = "GPU_BLOCKED_C5_NATIVE_COST_CODE_PREPARATION_ONLY"


def blocked_document():
    return dict(status=CPU_PREPARATION_BLOCKED, gpu_started=False, actual_gpu_runs=0,
        gpu_launch_allowed=False, native_execution_verified=False, native_cost_qualified=False,
        full_runtime_cost_qualified=False, on_observation_cost_measured=False,
        valid_native_receipt=None, effective_cost_upper_ns=None, effective_step_budget_ns=None)


def block_gpu_start():
    raise RuntimeError(CPU_PREPARATION_BLOCKED)


ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004'
CANDIDATE = 'artifacts/prefix_io_v1/server11-c5-native-cost-preparation-cpu-20261004/common_candidate'
LABEL = 'server11-c5-native-common-cost-preparation01'
GPU = None
PERMISSION = None  # No prior human authorization is inherited by this CPU preparation.
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
LOCK = None  # CPU source lock is supplied separately; no effective GPU lock exists.
GUARD = 'experiments/prefix_io_v1/scripts/run_gpu_stage.py'
RUN = 'experiments/prefix_io_v1/runs/' + LABEL
ANCESTOR = None  # Source ancestry is SOURCE_INHERITANCE.json, never an old GPU authority.
ANCESTOR_SHA = None


def safe(relative):
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or '\\' in relative or any(x in ('', '.', '..') for x in relative.split('/'))):
        raise ValueError('bounded project-relative path required')
    path = ROOT / relative
    cursor = path
    while cursor != ROOT:
        if cursor.is_symlink():
            raise ValueError('source/artifact symlink: ' + relative)
        cursor = cursor.parent
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('outside project')
    return path


def read(relative):
    return json.loads(safe(relative).read_bytes())


def ref(relative):
    path = safe(relative)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return dict(path=relative, bytes=path.stat().st_size, sha256=h.hexdigest())


def put(name, document):
    relative = D + '/' + name
    with safe(relative).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return ref(relative)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def check_sources(phase):
    block_gpu_start()
    lock = read(LOCK)
    failed = []
    for row in lock['files']:
        try:
            if ref(row['path']) != row:
                failed.append(dict(path=row['path'], reason='bytes/SHA mismatch'))
        except Exception as exc:
            failed.append(dict(path=row['path'], reason=repr(exc)))
    receipt = dict(phase=phase, source_lock_ref=ref(LOCK),
                   files_verified=len(lock['files'])-len(failed), failed=failed,
                   actual_verification_utc=now(), GPU_operations=0)
    row = put('SOURCE_' + phase.upper() + '_VERIFICATION.json', receipt)
    if failed:
        raise RuntimeError('source closure failed; see ' + row['path'])
    return row


def prepare():
    block_gpu_start()
    if ref(ANCESTOR)['sha256'] != ANCESTOR_SHA:
        raise RuntimeError('immutable ancestor mismatch')
    config = dict(root=str(ROOT), label=LABEL, gpu_uuid=GPU,
        purpose='CURRENT_CONTEXT_NATIVE_FULL_STEP_SSD_READ_DIAGNOSTIC_ONLY',
        seconds_limit=1200, reserved_seconds=1220, active_ledger=LEDGER,
        storage='experiments/prefix_io_v1/runs/server11-native-cost-01-private-storage',
        out=RUN+'/details', source_lock=LOCK,
        input_manifest='artifacts/prefix_io_v1/server11-native-cost-v1-20261003/SSD_INPUT_MANIFEST.json',
        collector_relative=CANDIDATE+'/native_full_step_collector.py')
    config_ref = put('NATIVE_COST_CONFIG.json', config)
    rows = {r['path']: r for r in read(ANCESTOR)['files']}
    additions = [PERMISSION, ANCESTOR,
        '.venv/lib/python3.12/site-packages/torch/cuda/streams.py']
    for directory in (D, CANDIDATE):
        for path in safe(directory).rglob('*'):
            if '__pycache__' in path.parts:
                continue
            if path.suffix in ('.py', '.md'):
                additions.append(path.relative_to(ROOT).as_posix())
    for name in ('CANDIDATE_MANIFEST.json', 'BASE_PACKAGE_COPY.json'):
        additions.append(CANDIDATE+'/'+name)
    additions += [config_ref['path'], config['input_manifest']]
    for group in read(config['input_manifest'])['groups']:
        additions += [config['storage']+'/'+r['path'] for r in group['files']]
    for relative in additions:
        value = ref(relative)
        if relative in rows and rows[relative] != value:
            raise RuntimeError('ancestor source may not be rewritten: '+relative)
        rows[relative] = value
    manifest = dict(schema_version=1, status='FROZEN_NATIVE_CONDITIONAL_DIAGNOSTIC',
        baseline_source_lock=ref(ANCESTOR), files=list(rows.values()),
        production_qualified=False, strategy_effect_verified=False)
    result = put('gpu-source-lock-native-cost.json', manifest)
    print(json.dumps(dict(source_lock_ref=result, source_count=len(rows), config_ref=config_ref)))


def scope(plan_relative):
    block_gpu_start()


def launch():
    block_gpu_start()
    document = read(D+'/SIX_PROCESS_AUTHORIZED_SCOPE.json')
    for key in ('permission_ref', 'source_lock_ref', 'plan_ref'):
        if ref(document[key]['path']) != document[key]:
            raise RuntimeError('scope reference drift '+key)
    ledger = read(LEDGER)
    if ledger['active_reservation'] is not None or ledger['gpu_wall_seconds'] + 1220 > 28800:
        raise RuntimeError('original cumulative budget unavailable')
    if safe(RUN).exists() or safe(D+'/GPU_LAUNCH_RECEIPT.json').exists():
        raise RuntimeError('existing launch: inspect, never retry this label')
    stat = os.statvfs(ROOT)
    free = stat.f_bavail * stat.f_frsize
    if free < document['primary_free_floor_bytes'] + document['primary_reserve_bytes']:
        raise RuntimeError('insufficient primary storage')
    telemetry = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,memory.free', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True, timeout=10).stdout
    devices = [line.split(',') for line in telemetry.strip().splitlines()]
    if len(devices) != 1 or devices[0][0].strip() != GPU or int(devices[0][2]) < 28000:
        raise RuntimeError('expected free GPU unavailable')
    processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=10).stdout
    if processes.strip():
        raise RuntimeError('existing compute process; do not interfere')
    before_ref = check_sources('before')
    command = ['.venv/bin/python','-B',GUARD,'--permissions-path',PERMISSION,
               '--label',LABEL,'--seconds','1200','--',
               '.venv/bin/python','-B',D+'/run_native_cost_experiment.py',
               '--execute','--config',D+'/NATIVE_COST_CONFIG.json']
    overrides = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                     PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    put('GPU_LAUNCH_INTENT.json', dict(command=command, environment_overrides=overrides, source_before_ref=before_ref,
         scope_ref=ref(D+'/SIX_PROCESS_AUTHORIZED_SCOPE.json'), ledger_before_ref=ref(LEDGER),
         ledger_gpu_wall_seconds_before=ledger['gpu_wall_seconds'], primary_free_bytes=free,
         gpu_telemetry=telemetry, compute_processes=processes, utc=now()))
    with safe(D+'/GPU_GUARD_STDOUT.log').open('xb') as out, safe(D+'/GPU_GUARD_STDERR.log').open('xb') as err:
        env = dict(os.environ); env.update(overrides)
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True, close_fds=True)
    print(json.dumps(put('GPU_LAUNCH_RECEIPT.json', dict(guard_pid=proc.pid, command=command,
          utc=now(), detached_original_guard=True, launch_is_not_success=True))))


def status():
    block_gpu_start()
    ledger = read(LEDGER)
    value = dict(active=ledger['active_reservation'], gpu_wall_seconds=ledger['gpu_wall_seconds'],
                 remaining_seconds=28800-ledger['gpu_wall_seconds'])
    if safe(RUN+'/result.json').exists():
        value['result'] = read(RUN+'/result.json')
    paths = sorted(safe(RUN).glob('details/windows/*/*.json')) if safe(RUN).exists() else []
    value['window_artifact_paths'] = [p.relative_to(ROOT).as_posix() for p in paths]
    for relative in (D+'/GPU_GUARD_STDERR.log', RUN+'/process.log'):
        path = safe(relative)
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size-3500))
                value[relative] = stream.read().decode('utf-8', errors='replace')
    print(json.dumps(value))


def main():
    print(json.dumps(blocked_document(), sort_keys=True))
    return 2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','scope','launch','status','after'])
    parser.add_argument('--plan-relative')
    args = parser.parse_args()
    if ROOT.resolve(strict=True) != ROOT:
        raise RuntimeError('fixed original workspace required')
    if args.action == 'prepare': prepare()
    elif args.action == 'scope': scope(args.plan_relative)
    elif args.action == 'launch': launch()
    elif args.action == 'status': status()
    else:
        if read(LEDGER)['active_reservation'] is not None or not safe(RUN+'/result.json').exists():
            raise RuntimeError('wait for original guard to close before source verification')
        print(json.dumps(check_sources('after')))


if __name__ == '__main__':
    raise SystemExit(main())

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

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server11-native-cost-v5-20261003'
LABEL = 'server11-native-cost-six-window-05'
GPU = 'GPU-38c1d1c7-2aba-3fa9-e1cd-fe4d3ae569c9'
PERMISSION = 'experiments/prefix_io_v1/configs/permissions.server11.native-cost.yaml'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
LOCK = D + '/gpu-source-lock-native-cost.json'
GUARD = 'experiments/prefix_io_v1/scripts/run_gpu_stage.py'
RUN = 'experiments/prefix_io_v1/runs/' + LABEL
ANCESTOR = 'artifacts/prefix_io_v1/server11-native-cost-v4-20261003/gpu-source-lock-native-cost.json'
ANCESTOR_SHA = 'e64115e0bffbd31aa9b17c9b2cb34ab50a5b670c6878cb1a5fe0345433129718'


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
    if ref(ANCESTOR)['sha256'] != ANCESTOR_SHA:
        raise RuntimeError('immutable ancestor mismatch')
    config = dict(root=str(ROOT), label=LABEL, gpu_uuid=GPU,
        purpose='CURRENT_CONTEXT_NATIVE_FULL_STEP_SSD_READ_DIAGNOSTIC_ONLY',
        seconds_limit=1200, reserved_seconds=1220, active_ledger=LEDGER,
        storage='experiments/prefix_io_v1/runs/server11-native-cost-01-private-storage',
        out=RUN+'/details', source_lock=LOCK,
        input_manifest='artifacts/prefix_io_v1/server11-native-cost-v1-20261003/SSD_INPUT_MANIFEST.json',
        collector_relative='artifacts/prefix_io_v1/server11-full-step-collector-v1-20261003/native_full_step_collector.py')
    config_ref = put('NATIVE_COST_CONFIG.json', config)
    rows = {r['path']: r for r in read(ANCESTOR)['files']}
    additions = [PERMISSION, ANCESTOR,
        '.venv/lib/python3.12/site-packages/torch/cuda/streams.py']
    for directory in (D, 'artifacts/prefix_io_v1/server11-full-step-collector-v1-20261003'):
        for path in safe(directory).iterdir():
            if path.suffix in ('.py', '.md'):
                additions.append(path.relative_to(ROOT).as_posix())
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
    plan_ref = ref(plan_relative)
    previous = 'artifacts/prefix_io_v1/server11-native-cost-v4-20261003/SIX_PROCESS_AUTHORIZED_SCOPE.json'
    value = dict(origin='direct_human_instruction_bounded_execution',
        verbatim_user_instruction='又为你申请了一个新的服务器，代码全部克隆到新的服务器，在新的服务器完成剩余的操作',
        credentials_omitted=True, previous_closed_scope_ref=ref(previous),
        decision='Preregister one real SSD file per preload action to match the original single-file issue unit; preserve iodepth four, all original engine paths and controls. New independent prompt families and seeds. Existing eight-file results remain unchanged.',
        SSD_action_operations=1, SSD_action_physical_bytes=917504, independently_frozen_prompt_first_tokens=[18100,19100,20100], independently_frozen_seeds=[1829,1830,1831],
        foreground_prompt_tokens=129, required_GPU_cached_tokens=128, selected_decode_offset=16, selected_pre_context=144,
        previous_guard_result_ref=ref('experiments/prefix_io_v1/runs/server11-native-cost-six-window-04/result.json'),
        previous_guard_outcome='Six full native windows passed independent verification at eight files. Guard 663.1885902825743 seconds; natural shutdown. This distinct one-file cell is required because the real ready queue is bounded by iodepth/open_lookahead four; eight-file action cost cannot be divided into per-file costs.',
        new_job_under_same_direct_human_instruction=True, prior_server11_GPU_attempts=4,
        permission_ref=ref(PERMISSION), source_lock_ref=ref(LOCK), plan_ref=plan_ref,
        label=LABEL, gpu_uuid=GPU, max_guard_jobs=1,
        max_model_processes=6, selected_order=['A0','B0','B1','A1','A2','B2'],
        max_one_token_primer_requests=6,
        max_full_warmup_requests=6, warmup_output_tokens_each=128,
        max_one_token_flush_requests=6,
        max_measured_requests=6, measured_output_tokens_each=128, maximum_total_output_tokens=1548,
        seconds_limit=1200, reserved_seconds=1220, cumulative_gpu_limit_seconds=28800,
        primary_reserve_bytes=2*1024**3, primary_free_floor_bytes=8*1024**3,
        no_downloads=True, no_system_or_driver_changes=True, no_existing_data_deletion=True,
        no_strategy_benefit_claim=True, fail_stops_remaining_windows=True,
        recorded_utc=now())
    print(json.dumps(put('SIX_PROCESS_AUTHORIZED_SCOPE.json', value)))


def launch():
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
    main()

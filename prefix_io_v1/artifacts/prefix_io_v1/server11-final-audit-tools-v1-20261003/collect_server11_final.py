"""Read closed evidence into an append-only summary; no model imports or GPU work."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
OUT = ROOT / 'artifacts/prefix_io_v1/server11-system-review-v1-20261003'

def read(relative):
    return json.loads((ROOT / relative).read_bytes())

def reference(path):
    value = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=len(value),
                sha256=hashlib.sha256(value).hexdigest())

def main():
    ledger = read('experiments/prefix_io_v1/gpu-budget-ledger.json')
    assert ledger['active_reservation'] is None, 'wait for original guard'
    runs = []
    for path in sorted((ROOT/'experiments/prefix_io_v1/runs').glob('server11-*/result.json')):
        result = json.loads(path.read_bytes())
        assert result['session_drained'] and not result['session_members_after_cleanup']
        runs.append(dict(reference=reference(path), result=result))
    rows, cpu, source_closures = [], [], []
    for version in (1, 2, 3):
        directory = ROOT / ('artifacts/prefix_io_v1/server11-p4-single-file-runtime-v%d-20261003' % version)
        for mode in ('off', 'shadow', 'on'):
            path = directory / ('QUALIFICATION_%s.json' % mode)
            if not path.exists():
                continue
            value = json.loads(path.read_bytes())
            keys = ('mode', 'qualification_passed', 'native_execution_verified', 'runtime_condition_qualified',
                    'semantic_conditions_passed', 'frozen_cost_migration_pass', 'frozen_cost_upper_ns',
                    'frozen_a_only_budget_ns', 'full_frames', 'full_output_tokens', 'selected_gpu_elapsed_ns',
                    'sum_full_step_gpu_ns', 'whole_request_ns', 'drain_ns', 'total_request_and_drain_ns',
                    'deferral', 'native_io', 'strategy_effect_verified', 'production_qualified',
                    'source_lock_ref', 'binding_ref', 'prompt_sha256', 'seed')
            row = {key: value.get(key) for key in keys}
            row.update(version=version, qualification_ref=reference(path))
            raw_path = ROOT / value['evidence_refs']['raw_result']['path']
            raw = json.loads(raw_path.read_bytes())
            row['conditional_single_file'] = (raw.get('p4_after_shutdown') or {}).get('conditional_single_file')
            row['output_ids_sha256'] = hashlib.sha256(json.dumps(value['output_token_ids']).encode()).hexdigest()
            rows.append(row)
            for phase in ('BEFORE', 'AFTER'):
                path = directory / ('SOURCE_%s_%s.json' % (mode, phase))
                source_closures.append(dict(reference=reference(path), result=json.loads(path.read_bytes())))
        if version == 3:
            candidate = ROOT / 'artifacts/prefix_io_v1/server11-p4-single-file-candidate-v3-20261003'
            review = ROOT / 'artifacts/prefix_io_v1/server11-p4-single-file-review-v3-20261003'
            for base in (candidate, directory, review):
                for path in sorted(base.glob('*CPU*_RESULT.json')):
                    cpu.append(dict(reference=reference(path), result=json.loads(path.read_bytes())))
                for path in sorted(base.glob('*CPU*_COMMAND.json')):
                    cpu.append(dict(reference=reference(path), command=json.loads(path.read_bytes())))
    for version in (1, 2, 3):
        arms = [row for row in rows if row['version'] == version]
        off = next((row for row in arms if row['mode'] == 'off'), None)
        if off:
            for row in arms:
                row['descriptive_total_change_vs_same_version_off_percent'] = (
                    100 * (row['total_request_and_drain_ns'] / off['total_request_and_drain_ns'] - 1))
    stat = os.statvfs(ROOT)
    proc = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                           '--format=csv,noheader'], capture_output=True, text=True, timeout=10)
    result = dict(schema_version=1, captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope='closed_server11_native_and_finite_single_file_evidence_only',
                  guarded_job_count=len(runs), guarded_jobs=runs,
                  initial_inherited_gpu_seconds=18829.89441077551,
                  new_guarded_seconds=sum(row['result']['elapsed_seconds'] for row in runs),
                  cumulative_gpu_seconds=ledger['gpu_wall_seconds'],
                  remaining_gpu_seconds=28800-ledger['gpu_wall_seconds'], active_reservation=None,
                  disk=dict(total_gib=stat.f_blocks*stat.f_frsize/1024**3,
                            free_gib=stat.f_bavail*stat.f_frsize/1024**3, free_inodes=stat.f_favail),
                  nvidia_smi=dict(exit=proc.returncode, compute_processes=proc.stdout.strip(), stderr=proc.stderr.strip()),
                  p4_rows=rows, cpu_execution_records=cpu, source_closures=source_closures,
                  native_cpu_guard=read('artifacts/prefix_io_v1/server11-p4-single-file-candidate-v3-20261003/cpu-native-guard.json'),
                  full_P4_complete=False, formal_P5_performance_proof=False, formal_SLO=None,
                  general_cost_production_qualified=False, D_J_release_credit_qualified=False,
                  GPU_operations_in_this_audit=0, deletions=0)
    assert abs(result['initial_inherited_gpu_seconds'] + result['new_guarded_seconds']
               - result['cumulative_gpu_seconds']) < 0.001
    OUT.mkdir(exist_ok=False)
    path = OUT / 'SERVER11_FINAL_EVIDENCE.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(reference=reference(path), guarded_job_count=len(runs),
                         remaining_gpu_seconds=result['remaining_gpu_seconds'], disk=result['disk'],
                         compute_processes=proc.stdout.strip())))

if __name__ == '__main__':
    main()

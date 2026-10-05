"""Summarize actual bounded server10 runs; never launches GPU work."""
import collections
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
DEST = ROOT / 'artifacts/prefix_io_v1/server10-kv-system-delivery-v1-20261003'
JOBS = (
    'server10-g3-reference-native-01', 'server10-g3-reference-paired-01',
    'server10-p4-native-off-01', 'server10-p4-native-shadow-01',
    'server10-g2-normal-off-01', 'server10-g2-normal-shadow-01',
    'server10-kv-byte-populate-01',
    'server10-kv-byte-populate-02', 'server10-kv-byte-paired-02',
)


def read(path):
    return json.loads(path.read_text())


def main():
    assert ROOT.resolve(strict=True) == ROOT
    rows, refs = [], []
    for name in JOBS:
        path = ROOT / 'experiments/prefix_io_v1/runs' / name / 'result.json'
        raw = path.read_bytes()
        result = json.loads(raw)
        expected = 78 if name == 'server10-kv-byte-populate-01' else 0
        assert result['exit'] == result['child_exit'] == expected, name
        assert result['timed_out'] is False and result['session_drained'] is True, name
        assert not result['session_members_after_cleanup'] and result['error'] is None, name
        rows.append(dict(job=name, guard_exit=result['exit'], child_exit=result['child_exit'],
                         seconds=result['elapsed_seconds'], timed_out=False, session_drained=True))
        refs.append(dict(path=str(path.relative_to(ROOT)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    captures = []
    for mode in ('populate', 'paired'):
        path = ROOT / 'experiments/prefix_io_v1/runs' / f'server10-kv-byte-{mode}-02/details'
        runtime = read(path / 'kv-diagnostic-runtime-state.json')
        actual = read(path / 'calibration-runtime-result.json')
        capture_path = path / 'kv-capture/native-kv-byte-receipt.json'
        capture = read(capture_path)
        assert runtime['status'] == 'RESTORED' and runtime['capture_installed'] is True
        assert runtime['capture_receipt'] == capture and runtime['cohort_binding']['status'] == 'PASS_ACTUAL_ORIGINAL_REQUEST_CAPTURE_COHORT'
        assert actual['original_engine_shutdown_returned'] is True and actual['original_shutdown_completed'] is True
        assert capture['status'] == 'PASS_NATIVE_KV_BYTE_DIAGNOSTIC' and not capture['failures']
        assert capture['diagnostic_real_bytes_verified'] is True and capture['timing_usable'] is False
        assert capture['production_qualified'] is False and capture['effect_verified'] is False
        drains = []
        for item in actual['native_post_shutdown']:
            aio, native = item['actual_aio'], item['actual_native']
            assert aio['accepted'] == aio['completed'] == aio['reaped']
            assert all(aio[k] == 0 for k in ('outstanding', 'pending', 'ready', 'unreaped'))
            assert all(value == 0 for value in native.values()) and item['actual_handler_active'] == 0
            assert item['snapshot']['aio']['closed'] and item['snapshot']['aio']['drained']
            drains.append(dict(aio=aio, native=native, handler_active=0, closed=True, drained=True))
        assert drains
        captures.append(dict(mode=mode, status=capture['status'],
                             stage_counts=dict(collections.Counter(r['stage'] for r in capture['records'])),
                             cohort_binding=runtime['cohort_binding'], final_drains=drains,
                             record_count=len(capture['records']), producer_file_count=len(capture['files']),
                             inherited_producer_file_count=len(capture['producer_files']),
                             original_shutdown_completed=True, timing_usable=False))
    ledger = read(ROOT / 'experiments/prefix_io_v1/gpu-budget-ledger.json')
    assert ledger['active_reservation'] is None and ledger['gpu_wall_seconds'] <= 28800
    stat = os.statvfs(ROOT)
    free = stat.f_bavail * stat.f_frsize
    assert free >= 8 * 1024**3
    compute = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
    assert compute.returncode == 0
    result = dict(status='PASS_BOUNDED_NEW_SERVER_SYSTEM_VERIFICATION',
                  GPU_runs=len(rows), successful_guard_runs=8, failed_guard_runs=1,
                  jobs=rows, source_guard_refs=refs, KV_diagnostics=captures,
                  new_server_gpu_guard_seconds=sum(r['seconds'] for r in rows),
                  cumulative_gpu_guard_seconds=ledger['gpu_wall_seconds'],
                  remaining_gpu_seconds=28800-ledger['gpu_wall_seconds'], active_reservation=None,
                  PRIMARY_free_bytes=free, storage_floor_bytes=8*1024**3,
                  compute_processes_after_jobs=compute.stdout.strip(),
                  full_P3_or_P4_completion=False, research_strategy_effect_verified=False,
                  new_machine_cost_qualified=False, physical_release_credit=False,
                  KV_diagnostic_timing_usable=False, original_failed_run_unchanged=True,
                  GPU_operations_by_this_collector=0,
                  scope='Existing engine correctness and lifecycle, fixed three prefixes; not a performance comparison.')
    with (DEST / 'ACTUAL_NINE_GPU_RUN_SYSTEM_SUMMARY.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('jobs','source_guard_refs','KV_diagnostics')}))


if __name__ == '__main__':
    main()

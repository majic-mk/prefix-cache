"""Join actual CPU logs, the verified revision and original idle GPU ledger.

No framework, SDK, shared object or GPU is loaded. A prepared calibration
entry is distinct from live launch authority and from a method-effect runner.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil

REL = 'artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'
LEDGER_SHA = '60368b94c494d15227eb528273e7b4a1a20119c6f371767ffde5cc96aa8f52f9'


def require(value, reason):
    if not value:
        raise ValueError('CPU_PREFLIGHT_REJECTED: ' + reason)


def safe(root, relative):
    require(type(relative) is str and relative and '\\' not in relative and ':' not in relative
            and not relative.startswith('/') and all(p not in ('', '.', '..') for p in relative.split('/')),
            'bounded project path')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink refused')
    return path


def ref(root, relative):
    path = safe(root, relative)
    require(path.is_file() and path.stat().st_size <= 4*1024**2, 'bounded regular evidence')
    raw = path.read_bytes()
    return dict(path=relative, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def read(root, relative):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(safe(root, relative).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda x: require(False, 'nonfinite JSON'))


def write(root, filename, document):
    path = safe(root, REL+'/'+filename)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return ref(root, REL+'/'+filename)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU invocation requires devices empty')
    spec = importlib.util.spec_from_file_location('_prospective_CPU_protocol', root/REL/'protocol/i_pilot_protocol.py')
    protocol_api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol_api)
    protocol = read(root, REL+'/protocol/I_PILOT_PROTOCOL.json')
    protocol_api.validate_protocol(protocol)
    freeze_tag = REL+'/CPU_CALIBRATION_SOURCE_FREEZE_01'
    freeze_result = read(root, freeze_tag+'_RESULT.json')
    command = read(root, freeze_tag+'_COMMAND.json')
    require(freeze_result.get('exit') == 0 and freeze_result.get('GPU_runs') == 0, 'actual freeze must pass')
    require(command == dict(argv=[str(root/'.venv/bin/python'), '-B', '-I', '-S',
        str(root/REL/'calibration_v2/control_native_cost_job.py'), 'freeze'], cwd=str(root), CUDA_VISIBLE_DEVICES=''),
        'actual expected native CPU freeze command')
    freeze_out = read(root, freeze_tag+'_STDOUT.log')
    require(freeze_out.get('status') == 'CPU_REVISION_ASSETS_FROZEN_UNBOUND'
            and freeze_out.get('actual_gpu_runs') == 0 and freeze_out.get('gpu_launch_allowed') is False,
            'freeze is unbound CPU evidence')
    lock_ref = freeze_out['revision_lock_ref']
    require(ref(root, lock_ref['path']) == lock_ref, 'revision bytes changed after actual full verification')
    lock = read(root, lock_ref['path'])
    rows = lock['files']
    require(len(rows) >= 4655 and len({r['path'] for r in rows}) == len(rows), 'complete unique inherited closure')
    require(lock['source_ancestry_is_not_authority'] is True and lock['gpu_launch_allowed'] is False,
            'immutable source lock grants no GPU authority')
    digest = hashlib.sha256(json.dumps(sorted(rows, key=lambda r:r['path']), sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    # freeze_revision really verifies all original rows, then _revision(full=True)
    # verifies the resulting entire closure. Attribute this proof to those logs;
    # do not claim a third whole-model byte audit in this helper.
    proof_ref = write(root, 'CALIBRATION_CPU_SOURCE_PROOF.json', dict(
        schema='i_pilot_actual_CPU_freeze_source_proof_v1', origin='actual_server_CPU',
        status='PASS_FULL_BYTES_IN_ORIGINAL_NATIVE_FREEZE', GPU_operations=0,
        files_verified=len(rows), total_bytes_verified=sum(r['bytes'] for r in rows),
        failed=[], source_rows_sha256=digest, source_lock_ref=lock_ref,
        verification_method='actual_native_freeze_revision_checked_rows_and_final_revision_full_true',
        actual_command_ref=ref(root, freeze_tag+'_COMMAND.json'),
        actual_result_ref=ref(root, freeze_tag+'_RESULT.json'),
        actual_stdout_ref=ref(root, freeze_tag+'_STDOUT.log'),
        actual_stderr_ref=ref(root, freeze_tag+'_STDERR.log')))
    tag = REL+'/CPU_CALIBRATION_UNIT_02'
    result = read(root, tag+'_RESULT.json')
    raw_log = safe(root, tag+'_STDERR.log').read_text(encoding='utf-8')
    runs = re.findall(r'^Ran (\d+) tests? in ', raw_log, flags=re.M)
    outcomes = re.findall(r'^test_.* \.\.\. (ok|FAIL|ERROR|skipped[^\n]*)$', raw_log, flags=re.M)
    require(result.get('exit') == 0 and result.get('GPU_runs') == 0 and len(runs) == 1,
            'actual entry tests must pass')
    count = int(runs[0])
    require(count == len(outcomes) == outcomes.count('ok') and count > 0 and raw_log.rstrip().endswith('OK'),
            'all entry tests passed without skip')
    tests_ref = write(root, 'CALIBRATION_ACTUAL_CPU_TEST_REPORT.json', dict(
        schema='i_pilot_actual_CPU_entry_test_report_v1', origin='actual_server_CPU',
        GPU_operations=0, exit_code=0, tests_passed=count, tests_failed=0, tests_skipped=0,
        stdout_ref=ref(root, tag+'_STDOUT.log'), stderr_ref=ref(root, tag+'_STDERR.log'),
        command_ref=ref(root, tag+'_COMMAND.json'), actual_result_ref=ref(root, tag+'_RESULT.json')))
    ledger_ref = ref(root, LEDGER)
    require(ledger_ref['sha256'] == LEDGER_SHA, 'original idle GPU ledger byte drift')
    ledger = read(root, LEDGER)
    used = ledger.get('gpu_wall_seconds')
    require(type(used) in (int,float) and math.isfinite(used) and 0 <= used <= 28800
            and ledger.get('active_reservation') is None and 28800-used >= 1220,
            'original calibration allowance unavailable')
    require(shutil.disk_usage(root/REL).free >= 10*1024**3, '8GiB free plus 2GiB calibration reserve')
    command = protocol_api.guard_command(protocol_api.job_rows()[0],
        permission_relative=REL+'/calibration_v2/EFFECTIVE_GPU_PERMISSION.json',
        runner_relative=REL+'/calibration_v2/run_native_cost_experiment.py')
    preflight_ref = write(root, 'CALIBRATION_CPU_PREFLIGHT.json', dict(
        schema='i_pilot_calibration_CPU_preflight_v1', status='CPU_CALIBRATION_PREFLIGHT_PASS',
        origin='actual_server_CPU', GPU_operations=0, source_lock_ref=lock_ref,
        source_proof_ref=proof_ref, cpu_test_result_ref=tests_ref, full_source_verified=True,
        entry_CPU_tests_passed=True, requires_live_original_guard_recheck=True,
        model_manifest_sha256=protocol_api.MODEL, original_ledger_ref=ledger_ref,
        guard_command=command, device_launch_ready=False,
        device_status='not_qualified_on_CPU; live resources and UUID binding required before launch',
        gpu_seconds_consumed=0, actual_remaining_seconds=28800-used,
        formal_effect_entry_ready=False, current_cost_upper_or_budget_changed=False))
    references = dict(source_lock_ref=lock_ref, source_proof_ref=proof_ref,
                      cpu_test_result_ref=tests_ref, preflight_ref=preflight_ref)
    write(root, 'CALIBRATION_ENTRY_REFERENCES.json', references)
    print(json.dumps(dict(status='ACTUAL_CPU_PREFLIGHT_RECORDS_READY_FOR_BYTE_JOIN', GPU_operations=0,
                          tests_passed=count, source_rows=len(rows), entry_references=references)))


if __name__ == '__main__':
    main()

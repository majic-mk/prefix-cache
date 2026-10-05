"""Join actual server CPU evidence without loading GPU or model libraries."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
OUT = ROOT / 'artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004'
LEDGER_SHA = '60368b94c494d15227eb528273e7b4a1a20119c6f371767ffde5cc96aa8f52f9'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(name):
    return json.loads((OUT / name).read_text())


def ref(name):
    p = OUT / name
    raw = p.read_bytes()
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def write(name, obj):
    with (OUT / name).open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only evidence join')
    specifications = [
        ('calibration_entry', 'CPU_CALIBRATION_UNIT_02', 109),
        ('interference_original_chain', 'CPU_I_BRIDGE_01', 20),
        ('owner_release_observation', 'CPU_OWNER_ADAPTER_02', 22),
        ('standing_authorization_review', 'CPU_CALIBRATION_AUTH_REVIEW_02', 21),
        ('prospective_protocol', 'CPU_PROTOCOL_TESTS_01', 22),
        ('strong_common_configuration', 'CPU_STRONG_BASELINE_01', 15),
    ]
    tests = []
    for component, tag, expected in specifications:
        result = read(tag + '_RESULT.json')
        require(result['exit'] == 0 and result['GPU_runs'] == 0, tag + ' actual CPU result')
        if component in ('owner_release_observation', 'strong_common_configuration'):
            report_name = ('SERVER_OWNER_PREPARATION_CPU_RESULT_V2.json'
                           if component == 'owner_release_observation' else 'SERVER_STRONG_CPU_RESULT.json')
            report = read(report_name)
            log = report['stderr']
            require(report['exit_code'] == 0, component + ' actual nested command result')
            log_ref = ref(report_name)
        else:
            log = (OUT / (tag + '_STDERR.log')).read_text()
            log_ref = ref(tag + '_STDERR.log')
        counts = re.findall(r'Ran (\d+) tests? in ', log)
        require(counts == [str(expected)] and re.search(r'\nOK\s*$', log), component + ' complete unittest result')
        tests.append(dict(component=component, passed=expected, failed=0, skipped=0,
                          result_ref=ref(tag + '_RESULT.json'), log_ref=log_ref,
                          command_ref=ref(tag + '_COMMAND.json')))
    require(sum(row['passed'] for row in tests) == 209, 'distinct final six-suite count')
    before, after = read('SOURCE_PROTECTION_BEFORE.json'), read('SOURCE_PROTECTION_AFTER.json')
    for key in ('source_lock_ref', 'source_map', 'original_ledger_ref', 'actual_completed_GPU_archive_ref'):
        require(before[key] == after[key], 'protected original inputs unchanged: ' + key)
    require(before['complete_source_rows_verified'] == after['complete_source_rows_verified'] == 4816,
            'full original protection proofs')
    ledger_path = ROOT / 'experiments/prefix_io_v1/gpu-budget-ledger.json'
    ledger_raw = ledger_path.read_bytes()
    require(hashlib.sha256(ledger_raw).hexdigest() == LEDGER_SHA, 'unchanged original cumulative budget')
    require(ledger_raw == (OUT / 'CPU_AUDIT_START_LEDGER_SNAPSHOT.json').read_bytes(), 'exact ledger snapshot match')
    ledger = json.loads(ledger_raw)
    require(ledger['active_reservation'] is None, 'no GPU reservation made by CPU preparation')
    prior = read('CPU_NEXT_PHASE_DECISION.json')
    require(prior['status'] == 'CPU_READY_FOR_GPU_CALIBRATION' and
            prior['device_launch_ready'] is False and prior['formal_goodput_allowed'] is False,
            'CPU join cannot qualify GPU or effects')
    strong = read('SERVER_STRONG_CPU_RESULT.json')
    require(strong['input_bytes_unchanged'] is True and strong['input_file_count'] == 53 and
            strong['prior_singlefile_planner_off_receipt_reusable_for_strong_effect'] is False,
            'actual strong configuration source proof and receipt domain boundary')
    tracked = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    require(tracked == '', 'original server tracked code unchanged')
    post = dict(schema='actual_CPU_preparation_post_state_v1', utc_epoch=time.time(),
                host=os.uname().nodename, device_nodes=sorted(str(p) for p in Path('/dev').glob('nvidia*')),
                free_primary_bytes=shutil.disk_usage('/root/autodl-tmp').free,
                tracked_status=tracked, original_ledger_sha256=LEDGER_SHA,
                active_reservation=ledger['active_reservation'], actual_GPU_runs_this_turn=0,
                deleted_data_files_this_turn=0, system_driver_package_changes_this_turn=0)
    write('CPU_PREPARATION_POST_STATE.json', post)
    decision = dict(
        schema='actual_server_bounded_CPU_preparation_decision_v1',
        status='PASS_BOUNDED_CPU_COMPONENTS_CALIBRATION_ENTRY_READY_EFFECT_BLOCKED',
        actual_final_server_CPU_tests=tests, distinct_final_tests_passed=209,
        failed_final_tests=0, skipped_final_tests=0, earlier_failed_attempts_retained=True,
        original_source_rows_protected_before_and_after=4816,
        new_calibration_source_rows_verified=4749,
        original_source_protection_before_ref=ref('SOURCE_PROTECTION_BEFORE.json'),
        original_source_protection_after_ref=ref('SOURCE_PROTECTION_AFTER.json'),
        calibration_source_proof_ref=ref('CALIBRATION_CPU_SOURCE_PROOF.json'),
        calibration_protocol_join_ref=ref('CPU_NEXT_PHASE_DECISION.json'),
        strong_config_server_result_ref=ref('SERVER_STRONG_CPU_RESULT.json'),
        post_state_ref=ref('CPU_PREPARATION_POST_STATE.json'),
        actual_GPU_runs_this_turn=0, actual_GPU_seconds_this_turn=0,
        original_cumulative_GPU_remaining_seconds=prior['actual_original_remaining_seconds'],
        new_GPU_reservations_created=0, calibration_entry_CPU_ready=True,
        strong_U_I_common_configuration_CPU_ready=True, live_strong_effect_runner_bound=False,
        current_device_nodes_present=bool(post['device_nodes']), device_launch_qualified=False,
        performance_benefit_proved=False, production_owner_release_capability=False,
        complete_P4_effect_preparation=False,
        next_allowed_scope='bounded mechanism calibration only after actual live-resource, source, UUID, storage and original-budget recheck',
        next_calibration_guard_seconds=1200, next_calibration_cleanup_reserve_seconds=20,
        diagnostic_calibration_planner_mode='off', diagnostic_receipt_covers_strong_planner_on_effect=False,
        additional_strong_planner_on_calibration_required=True,
        formal_U_I_effect_missing_gates=[
            'bind actual planner-on effect runner using the tested common configuration and an existing natural service trace',
            'freeze independent development deadline, reserve, budget and legitimate evaluation SLO before evaluation',
            'obtain new empirical cost coverage matching the strong planner-on configuration',
            'observe an ordinary legal shadow candidate without relaxing the old cost gate',
            'complete actual on lifecycle, capacity fences, accounting and output verification',
            'complete unselected paired U/I evaluation with all failures and drain costs included'],
        independent_deadline_ns=None, development_budget_ns=None, evaluation_SLO_ns=None,
        conditional_protocol_max_seconds=3780, conditional_protocol_is_actual_reservation=False,
        conditional_protocol_includes_extra_strong_calibration=False,
        all_effect_runs_fit_remaining_budget_guaranteed=False,
        CPU_fixtures_are_GPU_measurements=False, old_failed_experiment_repaired=False)
    write('FINAL_CPU_STAGE_DECISION.json', decision)
    print(json.dumps({k: decision[k] for k in ('status', 'distinct_final_tests_passed',
          'actual_GPU_runs_this_turn', 'calibration_entry_CPU_ready', 'live_strong_effect_runner_bound',
          'performance_benefit_proved')}, sort_keys=True))


if __name__ == '__main__':
    main()

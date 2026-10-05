"""Prospective I-only pilot contract. Standard library; never launches a job.

The old failed envelope is retained verbatim. A new table is a new experiment,
not a repair of old held-out observations. Fixtures are rejected as evidence.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'i_pilot_prospective_protocol_v1'
REMOTE = 'artifacts/prefix_io_v1/server12-i-pilot-cpu-preparation-20261004'
REMAINING = 4121.104761094321
MODEL = '9b1c79986aa1fd62eb9064ecb8b4e2a03fc235a34ac2d69c33a4fcb531a91017'
OLD = dict(cost_upper_ns=16238752, step_budget_ns=13171328,
           ordinary_cost_admission_fits=False, failed_records_retained=True,
           no_refit=True, no_threshold_widening=True)
FAMILIES = (
    ('cal0', 'calibration_fit', 40100, 4029),
    ('cal1', 'calibration_fit', 41100, 4030),
    ('cal2', 'calibration_holdout', 42100, 4031),
    ('dev0', 'development', 43100, 4329),
    ('dev1', 'development', 44100, 4430),
    ('eval0', 'evaluation', 45100, 4529),
    ('eval1', 'evaluation', 46100, 4630),
)
SLOTS = (
    ('cal01', 'calibration', 'off', ['cal0', 'cal1', 'cal2'], 1200),
    ('dev01', 'development', 'off', ['dev0'], 300),
    ('dev02', 'development', 'off', ['dev1'], 300),
    ('shadow01', 'shadow_qualification', 'shadow', ['dev0'], 300),
    ('on01', 'on_lifecycle_qualification', 'on', ['dev1'], 300),
    ('eval00-off', 'evaluation', 'off', ['eval0'], 300),
    ('eval00-on', 'evaluation', 'on', ['eval0'], 300),
    ('eval01-on', 'evaluation', 'on', ['eval1'], 300),
    ('eval01-off', 'evaluation', 'off', ['eval1'], 300),
)


def require(value, message):
    if not value:
        raise ValueError('I_PILOT_PROTOCOL_REJECTED: ' + message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label)
    return value


def exact(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(exact(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return left == right


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def parse(raw):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, 'duplicate JSON key')
            output[key] = value
        return output
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'nonfinite JSON'))


def safe(root, relative):
    require(type(relative) is str and relative and '\\' not in relative and ':' not in relative
            and not relative.startswith('/') and all(p not in ('', '.', '..') for p in relative.split('/')),
            'project-relative POSIX path')
    root = Path(root).resolve(strict=True)
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'symlink refused')
    require(path.resolve().is_relative_to(root), 'outside project')
    return path


def file_ref(root, relative):
    path = safe(root, relative)
    require(path.is_file() and path.stat().st_size <= 32 * 1024**2, 'bounded regular evidence')
    return dict(path=relative, bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def check_ref(root, row):
    require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'exact file reference')
    integer(row['bytes'], 'typed reference bytes', 1)
    require(type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'SHA-256')
    require(file_ref(root, row['path']) == row, 'actual referenced bytes changed')
    return parse(safe(root, row['path']).read_bytes())


def family_rows():
    return [dict(id=ident, partition=partition, prompt_token_ids=[first] + list(range(1001, 1129)),
                 seed=seed, workload_sha256=digest(dict(prompt=[first] + list(range(1001, 1129)), seed=seed)))
            for ident, partition, first, seed in FAMILIES]


def job_rows():
    output = []
    for ident, phase, mode, families, seconds in SLOTS:
        label = 'server12-i-pilot-' + ident
        output.append(dict(id=ident, label=label, phase=phase, mode=mode, family_ids=families,
                           window_order=[['A', 'B'], ['B', 'A'], ['A', 'B']] if phase == 'calibration' else [['B']],
                           execution_seconds=seconds, cleanup_seconds=20, reserved_seconds=seconds + 20,
                           config_relative=(REMOTE + '/calibration_v2/NATIVE_COST_CONFIG.json'
                                            if phase == 'calibration' else REMOTE + '/CONFIG_' + ident + '.json'),
                           run_relative='experiments/prefix_io_v1/runs/' + label,
                           source_proof_before=(REMOTE + '/calibration_v2/SOURCE_BEFORE_VERIFICATION.json'
                                                if phase == 'calibration' else REMOTE + '/SOURCE_' + ident + '_BEFORE.json'),
                           source_proof_launch=(REMOTE + '/calibration_v2/SOURCE_LAUNCH_VERIFICATION.json'
                                                if phase == 'calibration' else REMOTE + '/SOURCE_' + ident + '_LAUNCH.json'),
                           source_proof_after=(REMOTE + '/calibration_v2/SOURCE_AFTER_VERIFICATION.json'
                                               if phase == 'calibration' else REMOTE + '/SOURCE_' + ident + '_AFTER.json')))
    return output


def prospective_protocol():
    return dict(
        schema=SCHEMA, scope='server12_i_only_prospective_pilot_v1',
        created_client_date='2026-10-04', origin='CPU_PROSPECTIVE_PREREGISTRATION_NO_NEW_GPU_DATA',
        remote_scope=REMOTE, frozen_old_experiment=OLD.copy(), families=family_rows(), jobs=job_rows(),
        model_manifest_sha256=MODEL,
        diagnostic_geometry=dict(prompt_tokens=129, output_tokens=128, selected_offset=16,
                                 selected_context_tokens=144, warmup_offsets=[1],
                                 legal_storage_units=1, ssd_read_bytes=917504, max_accepted_parents=8,
                                 prefill_tokens=0, decode_batch=1, bf16=True, kernel_mode='eager'),
        strategy=dict(baseline='U_same_native_executor_and_all_existing_features',
                      off_bridge=None, on_policy='interference', dependency_sorting=False,
                      single_file_dispatch_controller=None, no_new_cache_or_model_executor=True,
                      fixed_pressure_are_required_before_paper_claims=True),
        budget=dict(original_ledger='experiments/prefix_io_v1/gpu-budget-ledger.json',
                    original_guard='experiments/prefix_io_v1/scripts/run_gpu_stage.py',
                    original_limit_seconds=28800, remaining_snapshot_seconds=REMAINING,
                    planned_max_reserved_seconds=3780, reservation_policy='original_guard_only',
                    live_recheck_before_each_job=True, attempt_consumed_on_failure=True,
                    retry_slots=0, remaining_after_all_max_reservations_seconds=REMAINING - 3780,
                    primary_reservation_per_short_job_bytes=134217728,
                    minimum_free_disk_bytes=8589934592, no_downloads=True,
                    system_or_driver_changes=False, existing_data_deletion=False,
                    persistent_user_GPU_authorization=True, per_round_reapproval_required=False),
        calibration=dict(fit_family_ids=['cal0', 'cal1'], heldout_family_ids=['cal2'],
                         fit_rule='ceil(mean(A_fit)) + max(0,ceil(mean(B_fit-A_fit))) + max_positive_fit_B_residual',
                         domain='GPU_execute_model_through_sample_tokens_current_stream_Event_elapsed_time_ns',
                         heldout_rule='one prospective B_holdout must be <= new fitted upper; failure stops, no refit',
                         old_data_input_to_fit=False, table_is_new_revision=True,
                         normal_workload_generalization_claim=False),
        development=dict(family_ids=['dev0', 'dev1'], freeze_before_first_on=True,
                         internal_step_budget_ns=None, non_GPU_reserve_ns=None,
                         declared_full_control_window_deadline_ns=None,
                         freeze_rule='independently declared development deadline minus measured development scheduler_sampling_output_and_control_reserve',
                         deadline_must_be_declared_before_development_records=True,
                         observed_A_max_is_service_SLO=False,
                         baseline_or_B_or_on_outcomes_must_not_trigger_deadline_widening=True,
                         heldout_or_evaluation_used_to_fit=False,
                         if_no_independent_deadline='BLOCK_EFFECT_ORDINARY_ON; shadow and off diagnostic only',
                         if_no_ordinary_admission='STOP_NO_CURRENT_INTERFERENCE_ADMISSION_OPPORTUNITY'),
        SLO=dict(TTFT_ns=None, request_ITL_P95_ns=None,
                 formal_goodput_allowed=False, step_budget_is_service_SLO=False,
                 freeze_service_targets_on_development_before_evaluation=True),
        evaluation=dict(family_ids=['eval0', 'eval1'], pairs=[['eval00-off', 'eval00-on'], ['eval01-on', 'eval01-off']],
                        primary='per_request_native_single_token_output_ITL_P95_raw_latency',
                        secondary=['TTFT_from_original_request_acceptance', 'drained_cohort_makespan',
                                   'all_request_throughput', 'tail_drain_time', 'actual_transfer_bytes',
                                   'progress_overrides', 'table_unknown_and_fallback_counts',
                                   'failed_cancelled_timed_out_requests', 'controller_CPU_time'],
                        fixed_length_ignore_eos_is_diagnostic=True,
                        injected_single_file_background_is_mechanism_diagnostic=True,
                        natural_service_stream_effect_claim_allowed=False,
                        no_token_independence_assumption=True, paired_order_is_predeclared=True,
                        all_scheduled_attempts_retained=True, no_best_run_selection=True,
                        two_pairs_do_not_establish_population_or_paper_effect=True),
        stage_gates=[
            'actual new source lock, CPU entry test evidence, original guard preflight, exact model and per-slot fresh directories',
            'calibration succeeds including never-fitted cal2 heldout and native shutdown/session drain',
            'development deadline declared prospectively, reserves measured, final budget frozen before on01',
            'shadow proposal source/epoch/legal-resource/cost coverage and ordinary candidate demonstrated',
            'on01 exact output/native progress/shutdown/session drain and total accounting valid',
            'only then four preregistered evaluation slots in their fixed order; no retry or replacement'],
        stop_rules=[
            'missing exact condition or cost coverage; preserve failure and do not promote',
            'no ordinary legal candidate under independent frozen development budget',
            'no actual strategy action difference; report no opportunity, not an unmeasured speedup',
            'any source drift, layout/model/GPU mismatch, unsafe capacity, foreign work, original guard failure or unsafe drain',
            'effect requires artificial throttle, a new executor, changed release protocol or deleting data',
            'future dependency effect requires normal resource wait and true owner release witness; this I-only pilot does not supply that proof'],
        GPU_operations_performed_by_protocol=0, GPU_launch_authorized_by_protocol=False,
        performance_benefit_proved=False, method_permanently_infeasible_proved=False)


def validate_protocol(document):
    require(type(document) is dict and exact(document, prospective_protocol()),
            'exact prospective families/order/budget/safety rules; outcome-dependent protocol mutations forbidden')
    require(type(document['budget']['planned_max_reserved_seconds']) is int, 'integer budget, not bool')
    require(sum(integer(job['reserved_seconds'], 'guard reservation', 1) for job in document['jobs']) == 3780,
            'bounded summed original guard reservations')
    require(3780 <= REMAINING, 'original remaining budget')
    return document


def guard_command(job, *, permission_relative, runner_relative):
    require(any(exact(job, row) for row in job_rows()), 'exact preregistered slot')
    for relative in (permission_relative, runner_relative):
        require(type(relative) is str and relative.startswith(REMOTE + '/') and
                '\\' not in relative and ':' not in relative and '..' not in relative.split('/'),
                'same-scope actual frozen permission/runner binding')
    return ['.venv/bin/python', '-B', 'experiments/prefix_io_v1/scripts/run_gpu_stage.py',
            '--permissions-path', permission_relative, '--label', job['label'],
            '--seconds', str(job['execution_seconds']), '--', '.venv/bin/python', '-B',
            runner_relative, '--execute', '--config', job['config_relative']]


def assess_calibration_preparation(root, protocol, *, source_lock_ref, source_proof_ref,
                                  cpu_test_result_ref, preflight_ref):
    """Join actual server source/test/preflight bytes. No launch or GPU import.

    A READY result prepares only cal01. All later slots remain prospective.
    Device presence and live guard authorization must be rechecked at launch.
    This join is optional: absent actual references keep readiness blocked.
    """
    validate_protocol(protocol)
    lock = check_ref(root, source_lock_ref)
    rows = lock.get('files')
    require(type(rows) is list and 4 <= len(rows) <= 10000, 'actual source lock rows')
    references = {}
    for row in rows:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'typed source reference')
        integer(row['bytes'], 'source reference bytes')
        require(type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']), 'source SHA')
        safe(root, row['path'])
        require(row['path'] not in references, 'unique source reference')
        references[row['path']] = row
    essential = [REMOTE + '/calibration_v2/' + name for name in
                 ('run_native_cost_experiment.py', 'control_native_cost_job.py', 'gpu_entry_binding.py')]
    essential.append('experiments/prefix_io_v1/scripts/run_gpu_stage.py')
    for relative in essential:
        require(relative in references and file_ref(root, relative) == references[relative],
                'actual source-pinned existing guard and executable calibration entry')
    require(references[essential[-1]]['sha256'] ==
            '3527b6218d3e6c5781de550ce1c7cdebfd354e37d80f41693010201a3586929a',
            'unchanged original guard; no second ledger')
    model_manifest_relative = 'artifacts/prefix_io_v1/new-server-03/modelscope-source/modelscope-download-plan.json'
    require(model_manifest_relative in references
            and references[model_manifest_relative]['sha256'] == MODEL
            and exact(file_ref(root, model_manifest_relative), references[model_manifest_relative]),
            'actual source-pinned model manifest bytes')
    tree = ast.parse(safe(root, essential[0]).read_bytes())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    require(constants.get('PROMPT_FIRST') == (40100, 41100, 42100)
            and constants.get('SEEDS') == (4029, 4030, 4031)
            and constants.get('LABEL') == 'server12-i-pilot-cal01'
            and constants.get('ORDER') == (('A', 'B'), ('B', 'A'), ('A', 'B')),
            'actual executable calibration constants bind the preregistered split/order')
    proof = check_ref(root, source_proof_ref)
    require(proof.get('failed') == [] and type(proof.get('files_verified')) is int
            and proof['files_verified'] == len(rows), 'complete actual source closure proof')
    require(proof.get('source_rows_sha256') == digest([references[path] for path in sorted(references)]),
            'actual proof identifies these exact frozen source rows')
    tests = check_ref(root, cpu_test_result_ref)
    require(type(tests.get('exit_code')) is int and tests['exit_code'] == 0
            and integer(tests.get('tests_passed'), 'actual tests passed', 1) > 0
            and type(tests.get('tests_failed')) is int and tests['tests_failed'] == 0
            and type(tests.get('tests_skipped')) is int and tests['tests_skipped'] == 0,
            'actual successful CPU entry tests, no skipped coverage')
    require(tests.get('origin') == 'actual_server_CPU' and type(tests.get('GPU_operations')) is int
            and tests['GPU_operations'] == 0,
            'actual CPU result, never fixture or synthetic GPU evidence')
    for key in ('stdout_ref', 'stderr_ref'):
        row = tests.get(key)
        require(type(row) is dict and row.get('bytes') is not None, 'actual test output bytes')
        require(exact(file_ref(root, row['path']), row), 'actual test stdout/stderr source bytes')
    preflight = check_ref(root, preflight_ref)
    require(preflight.get('schema') == 'i_pilot_calibration_CPU_preflight_v1'
            and preflight.get('status') == 'CPU_CALIBRATION_PREFLIGHT_PASS'
            and preflight.get('origin') == 'actual_server_CPU'
            and type(preflight.get('GPU_operations')) is int and preflight['GPU_operations'] == 0
            and preflight.get('full_source_verified') is True
            and preflight.get('entry_CPU_tests_passed') is True
            and preflight.get('requires_live_original_guard_recheck') is True,
            'actual current CPU guard preflight only')
    require(preflight.get('source_lock_ref') == source_lock_ref
            and preflight.get('source_proof_ref') == source_proof_ref
            and preflight.get('cpu_test_result_ref') == cpu_test_result_ref,
            'same byte-pinned source proof and entry tests')
    require(preflight.get('model_manifest_sha256') == MODEL, 'same actual model identity')
    ledger_ref = preflight.get('original_ledger_ref')
    require(type(ledger_ref) is dict and ledger_ref.get('path') == protocol['budget']['original_ledger'],
            'same original cumulative ledger, never a shadow ledger')
    ledger = check_ref(root, ledger_ref)
    seconds = ledger.get('gpu_wall_seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= 28800
            and ledger.get('active_reservation') is None and 28800 - seconds >= 1220,
            'actual idle original ledger can reserve cal01')
    command = guard_command(job_rows()[0],
                            permission_relative=REMOTE + '/calibration_v2/EFFECTIVE_GPU_PERMISSION.json',
                            runner_relative=REMOTE + '/calibration_v2/run_native_cost_experiment.py')
    require(preflight.get('guard_command') == command, 'actual original guard calibration command')
    require(type(preflight.get('device_launch_ready')) is bool,
            'device unavailable remains explicit and does not fabricate GPU readiness')
    result = evidence_status(protocol, runtime_ready=True)
    result.update(status='CPU_READY_FOR_GPU_CALIBRATION', calibration_entry_ready=True, paired_effect_runner_bound=False,
                  device_launch_ready=preflight['device_launch_ready'],
                  GPU_launch_allowed_by_CPU_join=False, live_recheck_required_before_execution=True,
                  actual_original_remaining_seconds=28800 - seconds,
                  calibration_guard_command=command,
                  verified_refs=dict(source_lock_ref=source_lock_ref, source_proof_ref=source_proof_ref,
                                     cpu_test_result_ref=cpu_test_result_ref, preflight_ref=preflight_ref,
                                     original_ledger_ref=ledger_ref))
    return result


def validate_development_freeze(document):
    """No effect eligibility without a prospective independent deadline.

    A-max is neither an ITL objective nor a deadline. Measured reserve uses
    development-only scheduler/sampling/output/control observations. Formal
    service SLO stays null in this mechanism pilot.
    """
    require(type(document) is dict and document.get('schema') == 'i_pilot_development_freeze_v1',
            'prospective development freeze')
    declared = integer(document.get('deadline_declared_ns'), 'declaration clock', 1)
    first = integer(document.get('first_development_observation_ns'), 'first development observation', 1)
    frozen = integer(document.get('frozen_before_on_ns'), 'freeze before on', 1)
    on = integer(document.get('first_on_observation_ns'), 'first on observation', 1)
    require(declared < first <= frozen < on, 'deadline precedes development; freeze precedes on')
    deadline = integer(document.get('declared_full_control_window_deadline_ns'), 'independent development deadline', 1)
    reserve = integer(document.get('development_non_GPU_reserve_ns'), 'development reserve', 1)
    require(deadline > reserve, 'positive internal budget after reserved control/output cost')
    require(type(document.get('internal_step_budget_ns')) is int
            and document['internal_step_budget_ns'] == deadline - reserve,
            'internal budget retains non-GPU reserve')
    require(document.get('deadline_origin') == 'independent_predeclared_development_contract'
            and document.get('source_partitions') == ['development']
            and document.get('used_old_heldout_or_evaluation') is False
            and document.get('deadline_widened_after_results') is False
            and document.get('service_TTFT_SLO_ns') is None
            and document.get('service_ITL_SLO_ns') is None,
            'independent data and no fabricated SLO or fitted evaluation')
    upper = integer(document.get('new_calibration_cost_upper_ns'), 'new validated cost upper', 1)
    return dict(status='ORDINARY_ADMISSION_CANDIDATE_EXISTS' if upper <= deadline - reserve
                else 'STOP_NO_CURRENT_INTERFERENCE_ADMISSION_OPPORTUNITY',
                ordinary_cost_admission_fits=upper <= deadline - reserve,
                internal_step_budget_ns=deadline - reserve, cost_upper_ns=upper,
                frozen_old_experiment_changed=False, formal_goodput_allowed=False)


def token_metrics(frontend):
    """Use only one actual content token per original engine output event.

    Cumulative outputs with a jump are chunk timing, not measured individual ITL.
    No division of chunk gaps or synthetic equal timestamps is permitted.
    """
    require(type(frontend) is dict, 'actual original frontend')
    started = integer(frontend.get('request_started_ns'), 'request acceptance timestamp', 1)
    finished = integer(frontend.get('request_completed_ns'), 'request completion timestamp', 1)
    rows, steps = frontend.get('token_times'), frontend.get('steps')
    require(type(rows) is list and len(rows) == 128 and type(steps) is list and 128 <= len(steps) <= 4096,
            'complete native token event and step evidence')
    emitted, previous, times = 0, started, []
    for step in steps:
        require(type(step) is dict, 'original step')
        before = integer(step.get('before_ns'), 'host step start', 1)
        after = integer(step.get('after_ns'), 'host step end', 1)
        require(previous <= before <= after <= finished, 'ordered observed frontend events')
        counts = step.get('cumulative_output_counts')
        require(type(counts) is list and len(counts) <= 1, 'sole original request')
        if counts:
            count = integer(counts[0], 'cumulative original content token count', 1)
            require(count == emitted + 1, 'batched/chunk output cannot become measured individual ITL')
            row = rows[emitted]
            require(type(row) is dict and type(row.get('token_ordinal')) is int and row['token_ordinal'] == emitted
                    and type(row.get('at_ns')) is int and row['at_ns'] == after,
                    'one actual content token maps to this original output event')
            times.append(after)
            emitted = count
        previous = after
    require(emitted == 128 and started <= times[0] < times[-1] <= finished,
            'complete actual ordered output events')
    itl = sorted(times[index] - times[index - 1] for index in range(1, len(times)))
    return dict(time_scope='host receipt of actual original single-token outputs; no client send clock',
                TTFT_ns=times[0] - started, request_ITL_P95_ns=itl[math.ceil(.95 * len(itl)) - 1],
                request_ITL_P99_ns=itl[math.ceil(.99 * len(itl)) - 1],
                output_tokens=128, actual_ITL_observations=127, ITL_independence_assumed=False)


def evidence_status(protocol, *, runtime_ready=False, calibrated=False, development_frozen=False,
                    ordinary_candidate=False, on_lifecycle=False, evaluation_complete=False):
    validate_protocol(protocol)
    for value in (runtime_ready, calibrated, development_frozen, ordinary_candidate, on_lifecycle, evaluation_complete):
        require(type(value) is bool, 'explicit boolean qualification')
    missing = [name for name, value in (
        ('actual_source_pinned_runtime_and_original_guard_preflight', runtime_ready),
        ('new_calibration_and_heldout_coverage', calibrated),
        ('independent_development_deadline_reserve_and_budget_frozen', development_frozen),
        ('shadow_ordinary_legal_candidate', ordinary_candidate),
        ('actual_on_lifecycle_and_accounting', on_lifecycle),
        ('four_complete_unselected_evaluation_records', evaluation_complete)) if not value]
    return dict(schema='i_pilot_CPU_protocol_decision_v1',
                status='CPU_PROTOCOL_COMPLETE_RUNTIME_BINDING_REQUIRED',
                effect_status='BLOCKED_FOR_EFFECT' if missing else 'DECLARED_GATES_AWAIT_ACTUAL_RECORD_BYTE_VALIDATION',
                missing_effect_gates=missing, protocol_validation_passed=True,
                formal_goodput_allowed=False, natural_service_stream_effect_verified=False,
                CPU_fixtures_are_GPU_data=False, GPU_operations_performed=0,
                original_failed_experiment_repaired=False, performance_benefit_proved=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--entry-references', type=Path,
                        help='actual JSON with source_lock_ref/source_proof_ref/cpu_test_result_ref/preflight_ref')
    parser.add_argument('--project-root', type=Path,
                        help='actual server project required only for byte-verified entry join')
    args = parser.parse_args(argv)
    document = validate_protocol(parse(args.protocol.read_bytes()))
    result = evidence_status(document)
    if args.entry_references:
        require(args.project_root is not None, 'actual project required for executable entry byte join')
        references = parse(args.entry_references.read_bytes())
        require(type(references) is dict and set(references) ==
                {'source_lock_ref', 'source_proof_ref', 'cpu_test_result_ref', 'preflight_ref'},
                'exact actual server entry references')
        result = assess_calibration_preparation(args.project_root, document, **references)
    if args.output:
        with args.output.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

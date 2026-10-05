"""Run exactly three predeclared off diagnostics through the original GPU guard.

The caller prepares separate live authorities, scopes and before proofs first.
This supervisor never creates grants, changes thresholds, retries an attempt,
kills another process, or turns a diagnostic cost observation into P4 eligibility.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path('/root/autodl-tmp/prefix-io-v1-handoff/project')
D = 'artifacts/prefix_io_v1/server12-c5-normal-repeatability-v1-20261004'
LOCK = D + '/COMMON_SOURCE_LOCK.json'
LEDGER = 'experiments/prefix_io_v1/gpu-budget-ledger.json'


def require(value, reason):
    if not value:
        raise ValueError('FIXED_REPEATABILITY_SUPERVISOR_REJECTED: ' + reason)


def safe(root, relative):
    require(type(relative) is str and relative and ':' not in relative and '\\' not in relative and
        not relative.startswith('/') and all(p not in ('', '.', '..') for p in relative.split('/')),
        'project-relative evidence path')
    path = root
    for part in relative.split('/'):
        path /= part
        require(not path.is_symlink(), 'source/evidence symlink refused')
    require(path.resolve().is_relative_to(root), 'outside project')
    return path


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, 'nonfinite JSON'))


def read(root, relative):
    path = safe(root, relative)
    require(path.is_file() and path.stat().st_size <= 32 * 1024**2, 'bounded JSON evidence')
    return parse(path.read_bytes())


def actual_ref(root, relative):
    path = safe(root, relative)
    require(path.is_file(), 'actual source/evidence file')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=relative, bytes=path.stat().st_size, sha256=digest)


def locked_rows(root):
    document = read(root, LOCK)
    require(document.get('schema') == 'c5_repeatability_common_source_lock_v1' and
        document.get('scope') == 'server12_c5_normal_repeatability_v1' and
        document.get('gpu_authority_issued') is False and type(document.get('files')) is list and
        1 <= len(document['files']) <= 10000, 'actual diagnostic shared source lock')
    result = {}
    for row in document['files']:
        require(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'} and
            type(row['bytes']) is int and row['bytes'] >= 0 and
            type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}', row['sha256']) is not None,
            'exact typed frozen reference')
        safe(root, row['path'])
        require(row['path'] not in result and row['path'] not in (LOCK, LEDGER), 'unique immutable source paths')
        result[row['path']] = row
    return result


def load(root, refs, filename, name):
    relative = D + '/' + filename
    require(relative in refs and actual_ref(root, relative) == refs[relative], 'actual locked supervisor dependency')
    path = safe(root, relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        require(actual_ref(root, relative) == refs[relative], 'dependency source drift during import')
        return module
    except BaseException:
        sys.modules.pop(name, None)
        raise


def validate_protocol(protocol):
    require(type(protocol) is dict and protocol.get('schema') == 'c5_fixed_workload_repeatability_protocol_v1' and
        protocol.get('scope') == 'server12_c5_normal_repeatability_v1' and
        type(protocol.get('repetitions')) is int and protocol['repetitions'] == 3 and
        protocol.get('mode') == 'off', 'exactly three preregistered off repetitions')
    for key, expected in (('seed', 2829), ('prompt_first_token', 28100), ('prompt_tokens', 129),
        ('complete_output_tokens', 128), ('complete_capture_frames', 128), ('measured_offset', 16),
        ('ssd_read_operations', 1), ('ssd_read_physical_bytes', 917504), ('max_accepted_parents', 8),
        ('cost_upper_ns', 16238752), ('step_budget_ns', 13171328)):
        require(type(protocol.get(key)) is int and protocol[key] == expected, 'fixed protocol field: ' + key)
    require(protocol.get('bridge_is_none') is True and protocol.get('thresholds_are_frozen') is True,
        'original bridge-off path and frozen thresholds')
    budget = protocol.get('budget')
    require(type(budget) is dict and all(type(budget.get(k)) is int and budget[k] == v for k, v in
        (('original_cumulative_seconds', 28800), ('execution_seconds_per_attempt', 300),
         ('cleanup_seconds_per_attempt', 20), ('planned_max_reserved_seconds', 960), ('replacement_or_retry_attempts', 0))),
        'original budget and three fixed no-retry attempts')
    jobs = protocol.get('jobs')
    require(type(jobs) is list and len(jobs) == 3, 'three actual planned slots')
    for index, job in enumerate(jobs):
        token = 'off' + str(index + 1).zfill(2)
        require(type(job) is dict and type(job.get('diagnostic_index')) is int and job['diagnostic_index'] == index and
            job.get('label') == 'server12-c5-native-repeat-' + token and job.get('config') == 'CONFIG_' + token + '.json',
            'fixed slot/index/label/config')
        for key, phase in (('source_before', 'BEFORE'), ('source_launch', 'LAUNCH'), ('source_after', 'AFTER')):
            require(job.get(key) == 'SOURCE_' + token + '_' + phase + '.json', 'fixed per-slot source proof')
    rule = protocol.get('decision_rule')
    require(type(rule) is dict and rule.get('cost_exceedance_alone_stops_planned_repetitions') is False and
        rule.get('pass_alone_stops_planned_repetitions') is False and rule.get('all_outcomes_retained') is True and
        rule.get('no_refit') is True and rule.get('no_upper_widening') is True and rule.get('no_budget_widening') is True and
        rule.get('normal_qualification_passed') is False and rule.get('permits_next_mode') is None,
        'no outcome selection, refit, widening or qualification promotion')
    return jobs


def decision(record, index, protocol):
    """Pure disposition of a public verifier result; never a GPU authorization."""
    validate_protocol(protocol)
    require(type(index) is int and 0 <= index < 3 and type(record) is dict and
        type(record.get('diagnostic_index')) is int and record['diagnostic_index'] == index,
        'same typed diagnostic slot')
    require(record.get('native_execution_verified') is True and record.get('diagnostic_evidence_valid') is True and
        record.get('lifecycle_valid') is True and type(record.get('full_output_tokens')) is int and
        record['full_output_tokens'] == 128 and type(record.get('full_capture_frames')) is int and
        record['full_capture_frames'] == 128, 'fully verified actual lifecycle/output evidence required')
    value = record.get('selected_gpu_elapsed_ns')
    require(type(value) is int and value > 0 and type(record.get('frozen_cost_upper_ns')) is int and
        record['frozen_cost_upper_ns'] == 16238752 and type(record.get('frozen_a_only_budget_ns')) is int and
        record['frozen_a_only_budget_ns'] == 13171328, 'actual selected timing and original bounds')
    coverage = value <= 16238752
    require(type(record.get('covered_original_upper')) is bool and record['covered_original_upper'] is coverage and
        record.get('status') == ('VALID_RECORD_COST_COVERED' if coverage else 'VALID_RECORD_COST_EXCEEDED'),
        'actual threshold comparison retained')
    require(record.get('normal_qualification_passed') is False and record.get('runtime_condition_qualified') is False and
        record.get('qualification_passed') is False and record.get('permits_next_mode') is None and
        record.get('P4_strategy_effect_verified') is False and record.get('performance_benefit_proved') is False and
        record.get('calibration_refit') is False and record.get('thresholds_changed') is False,
        'diagnostic cannot promote or revise the failed normal gate')
    return dict(action='CONTINUE_FIXED_PROTOCOL' if index < 2 else 'FIXED_PROTOCOL_COMPLETE',
        next_diagnostic_index=index + 1 if index < 2 else None, observed_coverage=coverage,
        GPU_authority_issued=False, normal_qualification_passed=False, permits_next_mode=None)


def put(root, relative, value):
    with safe(root, relative).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return actual_ref(root, relative)


def guard_idle(root):
    ledger = read(root, LEDGER)
    seconds = ledger.get('gpu_wall_seconds')
    require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= 28800 and
        ledger.get('active_reservation') is None, 'actual original guard must be idle and within budget')
    return ledger


def run(root, *, execute=False):
    root = Path(root).resolve(strict=True)
    require(root == ROOT, 'fixed actual server project')
    refs = locked_rows(root)
    require(actual_ref(root, D + '/run_fixed_repeatability.py') == refs.get(D + '/run_fixed_repeatability.py'),
        'actual frozen supervisor source')
    verifier = load(root, refs, 'verify_repeatability.py', '_fixed_repeatability_public_verifier')
    protocol_ref = verifier.prepare_protocol(root)
    require(protocol_ref == refs.get(D + '/PROTOCOL.json'), 'source-pinned actual preregistration')
    protocol = read(root, protocol_ref['path'])
    jobs = validate_protocol(protocol)
    if not execute:
        return dict(status='CPU_INSPECTION_ONLY_NO_GPU_LAUNCH', protocol_ref=protocol_ref, planned_jobs=jobs,
            planned_max_GPU_seconds_reserved=960, GPU_authority_issued=False, GPU_jobs_started=0,
            normal_qualification_passed=False, permits_next_mode=None)
    require(all(not safe(root, 'experiments/prefix_io_v1/runs/' + job['label']).exists() for job in jobs),
        'fresh fixed three-job protocol only; existing attempts are never retried')
    require(all(not safe(root, D + '/DIAGNOSTIC_RESULT_off' + str(i+1).zfill(2) + '.json').exists() for i in range(3)),
        'new append-only diagnostic reports')
    guard_idle(root)
    controller = load(root, refs, 'control_p4_single_file.py', '_fixed_repeatability_actual_controller')
    require(controller.D == D and controller.LOCK == LOCK and controller.SCOPE == protocol['scope'],
        'actual diagnostic controller source/scope')
    # All authorities and proofs must already exist. This does not write grants,
    # create new GPU permissions, infer consent from flags, or perform a probe.
    for index, job in enumerate(jobs):
        config, actual_refs, authority = controller.verify_metadata_configuration(root, D + '/' + job['config'])
        require(actual_refs == refs and config['diagnostic_index'] == index and config['protocol_ref'] == protocol_ref,
            'fresh same-source per-attempt configuration')
        controller.verify_scope(root, config, authority)
    started = []
    records = []
    for index, job in enumerate(jobs):
        token = 'off' + str(index + 1).zfill(2)
        try:
            guard_idle(root)
            launch_ref = controller.launch(index, root)
            started.append(dict(diagnostic_index=index, actual_launch_receipt=launch_ref))
            deadline = time.monotonic() + 350
            last_message = 0.0
            result_relative = 'experiments/prefix_io_v1/runs/' + job['label'] + '/result.json'
            while not safe(root, result_relative).exists():
                elapsed = time.monotonic()
                require(elapsed <= deadline, 'original guard completion not published within execution/cleanup allowance')
                if elapsed - last_message >= 30:
                    print(json.dumps(dict(status='WAITING_ORIGINAL_GPU_GUARD', diagnostic_index=index,
                        label=job['label'], authority_issued_by_supervisor=False)), flush=True)
                    last_message = elapsed
                time.sleep(1)
            guard_idle(root)
            controller.after(index, root)
            record = verifier.verify_repetition(root, index)
            disposition = decision(record, index, protocol)
            output = put(root, D + '/DIAGNOSTIC_RESULT_' + token + '.json', record)
            records.append(record)
            print(json.dumps(dict(status=record['status'], diagnostic_index=index,
                selected_gpu_elapsed_ns=record['selected_gpu_elapsed_ns'], report_ref=output,
                disposition=disposition)), flush=True)
        except Exception as exc:
            stop = dict(status='HARD_STOP_FIXED_DIAGNOSTIC_EVIDENCE_INVALID_OR_MISSING', diagnostic_index=index,
                reason=str(exc), started_attempts=started, actual_valid_records=records,
                retries_performed=0, remaining_slots_retained_as_missing=True,
                unqualified_slots=[dict(diagnostic_index=slot, label=jobs[slot]['label'],
                    status='INVALID_OR_MISSING_ACTUAL_EVIDENCE' if slot == index else 'NOT_LAUNCHED_AFTER_HARD_STOP',
                    native_execution_verified=False, selected_gpu_elapsed_ns=None)
                    for slot in range(index, 3)],
                normal_qualification_passed=False, permits_next_mode=None,
                supervisor_issued_GPU_authority=False, original_guard_still_owns_cleanup=True)
            stop_ref = put(root, D + '/REPEATABILITY_HARD_STOP_' + token + '.json', stop)
            return dict(stop, stop_ref=stop_ref)
    summary = verifier.summarize(root)
    require(summary.get('actual_valid_records') == 3 and summary.get('normal_qualification_passed') is False and
        summary.get('permits_next_mode') is None, 'all actual preregistered outcomes and unchanged qualification gate')
    output = put(root, D + '/REPEATABILITY_SUMMARY.json', summary)
    return dict(status='FIXED_THREE_DIAGNOSTICS_RECORDED_NOT_NORMAL_OR_P4_QUALIFICATION',
        summary_ref=output, classification=summary['classification'], actual_attempts_started=len(started),
        actual_valid_records=3, retries_performed=0, calibration_refit=False, thresholds_changed=False,
        normal_qualification_passed=False, permits_next_mode=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    result = run(args.project_root, execute=args.execute)
    print(json.dumps(result, sort_keys=True))
    return 2 if result['status'].startswith('HARD_STOP_') else 0


if __name__ == '__main__':
    raise SystemExit(main())

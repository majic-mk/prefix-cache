"""Independent stdlib replay of the preregistered CPU-only comparison gates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

PROTOCOL_SHA256 = '6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218'
PIPELINE = ('_pump_once', '_schedule_work', '_drain_cuda_copies', '_poll_ring_completions',
            '_flush_copy_batch', '_finish_jobs', '_intake', '_drain_incoming',
            '_drain_ready_preload_fds', '_drain_ready_load_fds')


def sha(b):
    return hashlib.sha256(b).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, expected=None):
    b = path.read_bytes()
    if expected:
        need(sha(b) == expected, 'SHA mismatch: ' + str(path))
    return json.loads(b)


def median(values):
    return statistics.median(values)


def evaluate(directory, race_receipt=None):
    plan = read(directory / 'PLAN.json')
    suite = read(directory / 'SUITE.json')
    need(plan['protocol_sha256'] == PROTOCOL_SHA256, 'preregistered protocol identity changed')
    protocol_bytes = (directory / 'PROTOCOL.json').read_bytes()
    need(sha(protocol_bytes) == PROTOCOL_SHA256, 'protocol file changed after run')
    need(json.loads(protocol_bytes) == plan['protocol'], 'plan protocol values differ')
    protocol = plan['protocol']
    need(plan['mode'] in ('smoke', 'score') and plan['scored'] == (plan['mode'] == 'score'), 'scoring mode differs')
    need(plan['GPU_runs'] == 0, 'CPU protocol contains GPU work')
    need(plan['premeasurement_settle_ns'] == 200000000, 'fixed setup settling changed')
    need(plan['synthetic_backend_completion_delay_ns'] == 1000000, 'physical fixture changed')
    scenarios = [protocol['main']] + protocol['sensitivity'] + protocol['functional_scenarios']
    expected = []
    for scenario in scenarios:
        if plan['mode'] == 'smoke':
            schedule = [('smoke', 0, a) for a in ('A', 'B')]
        else:
            schedule = [('warmup', i, a) for i, a in enumerate(protocol['warmup_order_per_scenario'])]
            schedule += [('score', i, a) for i in range(scenario['pairs']) for a in protocol['pair_order'][i]]
        expected += [(scenario['name'], phase, i, a) for phase, i, a in schedule]
    records = suite['records']
    actual = [(r['scenario'], r['phase'], r['pair_or_warmup'], r['arm']) for r in records]
    need(actual == expected[:len(actual)], 'fixed order changed, skipped or duplicated')
    if suite['status'] == 'COMPLETE':
        need(actual == expected, 'complete suite is missing trials')
    else:
        need(suite['status'] == 'STOPPED_ON_FAILURE', 'unknown suite outcome')
    raw = []
    identities = {}
    functional = []
    hashes = []
    for r in records:
        path = directory / r['result_file']
        if not path.exists():
            functional.append({'label': r['label'], 'failure': 'trial result absent', 'exit': r['exit']})
            continue
        d = read(path, r['result_sha256'])
        hashes.append({'file': path.name, 'sha256': r['result_sha256']})
        need((d['arm'], d['scenario']) == (r['arm'], r['scenario']), 'trial identity mismatch')
        ident = d['source_identity']
        arm = d['arm']
        need(ident['sha256'] == plan['candidate_manifest'][arm]['reactor'], 'runtime reactor differs')
        need(ident['actual_collector']['sha256'] == plan['candidate_manifest'][arm]['collector'], 'runtime collector differs')
        need(set(ident['actual_original_method_code_files'].values()) == {ident['path']}, 'native method code origin differs')
        need(ident['actual_collector']['accessor_code_filename'] == ident['actual_collector']['path'], 'collector accessor code origin differs')
        state_identity = {'reactor': ident['sha256'], 'collector': ident['actual_collector']['sha256'],
                          'modules': ident['loaded_modules'], 'helpers': ident['fixture_helpers'],
                          'AST': ident['method_AST_SHA256'], 'fixtures': ident['physical_fixtures'],
                          'wrappers': ident['transparent_count_wrappers']}
        if arm in identities:
            need(state_identity == identities[arm], 'runtime identity changed between trials')
        else:
            identities[arm] = state_identity
        need(d['GPU_operations'] == 0 and not d['cuda_initialized'] and not d['forbidden_imports'], 'non-CPU execution')
        for key in ('worker_cpu_ns', 'producer_cpu_ns', 'worker_plus_producer_cpu_ns'):
            need(type(d[key]) is int and d[key] >= 0, 'thread CPU must be a nonnegative integer: ' + key)
        need(d['worker_cpu_ns'] == d['worker']['cpu_ns'] and d['producer_cpu_ns'] == d['producer']['cpu_ns'], 'thread CPU raw fields disagree')
        need(d['worker_plus_producer_cpu_ns'] == d['worker_cpu_ns'] + d['producer_cpu_ns'], 'worker plus producer CPU sum differs')
        need(d['producer']['prepare_callback_included_in_producer_cpu'] is True and
             type(d['producer']['prepare_callback_cpu_ns']) is int and
             0 <= d['producer']['prepare_callback_cpu_ns'] <= d['producer_cpu_ns'], 'per-step prepare CPU not included')
        primary = '_MandatoryWait' if d['scenario'] == 'mandatory_20ms' else 'STOP' if d['scenario'] == 'stop_20ms' else None
        enqueue = next((e['at_ns'] for e in d['queue']['put'] if e['kind'] == primary), None)
        signed = None if enqueue is None or d['accepted_ns'] is None else d['accepted_ns'] - enqueue
        already = signed is not None and signed < 0
        need(d['enqueue_to_progress_signed_ns'] == signed and d['already_in_progress_at_control_enqueue'] == already,
             'raw control progress timing differs')
        need(d['enqueue_to_progress_eligible'] == (signed is not None and not already) and
             d['enqueue_to_progress_ns'] == (None if already else signed), 'negative control latency cannot qualify')
        errors = list(d['failures'])
        if not d['ready_arrival_unchanged']:
            errors.append('original ready age changed')
        if not all(d['counts'].get(name, 0) > 0 for name in PIPELINE):
            errors.append('full native pipeline stage not executed')
        if d['bridge']['defer'] < 1 or d['bridge']['preview'] < 1 or d['counts'].get('live_borrow', 0) < 1:
            errors.append('real preview/live path not exercised')
        if d['ring']['simulated_completed'] != 1 or d['counts'].get('physical_fixture_submit') != 1:
            errors.append('simulation completion not exactly once')
        if d['pool']['free'] != 16 or d['pool']['reserves'] != d['pool']['releases']:
            errors.append('synthetic resources not balanced')
        if d['payload_sha256'] != sha(b'CPU_SIMULATED_PREFIX_PAYLOAD_V1'):
            errors.append('synthetic payload mismatch')
        final = d['native_final_state']
        if any(final[k] != 0 for k in ('active', 'inflight', 'pending_copies', 'copy_ready', 'ready_load', 'ready_preload', 'queue_size', 'accepted_parents')):
            errors.append('final native state not drained')
        if not final['original_STOP_observed'] or final['worker_alive'] or final['producer_alive']:
            errors.append('original STOP/threads not complete')
        if d['scenario'] == 'mandatory_20ms' and (final['mandatory_future_done'] is not True or final['mandatory_future_result'] != 917504 or final['parent_retired_count'] != 1):
            errors.append('mandatory parent not completed/retired once')
        if d['scenario'] == 'original_deadline_100ms' and (d['deadline_lateness_ns'] is None or d['deadline_lateness_ns'] < 0):
            errors.append('deadline progressed early or not measured')
        if r['exit'] != 0 or d['status'] != 'PASS':
            errors.append('trial did not pass')
        if errors:
            functional.append({'label': r['label'], 'errors': errors})
        raw.append((r, d))
    if set(identities) == {'A', 'B'}:
        for method, value in identities['A']['AST'].items():
            if method not in ('_run', '_intake', '_prefix_single_file_retry_key'):
                need(identities['B']['AST'][method] == value, 'original native method unexpectedly changed: ' + method)
        common = set(identities['A']['modules']) & set(identities['B']['modules'])
        for name in common:
            if name.startswith('prefix_io_control'):
                need(identities['A']['modules'][name]['sha256'] == identities['B']['modules'][name]['sha256'],
                     'policy/control module changed across arms: ' + name)
    functional_pass = suite['status'] == 'COMPLETE' and not functional
    summaries = {}
    gates = []
    resource_limits = []
    measurement_ineligible = [{'label': r['label'], 'reason': 'original progress preceded control enqueue',
                              'signed_delta_ns': d['enqueue_to_progress_signed_ns']}
                             for r, d in raw if r['phase'] == 'score' and d['already_in_progress_at_control_enqueue']]
    for scenario in scenarios:
        scored = ([(r, d) for r, d in raw if r['phase'] == 'score' and r['scenario'] == scenario['name']]
                  if suite['status'] == 'COMPLETE' else [])
        if not scored:
            continue
        pairs = []
        for i in range(scenario['pairs']):
            pair = {r['arm']: d for r, d in scored if r['pair_or_warmup'] == i}
            need(set(pair) == {'A', 'B'}, 'scored pair incomplete')
            pair_result = {'index': i, 'order': protocol['pair_order'][i], 'arms': {}, 'delta_B_minus_A': {}}
            for arm, d in pair.items():
                pair_result['arms'][arm] = {key: d[key] for key in ('worker_cpu_ns', 'producer_cpu_ns', 'worker_plus_producer_cpu_ns',
                    'wall_ns', 'enqueue_to_progress_ns', 'deadline_lateness_ns', 'nr_throttled_delta')}
                pair_result['arms'][arm]['cpu_quota_limited'] = d['cgroup_before']['limited']
                pair_result['arms'][arm]['pump_calls'] = d['counts']['_pump_once']
                pair_result['arms'][arm]['defer_calls'] = d['bridge']['defer']
                pair_result['arms'][arm]['live_calls'] = d['counts']['live_borrow']
            for key in ('worker_cpu_ns', 'producer_cpu_ns', 'worker_plus_producer_cpu_ns', 'wall_ns', 'enqueue_to_progress_ns'):
                pair_result['delta_B_minus_A'][key] = None if any(pair[a][key] is None for a in ('A', 'B')) else pair['B'][key] - pair['A'][key]
            def unthrottled(d):
                limited = d['cgroup_before']['limited']
                return limited is False or (limited is True and d['nr_throttled_delta'] == 0)
            pair_result['both_unthrottled_or_unlimited'] = all(unthrottled(pair[a]) for a in ('A', 'B'))
            pairs.append(pair_result)
        def deltas(key):
            return [r['delta_B_minus_A'][key] for r in pairs]
        summary = {'pairs': pairs, 'paired_medians_B_minus_A': {},
                   'unthrottled_pairs': sum(r['both_unthrottled_or_unlimited'] for r in pairs)}
        for key in ('worker_cpu_ns', 'producer_cpu_ns', 'worker_plus_producer_cpu_ns', 'wall_ns', 'enqueue_to_progress_ns'):
            values = deltas(key)
            summary['paired_medians_B_minus_A'][key] = None if any(v is None for v in values) else median(values)
        cpu = deltas('worker_plus_producer_cpu_ns')
        if scenario['name'] == protocol['main']['name']:
            tests = [('main_worker_CPU', median(deltas('worker_cpu_ns')) < 0),
                     ('main_total_CPU', median(cpu) < 0),
                     ('main_9_of_12', sum(v < 0 for v in cpu) >= protocol['cpu_gate']['main_worker_plus_producer_negative_pairs_minimum']),
                     ('main_first_half', median(cpu[:6]) < 0), ('main_second_half', median(cpu[6:]) < 0),
                     ('main_wall', median(deltas('wall_ns')) <= protocol['latency_gate']['main_wall_paired_median_increment_limit_ns'])]
            required_unthrottled = protocol['latency_gate']['main_unthrottled_pairs_minimum_if_CPU_quota']
        else:
            tests = []
            required_unthrottled = protocol['latency_gate']['other_unthrottled_pairs_minimum_if_CPU_quota']
            if scenario['name'] in [s['name'] for s in protocol['sensitivity']]:
                tests.append((scenario['name'] + '_total_CPU_nonregression', median(cpu) <= 0))
            elif scenario['name'] in ('mandatory_20ms', 'stop_20ms'):
                name = 'mandatory' if scenario['name'] == 'mandatory_20ms' else 'STOP'
                values = deltas('enqueue_to_progress_ns')
                tests.append((scenario['name'] + '_progress_latency', all(v is not None for v in values) and median(values) <= protocol['latency_gate'][name + '_enqueue_to_progress_paired_median_increment_limit_ns']))
            else:
                values = [r['arms']['B']['deadline_lateness_ns'] for r in pairs]
                tests.append(('original_deadline_lateness', all(v is not None and v >= 0 for v in values) and median(values) <= protocol['latency_gate']['deadline_median_lateness_limit_ns']))
        for name, ok in tests:
            gates.append({'name': name, 'passed': bool(ok)})
        if summary['unthrottled_pairs'] < required_unthrottled:
            resource_limits.append({'scenario': scenario['name'], 'unthrottled_pairs': summary['unthrottled_pairs'], 'required': required_unthrottled})
        summaries[scenario['name']] = summary
    race = {'status': 'NOT_PROVIDED', 'passed': False}
    if race_receipt:
        b = race_receipt.read_bytes()
        record = json.loads(b)
        expected = plan['candidate_manifest']['B']['reactor']
        race = {'path': str(race_receipt), 'sha256': sha(b), 'status': record.get('status'),
                'passed': record.get('status') == 'PASS' and record.get('candidate_reactor_sha256') == expected
                          and record.get('candidate_collector_sha256') == plan['candidate_manifest']['B']['collector']
                          and record.get('gpu_workloads_run') == 0 and record.get('failed') == 0
                          and type(record.get('passed')) is int and record['passed'] > 0,
                'scope': 'Bound upstream independent race receipt; this analyzer does not execute that suite'}
    scored_complete = plan['mode'] == 'score' and functional_pass and len(summaries) == 6
    cpu_and_latency_pass = scored_complete and all(g['passed'] for g in gates) and not resource_limits
    return {'schema_version': 1, 'scope': 'CPU_SYNTHETIC_ENGINEERING_ONLY_NO_GPU_OR_PAPER_GAIN',
            'protocol_sha256': PROTOCOL_SHA256, 'plan_sha256': sha((directory / 'PLAN.json').read_bytes()),
            'suite_sha256': sha((directory / 'SUITE.json').read_bytes()), 'mode': plan['mode'],
            'runtime_identity_verified': True, 'full_original_control_flow_verified': True,
            'trial_count': len(raw), 'raw_sha_bindings': hashes,
            'functional_pass': functional_pass, 'functional_failures': functional,
            'scored_complete': scored_complete, 'gates': gates, 'resource_limited': resource_limits,
            'latency_measurement_ineligible': measurement_ineligible,
            'CPU_and_latency_gates_passed': cpu_and_latency_pass, 'independent_race': race,
            'CPU_candidate_qualified': cpu_and_latency_pass and race['passed'], 'scenarios': summaries,
            'GPU_runs_allowed': 0, 'GPU_effect_verified': False, 'D_J_release_credit': False,
            'P4_complete': False, 'paper_readiness': False,
            'decision': ('SMOKE_ONLY_NOT_SCORED' if plan['mode'] == 'smoke' else
                         'CPU_LATENCY_MEASUREMENT_INELIGIBLE_NO_UPGRADE' if functional_pass and measurement_ineligible else
                         'RETAIN_OFF_STOP_OR_RESOURCE_LIMITED' if not cpu_and_latency_pass else
                         'CPU_MECHANISM_PASSES_RACE_PENDING' if not race['passed'] else 'CPU_MECHANISM_QUALIFIED_ONLY'),
            'limitations': ['Exact original control methods execute with explicitly synthetic physical ring/queue/pool events.',
                            'Per-call counting costs are included symmetrically; these are instrumented CPU costs, not native GPU-serving latency.',
                            'Fresh process setup is excluded and followed by fixed 200ms settling; all cgroup throttle samples remain in results.',
                            'No statistical confidence interval, service SLO, real bottleneck controllability or GPU acceleration claim.']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--race-receipt', type=Path)
    args = p.parse_args()
    result = evaluate(args.directory.resolve(), args.race_receipt)
    result['command'] = sys.argv
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({k: result[k] for k in ('decision', 'trial_count', 'functional_pass', 'CPU_and_latency_gates_passed', 'CPU_candidate_qualified')}))
    return 0 if result['functional_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Read-only resource reanalysis of the one frozen server CPU comparison.

This script only reads JSON. It never loads or reruns the reactor/collector,
benchmark, model, GPU runtime or a subprocess. New output files are exclusive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

PROTOCOL_SHA = '6c00cf687f4903988a363e18a178ccca413076ced81920271bf9c962abde1218'
ANALYSIS_SHA = 'be79e7b1d9c539804d4b8a141a8e69f3219a6a6fd97cbb41efc6681a69512fe6'
CPU_KEYS = ('worker_cpu_ns', 'producer_cpu_ns', 'worker_plus_producer_cpu_ns')
MEASURES = CPU_KEYS + ('wall_ns', 'enqueue_to_progress_ns', 'deadline_lateness_ns',
    'start_to_submit_ns', 'start_to_complete_ns', 'producer_lateness_ns',
    'prepare_callback_cpu_ns', 'pump_calls', 'preview_calls', 'live_calls')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, expected=None):
    raw = path.read_bytes()
    if expected is not None:
        require(digest(raw) == expected, 'SHA mismatch: ' + str(path))
    return json.loads(raw), {'file': path.name, 'sha256': digest(raw), 'bytes': len(raw)}


def distribution(values):
    values = [x for x in values if x is not None]
    return None if not values else {'n': len(values), 'min': min(values),
        'median': statistics.median(values), 'max': max(values)}


def derived(record, data):
    for key in CPU_KEYS:
        require(type(data[key]) is int and data[key] >= 0, 'invalid thread CPU: ' + record['label'])
    require(data['worker_cpu_ns'] == data['worker']['cpu_ns'] and
        data['producer_cpu_ns'] == data['producer']['cpu_ns'] and
        data['worker_plus_producer_cpu_ns'] == data['worker_cpu_ns'] + data['producer_cpu_ns'],
        'CPU sum/raw mismatch: ' + record['label'])
    require(data['producer']['prepare_callback_included_in_producer_cpu'] is True,
        'per-step prepare CPU omitted')
    require(data['status'] == 'PASS' and not data['failures'] and record['exit'] == 0,
        'source trial was not a passing trial')
    require(data['GPU_operations'] == 0 and not data['cuda_initialized'] and
        not data['forbidden_imports'], 'source execution was not CPU only')
    row = {'label': record['label'], 'phase': record['phase'], 'pair_or_warmup': record['pair_or_warmup'],
        'scenario': data['scenario'], 'arm': data['arm'],
        **{key: data[key] for key in CPU_KEYS + ('wall_ns', 'enqueue_to_progress_ns', 'deadline_lateness_ns')},
        'start_to_submit_ns': data['accepted_ns'] - data['ready_arrived_ns'],
        'start_to_complete_ns': data['completed_ns'] - data['ready_arrived_ns'],
        'producer_lateness_ns': None if data['producer']['event_actual_ns'] is None else
            data['producer']['event_actual_ns'] - data['producer']['event_planned_ns'],
        'prepare_callback_cpu_ns': data['producer']['prepare_callback_cpu_ns'],
        'pump_calls': data['counts']['_pump_once'], 'preview_calls': data['counts']['bridge_preview'],
        'live_calls': data['counts']['live_borrow'],
        'already_in_progress_at_control_enqueue': data['already_in_progress_at_control_enqueue']}
    if row['scenario'] == 'original_deadline_100ms':
        require(row['start_to_submit_ns'] - 100000000 == row['deadline_lateness_ns'] >= 0,
            'original 100ms deadline binding differs')
    before, after = data['cgroup_before'], data['cgroup_after']
    require(before['quota'] == after['quota'] and before['limited'] == after['limited'], 'quota changed during trial')
    s0, s1 = before.get('stat'), after.get('stat')
    keys = ('nr_throttled', 'throttled_usec', 'usage_usec', 'nr_periods')
    delta = {key: None if s0 is None or s1 is None or key not in s0 or key not in s1 else s1[key] - s0[key] for key in keys}
    require(all(value is None or type(value) is int and value >= 0 for value in delta.values()), 'invalid cgroup counters')
    require(delta['nr_throttled'] == data['nr_throttled_delta'], 'recorded throttle delta differs')
    row.update(quota=before['quota'], quota_limited=before['limited'], cgroup_counter_deltas=delta,
        throttle_count_known=delta['nr_throttled'] is not None,
        throttled=None if delta['nr_throttled'] is None else delta['nr_throttled'] > 0,
        unthrottled_or_unlimited=before['limited'] is False or before['limited'] is True and delta['nr_throttled'] == 0)
    return row


def audit(benchmark_root):
    score = benchmark_root / 'server-score01'
    analysis, analysis_binding = read(benchmark_root / 'SERVER_SCORE_ANALYSIS.json', ANALYSIS_SHA)
    plan, plan_binding = read(score / 'PLAN.json', analysis['plan_sha256'])
    suite, suite_binding = read(score / 'SUITE.json', analysis['suite_sha256'])
    protocol, protocol_binding = read(score / 'PROTOCOL.json', PROTOCOL_SHA)
    require(plan['protocol'] == protocol and plan['protocol_sha256'] == PROTOCOL_SHA, 'frozen protocol differs')
    require(plan['mode'] == suite['mode'] == 'score' and suite['status'] == 'COMPLETE', 'formal run incomplete')
    require(analysis['functional_pass'] and analysis['scored_complete'], 'source qualification incomplete')
    scenarios = [protocol['main']] + protocol['sensitivity'] + protocol['functional_scenarios']
    expected_order = []
    for scenario in scenarios:
        expected_order += [(scenario['name'], 'warmup', i, arm) for i, arm in enumerate(protocol['warmup_order_per_scenario'])]
        expected_order += [(scenario['name'], 'score', i, arm) for i in range(scenario['pairs']) for arm in protocol['pair_order'][i]]
    records = suite['records']
    require([(r['scenario'], r['phase'], r['pair_or_warmup'], r['arm']) for r in records] == expected_order,
        'frozen ordered run grid differs')
    bindings = {r['file']: r['sha256'] for r in analysis['raw_sha_bindings']}
    require(len(bindings) == len(records) == len(analysis['raw_sha_bindings']) == 132, '132 distinct raw results required')
    require(set(bindings) == {r['result_file'] for r in records}, 'analysis/suite raw file sets differ')
    rows, raw_bindings = [], []
    for record in records:
        filename = record['result_file']
        require(Path(filename).name == filename, 'raw filename must remain inside score directory')
        require(record['result_sha256'] == bindings[filename], 'analysis/suite raw SHA differs')
        data, binding = read(score / filename, bindings[filename])
        require((data['scenario'], data['arm']) == (record['scenario'], record['arm']), 'raw identity differs')
        manifest = plan['candidate_manifest'][data['arm']]
        require(data['source_identity']['sha256'] == manifest['reactor'] and
            data['source_identity']['actual_collector']['sha256'] == manifest['collector'], 'runtime source binding differs')
        rows.append(derived(record, data))
        raw_bindings.append(binding)
    summaries, resources = {}, []
    for scenario in scenarios:
        name = scenario['name']
        scored = [row for row in rows if row['phase'] == 'score' and row['scenario'] == name]
        pairs = []
        for i in range(scenario['pairs']):
            pair = {row['arm']: row for row in scored if row['pair_or_warmup'] == i}
            require(set(pair) == {'A', 'B'}, 'incomplete paired sample')
            pairs.append({'index': i, 'order': protocol['pair_order'][i],
                'both_unthrottled_or_unlimited': all(row['unthrottled_or_unlimited'] for row in pair.values()),
                'delta_B_minus_A': {key: None if any(pair[arm][key] is None for arm in 'AB') else pair['B'][key] - pair['A'][key]
                    for key in CPU_KEYS + ('wall_ns', 'enqueue_to_progress_ns')},
                'arms': pair})
        arms = {}
        for arm in 'AB':
            selected = [row for row in scored if row['arm'] == arm]
            arms[arm] = {'n': len(selected), 'throttled_trials': sum(row['throttled'] is True for row in selected),
                'unknown_throttle_trials': sum(not row['throttle_count_known'] for row in selected),
                'distributions': {key: distribution([row[key] for row in selected]) for key in MEASURES},
                'cgroup_counter_distributions': {key: distribution([row['cgroup_counter_deltas'][key] for row in selected])
                    for key in ('nr_throttled', 'throttled_usec', 'usage_usec', 'nr_periods')}}
        medians = {}
        for key in CPU_KEYS + ('wall_ns', 'enqueue_to_progress_ns'):
            values = [p['delta_B_minus_A'][key] for p in pairs]
            medians[key] = None if any(value is None for value in values) else statistics.median(values)
        require(medians == analysis['scenarios'][name]['paired_medians_B_minus_A'], 'paired medians differ from frozen analysis')
        unthrottled = sum(p['both_unthrottled_or_unlimited'] for p in pairs)
        require(unthrottled == analysis['scenarios'][name]['unthrottled_pairs'], 'paired throttle classification differs')
        minimum = protocol['latency_gate']['main_unthrottled_pairs_minimum_if_CPU_quota' if name == protocol['main']['name'] else 'other_unthrottled_pairs_minimum_if_CPU_quota']
        if unthrottled < minimum:
            resources.append({'scenario': name, 'unthrottled_pairs': unthrottled, 'required': minimum})
        deltas = [p['delta_B_minus_A']['worker_plus_producer_cpu_ns'] for p in pairs]
        summaries[name] = {'pairs': scenario['pairs'], 'arms': arms, 'paired_medians_B_minus_A': medians,
            'total_CPU_negative_pairs': sum(value < 0 for value in deltas), 'total_CPU_delta_distribution_ns': distribution(deltas),
            'first_half_CPU_delta_median_ns': statistics.median(deltas[:len(deltas)//2]),
            'last_half_CPU_delta_median_ns': statistics.median(deltas[len(deltas)//2:]),
            'both_unthrottled_pairs': unthrottled, 'required_unthrottled_pairs': minimum, 'pair_details': pairs}
    require(resources == analysis['resource_limited'], 'resource-limited gate differs from original analysis')
    require(not analysis['CPU_candidate_qualified'] and not analysis['CPU_and_latency_gates_passed'], 'unexpected previous upgrade')
    scored = [row for row in rows if row['phase'] == 'score']
    require(len(scored) == 84 and len(scored)//2 == 42, 'scored sample count differs')
    return {'schema_version': 1, 'scope': 'READ_ONLY_EXISTING_CPU_SYNTHETIC_RESOURCE_AUDIT',
        'audit_runs_benchmark': False, 'audit_GPU_operations': 0, 'source_GPU_operations': 0,
        'input_bindings': {'analysis': analysis_binding, 'plan': plan_binding, 'suite': suite_binding, 'protocol': protocol_binding},
        'candidate_manifest': plan['candidate_manifest'], 'verified_raw_count': len(rows), 'verified_raw_bindings': raw_bindings,
        'warmup_count': len(rows)-len(scored), 'scored_trials': len(scored), 'scored_pairs': len(scored)//2,
        'scored_arm_totals': {arm: {'trials': sum(row['arm'] == arm for row in scored),
            'throttled_trials': sum(row['arm'] == arm and row['throttled'] is True for row in scored),
            'unknown_throttle_trials': sum(row['arm'] == arm and not row['throttle_count_known'] for row in scored)} for arm in 'AB'},
        'scenarios': summaries, 'warmup_details': [row for row in rows if row['phase'] == 'warmup'],
        'preserved_qualification': {key: analysis[key] for key in ('decision', 'functional_pass', 'scored_complete',
            'CPU_and_latency_gates_passed', 'CPU_candidate_qualified', 'resource_limited', 'latency_measurement_ineligible',
            'GPU_runs_allowed', 'GPU_effect_verified', 'D_J_release_credit', 'P4_complete', 'paper_readiness')},
        'definitions': {'wall_ns': 'Shared ready/start t0 until both worker and producer join; includes external step holding time.',
            'deadline_lateness_ns': 'First original synthetic read submission minus unchanged ready arrival plus 100 ms deadline.',
            'deadline_wall': 'Producer deliberately holds step end until t0+120 ms. Around 120 ms wall is not 20 ms late dispatch.',
            'throttle': 'cgroup nr_throttled counter increment during trial, not a per-thread counter or direct request-delay attribution.',
            'CPU': 'Instrumented native control flow with synthetic physical backend; worker plus producer includes preparation/notification.',
            'paired': 'All fixed-order pairs retained, including throttled samples; no phase selection or threshold changes.'},
        'limits': ['No new performance samples or GPU work were produced.',
            'Cgroup counters alone do not establish exclusive attribution or an exact removable latency.',
            'CPU control savings do not establish GPU execution speed, service latency/throughput or paper-level efficacy.',
            'No claim of true CUDA/DMA lifecycle, D/J release credit or controllable native I/O queueing is added.'],
        'next_stage': 'Preserve off and this closed score. Separate CPU launcher preparation may proceed; no rerun, automatic GPU run or qualification upgrade.'}


def report(data):
    def ms(value):
        return 'unknown' if value is None else f'{value / 1e6:.3f}'
    lines = ['# Existing CPU comparison: resource audit', '',
        'This is a read-only replay of 132 SHA-bound historical results (48 warmups, 84 scored trials, 42 pairs). It creates no new benchmark sample. The frozen qualification remains unchanged.', '',
        '| Scenario | Pairs | A throttled | B throttled | Both unthrottled | Required | A/B total CPU median (ms) | Paired CPU B-A (ms) |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for name, s in data['scenarios'].items():
        a, b = s['arms']['A'], s['arms']['B']
        lines.append(f"| {name} | {s['pairs']} | {a['throttled_trials']}/{a['n']} | {b['throttled_trials']}/{b['n']} | {s['both_unthrottled_pairs']} | {s['required_unthrottled_pairs']} | {ms(a['distributions']['worker_plus_producer_cpu_ns']['median'])} / {ms(b['distributions']['worker_plus_producer_cpu_ns']['median'])} | {ms(s['paired_medians_B_minus_A']['worker_plus_producer_cpu_ns'])} |")
    lines += ['', '## Longer scenarios: absolute distributions', '',
        '| Scenario | Arm | Wall min/median/max (ms) | Submit from ready min/median/max (ms) |',
        '|---|---|---|---|']
    for name in ('step_end_80ms', 'original_deadline_100ms'):
        for arm in 'AB':
            d = data['scenarios'][name]['arms'][arm]['distributions']
            values = [' / '.join(ms(d[key][stat]) for stat in ('min', 'median', 'max')) for key in ('wall_ns', 'start_to_submit_ns')]
            lines.append(f'| {name} | {arm} | {values[0]} | {values[1]} |')
    main = data['scenarios']['step_end_40ms']
    lines += ['', '## Interpretation and limits', '',
        f"The main 40 ms scenario has CPU savings in all {main['total_CPU_negative_pairs']}/12 pairs; paired worker-plus-producer median delta is {ms(main['paired_medians_B_minus_A']['worker_plus_producer_cpu_ns'])} ms. Producer CPU increases by {ms(main['paired_medians_B_minus_A']['producer_cpu_ns'])} ms and is included. Wall paired median changes by +{ms(main['paired_medians_B_minus_A']['wall_ns'])} ms. This supports reduced synthetic control-path polling CPU, not inference acceleration.", '',
        'B has no new throttling in its 42 scored trials. The two resource-limited scenarios are limited by A samples under the frozen requirement that both arms in a pair be unthrottled. This is insufficient upgrade evidence under the preregistration, not evidence that B lost native progress. The original gate is not relaxed and throttled trials are not discarded.', '',
        'For the deadline scenario, subtract 100 ms from the submission column to obtain deadline lateness. The approximately 120 ms wall includes a producer deliberately holding the external step until 120 ms; it is not the deadline response latency. No scored submission precedes the original deadline.', '',
        'Cgroup throttle duration is an aggregate counter and cannot be assigned one-for-one to a thread, request or removable waiting time. No additional causal claim is inferred from it.', '',
        'The physical backend, event completion, payload and pool remain CPU fixtures. Transparent counter overhead remains measured. No true CUDA/DMA lifecycle, D/J release, real queue controllability or end-to-end model gain is established.', '',
        'Keep the original off default and failed upgrade qualification. Do not repeat this closed score, adjust start phase/settling to obtain favorable quota observations, or automatically launch GPU. Separate bounded CPU launcher preparation belongs to another task.', '',
        '## Evidence', '',
        f"Original analysis SHA-256: `{data['input_bindings']['analysis']['sha256']}`.",
        f"Frozen protocol SHA-256: `{data['input_bindings']['protocol']['sha256']}`.",
        'The accompanying JSON stores every raw SHA, all scored pair details, all warmup records, input hashes, execution command and script hash. No source input is written.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-root', type=Path, required=True,
        help='Existing directory containing SERVER_SCORE_ANALYSIS.json and server-score01/')
    parser.add_argument('--output-dir', type=Path, required=True, help='New audit output directory; never overwritten')
    args = parser.parse_args()
    result = audit(args.benchmark_root.resolve())
    result['audit_command'] = sys.argv
    result['audit_python'] = sys.version
    result['audit_script_sha256'] = digest(Path(__file__).read_bytes())
    result['benchmark_root'] = str(args.benchmark_root.resolve())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'RESOURCE_AUDIT.json').open('x', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write('\n')
    with (out / 'RESOURCE_AUDIT.md').open('x', encoding='utf-8') as file:
        file.write(report(result))
    print(json.dumps({'status': 'READ_ONLY_AUDIT_PASS', 'verified_raw_count': result['verified_raw_count'],
        'scored_pairs': result['scored_pairs'], 'arm_totals': result['scored_arm_totals'],
        'preserved_CPU_candidate_qualified': result['preserved_qualification']['CPU_candidate_qualified'],
        'GPU_runs_allowed': 0, 'output': str(out)}))


if __name__ == '__main__':
    main()

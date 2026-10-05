"""Validator counterexamples. All invented timings remain in memory only.

Real smoke evidence supplies source/structure; synthetic scoring values never
become experiment artifacts or performance claims.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cpu_comparison_analyzer_tested', HERE / 'analyze_cpu_comparison.py')
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)
SMOKE = Path(os.environ.get('CPU_COMPARE_SMOKE_DIR', str(HERE / 'smoke_dev/suite05')))


def packed(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True).encode()


class MemoryPath:
    def __init__(self, files, name=''):
        self.files, self.key = files, name
    def __truediv__(self, name):
        return MemoryPath(self.files, name)
    @property
    def name(self):
        return self.key
    def read_bytes(self):
        return self.files[self.key]
    def exists(self):
        return self.key in self.files
    def __str__(self):
        return 'MEMORY_TEST_ONLY/' + self.key


def synthetic_complete():
    plan = json.loads((SMOKE / 'PLAN.json').read_bytes())
    plan.update(mode='score', scored=True)
    protocol = plan['protocol']
    seeds = {(r['scenario'], r['arm']): json.loads((SMOKE / r['result_file']).read_bytes())
             for r in json.loads((SMOKE / 'SUITE.json').read_bytes())['records']}
    files = {'PLAN.json': packed(plan), 'PROTOCOL.json': (SMOKE / 'PROTOCOL.json').read_bytes()}
    records = []
    scenarios = [protocol['main']] + protocol['sensitivity'] + protocol['functional_scenarios']
    for s in scenarios:
        schedule = [('warmup', i, a) for i, a in enumerate(protocol['warmup_order_per_scenario'])]
        schedule += [('score', i, a) for i in range(s['pairs']) for a in protocol['pair_order'][i]]
        for phase, i, arm in schedule:
            label = '%s-%s-%02d-%s' % (s['name'], phase, i, arm)
            d = copy.deepcopy(seeds[s['name'], arm])
            d['worker_cpu_ns'] = 1000 if arm == 'A' else 100
            d['producer_cpu_ns'] = 100
            d['worker_plus_producer_cpu_ns'] = d['worker_cpu_ns'] + 100
            d['worker']['cpu_ns'] = d['worker_cpu_ns']
            d['producer']['cpu_ns'] = 100
            d['producer']['prepare_callback_cpu_ns'] = 10
            d['wall_ns'] = 200000000
            if s['name'] in ('mandatory_20ms', 'stop_20ms'):
                kind = '_MandatoryWait' if s['name'] == 'mandatory_20ms' else 'STOP'
                enq = next(e['at_ns'] for e in d['queue']['put'] if e['kind'] == kind)
                d['accepted_ns'] = enq + 1000000
                d.update(enqueue_to_progress_ns=1000000, enqueue_to_progress_signed_ns=1000000,
                         already_in_progress_at_control_enqueue=False, enqueue_to_progress_eligible=True)
            if s['name'] == 'original_deadline_100ms':
                d['deadline_lateness_ns'] = 1000000
            d['cgroup_before']['limited'] = False
            d['nr_throttled_delta'] = 0
            b = packed(d)
            files[label + '.json'] = b
            records.append({'label': label, 'phase': phase, 'pair_or_warmup': i, 'arm': arm,
                            'scenario': s['name'], 'exit': 0, 'result_file': label + '.json',
                            'result_sha256': hashlib.sha256(b).hexdigest()})
    files['SUITE.json'] = packed({'status': 'COMPLETE', 'mode': 'score', 'records': records})
    return files


def mutate(files, select, change):
    suite = json.loads(files['SUITE.json'])
    for r in suite['records']:
        if select(r):
            d = json.loads(files[r['result_file']])
            change(d)
            b = packed(d)
            files[r['result_file']] = b
            r['result_sha256'] = hashlib.sha256(b).hexdigest()
    files['SUITE.json'] = packed(suite)


class AnalyzerCounterexamples(unittest.TestCase):
    def test_actual_smoke_is_never_scored(self):
        result = analyzer.evaluate(SMOKE)
        self.assertTrue(result['functional_pass'])
        self.assertFalse(result['scored_complete'])
        self.assertEqual(result['decision'], 'SMOKE_ONLY_NOT_SCORED')

    def test_in_memory_pass_never_grants_GPU_or_skips_race(self):
        result = analyzer.evaluate(MemoryPath(synthetic_complete()))
        self.assertTrue(result['CPU_and_latency_gates_passed'])
        self.assertFalse(result['CPU_candidate_qualified'])
        self.assertEqual(result['GPU_runs_allowed'], 0)

    def test_corrupt_historical_raw_hash_rejected(self):
        files = synthetic_complete()
        record = json.loads(files['SUITE.json'])['records'][0]
        files[record['result_file']] += b' '
        with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
            analyzer.evaluate(MemoryPath(files))

    def test_wrong_actual_collector_rejected_even_rehashed(self):
        files = synthetic_complete()
        mutate(files, lambda r: r['arm'] == 'B', lambda d: d['source_identity']['actual_collector'].update(sha256='0'*64))
        with self.assertRaisesRegex(ValueError, 'collector differs'):
            analyzer.evaluate(MemoryPath(files))

    def test_missing_original_pump_stage_rejected(self):
        files = synthetic_complete()
        mutate(files, lambda r: r['arm'] == 'B', lambda d: d['counts'].update(_poll_ring_completions=0))
        self.assertFalse(analyzer.evaluate(MemoryPath(files))['functional_pass'])

    def test_cost_moved_to_producer_cannot_pass(self):
        files = synthetic_complete()
        def change(d):
            d['producer_cpu_ns'] = 2000
            d['producer']['cpu_ns'] = 2000
            d['worker_plus_producer_cpu_ns'] = 2100
        mutate(files, lambda r: r['arm'] == 'B', change)
        result = analyzer.evaluate(MemoryPath(files))
        self.assertFalse(result['CPU_and_latency_gates_passed'])
        self.assertTrue(next(g['passed'] for g in result['gates'] if g['name'] == 'main_worker_CPU'))
        self.assertFalse(next(g['passed'] for g in result['gates'] if g['name'] == 'main_total_CPU'))

    def test_aggregate_cannot_hide_producer_cost(self):
        files = synthetic_complete()
        mutate(files, lambda r: r['arm'] == 'B', lambda d: d.update(worker_plus_producer_cpu_ns=0))
        with self.assertRaisesRegex(ValueError, 'CPU sum differs'):
            analyzer.evaluate(MemoryPath(files))

    def test_negative_or_boolean_CPU_rejected(self):
        for value in (-1, True):
            files = synthetic_complete()
            mutate(files, lambda r: r['arm'] == 'B', lambda d: d.update(producer_cpu_ns=value))
            with self.assertRaisesRegex(ValueError, 'nonnegative integer'):
                analyzer.evaluate(MemoryPath(files))

    def test_prepare_cost_not_omitted(self):
        files = synthetic_complete()
        mutate(files, lambda r: r['arm'] == 'B', lambda d: d['producer'].update(prepare_callback_included_in_producer_cpu=False))
        with self.assertRaisesRegex(ValueError, 'prepare CPU not included'):
            analyzer.evaluate(MemoryPath(files))

    def test_already_progressing_kept_but_cannot_qualify_latency(self):
        files = synthetic_complete()
        def change(d):
            enq = next(e['at_ns'] for e in d['queue']['put'] if e['kind'] == '_MandatoryWait')
            d.update(accepted_ns=enq-10, enqueue_to_progress_signed_ns=-10,
                     enqueue_to_progress_ns=None, already_in_progress_at_control_enqueue=True,
                     enqueue_to_progress_eligible=False)
        mutate(files, lambda r: r['scenario'] == 'mandatory_20ms', change)
        result = analyzer.evaluate(MemoryPath(files))
        self.assertTrue(result['functional_pass'])
        self.assertFalse(result['CPU_and_latency_gates_passed'])
        self.assertTrue(result['latency_measurement_ineligible'])
        self.assertEqual(result['decision'], 'CPU_LATENCY_MEASUREMENT_INELIGIBLE_NO_UPGRADE')

    def test_negative_latency_cannot_be_scored(self):
        files = synthetic_complete()
        mutate(files, lambda r: r['scenario'] == 'mandatory_20ms', lambda d: d.update(enqueue_to_progress_ns=-10))
        with self.assertRaisesRegex(ValueError, 'negative control latency'):
            analyzer.evaluate(MemoryPath(files))

    def test_race_receipt_must_bind_collector_and_reactor(self):
        files = synthetic_complete()
        plan = json.loads(files['PLAN.json'])
        receipt = {'status': 'PASS', 'candidate_reactor_sha256': plan['candidate_manifest']['B']['reactor'],
                   'candidate_collector_sha256': '0'*64, 'gpu_workloads_run': 0, 'failed': 0, 'passed': 32}
        files['race.json'] = packed(receipt)
        result = analyzer.evaluate(MemoryPath(files), MemoryPath(files, 'race.json'))
        self.assertFalse(result['CPU_candidate_qualified'])
        receipt['candidate_collector_sha256'] = plan['candidate_manifest']['B']['collector']
        files['race.json'] = packed(receipt)
        result = analyzer.evaluate(MemoryPath(files), MemoryPath(files, 'race.json'))
        self.assertTrue(result['CPU_candidate_qualified'])
        self.assertEqual(result['GPU_runs_allowed'], 0)

    def test_throttle_not_discarded_or_treated_as_zero(self):
        files = synthetic_complete()
        def change(d):
            d['cgroup_before']['limited'] = True
            d['nr_throttled_delta'] = 1
        mutate(files, lambda r: r['phase'] == 'score' and r['scenario'] == 'step_end_40ms' and r['pair_or_warmup'] < 4, change)
        result = analyzer.evaluate(MemoryPath(files))
        self.assertEqual(result['resource_limited'][0]['unthrottled_pairs'], 8)
        self.assertFalse(result['CPU_and_latency_gates_passed'])

    def test_early_deadline_or_unretired_parent_rejected(self):
        for kind in ('deadline', 'parent'):
            files = synthetic_complete()
            if kind == 'deadline':
                mutate(files, lambda r: r['scenario'] == 'original_deadline_100ms', lambda d: d.update(deadline_lateness_ns=-1))
            else:
                mutate(files, lambda r: r['scenario'] == 'mandatory_20ms', lambda d: d['native_final_state'].update(accepted_parents=1))
            self.assertFalse(analyzer.evaluate(MemoryPath(files))['functional_pass'])

    def test_missing_or_reordered_pair_rejected(self):
        files = synthetic_complete()
        suite = json.loads(files['SUITE.json'])
        suite['records'][8], suite['records'][9] = suite['records'][9], suite['records'][8]
        files['SUITE.json'] = packed(suite)
        with self.assertRaisesRegex(ValueError, 'fixed order'):
            analyzer.evaluate(MemoryPath(files))

    def test_stopped_partial_suite_cannot_invent_complete_comparison(self):
        files = synthetic_complete()
        suite = json.loads(files['SUITE.json'])
        suite['records'] = suite['records'][:9]
        suite['status'] = 'STOPPED_ON_FAILURE'
        files['SUITE.json'] = packed(suite)
        result = analyzer.evaluate(MemoryPath(files))
        self.assertFalse(result['scored_complete'])
        self.assertFalse(result['functional_pass'])


if __name__ == '__main__':
    unittest.main()

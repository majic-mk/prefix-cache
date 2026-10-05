"""CPU-only source/history contracts; no native receipt is issued by fixtures."""
import ast
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_test_normal_history_adapter', HERE / 'historical_calibration_binding.py')
H = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = H
spec.loader.exec_module(H)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def cpu_event(identity='cpu-fixture-historical', elapsed=2.0):
    return dict(reservation_id=identity, elapsed_seconds=elapsed,
        origin='CPU_fixture_only_not_native_guard', native_execution_verified=False)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.event = cpu_event()
        self.snapshot = dict(active_reservation=None, events=[self.event], gpu_wall_seconds=12.0)
        self.current = deepcopy(self.snapshot)

    def verify(self):
        return H.verify_ledger_extension(self.snapshot, self.current, self.event, ledger_before_seconds=10.0)

    def reject(self):
        with self.assertRaises((ValueError, TypeError)):
            self.verify()

    def test_cpu_fixture_identity_relation_does_not_qualify_native(self):
        value = self.verify()
        self.assertEqual(value['appended_events'], 0)
        self.assertFalse(value['new_gpu_authority_issued'])
        self.assertFalse(value['normal_native_execution_qualified'])
        self.assertEqual(value['GPU_operations'], 0)

    def test_cpu_fixture_appended_relation(self):
        self.current['events'].append(cpu_event('cpu-fixture-tail', 3.0))
        self.current['gpu_wall_seconds'] = 15.0
        self.assertEqual(self.verify()['appended_events'], 1)

    def test_cpu_fixture_new_active_relation_metadata_only(self):
        self.current['active_reservation'] = dict(id='cpu-fixture-new-active', reserved_seconds=320,
            seconds_limit=300, state='running')
        self.assertEqual(self.verify()['current_active_status'], 'ACTIVE_NEW_RESERVATION_METADATA_ONLY')

    def test_actual_snapshot_raw_hash_and_idle_relation_metadata_only(self):
        path = HERE / 'GPU03_LEDGER_SNAPSHOT.json'
        raw = path.read_bytes()
        self.assertEqual(len(raw), H.SNAPSHOT_BYTES)
        self.assertEqual(sha256(raw).hexdigest(), H.SNAPSHOT_SHA)
        actual = H.parse(raw)
        event = actual['events'][-1]
        value = H.verify_ledger_extension(actual, deepcopy(actual), event,
            ledger_before_seconds=22380.692507382948)
        self.assertEqual(value['snapshot_events'], 262)
        self.assertEqual(event['reservation_id'], H.HISTORICAL_RESERVATION)
        self.assertFalse(value['normal_native_execution_qualified'])

    def test_actual_prefix_with_cpu_fixture_tail_and_active(self):
        actual = H.parse((HERE / 'GPU03_LEDGER_SNAPSHOT.json').read_bytes())
        current = deepcopy(actual)
        current['events'].append(cpu_event('cpu-fixture-tail-only', 1.5))
        current['gpu_wall_seconds'] += 1.5
        current['active_reservation'] = dict(id='cpu-fixture-active-only', reserved_seconds=320,
            seconds_limit=300, state='running')
        result = H.verify_ledger_extension(actual, current, actual['events'][-1],
            ledger_before_seconds=22380.692507382948)
        self.assertEqual(result['appended_events'], 1)
        self.assertFalse(result['normal_native_execution_qualified'])

    def test_prefix_mutation(self):
        self.current['events'][0]['origin'] = 'mutated'
        self.reject()

    def test_prefix_truncation(self):
        self.current['events'] = []
        self.reject()

    def test_prefix_typed_bool_numeric_drift(self):
        self.current['events'][0]['elapsed_seconds'] = True
        self.reject()

    def test_duplicate_tail_reservation(self):
        self.current['events'].append(deepcopy(self.event))
        self.current['gpu_wall_seconds'] += 2
        self.reject()

    def test_unknown_tail_reservation(self):
        self.current['events'].append(dict(elapsed_seconds=1.0))
        self.current['gpu_wall_seconds'] += 1
        self.reject()

    def test_changed_legacy_no_id_record(self):
        real = H.parse((HERE / 'GPU03_LEDGER_SNAPSHOT.json').read_bytes())
        real['events'][0]['label'] = 'changed'
        with self.assertRaises(ValueError):
            H.event_ids(real['events'], allow_legacy_first=True)

    def test_negative_tail_elapsed(self):
        self.current['events'].append(cpu_event('cpu-fixture-tail', -1.0))
        self.current['gpu_wall_seconds'] -= 1
        self.reject()

    def test_nonfinite_tail_elapsed(self):
        self.current['events'].append(cpu_event('cpu-fixture-tail', float('nan')))
        self.reject()

    def test_changed_wall_sum(self):
        self.current['gpu_wall_seconds'] += 1
        self.reject()

    def test_changed_historical_delta(self):
        self.snapshot['gpu_wall_seconds'] = 13.0
        self.current['gpu_wall_seconds'] = 13.0
        self.reject()

    def test_budget_overflow(self):
        self.current['events'].append(cpu_event('cpu-fixture-tail', 30000.0))
        self.current['gpu_wall_seconds'] += 30000.0
        self.reject()

    def test_completed_reservation_still_active(self):
        self.current['active_reservation'] = dict(id=self.event['reservation_id'], reserved_seconds=320,
            seconds_limit=300)
        self.reject()

    def test_active_cleanup_unresolved(self):
        self.current['active_reservation'] = dict(id='cpu-fixture-new-active', reserved_seconds=320,
            seconds_limit=300, state='cleanup_unresolved')
        self.reject()

    def test_active_bool_reserved(self):
        self.current['active_reservation'] = dict(id='cpu-fixture-new-active', reserved_seconds=True,
            seconds_limit=1)
        self.reject()

    def test_active_larger_than_remaining(self):
        self.current['active_reservation'] = dict(id='cpu-fixture-new-active', reserved_seconds=30000,
            seconds_limit=300)
        self.reject()

    def test_active_not_permitted(self):
        self.current['active_reservation'] = dict(id='cpu-fixture-new-active', reserved_seconds=320,
            seconds_limit=300)
        with self.assertRaises(ValueError):
            H.verify_ledger_extension(self.snapshot, self.current, self.event,
                ledger_before_seconds=10.0, allow_active=False)

    def test_pure_history_cannot_authorize_gpu(self):
        with self.assertRaisesRegex(ValueError, 'no new GPU execution authority'):
            H.verify_active_guard(None, None, None)

    def test_duplicate_json_rejected(self):
        with self.assertRaises(ValueError):
            H.parse('{"gpu_wall_seconds":1,"gpu_wall_seconds":2}')

    def test_native_derivative_changes_only_binding_api(self):
        remote = Path(os.environ.get('C5_NORMAL_PROJECT_ROOT', '/nonexistent')) / H.G
        old = remote if remote.exists() else HERE.parent / 'notification_v5_gpu_entry_path_revision'
        raw = (old / 'native_conditional_cost.py').read_bytes()
        before = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
            if isinstance(node, ast.FunctionDef)}
        after = {node.name: ast.dump(node, include_attributes=False)
            for node in ast.parse((HERE / 'native_conditional_cost.py').read_bytes()).body
            if isinstance(node, ast.FunctionDef)}
        self.assertEqual(set(before), set(after))
        self.assertEqual([name for name in before if before[name] != after[name]], ['binding_api'])
        self.assertEqual((HERE / 'prepare_and_verify_native_cost.py').read_bytes(),
            (old / 'prepare_and_verify_native_cost.py').read_bytes())
        original = (old / 'gpu_entry_binding.py').read_bytes()
        self.assertEqual((len(original), sha256(original).hexdigest()), (H.ORIGINAL_BYTES, H.ORIGINAL_SHA))
        fn = next(node for node in ast.parse(original).body
            if isinstance(node, ast.FunctionDef) and node.name == 'verify_completed_guard')
        stop = next(i for i, node in enumerate(fn.body) if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'ledger')
        self.assertGreater(stop, 10)

    def test_public_canonical_rejects_private_construction_and_pins_sources(self):
        public = module('_test_normal_history_public', 'p4_single_file_receipt.py')
        with self.assertRaisesRegex(ValueError, 'load_verified_single_file'):
            public.ExactSingleFileReceipt()
        for reference, filename in ((public._VERIFIER, 'native_conditional_cost.py'),
            (public._SERIALIZER, 'prepare_and_verify_native_cost.py'),
            (public._HISTORY, 'historical_calibration_binding.py'),
            (public._SNAPSHOT, 'GPU03_LEDGER_SNAPSHOT.json')):
            raw = (HERE / filename).read_bytes()
            self.assertEqual((len(raw), sha256(raw).hexdigest()), (reference.bytes, reference.sha256))
        self.assertTrue(public._COMMON_OVERLAY.startswith(H.G + '/'))
        self.assertFalse(any(name == prefix or name.startswith(prefix+'.')
            for name in sys.modules for prefix in ('torch', 'vllm', 'py_kvcache')))


if __name__ == '__main__':
    unittest.main(verbosity=2)

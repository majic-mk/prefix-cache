"""CPU rejection tests. Fixtures never issue a receipt or qualify a GPU path."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


HERE = Path(__file__).resolve().parent
E = module(HERE / 'entry_receipt_session.py', '_dedup_cpu_test_source')
OLD = HERE.parent / 'normal_native_cpu'
H = module(OLD / 'historical_calibration_binding.py', '_dedup_cpu_actual_history')
Q = module(OLD / 'p4_single_file_receipt.py', '_dedup_cpu_actual_canonical')


class IntegrityRejectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'source.py').write_text('# explicitly synthetic integrity file\n', encoding='utf-8')
        (self.root / 'authority.json').write_text('{"fixture_only":true,"authorization":false}\n', encoding='utf-8')
        self.rows = {name: E.file_ref(self.root, name) for name in ('source.py', 'authority.json')}

    def tearDown(self):
        self.temp.cleanup()

    def test_full_byte_proof_is_deterministic_without_qualification(self):
        self.assertEqual(E.verify_integrity(self.root, self.rows), E.verify_integrity(self.root, self.rows))

    def test_same_length_source_drift_rejected(self):
        path = self.root / 'source.py'
        path.write_bytes(path.read_bytes().replace(b'synthetic', b'SYNTHETIC'))
        with self.assertRaisesRegex(ValueError, 'immutable bytes changed'):
            E.verify_integrity(self.root, self.rows)

    def test_authority_byte_drift_rejected(self):
        path = self.root / 'authority.json'
        path.write_bytes(path.read_bytes().replace(b'false', b'true '))
        with self.assertRaisesRegex(ValueError, 'immutable bytes changed'):
            E.verify_integrity(self.root, self.rows)

    def test_row_alias_rejected(self):
        with self.assertRaisesRegex(ValueError, 'row-key path mismatch'):
            E.verify_integrity(self.root, {'renamed.py': self.rows['source.py']})

    def test_traversal_rejected(self):
        with self.assertRaisesRegex(ValueError, 'bounded relative'):
            E.file_ref(self.root, '../source.py')

    def test_absolute_path_rejected(self):
        with self.assertRaisesRegex(ValueError, 'bounded relative'):
            E.file_ref(self.root, str(self.root / 'source.py'))

    def test_bool_byte_size_rejected(self):
        row = dict(self.rows['source.py'], bytes=True)
        with self.assertRaisesRegex(ValueError, 'exact reference schema'):
            E.verify_ref(self.root, row)

    def test_duplicate_json_rejected(self):
        path = self.root / 'duplicate.json'
        path.write_text('{"a":1,"a":2}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'duplicate JSON'):
            E.read_json(path)

    def test_factory_refuses_fixture_project(self):
        with self.assertRaisesRegex(ValueError, 'actual fixed project'):
            E.CompletedReceiptSession(self.root)

    def test_no_public_constructor_promotion(self):
        with self.assertRaisesRegex(ValueError, 'load_verified_single_file'):
            Q.ExactSingleFileReceipt()

    def test_active_path_is_unimplemented(self):
        with self.assertRaisesRegex(ValueError, 'not implemented or qualified'):
            E.CompletedReceiptSession.verify_active_guard(None)


class ProcessIdentityRejectTests(unittest.TestCase):
    def test_actual_comparison_has_no_receipt_issuer(self):
        observed = dict(pid=19, sid=13, process_group=13)
        self.assertTrue(E.validate_process_identity(observed, dict(observed)))

    def test_each_process_coordinate_rechecked(self):
        expected = dict(pid=19, sid=13, process_group=13)
        for key in expected:
            with self.subTest(key=key):
                actual = dict(expected); actual[key] += 1
                with self.assertRaisesRegex(ValueError, 'cannot cross PID'):
                    E.validate_process_identity(expected, actual)

    def test_bool_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cannot cross PID'):
            E.validate_process_identity(dict(pid=1, sid=1, process_group=1),
                dict(pid=True, sid=1, process_group=1))


class ActualLedgerMetadataRejectTests(unittest.TestCase):
    """Use real immutable calibration snapshot, without invoking an issuer."""
    def setUp(self):
        self.snapshot = json.loads((OLD / 'SERVER12_LEDGER_SNAPSHOT.json').read_bytes())
        self.event = self.snapshot['events'][-1]
        self.before = self.snapshot['gpu_wall_seconds'] - self.event['elapsed_seconds']

    def check(self, actual):
        return H.verify_ledger_extension(self.snapshot, actual, self.event,
            ledger_before_seconds=self.before, allow_active=False)

    def test_real_completed_prefix_metadata(self):
        result = self.check(deepcopy(self.snapshot))
        self.assertEqual(result['GPU_operations'], 0)
        self.assertFalse(result['normal_native_execution_qualified'])

    def test_historical_prefix_edit_rejected(self):
        actual = deepcopy(self.snapshot)
        actual['events'][1]['label'] += '-altered'
        with self.assertRaisesRegex(ValueError, 'prefix changed'):
            self.check(actual)

    def test_ledger_truncation_rejected(self):
        actual = deepcopy(self.snapshot); actual['events'].pop()
        with self.assertRaisesRegex(ValueError, 'bounded complete ledger prefix'):
            self.check(actual)

    def test_ledger_wall_rewrite_rejected(self):
        actual = deepcopy(self.snapshot); actual['gpu_wall_seconds'] += 1
        with self.assertRaisesRegex(ValueError, 'exact appended-event accounting'):
            self.check(actual)

    def test_any_active_guard_rejected(self):
        actual = deepcopy(self.snapshot)
        actual['active_reservation'] = dict(id='synthetic-untrusted-active', reserved_seconds=320, seconds_limit=300)
        with self.assertRaisesRegex(ValueError, 'active reservation permitted'):
            self.check(actual)

    def test_duplicate_completed_event_rejected(self):
        actual = deepcopy(self.snapshot); actual['events'].append(deepcopy(self.event))
        actual['gpu_wall_seconds'] += self.event['elapsed_seconds']
        with self.assertRaisesRegex(ValueError, 'unique completed reservation'):
            self.check(actual)


if __name__ == '__main__':
    unittest.main()

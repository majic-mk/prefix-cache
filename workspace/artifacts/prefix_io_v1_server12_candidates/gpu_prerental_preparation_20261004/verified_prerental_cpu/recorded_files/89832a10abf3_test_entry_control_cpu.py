"""CPU rejection tests for bounded launch metadata; never launch a GPU."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('entry_control_under_test', HERE/'entry_control.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


class EntryControlTests(unittest.TestCase):
    def test_original_budget_accepts_current_remaining_without_reservation(self):
        ledger = dict(gpu_wall_seconds=24678.89523890568, active_reservation=None)
        before = json.dumps(ledger)
        self.assertAlmostEqual(E.budget(ledger), 4121.104761094321)
        self.assertEqual(json.dumps(ledger), before)

    def test_nonfinite_boolean_negative_excess_usage_rejected(self):
        for value in (float('nan'), float('inf'), float('-inf'), True, False, -1, 28801):
            with self.subTest(value=value), self.assertRaises(ValueError):
                E.budget(dict(gpu_wall_seconds=value, active_reservation=None))

    def test_cleanup_reserve_is_required(self):
        self.assertEqual(E.budget(dict(gpu_wall_seconds=28480, active_reservation=None)), 320)
        with self.assertRaises(ValueError):
            E.budget(dict(gpu_wall_seconds=28480.1, active_reservation=None))

    def test_unresolved_reservation_rejected(self):
        with self.assertRaises(ValueError):
            E.budget(dict(gpu_wall_seconds=0, active_reservation={'state':'cleanup_unresolved'}))

    def test_permission_never_expands_system_download_payment_or_rental(self):
        base = dict(allow_gpu_runs=True, max_gpu_hours=8, allow_driver_or_system_changes=False,
                    allow_model_downloads=True, approved_gpu_ids=['old-device'],
                    approved_auxiliary_storage={'root':'fixture'}, approved_dependency_root='fixture-root',
                    approved_experiment_root='fixture-runs')
        saved = dict(base)
        gpu = 'GPU-00000000-0000-0000-0000-000000000001'
        value = E.effective_permission(base, gpu)
        self.assertEqual(value['max_gpu_hours'], 8)
        self.assertEqual(value['approved_gpu_ids'], [gpu])
        self.assertEqual(value['approved_dependency_root'], 'fixture-root')
        self.assertEqual(value['approved_experiment_root'], 'fixture-runs')
        self.assertNotIn('approved_auxiliary_storage', value)
        for key in ('allow_model_downloads','allow_driver_or_system_changes','allow_shared_data_deletion',
                    'allow_payment','allow_new_cloud_rental','allow_remote_push'):
            self.assertIs(value[key], False)
        self.assertEqual(base, saved)

    def test_wrong_or_missing_device_and_budget_rejected(self):
        for gpu in ('GPU-fixture', None, '', 'GPU-00000000-0000-0000-0000-000000000001,other'):
            with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                E.effective_permission(dict(allow_gpu_runs=True,max_gpu_hours=8), gpu)
        for hours in (True, 9, None, float('nan'), float('inf')):
            with self.subTest(hours=hours), self.assertRaises(ValueError):
                E.effective_permission(dict(allow_gpu_runs=True,max_gpu_hours=hours),
                                       'GPU-00000000-0000-0000-0000-000000000001')

    def test_traversal_absolute_ambiguous_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            for name in ('../x', 'a/../x', '/tmp/x', 'a//x', 'a\\x', 'C:/x', './x'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    E.safe(root, name, existing=False)
            (root/'a').mkdir()
            self.assertEqual(E.safe(root,'a/x',existing=False),root/'a/x')

    def test_duplicate_and_nonfinite_JSON_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'record.json'
            for text in ('{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    E.document(path)

    def test_write_is_append_only(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'record.json'
            E.write(path, {'fixture':True})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                E.write(path, {'fixture':False})
            self.assertEqual(path.read_bytes(),before)

    def test_no_gpu_model_backend_network_client_imports(self):
        self.assertFalse(any(name.split('.')[0] in {'torch','vllm','py_kvcache','cupy','openai'} for name in sys.modules))


if __name__ == '__main__':
    unittest.main(verbosity=2)

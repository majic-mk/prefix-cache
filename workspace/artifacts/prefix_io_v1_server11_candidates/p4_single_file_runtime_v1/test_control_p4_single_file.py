"""CPU launch-gate tests: subprocess and GPU telemetry are always mocked."""
import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch, Mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_cpu_only_single_file_control', HERE / 'control_p4_single_file.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(patch.object(C, 'ROOT', self.root))
        self.gpu = self.stack.enter_context(patch.object(C.subprocess, 'run',
            side_effect=AssertionError('unexpected external/GPU call')))
        self.spawn = self.stack.enter_context(patch.object(C.subprocess, 'Popen',
            side_effect=AssertionError('unexpected dispatch')))
        self.write(C.LEDGER, dict(active_reservation=None, gpu_wall_seconds=0))
        self.write(C.LOCK, dict(files=[]))
        self.write(C.BINDING, dict(cpu_fixture=True))
        self.write(C.PERMISSION, dict(cpu_fixture=True))
        with contextlib.redirect_stdout(__import__('io').StringIO()): C.prepare('off')

    def tearDown(self):
        self.stack.close(); self.temp.cleanup()

    def write(self, relative, value):
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def assert_no_dispatch(self):
        self.gpu.assert_not_called(); self.spawn.assert_not_called()

    def test_active_reservation_refuses_before_any_external_call(self):
        self.write(C.LEDGER, dict(active_reservation=dict(label='other'), gpu_wall_seconds=0))
        with self.assertRaisesRegex(ValueError, 'budget'): C.launch('off')
        self.assert_no_dispatch()

    def test_full_reserved_budget_must_fit(self):
        self.write(C.LEDGER, dict(active_reservation=None, gpu_wall_seconds=28800-319))
        with self.assertRaisesRegex(ValueError, 'budget'): C.launch('off')
        self.assert_no_dispatch()

    def test_existing_run_or_intent_never_retries(self):
        for kind in ('run', 'intent'):
            target = ('experiments/prefix_io_v1/runs/'+C.label('off') if kind=='run'
                      else C.D+'/LAUNCH_INTENT_off.json')
            path=self.root/target; path.parent.mkdir(parents=True,exist_ok=True)
            if kind=='run': path.mkdir()
            else: path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'never blindly retried'): C.launch('off')
        self.assert_no_dispatch()

    def test_changed_scoped_config_refuses_before_gpu(self):
        config=C.read(C.config_path('off')); config['seconds_limit']=301
        self.write(C.config_path('off'),config)
        with self.assertRaisesRegex(ValueError, 'reference changed'): C.launch('off')
        self.assert_no_dispatch()

    def test_config_has_runtime_binding_and_scope_separately_pins_it(self):
        config=C.read(C.config_path('off')); scope=C.read(C.D+'/SCOPE_off.json')
        self.assertEqual(config['runtime_binding_relative'], C.C+'/single_file_runtime_binding.py')
        self.assertEqual(scope['config_ref'], C.ref(C.config_path('off')))
        self.assertNotIn(C.config_path('off'), [r['path'] for r in C.read(C.LOCK)['files']])
        self.assertEqual((scope['seconds_limit'],scope['reserved_seconds']), (300,320))
        self.assertEqual(scope['maximum_gpu_jobs'], 1)
        self.assert_no_dispatch()

    def test_predecessor_runtime_qualification_required(self):
        cases = [dict(scope='server11_p4_single_file_runtime_qualification_v1',mode='off',
                      native_execution_verified=True,runtime_condition_qualified=False,permits_next_mode='shadow'),
                 dict(scope='wrong',mode='off',native_execution_verified=True,
                      runtime_condition_qualified=True,permits_next_mode='shadow'),
                 dict(scope='server11_p4_single_file_runtime_qualification_v1',mode='off',
                      native_execution_verified=True,runtime_condition_qualified=True,permits_next_mode='on')]
        for receipt in cases:
            self.write(C.result_path('off'),receipt)
            with self.assertRaises(ValueError): C.prepare('shadow')
        self.assert_no_dispatch()

    def test_success_records_intent_then_only_one_original_guard(self):
        self.gpu.side_effect = [NS(stdout=C.GPU+', fake CPU fixture GPU, 30000\n'), NS(stdout='')]
        self.spawn.side_effect=None; self.spawn.return_value=NS(pid=12345)
        self.stack.enter_context(patch.object(C.os,'statvfs',create=True,
            return_value=NS(f_bavail=100*1024**3,f_frsize=1)))
        # The synthetic lock's source read is still real local CPU hashing.
        with contextlib.redirect_stdout(__import__('io').StringIO()): C.launch('off')
        self.spawn.assert_called_once()
        args, kwargs = self.spawn.call_args
        self.assertEqual(args[0][2], C.GUARD)
        self.assertEqual(args[0][args[0].index('--seconds')+1], '300')
        self.assertTrue(kwargs['start_new_session'])
        self.assertTrue((self.root/C.D/'LAUNCH_INTENT_off.json').is_file())
        with self.assertRaisesRegex(ValueError, 'never blindly retried'): C.launch('off')
        self.assertEqual(self.spawn.call_count, 1)
        self.assertEqual(self.gpu.call_count, 2)


if __name__ == '__main__': unittest.main()

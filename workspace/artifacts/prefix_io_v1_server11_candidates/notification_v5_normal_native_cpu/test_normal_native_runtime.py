"""CPU-only runtime boundary/clock tests; no receipt issuance or GPU imports.

Temporary minimal modules and clock values have explicit cpu_fixture origin.
They test source confinement and unknown/overlapping CPU accounting only.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = load('_cpu_test_normal_runtime_runner', 'run_p4_single_file_experiment.py')
V = load('_cpu_test_normal_runtime_verifier', 'verify_p4_single_file.py')
M = load('_cpu_test_normal_gross_cost', 'observation_cpu_cost.py')


def source_rows():
    return {name: dict(path='cpu_fixture/' + name + '.py', bytes=1, sha256='a'*64)
            for name in ('runner', 'cost_meter', 'collector', 'reactor', 'notification_adapter')}


class CPUThread:
    def __init__(self, ident=22, tid=220):
        self.ident, self.native_id, self.live = ident, tid, True
    def is_alive(self):
        return self.live


class MeterTests(unittest.TestCase):
    def make(self, *, mode='on', owner=None, aio=None, reader=None,
             wall=(100, 900), process=(1000, 2000), thread=(10, 40)):
        calls = []
        def clock(name, values):
            iterator = iter(values)
            def read():
                calls.append(name)
                return next(iterator)
            return read
        counts = {}
        def native(tid):
            calls.append('native:' + str(tid))
            counts[tid] = counts.get(tid, 0) + 1
            return dict(native_tid=tid, start_ticks=7, cpu_ns=100*counts[tid],
                        resolution_ns=10_000_000, clock_source='linux_proc_task_stat')
        meter = M.CompletePathCPU(mode=mode, run_id='cpu-fixture-run', request_id='cpu-fixture-request',
            source_refs=source_rows(), origin='cpu_fixture', wall_clock=clock('wall', wall),
            process_clock=clock('process', process), thread_clock=clock('main', thread),
            native_reader=reader or native)
        reactor = types.SimpleNamespace(_worker=owner, ring=types.SimpleNamespace(_worker=aio))
        return meter, reactor, calls

    @staticmethod
    def stop(meter, *, outcome=None):
        rows = [] if outcome is None else [dict(outcome=outcome)]
        evidence = dict(mode=meter.mode, run_id=meter.run_id, request_id=meter.request_id, rows=rows)
        meter.stop(full_frame_count=128, action_count=1, notification_evidence=evidence)
        return meter.export(), evidence

    def test_all_arms_measure_same_span_without_qualification(self):
        for mode in ('off', 'shadow', 'on'):
            with self.subTest(mode=mode):
                meter, reactor, calls = self.make(mode=mode)
                meter.start(reactor); doc, _ = self.stop(meter)
                self.assertEqual(doc['process_cpu_ns'], 1000)
                self.assertEqual(doc['wall_ns'], 800)
                self.assertEqual(calls, ['wall', 'process', 'main', 'main', 'process', 'wall'])
                self.assertEqual(doc['origin'], 'cpu_fixture')
                self.assertFalse(doc['full_runtime_cost_qualified'])
                self.assertFalse(doc['net_observation_cost_qualified'])
                self.assertIsNone(doc['valid_native_cost_upper_ns'])
                self.assertIsNone(doc['valid_native_step_budget_ns'])

    def test_missing_workers_unknown_not_zero(self):
        meter, reactor, _ = self.make(); meter.start(reactor); doc, _ = self.stop(meter)
        for row in doc['thread_cpu_diagnostics'][1:]:
            self.assertEqual(row['status'], 'UNAVAILABLE')
            self.assertIsNone(row['cpu_ns']); self.assertTrue(row['reason'])

    def test_native_diagnostics_enclosed_and_not_added(self):
        owner, aio = CPUThread(), CPUThread(33, 330)
        meter, reactor, calls = self.make(owner=owner, aio=aio)
        meter.start(reactor); doc, _ = self.stop(meter)
        self.assertEqual(calls, ['wall', 'process', 'main', 'native:220', 'native:330',
                                 'main', 'native:220', 'native:330', 'process', 'wall'])
        self.assertEqual([row['cpu_ns'] for row in doc['thread_cpu_diagnostics']], [30, 100, 100])
        self.assertEqual(doc['process_cpu_ns'], 1000)
        self.assertTrue(doc['thread_values_are_not_additive'])
        self.assertTrue(doc['thread_diagnostics_included_in_process_cpu'])

    def test_thread_end_after_start_unknown(self):
        owner = CPUThread(); meter, reactor, _ = self.make(owner=owner)
        meter.start(reactor); owner.live = False; doc, _ = self.stop(meter)
        self.assertEqual(doc['thread_cpu_diagnostics'][1]['status'], 'UNAVAILABLE_AT_END')
        self.assertIsNone(doc['thread_cpu_diagnostics'][1]['cpu_ns'])

    def test_tid_lifetime_reuse_unknown(self):
        owner = CPUThread(); reads = []
        def reader(tid):
            reads.append(tid)
            return dict(native_tid=tid, start_ticks=len(reads), cpu_ns=100*len(reads),
                        resolution_ns=10_000_000, clock_source='linux_proc_task_stat')
        meter, reactor, _ = self.make(owner=owner, reader=reader)
        meter.start(reactor); doc, _ = self.stop(meter)
        row = doc['thread_cpu_diagnostics'][1]
        self.assertEqual(row['status'], 'UNAVAILABLE_AT_END'); self.assertIsNone(row['cpu_ns'])

    def test_unreadable_proc_counter_unknown(self):
        def reader(_tid):
            raise FileNotFoundError('cpu_fixture exited task')
        meter, reactor, _ = self.make(owner=CPUThread(), reader=reader)
        meter.start(reactor); doc, _ = self.stop(meter)
        self.assertEqual(doc['thread_cpu_diagnostics'][1]['status'], 'UNAVAILABLE')
        self.assertIsNone(doc['thread_cpu_diagnostics'][1]['cpu_ns'])

    def test_alias_tid_is_explicit_overlap(self):
        thread = CPUThread(); meter, reactor, _ = self.make(owner=thread, aio=thread)
        meter.start(reactor); doc, _ = self.stop(meter)
        self.assertTrue(doc['aliased_native_tids']); self.assertEqual(doc['process_cpu_ns'], 1000)

    def test_unexercised_on_is_not_exercised(self):
        meter, reactor, _ = self.make(); meter.start(reactor); doc, _ = self.stop(meter)
        self.assertEqual(doc['notification_wait_status'], 'NOT_EXERCISED')
        self.assertFalse(doc['on_observation_cost_qualified'])

    def test_actual_wait_outcome_only_marks_path(self):
        for outcome in ('matching_wake', 'original_deadline_timeout'):
            meter, reactor, _ = self.make(); meter.start(reactor); doc, _ = self.stop(meter, outcome=outcome)
            self.assertEqual(doc['notification_wait_status'], 'EXERCISED')
            self.assertFalse(doc['full_runtime_cost_qualified'])

    def test_unrelated_wake_does_not_exercise_wait(self):
        meter, reactor, _ = self.make(); meter.start(reactor); doc, _ = self.stop(meter, outcome='unrelated_wake')
        self.assertEqual(doc['notification_wait_status'], 'NOT_EXERCISED')

    def test_backwards_process_clock_rejected(self):
        meter, reactor, _ = self.make(process=(1000, 999)); meter.start(reactor)
        with self.assertRaisesRegex(ValueError, 'monotonic'): self.stop(meter)

    def test_backwards_wall_clock_rejected(self):
        meter, reactor, _ = self.make(wall=(100, 99)); meter.start(reactor)
        with self.assertRaisesRegex(ValueError, 'monotonic'): self.stop(meter)

    def test_thread_clock_backwards_is_unknown(self):
        meter, reactor, _ = self.make(thread=(40, 10)); meter.start(reactor); doc, _ = self.stop(meter)
        self.assertEqual(doc['thread_cpu_diagnostics'][0]['status'], 'UNAVAILABLE_AT_END')
        self.assertIsNone(doc['thread_cpu_diagnostics'][0]['cpu_ns'])

    def test_lifecycle_start_stop_once(self):
        meter, reactor, _ = self.make()
        with self.assertRaises(ValueError): meter.export()
        meter.start(reactor)
        with self.assertRaises(ValueError): meter.start(reactor)
        self.stop(meter)
        with self.assertRaises(ValueError): self.stop(meter)

    def test_fixture_never_passes_native_cost_validation(self):
        meter, reactor, _ = self.make(); meter.start(reactor); doc, notification = self.stop(meter)
        with self.assertRaisesRegex(ValueError, 'origin'):
            M.validate_gross_cost(doc, mode='on', run_id=meter.run_id, request_id=meter.request_id,
                source_refs=source_rows(), full_frame_count=128, action_count=1, notification_evidence=notification)

    def test_native_origin_refuses_injected_clock(self):
        with self.assertRaisesRegex(ValueError, 'builtin'):
            M.CompletePathCPU(mode='on', run_id='cpu-fixture-run', request_id='cpu-fixture-request',
                source_refs=source_rows(), wall_clock=lambda: 1)

    def test_same_mode_notification_identity_required(self):
        meter, reactor, _ = self.make(); meter.start(reactor)
        with self.assertRaisesRegex(ValueError, 'same-mode'):
            meter.stop(full_frame_count=128, action_count=1,
                notification_evidence=dict(mode='shadow', run_id=meter.run_id, request_id=meter.request_id, rows=[]))

    def test_frame_boolean_rejected(self):
        meter, reactor, _ = self.make(); meter.start(reactor)
        with self.assertRaises(ValueError):
            meter.stop(full_frame_count=True, action_count=1,
                notification_evidence=dict(mode='on', run_id=meter.run_id, request_id=meter.request_id, rows=[]))

    def test_descriptive_net_keeps_negative_difference_and_no_grant(self):
        meter, reactor, _ = self.make(); meter.start(reactor); current, _ = self.stop(meter)
        baseline = deepcopy(current); baseline.update(mode='off', process_cpu_ns=1300)
        delta = M.descriptive_net(current, baseline)
        self.assertEqual(delta['process_cpu_difference_ns'], -300)
        self.assertFalse(delta['full_runtime_cost_qualified']); self.assertFalse(delta['performance_effect_verified'])
        self.assertEqual(delta['sample_pairs'], 1)

    def test_descriptive_net_source_mismatch_rejected(self):
        meter, reactor, _ = self.make(); meter.start(reactor); current, _ = self.stop(meter)
        baseline = deepcopy(current); baseline['source_refs']['reactor']['sha256'] = 'b'*64
        with self.assertRaises(ValueError): M.descriptive_net(current, baseline)

    def test_fixed_proc_reader_includes_starttime_and_ticks(self):
        fields = ['S'] + ['0']*19
        fields[11], fields[12], fields[19] = '4', '3', '9'
        raw = '220 (cpu fixture worker) ' + ' '.join(fields)
        with patch.object(M.Path, 'read_text', return_value=raw), patch.object(M.os, 'sysconf', create=True, return_value=100):
            row = M.native_task_cpu(220)
        self.assertEqual(row['cpu_ns'], 70_000_000)
        self.assertEqual(row['start_ticks'], 9); self.assertEqual(row['resolution_ns'], 10_000_000)

    def test_proc_wrong_tid_refused(self):
        with patch.object(M.Path, 'read_text', return_value='330 (wrong) S ' + '0 '*19):
            with self.assertRaises(ValueError): M.native_task_cpu(220)


class SourceAndGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.saved = {name: value for name, value in sys.modules.items()
                      if name == 'prefix_io_control' or name.startswith('prefix_io_control.')}
        for name in self.saved: sys.modules.pop(name)
        self.addCleanup(self.restore)
        self.refs = {}
        self.add(R.CONTROL_PACKAGE, '"""Explicit temporary CPU fixture package."""\n')
        self.add(R.CANONICAL, '"""CPU fixture only; no receipt issuer or authority."""\n'
            'class ExactSingleFileReceipt:\n    pass\n'
            'def load_verified_single_file(*args):\n    raise ValueError("cpu_fixture cannot issue receipt")\n')

    def restore(self):
        for name in list(sys.modules):
            if name == 'prefix_io_control' or name.startswith('prefix_io_control.'):
                sys.modules.pop(name)
        sys.modules.update(self.saved)

    def add(self, relative, source):
        path = self.root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source, encoding='utf-8')
        self.refs[relative] = R.file_ref(self.root, relative)
        return path

    def test_explicit_cpu_class_alias_registered_once_without_receipt(self):
        canonical = R.register_canonical_receipt(self.root, self.refs)
        from prefix_io_control.p4_single_file_receipt import ExactSingleFileReceipt
        self.assertIs(ExactSingleFileReceipt, canonical.ExactSingleFileReceipt)
        self.assertEqual(ExactSingleFileReceipt.__module__, R.CANONICAL_MODULE)
        self.assertEqual(R.loaded_native_source(self.root, self.refs, R.CANONICAL_MODULE, canonical)['source_ref'], self.refs[R.CANONICAL])
        with self.assertRaisesRegex(ValueError, 'cannot issue'): canonical.load_verified_single_file()
        with self.assertRaisesRegex(ValueError, 'fresh'): R.register_canonical_receipt(self.root, self.refs)

    def test_canonical_source_drift_refused_before_load(self):
        (self.root / R.CANONICAL).write_text('raise AssertionError("never executed")', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'drift'): R.register_canonical_receipt(self.root, self.refs)

    def test_actual_public_canonical_registers_class_without_issuing_receipt(self):
        # Only import the real source and public class; no fixture is submitted
        # to its loader, and no live GPU evidence/authority is manufactured.
        actual = HERE / 'p4_single_file_receipt.py'
        self.add(R.CANONICAL, actual.read_text(encoding='utf-8'))
        canonical = R.register_canonical_receipt(self.root, self.refs)
        from prefix_io_control.p4_single_file_receipt import ExactSingleFileReceipt
        self.assertIs(ExactSingleFileReceipt, canonical.ExactSingleFileReceipt)
        self.assertEqual(ExactSingleFileReceipt.__module__, R.CANONICAL_MODULE)
        self.assertEqual(R.loaded_native_source(self.root, self.refs, R.CANONICAL_MODULE, canonical)['source_ref'], self.refs[R.CANONICAL])
        with self.assertRaisesRegex(ValueError, 'source-bound'): ExactSingleFileReceipt()

    def test_malformed_class_module_refused_and_alias_removed(self):
        self.add(R.CANONICAL, 'class ExactSingleFileReceipt:\n    __module__="other.alias"\n'
            'def load_verified_single_file(*a):\n    raise ValueError("cpu_fixture")\n')
        with self.assertRaisesRegex(ValueError, 'exact public'): R.register_canonical_receipt(self.root, self.refs)
        self.assertNotIn(R.CANONICAL_MODULE, sys.modules)

    def test_arbitrary_normal_module_cannot_use_canonical_exception(self):
        canonical = R.register_canonical_receipt(self.root, self.refs)
        with self.assertRaises(ValueError):
            R.loaded_native_source(self.root, self.refs, 'prefix_io_control.other', canonical)

    def test_canonical_module_name_mismatch_refused(self):
        canonical = R.register_canonical_receipt(self.root, self.refs)
        canonical.__name__ = 'other.alias'
        with self.assertRaisesRegex(ValueError, 'actual imported'):
            R.loaded_native_source(self.root, self.refs, R.CANONICAL_MODULE, canonical)

    def test_old_overlay_canonical_cannot_use_new_alias(self):
        canonical = R.register_canonical_receipt(self.root, self.refs)
        old = self.add(R.OVERLAY + '/old_receipt.py', '# cpu_fixture old path\n')
        canonical.__file__ = str(old)
        with self.assertRaisesRegex(ValueError, 'exact source-bound'):
            R.loaded_native_source(self.root, self.refs, R.CANONICAL_MODULE, canonical)

    def test_other_package_source_outside_g_rejected(self):
        path = self.add(R.DELIVERY + '/other.py', '# cpu_fixture outside G\n')
        module = types.SimpleNamespace(__name__='py_kvcache.other', __file__=str(path))
        with self.assertRaisesRegex(ValueError, 'escaped'):
            R.loaded_native_source(self.root, self.refs, 'py_kvcache.other', module)

    def test_real_cli_path_missing_authority_rejected_before_import(self):
        fake = types.SimpleNamespace(verify_configuration=lambda *_args: (_ for _ in ()).throw(ValueError('cpu_fixture missing authority')))
        with patch.object(R, 'controller_api', return_value=fake), patch.object(R, 'execute_window', side_effect=AssertionError('must not execute')):
            with self.assertRaisesRegex(ValueError, 'missing authority'):
                R.main(['--execute', '--config', R.config_relative('off')])

    def test_direct_execute_rejects_gate_before_model_import(self):
        with patch.object(R, 'ROOT', self.root), patch.object(R, 'load_configuration', side_effect=ValueError('cpu_fixture missing authority')):
            with self.assertRaisesRegex(ValueError, 'missing authority'):
                R.execute_window(self.root, dict(mode='off'), {}, {})

    def test_verifier_rejects_missing_authority_before_native_contract(self):
        fake = types.SimpleNamespace(verify_configuration=lambda *_args: (_ for _ in ()).throw(ValueError('cpu_fixture missing authority')))
        with patch.object(V, 'controller_api', return_value=fake), patch.object(V, 'contract', side_effect=AssertionError('must not load native')):
            with self.assertRaisesRegex(ValueError, 'missing authority'):
                V.verify_runtime(self.root, config_ref=dict(path='cpu_fixture_config.json'), result_ref={}, guard_ref={}, before_ref={}, after_ref={})

    def test_source_traversal_refused(self):
        with self.assertRaises(ValueError): R.safe(self.root, '../outside.py')

    def test_cpu_runtime_imports_no_gpu_packages(self):
        self.assertFalse(any(name == 'torch' or name.startswith(('torch.', 'vllm.', 'py_kvcache.')) or name == 'vllm' for name in sys.modules))


class DeferralEvidenceTests(unittest.TestCase):
    @staticmethod
    def snapshots(mode='on', proposed=0, blocked=0):
        state = dict(enabled=True, shadow=(mode=='shadow'), generic_production_qualified=False,
            evidence_binding_sha256='a'*64, previews=0, proposed_deferrals=0,
            actual_blocked_attempts=0, actual_decisions=[])
        initial = dict(valid=True, fault=None, mode='interference', new_work_queues=0,
            held_job_or_resource_owners=False, interference_production_qualified=False,
            conditional_single_file=state)
        final = deepcopy(initial); final['conditional_single_file'].update(previews=proposed, proposed_deferrals=proposed, actual_blocked_attempts=blocked)
        return initial, final, deepcopy(final)

    def verify(self, mode, snapshots):
        return V.validate_deferral(mode, *snapshots, selected={}, accepted={}, trigger={}, binding_sha='a'*64)

    def test_on_without_proposal_preserves_not_exercised(self):
        result = self.verify('on', self.snapshots())
        self.assertEqual(result['status'], 'NOT_EXERCISED'); self.assertFalse(result['actual_deferral_exercised'])

    def test_on_proposal_without_actual_block_never_qualifies(self):
        result = self.verify('on', self.snapshots(proposed=1))
        self.assertTrue(result['qualified_proposal_observed']); self.assertFalse(result['actual_deferral_exercised'])
        self.assertEqual(result['status'], 'NOT_EXERCISED')

    def test_shadow_without_proposal_cannot_advance(self):
        result = self.verify('shadow', self.snapshots(mode='shadow'))
        self.assertFalse(result['qualified_proposal_observed']); self.assertFalse(result['actual_deferral_exercised'])

    def test_off_bridge_must_remain_none(self):
        result = self.verify('off', (None, None, None)); self.assertTrue(result['off_bridge_is_none'])
        with self.assertRaises(ValueError): self.verify('off', self.snapshots())

    def test_original_migration_threshold_remains_strict(self):
        self.assertEqual(V.migration_gate('off', 21, limit_ns=20), (False, False))
        self.assertEqual(V.migration_gate('shadow', 20, limit_ns=20), (True, True))
        with self.assertRaises(ValueError): V.migration_gate('off', True, limit_ns=20)


if __name__ == '__main__':
    unittest.main()

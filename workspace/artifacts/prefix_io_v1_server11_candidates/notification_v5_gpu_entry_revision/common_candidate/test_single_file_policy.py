"""CPU protocol fixtures; never GPU evidence or native factory receipts."""
import importlib.util
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = (Path(os.environ['SERVER11_AUTHOR_SOURCE_ROOT']).resolve()
        if os.environ.get('SERVER11_AUTHOR_SOURCE_ROOT') else
        ROOT / 'artifacts/prefix_io_v1_server08_p4_cpu_extended_20261001/verified-v3/source')
PACKAGE = 'third_party/work/prefix-io-p4-02-cpu/src'
sys.path.insert(0, str(BASE / PACKAGE))
import prefix_io_control
prefix_io_control.__path__ = [str(HERE / 'source' / PACKAGE / 'prefix_io_control'),
                             *prefix_io_control.__path__]
from prefix_io_control.p4_policy import P4Policy, make_p4_policy
from prefix_io_control.p4_bridge import NativeP4Bridge
from prefix_io_control.p4_types import P4Config, WorkDescriptor, SystemSnapshot
from prefix_io_control.dispatch_shadow import ShadowState
from prefix_io_control.dispatch_budget import ZERO, Amount

spec = importlib.util.spec_from_file_location('_cpu_only_candidate_collector', HERE / 'native_full_step_collector.py')
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def fixture_receipt(cost=200, budget=100):
    # Deliberate test-only memory fixture; real installation can only call the
    # separately pinned native verifier factory. This object is not serialized.
    from prefix_io_control.p4_single_file_receipt import ExactSingleFileReceipt, FileRef
    value = object.__new__(ExactSingleFileReceipt)
    fields = dict(signature=('m', 'GPU-fixture', 'layout', 'kernel', 1, 1, 0, 144, 917504),
        cost_upper_ns=cost, step_budget_ns=budget, calibration_source_lock_sha256='0'*64,
        common_source_refs=(), runtime_common_refs=(), runtime_overlay_refs=(),
        cuda_event_source_sha256='1'*64, calibration_native_source_sha256='2'*64,
        binding_ref=FileRef('fixture.json', 1, '3'*64))
    for key, item in fields.items(): object.__setattr__(value, key, item)
    return value


class SingleFilePolicyTests(unittest.TestCase):
    def setUp(self):
        self.receipt = fixture_receipt()
        self.policy = P4Policy('r', P4Config('interference', 100, 1000, 100), single_file=self.receipt)
        self.state = (5, 10, 20, 1, 1, 0, 144)
        self.snapshot = SystemSnapshot('r', 1, 25,
            frozenset(('native_ready_work', 'conditional_single_file_live_step')),
            native_state=ShadowState('r', 25, ZERO, 10000000, 0, True),
            load_signature=self.receipt.signature)
        self.work = WorkDescriptor('r', 1, 0, 'preload:1:'+'a'*64, 'ssd_read',
            917504, 0, 5, False, minimum_unit_bytes=917504)

    def preview(self, **changes):
        return self.policy.issue_preview(changes.pop('work', self.work),
            changes.pop('snapshot', self.snapshot), now_ns=changes.pop('now_ns', 30),
            expected_epoch=1, _single_file_state=changes.pop('state', self.state), **changes)

    def test_exact_cost_defers_without_general_production(self):
        result = self.preview()
        self.assertEqual(result.action, 'defer')
        self.assertFalse(result.production_qualified)
        self.assertFalse(self.policy.production_interference_qualified)

    def test_no_generic_order_or_batch_permission(self):
        for result in (self.policy.choose(self.snapshot, (self.work,), now_ns=30, expected_epoch=1),
                       self.policy.choose_batch(self.snapshot, (self.work,), now_ns=30, expected_epoch=1)):
            self.assertEqual(result.action, 'native_fallback')

    def test_cost_below_predeclared_budget_issues(self):
        self.policy = P4Policy('r', P4Config('interference', 100, 1000, 100),
                              single_file=fixture_receipt(90, 100))
        self.assertEqual(self.preview().action, 'issue')

    def test_unknown_or_other_condition_falls_back(self):
        states = [None, (5, 10, 20, 1, 1, 0, 145), (5, 10, 20, 2, 2, 0, 144),
                  (5, 10, 20, 1, 1, 1, 144), (5, 10, 35, 1, 1, 0, 144)]
        for state in states:
            with self.subTest(state=state): self.assertEqual(self.preview(state=state).action, 'native_fallback')
        for work in (replace(self.work, nbytes=2*917504), replace(self.work, stage='h2d'),
                     replace(self.work, parent_id=1), replace(self.work, child_id='ordinary'),
                     replace(self.work, minimum_unit_bytes=1)):
            with self.subTest(work=work): self.assertEqual(self.preview(work=work).action, 'native_fallback')

    def test_existing_io_not_zero_and_wrong_signature_reject(self):
        nonzero = (Amount(1, 917504), *ZERO[1:])
        for snapshot in (replace(self.snapshot, native_state=replace(self.snapshot.native_state, inflight=nonzero)),
                         replace(self.snapshot, load_signature=('different',)),
                         replace(self.snapshot, capabilities=frozenset(('native_ready_work',)))):
            self.assertEqual(self.preview(snapshot=snapshot).action, 'native_fallback')

    def test_stale_explicit_cpu_mock_and_generation_reject(self):
        self.assertEqual(self.preview(now_ns=200).action, 'native_fallback')
        self.assertEqual(self.preview(execution='cpu_mock').action, 'native_fallback')
        self.assertEqual(self.preview(snapshot=replace(self.snapshot, snapshot_epoch=2)).action, 'native_fallback')

    def test_original_progress_and_age_override_remain(self):
        for progress in ('mandatory', 'continuation', 'shutdown', 'mandatory_support'):
            self.assertTrue(self.preview(progress=progress).progress_override)
        self.policy = P4Policy('r', P4Config('interference', 100, 20, 100), single_file=self.receipt)
        self.assertTrue(self.preview().progress_override)

    def test_receipt_cannot_change_mode_budget_or_generic_table(self):
        for config in (P4Config('joint', 100, 1000, 100), P4Config('interference', 100, 1000, 101)):
            with self.assertRaises(ValueError): P4Policy('r', config, single_file=self.receipt)
        with self.assertRaises(ValueError): P4Policy('r', P4Config('interference',100,1000,100),single_file={})
        self.assertIsNone(make_p4_policy('r', P4Config('off',100,1000)))

    def test_shadow_records_prediction_without_real_block(self):
        bridge = NativeP4Bridge(self.policy); bridge.bind()
        bridge._single_file_runtime = NS(signature_prefix=self.receipt.signature[:4])
        bridge._single_file_state = lambda: self.state  # explicit CPU-only fake
        result = bridge.preview_issue(self.work, self.snapshot, now_ns=30)
        self.assertEqual(result.action, 'native_fallback')
        self.assertEqual(bridge.single_file_deferrals, 1)
        self.assertEqual(bridge.single_file_blocked_attempts, 0)
        self.assertFalse(bridge.record_single_file_deferral(self.work.work_id, at_ns=31))

    def test_end_during_preview_and_before_block_restore_native(self):
        bridge = NativeP4Bridge(self.policy); bridge.bind(); bridge.single_file_shadow = False
        bridge._single_file_runtime = NS(signature_prefix=self.receipt.signature[:4])
        states = iter((self.state, None))
        bridge._single_file_state = lambda: next(states)
        self.assertEqual(bridge.preview_issue(self.work,self.snapshot,now_ns=30).action,'native_fallback')
        bridge._last_single_file_step = self.state; bridge._single_file_state = lambda: None
        self.assertFalse(bridge.record_single_file_deferral(self.work.work_id, at_ns=31))

    def test_actual_decisions_keep_first_last_and_bounded_identity(self):
        bridge = NativeP4Bridge(self.policy); bridge.bind(); bridge.single_file_shadow = False
        bridge._single_file_runtime = NS(signature_prefix=self.receipt.signature[:4])
        bridge._single_file_state = lambda: self.state
        self.assertEqual(bridge.preview_issue(self.work,self.snapshot,now_ns=30).action,'defer')
        self.assertTrue(bridge.record_single_file_deferral(self.work.work_id, at_ns=31))
        self.assertTrue(bridge.record_single_file_deferral(self.work.work_id, at_ns=34))
        row = bridge.single_file_decisions[0]
        self.assertEqual((row['first_defer_ns'],row['last_defer_ns'],row['count']), (31,34,2))
        self.assertEqual(bridge.single_file_blocked_attempts, 2)


class LivePreparedStateTests(unittest.TestCase):
    def setUp(self):
        self.capture = collector.FullStepCapture(run_id='cpu-test-only', origin='native_gpu_recording',
            selected_offsets=(16,), action=None)
        self.start = NS(record_before_ns=10, completed_query_ns=20)
        self.end = NS(record_before_ns=None)
        self.pending = dict(phase='prepared',ordinal=5,frame=NS(step_kind='decode',batch=1,
            active_decode=1,prefill_tokens=0,context_length=144))
        self.capture.observer = NS(enabled=True, events=NS(active=(5,self.start,self.end)),
            scalar=NS(enabled=True, _adapter=NS(_pending=self.pending)))

    def test_value_is_scalar_and_uses_original_query_time(self):
        self.assertEqual(self.capture.current_single_file_step(), (5,10,20,1,1,0,144))
        self.pending['phase']='awaiting_sample'
        self.assertEqual(self.capture.current_single_file_step()[2],20)

    def test_end_unknown_start_exception_and_detach_invalidate(self):
        self.start.completed_query_ns=None
        self.assertIsNone(self.capture.current_single_file_step())
        self.start.completed_query_ns=20; self.end.record_before_ns=21
        self.assertIsNone(self.capture.current_single_file_step())
        self.end.record_before_ns=None; self.capture.valid=False
        self.assertIsNone(self.capture.current_single_file_step())
        self.capture.valid=True; self.capture._detached=True
        self.assertIsNone(self.capture.current_single_file_step())

    def test_cpu_fixture_and_wrong_phase_never_create_live_value(self):
        self.capture.origin='cpu_fixture'
        self.assertIsNone(self.capture.current_single_file_step())
        self.capture.origin='native_gpu_recording'; self.pending['phase']='begun'
        self.assertIsNone(self.capture.current_single_file_step())


if __name__ == '__main__': unittest.main()

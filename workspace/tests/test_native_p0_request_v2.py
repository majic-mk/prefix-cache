"""P0 driver CPU orchestration; fake model/transport/costs are not GPU evidence."""
from dataclasses import replace
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from probekv.native_p0_request_v2 import execute_p0_request, P0RequestFailure
from probekv.v8_schema10_native_adapter import NativeRequestContext
from probekv.v8_schema10_measured_costs import MeasuredRequestCostProvider
from probekv.v8_schema10_cost_provider import EXECUTION_SHAPE_KEY, UnsupportedTimelineCost
from probekv.v8_schema6_planner import DeterministicJointTimelineEstimator
from tests import test_native_p0_operation_v2 as operation_fixtures
from tests import test_native_comparison_publication_v2 as publication_fixtures
from tests.test_native_consumption_v2 import CPUTransport
from probekv.native_p0_operation_v2 import P0ComparisonActionV2
from probekv.source_manifest_v2 import request_input_digest
from probekv.v8_schema6_hbm import UnifiedHBMReservationManager
from contextlib import ExitStack


class NativeP0RequestTests(unittest.TestCase):
    def setUp(self):
        self.f = operation_fixtures.NativeP0OperationTests()
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)

    def setup_request(self, *, costs=False):
        store, c, snap, profile, action = self.f.setup_action()
        c.engine.session.current_layer = 0
        c.arrival_ns = time.perf_counter_ns()
        c.adapter.costs = None
        c.commit_reuse = NativeRequestContext.commit_reuse.__get__(c)
        c.advance_to_depth = lambda d: setattr(c.engine.session, 'current_layer', d)
        c.configure_target_capture_v2 = Mock(return_value={'cpu_test_capture': True})
        c.finish_calls = 0
        def finish(callback):
            c.finish_calls += 1
            callback()
            c.finished = True
            return {'token_ids': [1, 2], 'answer': 'CPU fake answer', 'quality_passed': None}
        c.finish = finish
        if costs:
            # Deliberately bypass disk constructor in CPU tests only. No fake
            # rows are written as real measurements; planner math is exercised.
            provider = MeasuredRequestCostProvider.__new__(MeasuredRequestCostProvider)
            provider.key_contract = EXECUTION_SHAPE_KEY
            provider.sha = 'a'*64
            provider.dense_reference = Mock(return_value=100000.)
            provider.joint_estimator = Mock(return_value=DeterministicJointTimelineEstimator(
                base_future_ms=1., dense_cost_ms_by_segment={'C': 100000.}, reuse_cost_ms_by_segment={'C': 1.}))
            c.adapter.costs = provider
        return store, c, snap, profile, (action,)

    def run_request(self, store, c, snap, profile, actions):
        with patch.object(c, 'configure_cuda_comparison_v2',
                          side_effect=lambda **kw: self.f.cpu_workspace(c, **kw)), \
                patch('probekv.native_consumption_v2.PhysicalLayerwiseSourceLoader', CPUTransport):
            return execute_p0_request(c, store, snap, profile, actions, host_capture_bytes=100000)

    def test_prepared_missing_costs_dense_answer_and_cleanup_not_fake_commit(self):
        store, c, snap, p, actions = self.setup_request()
        result = self.run_request(store, c, snap, p, actions)
        self.assertEqual(result['status'], 'COMPLETED')
        self.assertEqual(result['final_admission']['status'], 'UNSUPPORTED')
        self.assertFalse(result['committed_boundaries'])
        self.assertEqual(result['answer']['token_ids'], [1, 2])
        self.assertEqual(c.finish_calls, 1)
        self.assertFalse(result['native_runtime_qualified'])
        self.assertTrue(result['cleanup']['passed'])
        self.assertEqual(c.adapter.hbm.active_reserved_bytes, 0)
        self.assertNotIn(snap.snapshot_id, store._snapshots)
        self.assertGreaterEqual(result['total_service_ms'], result['timing']['ttft_ms'])

    def test_measured_planner_accepts_then_actual_context_commit_once(self):
        store, c, snap, p, actions = self.setup_request(costs=True)
        engine = c.engine
        result = self.run_request(store, c, snap, p, actions)
        self.assertEqual(result['final_admission']['accepted'], ['C'])
        self.assertEqual(result['committed_boundaries'], {'C': 2})
        self.assertEqual(len(engine.commit_calls), 1)
        self.assertEqual(engine.commit_calls[0]['repair_positions'], (2,))
        self.assertEqual(result['extra_forward_count'], 0)

    def test_missing_joint_cell_is_unsupported_and_dense(self):
        store, c, snap, p, actions = self.setup_request(costs=True)
        c.adapter.costs.joint_estimator.side_effect = UnsupportedTimelineCost('unmeasured shape')
        result = self.run_request(store, c, snap, p, actions)
        self.assertEqual(result['final_admission']['status'], 'UNSUPPORTED')
        self.assertFalse(result['committed_boundaries'])
        self.assertEqual(c.finish_calls, 1)

    def test_missing_matched_dense_reference_never_substitutes_zero(self):
        store, c, snap, p, actions = self.setup_request(costs=True)
        c.adapter.costs.dense_reference.return_value = None
        result = self.run_request(store, c, snap, p, actions)
        self.assertEqual(result['final_admission']['reason'], 'missing_matched_dense_reference')
        c.adapter.costs.joint_estimator.assert_not_called()

    def test_cost_rejection_does_not_trigger_growth(self):
        store, c, snap, p, actions = self.setup_request(costs=True)
        c.adapter.costs.dense_reference.return_value = .00001
        result = self.run_request(store, c, snap, p, actions)
        self.assertFalse(result['committed_boundaries'])
        self.assertEqual(store.storage_audit()['source_count'], 1)
        self.assertEqual(result['publication']['status'], 'SKIPPED')

    def test_readonly_rejects_repair_change_or_nonfresh_context_before_execution(self):
        store, c, snap, p, actions = self.setup_request()
        for name, value in (('repair_ratio', 1.), ('arrival_ns', 0)):
            old = getattr(c, name); setattr(c, name, value)
            with self.assertRaises(ValueError): self.run_request(store, c, snap, p, actions)
            setattr(c, name, old)
        c.configure_target_capture_v2.assert_not_called()
        self.assertFalse(c.closed)
        self.assertIn(snap.snapshot_id, store._snapshots)

    def test_legacy_cost_contract_rejected_before_cuda(self):
        store, c, snap, p, actions = self.setup_request(costs=True)
        c.adapter.costs.key_contract = 'legacy_identity_v1'
        with self.assertRaises(ValueError): self.run_request(store, c, snap, p, actions)
        c.configure_target_capture_v2.assert_not_called()

    def test_bad_frozen_candidate_action_not_replaced_by_live_lookup(self):
        store, c, snap, p, actions = self.setup_request()
        with self.assertRaises(ValueError):
            self.run_request(store, c, snap, p, (replace(actions[0], source_ids=()),))
        self.assertFalse(c.closed)

    def test_cold_content_miss_needs_no_current_projection_and_stays_dense(self):
        _, c, _, p, actions = self.setup_request()
        store = self.f.store(name='empty')
        snap = store.begin_request(c.request['request_id'])
        action = replace(actions[0], source_ids=(), snapshot_id=snap.snapshot_id)
        with patch.object(c, 'observe_current_k', side_effect=AssertionError('miss needs no projection')):
            result = self.run_request(store, c, snap, p, (action,))
        self.assertEqual(result['comparison_receipts']['C']['stored_ids'], ())
        self.assertEqual(result['final_admission']['reason'], 'no_prepared_compatible_source')
        self.assertFalse(result['selected_sources'])

    def test_missing_first_token_retains_answer_but_refuses_publication(self):
        store, c, snap, p, actions = self.setup_request()
        def finish(callback):
            c.finished = True
            return {'answer': 'completed but missing endpoint'}
        c.finish = finish
        with self.assertRaises(P0RequestFailure) as caught:
            self.run_request(store, c, snap, p, actions)
        audit = caught.exception.audit
        self.assertEqual(audit['answer']['answer'], 'completed but missing endpoint')
        self.assertIsNone(audit['publication'])
        self.assertTrue(audit['cleanup']['passed'])

    def test_model_failure_preserves_partial_evidence_and_releases(self):
        store, c, snap, p, actions = self.setup_request()
        c.finish = Mock(side_effect=RuntimeError('model failed'))
        with self.assertRaises(P0RequestFailure) as caught:
            self.run_request(store, c, snap, p, actions)
        audit = caught.exception.audit
        self.assertIn('C', audit['comparison_receipts'])
        self.assertIsNone(audit['publication'])
        self.assertTrue(audit['cleanup']['passed'])
        self.assertNotIn(snap.snapshot_id, store._snapshots)

    def test_failed_cleanup_quarantines_snapshot_and_reports_answer(self):
        store, c, snap, p, actions = self.setup_request()
        c.synchronize.side_effect = RuntimeError('device lost')
        with self.assertRaises(P0RequestFailure) as caught:
            self.run_request(store, c, snap, p, actions)
        audit = caught.exception.audit
        self.assertFalse(audit['cleanup']['passed'])
        self.assertIn(snap.snapshot_id, store._snapshots)
        self.assertTrue(c.source_consumption_v2.quarantined)
        self.assertGreater(c.adapter.hbm.active_reserved_bytes, 0)
        # Explicit test-only recovery; never a real failed-CUDA recovery policy.
        c.synchronize.side_effect = None
        c.source_consumption_v2.quarantined = False
        c.comparison_workspace_v2.lease.quarantined = False
        c.close(); store.end_request(snap)

    def test_stale_snapshot_is_dense_without_unregistered_retry(self):
        store, c, snap, p, actions = self.setup_request(costs=True)
        def changed(context):
            context.generation += 1
            return DeterministicJointTimelineEstimator(base_future_ms=1.,
                dense_cost_ms_by_segment={'C': 100000.}, reuse_cost_ms_by_segment={'C': 1.})
        c.adapter.costs.joint_estimator.side_effect = changed
        result = self.run_request(store, c, snap, p, actions)
        self.assertEqual(result['final_admission']['reason'], 'stale_planner_snapshot')
        self.assertEqual(c.adapter.costs.joint_estimator.call_count, 1)
        self.assertFalse(result['committed_boundaries'])

    def captured_request(self):
        f = publication_fixtures.NativeComparisonPublicationTests()
        f.setUp(); self.addCleanup(f.doCleanups)
        store = f.fixture.store()
        c, model, session = f.context()
        capture = c.source_capture_v2
        c.source_capture_v2 = None
        c.configure_target_capture_v2 = lambda **kw: (setattr(c, 'source_capture_v2', capture) or capture.audit())
        c.adapter.depths = (1, 2)
        c.adapter.spec.checkpoints = (1, 2)
        c.adapter.provenance['tokenizer_hash'] = 'tokenizer'
        c.adapter.hbm = UnifiedHBMReservationManager(allocator_capacity_bytes=300000, safety_bytes=0)
        c.adapter.check_deadline = Mock()
        c.adapter.costs = None
        c.adapter.native_repair_metric = 'normalized_kv_deviation'
        c.adapter.inner.old_kvs = []
        for layer in model.layers:
            layer.self_attn.hack_kv = []
        c.closed = c.selection_closed = False
        c.probe_fallback_reason = None
        c.prepared = {}; c.frozen = {}; c.supports = {}; c.replica_reservations = {}
        c.hot_replicas = {}; c.hot_leases = ExitStack()
        c.workspace = c.source_capture_workspace_v2 = None
        c.source_consumption_v2 = None
        c.repair_ratio = .15
        c.actual_repair_check_sunk_ms = 0.
        c.native = NS(sequence=NS(seq_id=1))
        c.arrival_ns = time.perf_counter_ns()
        c.engine.source_loader = NS(pool=object(), device='cuda', integrity_mode='online_immutable')
        c.engine.prefetch_window = 0; c.engine.tickets = {}
        snapshot = store.begin_request(c.request['request_id'])
        p = f.profile()
        action = P0ComparisonActionV2(c.request['request_id'], 'C', snapshot.snapshot_id,
            request_input_digest(c.request['token_ids'], tuple(range(len(c.request['token_ids'])))),
            1, (), p.binding_digest, 100000, 100000)
        self.addCleanup(c.close)
        return store, c, snapshot, p, (action,), model

    def test_T18_T19_actual_capture_publishes_post_answer_for_next_snapshot_only(self):
        store, c, snapshot, p, actions, model = self.captured_request()
        original_commit = store.commit_publication
        visibility = []
        def commit(*args, **kwargs):
            self.assertTrue(c.finished)
            result = original_commit(*args, **kwargs)
            visibility.append(store.lookup(snapshot, (2, 3, 4)))
            return result
        with patch.object(store, 'commit_publication', side_effect=commit):
            result = self.run_request(store, c, snapshot, p, actions)
        self.assertEqual(result['publication']['targets']['C']['status'], 'PUBLISHED')
        self.assertEqual(visibility, [()])
        self.assertEqual(model.real_block_calls, 3)
        self.assertEqual([b.self_attn.qkv_proj.calls for b in model.layers], [1, 1, 1])
        reader = store.begin_request('future')
        rows = store.lookup(reader, (2, 3, 4))
        self.assertEqual(len(rows), 1)
        with store.leased_target(reader, rows[0]['source_id']) as layers:
            self.assertEqual(tuple(layers[0][0].shape), (3, 1, 2))
        store.end_request(reader)
        self.assertEqual(result['extra_forward_count'], 0)
        self.assertEqual(store.storage_audit()['prefix_shadow_bytes'], 0)

    def test_post_answer_uncertain_publication_preserves_output_and_stops_next_work(self):
        store, c, snapshot, p, actions, model = self.captured_request()
        with patch.object(store, 'commit_publication', side_effect=OSError('uncertain durable write')):
            result = self.run_request(store, c, snapshot, p, actions)
        self.assertEqual(result['status'], 'RECOVERY_REQUIRED')
        self.assertEqual(result['publication']['targets']['C']['status'], 'COMMIT_UNCERTAIN')
        self.assertIsNotNone(result['answer'])
        self.assertEqual(model.real_block_calls, 3)
        self.assertTrue(result['cleanup']['passed'])

    def test_complete_request_inventory_includes_nonselected_dense_segment(self):
        store, c, snap, p, actions = self.setup_request(costs=True)
        c.segments['D'] = dict(segment_id='D', positions=(4,), token_ids=(14,))
        c.execution_inventory['D'] = NS(comparison_eligible=True, remaining_positions=(4,))
        estimator = DeterministicJointTimelineEstimator(base_future_ms=1.,
            dense_cost_ms_by_segment={'C': 100000., 'D': 7.}, reuse_cost_ms_by_segment={'C': 1.})
        estimator.estimate = Mock(wraps=estimator.estimate)
        c.adapter.costs.joint_estimator.return_value = estimator
        result = self.run_request(store, c, snap, p, actions)
        query = estimator.estimate.call_args_list[0].args[0]
        self.assertEqual(query.inventory_segment_ids, ('C', 'D'))
        self.assertEqual(query.dense_fallback_segment_ids, ('D',))
        self.assertEqual(result['final_admission']['accepted'], ['C'])


if __name__ == '__main__': unittest.main()

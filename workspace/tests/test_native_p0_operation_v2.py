"""Bounded native operation orchestration using explicit CPU test transports."""
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

import torch

from probekv.cuda_comparison_v2 import CudaComparisonWorkspaceV2, ComparisonReservationV2
from probekv.native_p0_operation_v2 import (P0ComparisonActionV2,
    validate_p0_comparison_action,execute_p0_comparison_preparation)
from probekv.source_manifest_v2 import request_input_digest
from tests import test_native_consumption_v2 as fixtures
from tests.test_cuda_comparison_v2 import MockEvent


class NativeP0OperationTests(unittest.TestCase):
    setUp=fixtures.NativeV2ConsumptionTests.setUp
    close_stores=fixtures.NativeV2ConsumptionTests.close_stores
    store=fixtures.NativeV2ConsumptionTests.store
    candidate=fixtures.NativeV2ConsumptionTests.candidate
    plan=fixtures.NativeV2ConsumptionTests.plan
    publish=fixtures.NativeV2ConsumptionTests.publish
    read_rows=fixtures.NativeV2ConsumptionTests.read_rows
    context=fixtures.NativeV2ConsumptionTests.context

    def setup_action(self):
        store=self.store();candidate=self.candidate('birth');source=self.publish(store,candidate)
        c,issuer=self.context(store,candidate,capacity=300000)
        snapshot,profile=issuer.snapshot,issuer.profile
        issuer.close()
        c.adapter.depths=(1,2);c.adapter.check_deadline=Mock()
        c.commit_reuse=Mock(side_effect=AssertionError('operation must not bypass FinalCommit'))
        action=P0ComparisonActionV2('consumer','C',snapshot.snapshot_id,
            request_input_digest(c.request['token_ids'],tuple(range(len(c.request['token_ids'])))),
            1,(source['source_id'],),profile.binding_digest,100000,100000)
        return store,c,snapshot,profile,action

    def cpu_workspace(self,c,*,capacity_bytes):
        # Deliberately bypass actual constructor; this cannot qualify CUDA.
        w=CudaComparisonWorkspaceV2.__new__(CudaComparisonWorkspaceV2)
        w.context=c;w.request_id=c.request['request_id'];w.device=torch.device('cpu')
        w.input_digest=request_input_digest(c.request['token_ids'],tuple(range(len(c.request['token_ids']))))
        w.lease=ComparisonReservationV2(c.adapter.hbm,w.request_id,capacity_bytes)
        w.events=[];w._active=False;w._failed=False;w.required_bytes=Mock(return_value=1000)
        w.torch=NS(bfloat16=torch.bfloat16,cuda=NS(device=lambda d:nullcontext(),Event=MockEvent,
            current_stream=lambda d:None,synchronize=Mock()))
        c.engine.session._clear_observation=Mock()
        c.comparison_workspace_v2=w
        return w

    def execute(self,store,c,snapshot,profile,action):
        with patch.object(c,'configure_cuda_comparison_v2',side_effect=lambda **kw:self.cpu_workspace(c,**kw)), \
                patch('probekv.native_consumption_v2.PhysicalLayerwiseSourceLoader',fixtures.CPUTransport):
            result=execute_p0_comparison_preparation(c,store,snapshot,profile,action)
        self.addCleanup(result['issuer'].close)
        return result

    def test_bounded_operation_compares_freezes_prepares_but_never_commits(self):
        store,c,snapshot,profile,action=self.setup_action()
        result=self.execute(store,c,snapshot,profile,action)
        self.assertEqual(result['audit']['outcome'],'PREPARED_NOT_COMMITTED')
        self.assertEqual(result['ticket'].source_id,action.source_ids[0])
        self.assertFalse(result['audit']['reuse_committed'])
        self.assertFalse(result['audit']['native_runtime_qualified'])
        self.assertEqual(result['audit']['extra_forward_count'],0)
        c.commit_reuse.assert_not_called()
        self.assertEqual(len(store.lookup(snapshot,[12,13])),1)
        c.close()
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertFalse(any(store._lease_counts.values()))

    def test_incompatible_winner_does_not_prepare_or_publish(self):
        store,c,snapshot,profile,action=self.setup_action()
        profile=replace(profile,tau_reuse=0.,tau_add=0.)
        action=replace(action,profile_binding_digest=profile.binding_digest)
        c.engine.session.observe_pre_rope_k=lambda d:torch.zeros((6,2,2),dtype=torch.bfloat16)
        result=self.execute(store,c,snapshot,profile,action)
        self.assertEqual(result['audit']['outcome'],'DENSE_REQUIRED')
        self.assertEqual(result['audit']['reason'],'no_compatible_winner')
        self.assertIsNone(result['ticket'])
        self.assertFalse(c.source_consumption_v2.loader.calls)
        self.assertEqual(len(store.lookup(snapshot,[12,13])),1)
        c.commit_reuse.assert_not_called()

    def test_wrong_input_pool_snapshot_or_profile_fails_before_model_or_cuda(self):
        store,c,snapshot,profile,action=self.setup_action()
        for changes in (dict(request_id='other'),dict(request_input_digest='f'*64),
                        dict(source_ids=('unknown',)),dict(snapshot_id='wrong'),
                        dict(profile_binding_digest='e'*64),dict(completed_depth=2)):
            with self.subTest(changes=changes), patch.object(c,'advance_to_depth',side_effect=AssertionError()), \
                    patch.object(c,'configure_cuda_comparison_v2',side_effect=AssertionError()):
                with self.assertRaises(ValueError):
                    execute_p0_comparison_preparation(c,store,snapshot,profile,replace(action,**changes))
        self.assertFalse(c.closed)
        self.assertFalse(c.adapter.hbm.active_reserved_bytes)

    def test_action_cannot_expand_candidate_count_or_execution_mode(self):
        _,_,_,_,action=self.setup_action()
        for changes in (dict(source_ids=('a',)*2),
                        dict(source_ids=('a','b','c','d','e')),dict(mode='full_sweep'),
                        dict(completed_depth=0),dict(cuda_workspace_bytes=0)):
            with self.subTest(changes=changes),self.assertRaises(ValueError):replace(action,**changes)

    def test_legacy_prefix_full_capture_and_existing_preparation_rejected(self):
        store,c,snapshot,profile,action=self.setup_action()
        for key,value in (('capture_original_full_prefill',True),('publish_exact_prefix_shadow',True),
                          ('prefetch_window',2),('use_gpu_hot_cache',True)):
            c.request[key]=value
            with self.assertRaises(ValueError):validate_p0_comparison_action(c,store,snapshot,profile,action)
            c.request.pop(key)
        c.frozen['C']='already-selected'
        with self.assertRaises(ValueError):validate_p0_comparison_action(c,store,snapshot,profile,action)

    def test_missing_selection_state_is_dense_not_growth_or_full_kv_fallback(self):
        store,c,snapshot,profile,action=self.setup_action()
        with patch.object(store,'read_selection',side_effect=KeyError('missing depth')):
            result=self.execute(store,c,snapshot,profile,action)
        self.assertEqual(result['audit']['reason'],'incomplete_comparison')
        self.assertFalse(c.source_consumption_v2.loader.calls)
        self.assertIsNone(result['ticket'])
        with self.assertRaises(ValueError):result['issuer'].publication_scope(result['receipt'])

    def test_mid_operation_failure_runs_cleanup_without_retry_or_receipt(self):
        store,c,snapshot,profile,action=self.setup_action()
        with patch.object(c,'configure_cuda_comparison_v2',side_effect=lambda **kw:self.cpu_workspace(c,**kw)), \
                patch('probekv.native_consumption_v2.PhysicalLayerwiseSourceLoader',fixtures.CPUTransport), \
                patch('probekv.selection_comparison.residual_scores',side_effect=RuntimeError('kernel failed')):
            with self.assertRaises(RuntimeError):execute_p0_comparison_preparation(c,store,snapshot,profile,action)
        self.assertTrue(c.closed)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertFalse(any(store._lease_counts.values()))
        self.assertEqual(len(store.lookup(snapshot,[12,13])),1)
        c.commit_reuse.assert_not_called()


if __name__=='__main__':unittest.main()

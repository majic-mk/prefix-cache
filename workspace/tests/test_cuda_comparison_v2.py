"""CPU lifecycle and mocked CUDA orchestration, NOT real CUDA evidence."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import unittest

import torch

from probekv.cuda_comparison_v2 import (ComparisonReservationV2,
    CudaComparisonWorkspaceV2, comparison_workspace_estimate)
from probekv.source_comparison_v2 import ComparisonSessionV2
from probekv.source_manifest_v2 import request_input_digest
from probekv.v8_schema6_hbm import UnifiedHBMReservationManager
from probekv.v8_schema10_native_adapter import NativeRequestContext
from tests import test_source_comparison_v2 as fixtures


class WorkspaceOwnershipTests(unittest.TestCase):
    def manager(self,capacity=1000):
        return UnifiedHBMReservationManager(allocator_capacity_bytes=capacity,safety_bytes=0)

    def test_estimate_includes_whole_projection_not_just_target_k(self):
        small=comparison_workspace_estimate(active_rows=100,hidden_width=32,query_width=32,kv_width=8,target_rows=10)
        large=comparison_workspace_estimate(active_rows=200,hidden_width=32,query_width=32,kv_width=8,target_rows=10)
        self.assertGreater(large,small)
        self.assertGreater(small,10*8*2)
        for change in (dict(target_rows=101),dict(active_rows=True),dict(kv_width=0)):
            args=dict(active_rows=100,hidden_width=32,query_width=32,kv_width=8,target_rows=10)
            args.update(change)
            with self.assertRaises(ValueError):comparison_workspace_estimate(**args)

    def test_shared_manager_idempotent_reservation_survives_scalar_result(self):
        manager=self.manager();lease=ComparisonReservationV2(manager,'request',800)
        r=lease.acquire(500)
        self.assertIs(lease.acquire(600),r)
        self.assertEqual(manager.active_reserved_bytes,800)
        events=[]
        lease.close(fence=lambda:events.append(('fence',manager.active_reserved_bytes)),
                    clear_observations=lambda:events.append(('clear',manager.active_reserved_bytes)))
        self.assertEqual(events,[('fence',800),('clear',800)])
        self.assertEqual(manager.active_reserved_bytes,0)
        lease.close(fence=Mock(side_effect=AssertionError()),clear_observations=Mock(side_effect=AssertionError()))
        with self.assertRaises(RuntimeError):lease.acquire(1)

    def test_capacity_and_shared_headroom_reject_before_allocation(self):
        manager=self.manager()
        lease=ComparisonReservationV2(manager,'request',800)
        with self.assertRaises(MemoryError):lease.acquire(801)
        self.assertEqual(manager.active_reserved_bytes,0)
        lease.acquire(500)
        other=ComparisonReservationV2(manager,'other',500)
        with self.assertRaises(MemoryError):other.acquire(500)
        self.assertIsNone(other.reservation)
        self.assertEqual(manager.active_reserved_bytes,800)

    def test_owner_kind_or_size_mutation_is_not_a_valid_lease(self):
        for field,value in (('owner_request_id','other'),('bytes',1),('released',True)):
            lease=ComparisonReservationV2(self.manager(),'request',800)
            r=lease.acquire(500);setattr(r,field,value)
            with self.assertRaises(ValueError):lease.acquire(500)

    def test_failed_fence_or_reference_clear_keeps_reservation_quarantined(self):
        for failure in ('fence','clear'):
            manager=self.manager();lease=ComparisonReservationV2(manager,'request',800)
            lease.acquire(500)
            fence=Mock(side_effect=RuntimeError('device failed') if failure=='fence' else None)
            clear=Mock(side_effect=RuntimeError('owner failed') if failure=='clear' else None)
            with self.assertRaises(RuntimeError):lease.close(fence=fence,clear_observations=clear)
            self.assertTrue(lease.quarantined)
            self.assertEqual(manager.active_reserved_bytes,800)
            if failure=='fence':clear.assert_not_called()
            with self.assertRaises(RuntimeError):lease.close(fence=Mock(),clear_observations=Mock())

    def test_real_constructor_rejects_cpu_without_touching_cuda(self):
        context=NativeRequestContext.__new__(NativeRequestContext)
        context.engine=NS(session=NS(hidden_states=torch.ones(2,4,dtype=torch.bfloat16)))
        context.closed=context.finished=False
        context.adapter=NS(torch=torch)
        with patch.object(torch.cuda,'Event',side_effect=AssertionError('no GPU allowed')):
            with self.assertRaisesRegex(ValueError,'real native BF16 CUDA'):
                CudaComparisonWorkspaceV2(context,capacity_bytes=1000)

    def test_close_does_not_release_a_replaced_reservation(self):
        manager=self.manager();lease=ComparisonReservationV2(manager,'request',800)
        r=lease.acquire(500)
        manager.reservations[r.reservation_id]=NS(bytes=800,released=False)
        fence=Mock();clear=Mock()
        with self.assertRaises(ValueError):lease.close(fence=fence,clear_observations=clear)
        self.assertTrue(lease.quarantined)
        fence.assert_not_called();clear.assert_not_called()
        self.assertFalse(manager.reservations[r.reservation_id].released)


class MockEvent:
    fail_fence=False
    elapsed=1.25
    def __init__(self,**kwargs):pass
    def record(self,stream):pass
    def synchronize(self):
        if self.fail_fence:raise RuntimeError('event fence failed')
    def elapsed_time(self,end):return self.elapsed


class MockedCudaComparisonTests(unittest.TestCase):
    # Borrow only fixture helpers; do not rediscover another TestCase's tests.
    setUp=fixtures.SourceComparisonReceiptTests.setUp
    close_stores=fixtures.SourceComparisonReceiptTests.close_stores
    store=fixtures.SourceComparisonReceiptTests.store
    candidate=fixtures.SourceComparisonReceiptTests.candidate
    plan=fixtures.SourceComparisonReceiptTests.plan
    publish=fixtures.SourceComparisonReceiptTests.publish
    read_rows=fixtures.SourceComparisonReceiptTests.read_rows
    profile=fixtures.SourceComparisonReceiptTests.profile
    context=fixtures.SourceComparisonReceiptTests.context
    setup_session=fixtures.SourceComparisonReceiptTests.setup_session

    def configured(self,count=2):
        store,snapshot,old,c=self.setup_session(count=count)
        old.close()
        c.closed=False;c._observation={}
        c.adapter.hbm=UnifiedHBMReservationManager(allocator_capacity_bytes=200000,safety_bytes=0)
        c.engine=NS(session=NS(_clear_observation=Mock()))
        # Explicit test-only object construction and CPU transport. The actual
        # constructor rejects these tensors; no GPU report is emitted here.
        w=CudaComparisonWorkspaceV2.__new__(CudaComparisonWorkspaceV2)
        w.context=c;w.request_id=c.request['request_id'];w.device=torch.device('cpu')
        w.input_digest=request_input_digest(c.request['token_ids'],tuple(range(len(c.request['token_ids']))))
        w.lease=ComparisonReservationV2(c.adapter.hbm,w.request_id,100000)
        w.events=[];w._active=False;w._failed=False
        w.required_bytes=Mock(return_value=50000)
        w.torch=NS(bfloat16=torch.bfloat16,cuda=NS(device=lambda d:nullcontext(),Event=MockEvent,
            current_stream=lambda d:None,synchronize=Mock()))
        c.comparison_workspace_v2=w
        issuer=ComparisonSessionV2(store,snapshot,self.profile(),workspace_bytes=100000,cuda_workspace=w)
        self.addCleanup(issuer.close)
        return store,c,issuer,w

    def test_same_arithmetic_with_reservation_before_observation_and_fenced_receipt(self):
        store,c,issuer,w=self.configured()
        current=c.observe_current_k.return_value
        def observe(*args):
            self.assertEqual(c.adapter.hbm.active_reserved_bytes,100000)
            return current
        c.observe_current_k.side_effect=observe
        receipt=issuer.compare_native(c,'target')
        self.assertEqual(receipt.counts,dict(stored=2,eligible=2,available=2,compared=2))
        self.assertEqual(len(receipt.measurement_digest),64)
        self.assertEqual(w.events[-1]['full_source_kv_bytes'],0)
        self.assertEqual(w.events[-1]['selection_state_h2d_bytes'],2*current.numel()*2)
        self.assertEqual(w.events[-1]['current_k_digest_d2h_bytes'],current.numel()*2)
        self.assertEqual(w.events[-1]['cuda_envelope_ms'],1.25) # MOCK, not hardware timing.
        self.assertGreaterEqual(receipt.host_ms,w.events[-1]['host_ms'])
        self.assertFalse(w.audit()['native_runtime_qualified'])
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,100000)
        w.close()
        c.engine.session._clear_observation.assert_called_once()
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_content_miss_does_not_reserve_project_transfer_or_time_cuda(self):
        _,c,issuer,w=self.configured(count=0)
        receipt=issuer.compare_native(c,'target')
        self.assertIsNone(receipt.measurement_digest)
        self.assertIsNone(receipt.current_k_digest)
        self.assertEqual(w.events,[])
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        c.observe_current_k.assert_not_called()

    def test_failed_final_fence_revokes_even_computed_receipt_and_quarantines(self):
        _,c,issuer,w=self.configured()
        with patch.object(MockEvent,'fail_fence',True):
            with self.assertRaises(RuntimeError):issuer.compare_native(c,'target')
        self.assertEqual(issuer._issued,{})
        self.assertTrue(w.lease.quarantined)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,100000)
        with self.assertRaises(RuntimeError):issuer.compare_native(c,'target')
        with self.assertRaises(RuntimeError):w.close()

    def test_arithmetic_failure_fences_and_does_not_authorize_growth(self):
        _,c,issuer,w=self.configured()
        with patch('probekv.selection_comparison.residual_scores',side_effect=RuntimeError('kernel failed')):
            with self.assertRaises(RuntimeError):issuer.compare_native(c,'target')
        w.torch.cuda.synchronize.assert_called_once()
        self.assertEqual(issuer._issued,{})
        self.assertEqual(w.events[-1]['outcome'],'FAILED')
        w.close()
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_wrong_request_and_changed_input_rejected_before_projection(self):
        _,c,issuer,w=self.configured()
        c.request['token_ids'][0]+=1
        with self.assertRaises(ValueError):issuer.compare_native(c,'target')
        c.observe_current_k.assert_not_called()
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_invalid_event_time_never_becomes_measurement_or_receipt(self):
        _,c,issuer,w=self.configured()
        with patch.object(MockEvent,'elapsed',float('nan')):
            with self.assertRaises(RuntimeError):issuer.compare_native(c,'target')
        self.assertEqual(issuer._issued,{})
        self.assertTrue(w.lease.quarantined)

    def test_budget_failure_precedes_current_projection(self):
        _,c,issuer,w=self.configured()
        w.required_bytes.return_value=100001
        with self.assertRaises(MemoryError):issuer.compare_native(c,'target')
        c.observe_current_k.assert_not_called()
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_current_device_validation_precedes_gpu_finite_arithmetic(self):
        _,c,issuer,w=self.configured()
        with patch.object(w,'validate_current',side_effect=ValueError('wrong device')), \
                patch.object(torch,'isfinite',side_effect=AssertionError('no wrong-device kernel')):
            with self.assertRaisesRegex(ValueError,'wrong device'):issuer.compare_native(c,'target')
        self.assertEqual(issuer._issued,{})
        w.close()


if __name__=='__main__':unittest.main()

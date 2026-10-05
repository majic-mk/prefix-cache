"""CPU fake transport + real file/receipt contracts; not real GPU execution."""
from contextlib import ExitStack
from dataclasses import replace, asdict
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from tests import test_source_store_v2 as fixtures
from probekv.cacheblend_v6_online_engine import LayerwiseLoadTicket
from probekv.native_source_capture_v2 import NativeTargetCaptureV2
from probekv.source_comparison_v2 import (ComparisonProfileBindingV2, ComparisonSessionV2,
    scorer_digest, runtime_binding_digest)
from probekv.source_provenance_v2 import KVImport, SourceOrigin
from probekv.v8_schema10_native_adapter import NativeRequestContext
from probekv.v8_schema6_hbm import UnifiedHBMReservationManager, HBMReservationKind


class CPUEvent:
    def query(self): return True
    def synchronize(self): pass


class CPUTransport:
    """Explicit test replacement; never allocates a CUDA tensor or event."""
    def __init__(self, pool, *, authorize, integrity_mode, device):
        self.pool, self.authorize = pool, authorize
        self.integrity_mode, self.device = integrity_mode, device
        self.calls = []
        self.fail = False

    def begin(self, **kw):
        self.calls.append(kw)
        layers = kw['canonical_layers']
        self.authorize(source_id=kw['source_id'], segment_id=kw['segment_id'],
                       bytes_required=layers.full_kv_bytes)
        if self.fail: raise RuntimeError('injected preparation failure')
        pairs = {i+1: layers[i] for i in range(len(layers))}
        return LayerwiseLoadTicket(kw['segment_id'], kw['source_id'], 0., layers.full_kv_bytes,
            pairs, CPUEvent(), {i:CPUEvent() for i in pairs}, '', '', tuple(kw['segment_positions']),
            expected_artifact_digest=kw['expected_artifact_digest'],
            expected_layer_count=len(layers), integrity_mode='online_immutable')


class CPUEngine:
    def __init__(self, current):
        self.source_loader = NS(pool=object(), device='cuda', integrity_mode='online_immutable')
        self.prefetch_window = 0
        self.tickets = {}
        self._source_row_indices = {}
        self.session = NS(current_layer=1, commits={}, source_handles={},
            active_positions=tuple(range(6)),
            observe_pre_rope_k=lambda d: current,
            observe_repair_check_pre_rope_kv=lambda d:(current,current+17))
        self.commit_calls = []

    def start_winner_prefetch(self, **kw):
        ticket = self.source_loader.begin(**kw)
        self.tickets[kw['segment_id']] = ticket
        self.session.source_handles[kw['segment_id']] = {'source_id':kw['source_id'],'ticket':ticket}
        return ticket

    def commit_ready_segment(self, **kw):
        self.commit_calls.append(kw)
        self.session.commits[kw['segment_id']] = NS(source_id=self.tickets[kw['segment_id']].source_id)


class NativeV2ConsumptionTests(unittest.TestCase):
    def controlled_parent_case(self):
        store=self.store();candidate=self.candidate('birth');result=self.publish(store,candidate)
        c,issuer=self.context(store,candidate,tau=0.)
        with patch('probekv.native_consumption_v2.PhysicalLayerwiseSourceLoader',CPUTransport):
            c.configure_source_consumption_v2(issuer)
        c.engine.session.observe_pre_rope_k(1).add_(1)
        c.request['max_new_tokens']=1
        target=dict(segment_id='T',positions=[4],token_ids=[14])
        c.request['segments']=[dict(c.segments['C']),target]
        c.source_capture_v2=NS(targets={'T':(4,)},bind_parent=Mock(),close=Mock())
        receipt=issuer.compare_native(c,'C')
        self.assertGreater(dict(receipt.scores)[receipt.winner_source_id],0.)
        job=dict(operation='controlled_lineage_birth',input_origin='controlled_provenance_diagnostic',
            parent_preparation_mode='preregistered_source_correctness_only',
            economic_admission_evaluated=False,production_reuse_commit_observed=False,
            request=c.request.copy(),comparison_profile=asdict(issuer.profile),
            upstream_segment_id='C',expected_parent_generation=0,
            target_id='T',expected_target_generation=1,
            sources_by_segment={'C':[dict(source_id=result['source_id'])],'T':[]})
        return store,c,issuer,receipt,job

    def test_controlled_parent_does_not_relax_normal_threshold_or_production_commit(self):
        _,c,issuer,r,j=self.controlled_parent_case()
        with self.assertRaisesRegex(ValueError,'threshold'):c.prepare_selected_v2(r)
        ticket=c.source_consumption_v2.prepare_controlled_parent(r,job=j)
        self.assertEqual(ticket.source_id,r.winner_source_id)
        self.assertEqual(issuer.profile.tau_reuse,0.)
        audit=c.source_consumption_v2.audit()['controlled_recipe_parents']['C']
        self.assertFalse(audit['residual_admission_evaluated'])
        self.assertFalse(audit['production_execution_allowed'])
        with self.assertRaisesRegex(ValueError,'production commit'):
            c.source_consumption_v2.assert_can_commit('C')

    def test_controlled_parent_requires_declared_mode_matching_request_and_profile(self):
        from copy import deepcopy
        _,c,_,r,j=self.controlled_parent_case()
        for key,value in [('parent_preparation_mode','residual_admitted'),
                          ('operation','source_request'),('production_reuse_commit_observed',True)]:
            bad=deepcopy(j);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                c.source_consumption_v2.prepare_controlled_parent(r,job=bad)
        bad=deepcopy(j);bad['comparison_profile']['tau_reuse']=99.
        with self.assertRaises(ValueError):c.source_consumption_v2.prepare_controlled_parent(r,job=bad)
        bad=deepcopy(j);bad['sources_by_segment']['C']=[dict(source_id='wrong-source')]
        with self.assertRaisesRegex(ValueError,'preregistered Source'):
            c.source_consumption_v2.prepare_controlled_parent(r,job=bad)
        self.assertFalse(c.prepared)

    def test_controlled_parent_rejects_stale_receipt_and_wrong_generation(self):
        _,c,_,r,j=self.controlled_parent_case()
        c.generation+=1
        with self.assertRaisesRegex(ValueError,'stale'):
            c.source_consumption_v2.prepare_controlled_parent(r,job=j)
        self.assertFalse(c.prepared)
        c.generation-=1;j['expected_parent_generation']=1;j['expected_target_generation']=2
        with self.assertRaisesRegex(ValueError,'generation'):
            c.source_consumption_v2.prepare_controlled_parent(r,job=j)

    setUp = fixtures.TargetSourceStoreTests.setUp
    close_stores = fixtures.TargetSourceStoreTests.close_stores
    store = fixtures.TargetSourceStoreTests.store
    candidate = fixtures.TargetSourceStoreTests.candidate
    plan = fixtures.TargetSourceStoreTests.plan
    publish = fixtures.TargetSourceStoreTests.publish
    read_rows = fixtures.TargetSourceStoreTests.read_rows

    def context(self, store, candidate, *, tau=10.0, capacity=100000):
        c = NativeRequestContext.__new__(NativeRequestContext)
        c.request = dict(request_id='consumer', token_ids=[10,11,12,13,14,15], prefetch_window=0)
        c.segments = {'C':dict(segment_id='C',positions=[2,3],token_ids=[12,13],content_key='legacy-not-v2')}
        c.execution_inventory = {'C':NS(comparison_eligible=True,remaining_positions=(2,3))}
        current = torch.zeros((6,2,2),dtype=torch.bfloat16)
        current[2:4] = candidate['selection_states'][1]
        c.engine = CPUEngine(current)
        c.adapter = NS(provenance={'model_signature':'model','tokenizer_hash':'tokenizer'},
            hbm=UnifiedHBMReservationManager(allocator_capacity_bytes=capacity,safety_bytes=0),
            spec=NS(num_layers=3), costs=NS(sha='a'*64),
            inner=NS(layers=[NS(self_attn=NS(num_kv_heads=2,head_dim=2,hack_kv=[])) for _ in range(3)],
                     cache_fuse_metadata={},old_kvs=[]),
            native_repair_metric='normalized_kv_deviation',
            store_provider=Mock(side_effect=AssertionError('must not touch legacy exact pool')))
        c.native=NS(sequence=NS(seq_id=1))
        c.synchronize=Mock()
        c.finished=c.closed=c.selection_closed=False
        c.cached_prefix_tokens=0; c.probe_fallback_reason=None; c.generation=1
        c.source_capture_v2=c.source_capture_workspace_v2=c.source_consumption_v2=None
        c.prepared={};c.frozen={};c.committed={};c.supports={};c.replica_reservations={}
        c.hot_replicas={};c.hot_leases=ExitStack()
        c.workspace=c.capture_reservation=None
        c._observation={};c.repair_ratio=.15;c.actual_repair_check_sunk_ms=0
        snapshot=store.begin_request('consumer')
        profile=ComparisonProfileBindingV2('model',store.config['policy'],1,.15,tau,tau,
            'b'*64,runtime_binding_digest(),scorer_digest())
        comparison=ComparisonSessionV2(store,snapshot,profile,workspace_bytes=100000)
        self.addCleanup(comparison.close)
        self.addCleanup(c.close)
        return c,comparison

    def configured(self, *, mixed=False, capacity=100000, tau=10.):
        store=self.store()
        candidate=self.candidate('birth',mixed=mixed)
        result=self.publish(store,candidate)
        c,comparison=self.context(store,candidate,capacity=capacity,tau=tau)
        with patch('probekv.native_consumption_v2.PhysicalLayerwiseSourceLoader',CPUTransport):
            audit=c.configure_source_consumption_v2(comparison)
        self.assertFalse(audit['native_runtime_qualified'])
        receipt=comparison.compare_native(c,'C')
        return store,c,comparison,receipt,result

    def test_actual_receipt_prepares_target_without_legacy_pool_or_commit(self):
        store,c,comparison,receipt,result=self.configured()
        ticket=c.prepare_selected_v2(receipt)
        self.assertEqual(ticket.source_id,result['source_id'])
        self.assertEqual(ticket.segment_positions,(2,3))
        self.assertEqual(c.frozen,{'C':result['source_id']})
        self.assertFalse(c.committed)
        self.assertFalse(c.engine.commit_calls)
        self.assertEqual(store._lease_counts[result['source_id']],1)
        self.assertGreater(c.adapter.hbm.active_reserved_bytes,0)
        c.register_ready_hot_replicas()
        c.adapter.store_provider.assert_not_called()
        c.close()
        self.assertEqual(store._lease_counts[result['source_id']],0)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        c.adapter.store_provider.assert_not_called()
        store.end_request(comparison.snapshot)

    def test_unissued_changed_stale_and_over_threshold_receipts_do_not_prepare(self):
        store,c,comparison,receipt,_=self.configured(tau=0.)
        with self.assertRaises(ValueError):c.prepare_selected_v2(replace(receipt,winner_source_id='forged'))
        c.generation+=1
        with self.assertRaises(ValueError):c.prepare_selected_v2(receipt)
        c.generation-=1
        self.assertFalse(c.source_consumption_v2.loader.calls)
        self.assertFalse(c.frozen)
        # Zero threshold accepts the exact state above. A separate request with
        # a real nonzero observation is rejected by the bound threshold.
        c.close();store.end_request(comparison.snapshot);comparison.close()
        candidate=self.candidate('other-birth',base=50)
        c2,issuer2=self.context(store,candidate,tau=0.)
        with patch('probekv.native_consumption_v2.PhysicalLayerwiseSourceLoader',CPUTransport):
            c2.configure_source_consumption_v2(issuer2)
        bad=issuer2.compare_native(c2,'C')
        self.assertGreater(dict(bad.scores)[bad.winner_source_id],0.)
        with self.assertRaisesRegex(ValueError,'threshold'):c2.prepare_selected_v2(bad)
        self.assertFalse(c2.source_consumption_v2.loader.calls)

    def test_frozen_source_cannot_be_prepared_twice_or_replaced_at_closure(self):
        _,c,_,receipt,_=self.configured()
        ticket=c.prepare_selected_v2(receipt)
        with self.assertRaises(ValueError):c.prepare_selected_v2(receipt)
        with self.assertRaises(ValueError):c.finish_selection({'C':'other'},c.prepared)
        with self.assertRaises(RuntimeError):c.prepare_winner('C','other',(),None)
        c.finish_selection(c.frozen,{'C':ticket})
        self.assertTrue(c.selection_closed)

    def test_token_and_model_geometry_checked_before_source_freeze(self):
        _,c,_,receipt,_=self.configured()
        c.segments['C']['token_ids']=[999,998]
        with self.assertRaises(ValueError):c.prepare_selected_v2(receipt)
        self.assertFalse(c.frozen)
        self.assertFalse(c.source_consumption_v2.loader.calls)

    def test_real_lease_reservation_and_request_owner_required_for_copy(self):
        _,c,_,receipt,result=self.configured()
        c.prepare_selected_v2(receipt)
        bridge=c.source_consumption_v2
        reservation=c.replica_reservations['C']
        with self.assertRaises(ValueError):bridge.authorize(source_id='other',segment_id='C',bytes_required=96)
        reservation.owner_request_id='other-request'
        with self.assertRaises(ValueError):bridge.authorize(source_id=result['source_id'],segment_id='C',bytes_required=96)
        reservation.owner_request_id='consumer'
        with self.assertRaises(ValueError):bridge.authorize(source_id=result['source_id'],segment_id='C',bytes_required=1)

    def test_hbm_failure_frees_lease_and_never_uses_second_source(self):
        store,c,_,receipt,result=self.configured(capacity=0)
        with self.assertRaises(MemoryError):c.prepare_selected_v2(receipt)
        self.assertEqual(c.frozen,{'C':result['source_id']})
        self.assertEqual(store._lease_counts[result['source_id']],0)
        self.assertFalse(c.prepared)
        self.assertFalse(c.source_consumption_v2.loader.calls)
        with self.assertRaises(ValueError):c.prepare_selected_v2(receipt)

    def test_lease_open_failure_records_irreversible_attempt_without_unowned_exit(self):
        store,c,_,receipt,result=self.configured()
        lease=MagicMock()
        lease.__enter__.side_effect=OSError('SSD unavailable')
        with patch.object(store,'leased_target',return_value=lease):
            with self.assertRaisesRegex(OSError,'SSD unavailable'):c.prepare_selected_v2(receipt)
        self.assertEqual(c.frozen,{'C':result['source_id']})
        self.assertIn('SSD unavailable',c.source_consumption_v2.audit()['failed_segments']['C'])
        lease.__exit__.assert_not_called()
        self.assertEqual(store._lease_counts.get(result['source_id'],0),0)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)
        self.assertFalse(c.source_consumption_v2.loader.calls)
        with self.assertRaises(ValueError):c.prepare_selected_v2(receipt)

    def test_diagnostic_partial_comparison_does_not_freeze_winner(self):
        store=self.store()
        a=self.candidate('birth-a');b=self.candidate('birth-b',base=7)
        first=self.publish(store,a);self.publish(store,b)
        c,comparison=self.context(store,a)
        with patch('probekv.native_consumption_v2.PhysicalLayerwiseSourceLoader',CPUTransport):
            c.configure_source_consumption_v2(comparison)
        receipt=comparison.compare_native(c,'C',candidate_ids=(first['source_id'],))
        self.assertEqual((len(receipt.eligible_ids),len(receipt.compared_ids)),(2,1))
        with self.assertRaisesRegex(ValueError,'full eligible'):c.prepare_selected_v2(receipt)
        self.assertFalse(c.frozen)
        self.assertFalse(c.source_consumption_v2.loader.calls)

    def test_upstream_request_mutation_invalidates_receipt_before_freeze(self):
        _,c,_,receipt,_=self.configured()
        c.request['token_ids'][0]=999
        with self.assertRaisesRegex(ValueError,'stale observation'):c.prepare_selected_v2(receipt)
        self.assertFalse(c.frozen)

    def test_current_absolute_positions_may_differ_from_source_birth_positions(self):
        store=self.store();candidate=self.candidate('birth');result=self.publish(store,candidate)
        c,comparison=self.context(store,candidate)
        c.request['token_ids']=[10,12,13,20,14,15]
        c.segments['C'].update(positions=[1,2],token_ids=[12,13])
        c.execution_inventory['C'].remaining_positions=(1,2)
        current=torch.zeros((6,2,2),dtype=torch.bfloat16)
        current[1:3]=candidate['selection_states'][1]
        c.engine.session.observe_pre_rope_k=lambda d:current
        with patch('probekv.native_consumption_v2.PhysicalLayerwiseSourceLoader',CPUTransport):
            c.configure_source_consumption_v2(comparison)
        receipt=comparison.compare_native(c,'C')
        ticket=c.prepare_selected_v2(receipt)
        self.assertEqual(ticket.segment_positions,(1,2))
        self.assertEqual(tuple(store._catalog['rows'][result['source_id']]['birth_target_positions']),(2,3))

    def test_copy_failure_fences_before_lease_and_hbm_release(self):
        store,c,_,receipt,result=self.configured()
        c.source_consumption_v2.loader.fail=True
        seen=[]
        c.synchronize.side_effect=lambda:seen.append((store._lease_counts[result['source_id']],c.adapter.hbm.active_reserved_bytes))
        with self.assertRaises(RuntimeError):c.prepare_selected_v2(receipt)
        self.assertTrue(seen and seen[0][0]==1 and seen[0][1]>0)
        self.assertEqual(store._lease_counts[result['source_id']],0)
        self.assertEqual(c.adapter.hbm.active_reserved_bytes,0)

    def test_fence_failure_quarantines_resources_instead_of_releasing(self):
        store,c,_,receipt,result=self.configured()
        bridge=c.source_consumption_v2
        bridge.loader.fail=True
        c.synchronize.side_effect=RuntimeError('CUDA fence failed')
        def restore_test_fence_only():
            c.synchronize.side_effect=None
            bridge.quarantined=False
        self.addCleanup(restore_test_fence_only)
        with self.assertRaisesRegex(RuntimeError,'quarantined'):c.prepare_selected_v2(receipt)
        self.assertTrue(bridge.audit()['resources_quarantined'])
        self.assertEqual(store._lease_counts[result['source_id']],1)
        self.assertGreater(c.adapter.hbm.active_reserved_bytes,0)
        with self.assertRaises(RuntimeError):bridge.close()

    def test_final_commit_preserves_snapshot_admission_and_hbm_promotion(self):
        _,c,_,receipt,_=self.configured()
        c.prepare_selected_v2(receipt)
        c.finish_selection(c.frozen,c.prepared)
        c.ready_for_final_commit(c.prepared)
        snapshot=c.planner_snapshot(c.adapter.hbm.epoch)
        c.generation+=1
        with self.assertRaises(RuntimeError):c.commit_reuse(NS(planner_snapshot=snapshot,accepted_ready_segment_ids=('C',)))
        self.assertFalse(c.engine.commit_calls)
        current=c.planner_snapshot(c.adapter.hbm.epoch)
        c.commit_reuse(NS(planner_snapshot=current,accepted_ready_segment_ids=('C',)))
        self.assertEqual(c.committed,{'C':2})
        self.assertEqual(c.replica_reservations['C'].kind,HBMReservationKind.COMMITTED_EXECUTION)
        self.assertFalse(c.source_consumption_v2.audit()['final_commit_admission_bypassed'])

    def test_commit_rejects_unclosed_selection_missing_support_and_not_ready(self):
        _,c,_,receipt,_=self.configured()
        ticket=c.prepare_selected_v2(receipt)
        def commit():
            c.commit_reuse(NS(planner_snapshot=c.planner_snapshot(c.adapter.hbm.epoch),
                              accepted_ready_segment_ids=('C',)))
        with self.assertRaisesRegex(ValueError,'selection closure'):commit()
        c.finish_selection(c.frozen,c.prepared)
        with self.assertRaisesRegex(ValueError,'repair support'):commit()
        c.ready_for_final_commit(c.prepared)
        ticket.layer_events[2]=NS(query=lambda:False)
        with self.assertRaisesRegex(ValueError,'readiness'):commit()
        ticket.layer_events[2]=CPUEvent()
        c.supports['C'][2]=(999,)
        with self.assertRaisesRegex(ValueError,'support'):commit()
        self.assertFalse(c.engine.commit_calls)

    def test_strict_publication_context_method_uses_issued_receipt_bridge(self):
        _,c,comparison,receipt,_=self.configured()
        with patch('probekv.native_publication_v2.publish_compared_targets_v2',return_value={'delegated':True}) as publish:
            self.assertEqual(c.publish_compared_targets_v2(comparison,{'C':receipt}),{'delegated':True})
        publish.assert_called_once_with(c,comparison,{'C':receipt})

    def test_consumption_only_never_offers_legacy_publication_or_extra_forward(self):
        _,c,_,_,_=self.configured()
        c.adapter.build_exact_dense_source=Mock(side_effect=AssertionError('no extra forward'))
        c.canonical_exports={'C':'legacy-placeholder'}
        self.assertIsNone(c.source_capture_v2)
        self.assertEqual(c.export_exact_dense(),{})
        self.assertEqual(c.deferred_canonical_builders(),{})
        self.assertEqual(c.materialization_metadata(),{})
        c.adapter.build_exact_dense_source.assert_not_called()
        c.adapter.store_provider.assert_not_called()

    def test_mixed_parent_is_bound_as_g1_and_its_new_dependency_remains_g2(self):
        store,c,_,receipt,result=self.configured(mixed=True)
        capture=NativeTargetCaptureV2(request_id='consumer',token_ids=c.request['token_ids'],
            model_signature='model',authorization_domain='domain',num_layers=3,
            targets={'later':(4,5)},selection_depths=(1,2),host_budget_bytes=10000,
            kv_heads=2,head_dim=2,registry=self.registry)
        c.source_capture_v2=capture
        c.prepare_selected_v2(receipt)
        parent=capture.parents[result['source_id']]
        self.assertEqual((parent.origin,parent.generation),(SourceOrigin.MIXED,1))
        for layer in (1,2,3):
            rows=tuple(range(6)) if layer==1 else (0,1,3,4,5)
            capture.ledger.record_layer(layer_1based=layer,qkv_rows=rows,attention_rows=rows,
                output_mlp_rows=rows,imported_kv=() if layer==1 else (KVImport(2,parent.source_id,parent.generation),),
                completion_reference='cpu-proof-only')
        self.assertEqual(capture.ledger.target_proof((4,5)).generation,2)
        with self.assertRaises(ValueError):c.bind_capture_parent_v2(replace(parent,generation=0,origin=SourceOrigin.EXACT))
        shape=c.source_measurement_shape('C',result['source_id'],1)
        self.assertEqual(shape['source_origin'],'MIXED_CONTEXT_FULL_SEGMENT')
        self.assertEqual(shape['source_generation'],1)
        self.assertEqual(shape['source_contract'],'target_source_v2')
        c.adapter.store_provider.assert_not_called()

    def test_prefix_hot_cache_and_windowed_ssd_are_explicitly_rejected(self):
        store=self.store();candidate=self.candidate('birth');self.publish(store,candidate)
        c,comparison=self.context(store,candidate)
        for field,value in (('prefetch_window',1),('use_gpu_hot_cache',True),('retain_gpu_hot_cache',True),
                            ('capture_original_full_prefill',True),('publish_exact_prefix_shadow',True)):
            c.request[field]=value
            with self.assertRaises(ValueError):c.configure_source_consumption_v2(comparison)
            c.request.pop(field)
        c.cached_prefix_tokens=1
        with self.assertRaises(ValueError):c.configure_source_consumption_v2(comparison)
        c.cached_prefix_tokens=0
        c.capture_reservation=object()
        with self.assertRaises(ValueError):c.configure_source_consumption_v2(comparison)
        c.capture_reservation=None
        c.native.prefix_shadow=('legacy-whole-prefix',)
        with self.assertRaises(ValueError):c.configure_source_consumption_v2(comparison)
        c.native.prefix_shadow=None
        # No configured v2 bridge means close follows the legacy route; the
        # test does not provide or permit a legacy store, so explicitly use a
        # valid bridge for ordinary lifecycle cleanup.
        with patch('probekv.native_consumption_v2.PhysicalLayerwiseSourceLoader',CPUTransport):
            c.configure_source_consumption_v2(comparison)


if __name__=='__main__':unittest.main()
